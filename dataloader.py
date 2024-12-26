import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
import random
from torchvision.datasets import CIFAR10
import matplotlib.pyplot as plt


# Denormalization class
class Denormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor


#on considère qu'on utilise la classe sur des dataset directement (Pytorch), donc on va load les données avec Pytorch
class CIFAR(Dataset):
    def __init__(self, dataset, transform=None, unsupervised=True):
        self.dataset = dataset
        self.transform = transform
        self.unsupervised = unsupervised
        
        # Define the mean and std for normalization
        self.mean_pix = [125.3 / 255.0, 123.0 / 255.0, 113.9 / 255.0]
        self.std_pix = [63.0 / 255.0, 62.1 / 255.0, 66.7 / 255.0]

        # Define transformations
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean_pix, std=self.std_pix),
        ])
        self.inv_transform = transforms.Compose([
            Denormalize(self.mean_pix, self.std_pix),
            lambda x: x.numpy() * 255.0,
            lambda x: x.transpose(1, 2, 0).astype(np.uint8),
        ])

    def __getitem__(self, index):
        img, label = self.dataset[index]

        if self.unsupervised:
            # Generate rotated images
            rotated_imgs = [
                self.transform(img),
                self.transform(rotate_img(img, 90).copy()),
                self.transform(rotate_img(img, 180).copy()),
                self.transform(rotate_img(img, 270).copy()),
            ]
            # Generate the rotation labels
            rotation_labels = torch.LongTensor([0, 1, 2, 3])
            return torch.stack(rotated_imgs, dim=0), rotation_labels
        else:
            # In supervised mode, no rotations 
            img = self.transform(img)
            return img, label

    def __len__(self):
        return len(self.dataset)

#on n'a pas besoin de créer une nouvelle classe Dataloader
def get_dataloader(dataset, batch_size=1, unsupervised=True, num_workers=0, shuffle=True):
    # Define transforms
    mean_pix = [125.3 / 255.0, 123.0 / 255.0, 113.9 / 255.0]
    std_pix = [63.0 / 255.0, 62.1 / 255.0, 66.7 / 255.0]
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean_pix, std=std_pix),
    ])

    # Create the dataset
    rotated_dataset = CIFAR(dataset=dataset, transform=transform, unsupervised=unsupervised)

    # Custom collate function for unsupervised mode
    def custom_collate_fn(batch):
        if unsupervised:
            images, labels = zip(*batch)
            images = torch.cat(images, dim=0)  # Flatten the rotated images
            labels = torch.cat(labels, dim=0)  # Flatten the labels
            return images, labels
        else:
            return torch.utils.data.dataloader.default_collate(batch)

    # Create DataLoader
    data_loader = DataLoader(
        dataset=rotated_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=custom_collate_fn if unsupervised else None,
    )
    

    return data_loader # return images, labels


"""
Takes in an image and a rotation. Returns the the image with the rotation applied.
"""
def rotate_img(img, rot):
    if rot == 0: # 0 degrees rotation
        return img
    elif rot == 90: # 90 degrees rotation
        return np.flipud(np.transpose(img, (1,0,2)))
    elif rot == 180: # 90 degrees rotation
        return np.fliplr(np.flipud(img))
    elif rot == 270: # 270 degrees rotation / or -90
        return np.transpose(np.flipud(img), (1,0,2))
    else:
        raise ValueError('rotation should be 0, 90, 180, or 270 degrees')
    
    
    
def retrieve_original_image(train_loader):# Assume train_loader is already created and outputs shape [batch_size*4, channels, height, width]
    images, rotation_labels = next(iter(train_loader))  # Fetch a batch of data

    # Number of images per batch and rotations per image
    batch_size = images.shape[0] // 4

    # Reshape to group by original images: [batch_size, 4, channels, height, width]
    grouped_images = images.view(batch_size, 4, *images.shape[1:])  # Now each entry has 4 rotations

    # De-transform and visualize each set of rotations for one image
    for i in range(batch_size):  # Iterate over each original image in the batch
        fig, axes = plt.subplots(1, 4, figsize=(12, 3))
        
        for j in range(4):  # Iterate over the 4 rotated versions of the image
            # De-transform the image
            de_transformed_image = train_loader.dataset.inv_transform(grouped_images[i, j])

            # Visualize
            axes[j].imshow(de_transformed_image)  # Ensure [H, W, C]
            axes[j].set_title(f"Rotation {j * 90}°")
            axes[j].axis("off")
        
        plt.suptitle(f"Original Image {i + 1} with Rotations")
        plt.show()

        # Break after visualizing the first image in the batch (optional)
        break