import os
import importlib.util
import argparse
import config
from dataloader import get_dataloader
from models.ResNet18 import create_model as create_resnet
from models.ClassifierModified import create_model as create_classifier
from train import train, test, get_optim, get_loss, plot_losses
import matplotlib.pyplot as plt
import torch

def main():
    # Argument parsing
    parser = argparse.ArgumentParser(description="Self-Supervised Learning Project")
    parser.add_argument("--exp",         type=str,  required=True, help="config file name without .py extension")
    parser.add_argument('--evaluate',    type=bool, default=False, help='Evaluate the model instead of training')
    parser.add_argument('--checkpoint',  type=int,  default=0,     help='checkpoint (epoch id) that will be loaded')
    parser.add_argument('--loadfrom',      type=str,  default='', help='path to classifier mdodel to test')
    args = parser.parse_args()

    # Combine configurations
    # Construct the path to the configuration file
    exp_config_file = os.path.join('.', 'config', args.exp + '.py')

    # Use importlib to load the module dynamically
    spec = importlib.util.spec_from_file_location("config", exp_config_file)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)

    #create opt dictionnary
    opt = config.opt
    
    opt['model']['loaded_from_epoch'] = args.checkpoint
    print("opt['model']['loaded_from_epoch']", opt['model']['loaded_from_epoch'])
    
    print(opt.keys())
    print("opt['model']", opt['model'])
    print("opt['model'].keys()", opt['model'].keys())

    # Initialize model
    if opt["model"]["type"] == "ResNet18":
        model = create_resnet(opt['model'])
    elif opt["model"]["type"] == "ClassifierModified" and opt["data"]["dataset_type"] == "CIFAR":
        #in that case we need to load the pretext model first from the checkpoint
        pretext_model = create_resnet(opt['pretext_model'])
        optim = get_optim(pretext_model, opt["train"])

        checkpoint_path = f"checkpoints/ResNet18/checkpoint_epoch_{args.checkpoint}.pt"  # Format the checkpoint file path
        print(f"Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path)  # Load the checkpoint file

        pretext_model.load_state_dict(checkpoint['model_state_dict'])  # Load model weights
        optim.load_state_dict(checkpoint['optimizer_state_dict'])  # Load optimizer state
        start_epoch = checkpoint['epoch']  # Load the saved epoch
        val_loss = checkpoint['valid_loss']  # Load the saved loss (optional)
        
        model = create_classifier(pretext_model, opt['model'])  # Use the pretext model to initialize the classifier

    else:
        pretext_model = create_resnet(opt['pretext_model'])
        optim = get_optim(pretext_model, opt["train"])

        checkpoint_path = f"checkpoints/checkpoint_epoch_{args.checkpoint}.pt"  # Format the checkpoint file path
        print(f"Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path)  # Load the checkpoint file

        pretext_model.load_state_dict(checkpoint['model_state_dict'])  # Load model weights
        optim.load_state_dict(checkpoint['optimizer_state_dict'])  # Load optimizer state
        start_epoch = checkpoint['epoch']  # Load the saved epoch
        val_loss = checkpoint['valid_loss']  # Load the saved loss (optional)
        print(f"Val Loss: {val_loss:.4f}")
        
        model = create_classifier(pretext_model, opt['model'])  # Use the pretext model to initialize the classifier

    print(f"nb of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    print(opt['device'])
    # Prepare DataLoader
    dataset_name = opt["data"]["dataset_name"]
    if dataset_name == "CIFAR10":
        from torchvision.datasets import CIFAR10
        train_full = CIFAR10(root=opt["data"]["dataset_path"], train=True, download=True) #train set 
        test_dataset = CIFAR10(root=opt["data"]["dataset_path"], train=False, download=True) #test set
        train_dataset, valid_dataset = torch.utils.data.random_split(train_full, [0.85, 0.15])
    else:
        from torchvision.datasets import Flowers102
        train_dataset = Flowers102(root=opt["data"]["dataset_path"], split='train', download=True) #train set
        valid_dataset = Flowers102(root=opt["data"]["dataset_path"], split='val', download=True) #validation set
        test_dataset = Flowers102(root=opt["data"]["dataset_path"], split='test', download=True) #test set
    
    
    train_loader = get_dataloader(
        dataset=train_dataset,
        batch_size=opt["train"]["batch_size"],
        unsupervised=opt["data"]["unsupervised"],
        num_workers=opt["train"]["num_workers"],
        shuffle=opt["train"]["shuffle"],
        dataset_type=opt["data"]["dataset_type"]
    )
    valid_loader = get_dataloader(
        dataset=valid_dataset,
        batch_size=opt["train"]["batch_size"],
        unsupervised=opt["data"]["unsupervised"],
        num_workers=opt["train"]["num_workers"],
        shuffle=opt["train"]["shuffle"],
        dataset_type=opt["data"]["dataset_type"]
    )
    test_loader = get_dataloader(
        dataset=test_dataset,
        batch_size=opt["train"]["batch_size"],
        unsupervised=opt["data"]["unsupervised"],
        num_workers=opt["train"]["num_workers"],
        shuffle=opt["train"]["shuffle"],
        dataset_type=opt["data"]["dataset_type"]
    )
   
    # Optimizer and Loss Function
    optim = get_optim(model, opt["train"])
    loss_fn = get_loss(opt["train"])

    # Train or Test
    if args.evaluate == False:
        print("Starting training...")
        train_losses, val_losses = train(model, train_loader, valid_loader, optim, loss_fn, opt, epochs=opt["train"]["epochs"])
        plot_losses(train_losses, val_losses)
    else:
        print("Starting testing...")
        #load the model to test 
        if opt["model"]["type"] == "ResNet18": #test the resnet18 model
            checkpoint = torch.load(f"checkpoints/ResNet18/checkpoint_epoch_{args.checkpoint}.pt")
        else: #test the classifier model
            #checkpoint = torch.load(f"checkpoints/{opt['model']['type']}_layer{opt['model']['num_couche']}_feat/checkpoint_epoch_{args.checkpoint}.pt")
            model_path = args.loadfrom
            model.load_state_dict(torch.load(model_path))
        model.load_state_dict(checkpoint['model_state_dict'])
        losses = test(model, train_loader, loss_fn, opt)
        

if __name__ == "__main__":
    main()
    
    

    


#on how to load the weights of a specific checkpoint:
#checkpoint = torch.load("checkpoint_epoch_3.pt")  # Example for epoch 3
#model.load_state_dict(checkpoint['model_state_dict'])
#optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
#epoch = checkpoint['epoch']
#train_loss = checkpoint['train_loss']

#print(f"Resumed from epoch {epoch}, Train Loss: {train_loss}")
