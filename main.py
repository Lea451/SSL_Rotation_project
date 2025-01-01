import os
import importlib.util
import argparse
import config
from dataloader import get_dataloader
from models.ResNet18 import create_model as create_resnet
from models.Classifier import create_model as create_classifier
from train import train, get_optim, get_loss, plot_losses
import torch

def main():
    # Argument parsing
    parser = argparse.ArgumentParser(description="Self-Supervised Learning Project")
    parser.add_argument("--exp",         type=str,  required=True, help="config file name without .py extension")
    parser.add_argument('--evaluate',    type=bool, default=False, help='Evaluate the model instead of training')
    parser.add_argument('--checkpoint',  type=int,  default=0,     help='checkpoint (epoch id) that will be loaded')
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

    # Initialize model
    if opt["model"]["type"] == "ResNet18":
        model = create_resnet(opt["model"])
    elif opt["model"]["type"] == "Classifier":
        model = create_classifier(opt["model"])
    else:
        raise ValueError(f"Unknown model type: {opt['model']['type']}")

    # Prepare DataLoader
    dataset_name = opt["data"]["dataset_name"]
    if dataset_name == "CIFAR10":
        from torchvision.datasets import CIFAR10
        dataset = CIFAR10(root=opt["data"]["dataset_path"], train=False, download=True) #do not train by default !
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    train_loader = get_dataloader(
        dataset=dataset,
        batch_size=opt["train"]["batch_size"],
        unsupervised=opt["data"]["unsupervised"],
        num_workers=opt["train"]["num_workers"],
        shuffle=opt["train"]["shuffle"]
    )

    # Optimizer and Loss Function
    optim = get_optim(model, opt["train"])
    loss_fn = get_loss(opt["train"])

    # Train or Test
    if args.evaluate == False:
        print("Starting training...")
        losses = train(model, train_loader, optim, loss_fn, epochs=opt["train"]["epochs"])
        plot_losses(losses)
    else:
        print("Testing mode is not implemented yet.")

if __name__ == "__main__":
    main()


#on how to load the weights of a specific checkpoint:
#checkpoint = torch.load("checkpoint_epoch_3.pt")  # Example for epoch 3
#model.load_state_dict(checkpoint['model_state_dict'])
#optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
#epoch = checkpoint['epoch']
#train_loss = checkpoint['train_loss']

#print(f"Resumed from epoch {epoch}, Train Loss: {train_loss}")
