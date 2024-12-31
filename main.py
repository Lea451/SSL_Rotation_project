import argparse
from config.config import config
from dataloader import get_dataloader
from models.ResNet18 import create_model as create_resnet
from models.Classifier import create_model as create_classifier
from train import train, get_optim, get_loss, plot_losses
import torch

def main():
    # Argument parsing
    parser = argparse.ArgumentParser(description="Self-Supervised Learning Project")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "test"], help="Execution mode")
    args = parser.parse_args()

    # Combine configurations
    opt = {**config['train_config'], **config['model_config']}

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
        dataset = CIFAR10(root=opt["data"]["dataset_path"], train=True, download=True)
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
    if args.mode == "train":
        print("Starting training...")
        losses = train(model, train_loader, optim, loss_fn, epochs=opt["train"]["epochs"])
        plot_losses(losses)
    elif args.mode == "test":
        print("Testing mode is not implemented yet.")

if __name__ == "__main__":
    main()
