import tqdm
import torch
import torch.nn as nn
from torch.optim import SGD
import matplotlib.pyplot as plt


def get_optim(model, opt):
    ''' Returns SGD Optimizer with given parameters '''
    #il faudra mettre les paramètres de opt, mais il faudra aussi configurer opt pour que ça marche
    return SGD(
        model.parameters(),
        lr=0.1,
        momentum=0.9,
        nesterov=True,
        weight_decay=5e-4,
    )

def get_loss(opt):
    ''' Returns CrossEntropyLoss '''
    #même remarque
    return nn.CrossEntropyLoss()

def train(model, train_loader, optim, loss_fn, epochs=1):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for epoch in range(epochs):
        model.train()
        losses = []

        for data, target in tqdm.tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs} - Training"):
            data, target = data.to(device), target.to(device)
            optim.zero_grad()
            output = model(data)
            loss = loss_fn(output, target)
            loss.backward()
            optim.step()
            losses.append(loss.item())

    return losses

def plot_losses(losses):
    ''' Plots the training losses '''
    plt.figure(figsize=(10, 6))
    plt.plot(losses, label="Batch Loss")
    plt.xlabel("Batch")
    plt.ylabel("Loss")
    plt.title("Loss Per Batch During Training")
    plt.legend()
    plt.grid()
    plt.show()

