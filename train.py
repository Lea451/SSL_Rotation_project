import os
import tqdm
import torch
import torch.nn as nn
from torch.optim import SGD
import matplotlib.pyplot as plt


def get_optim(model, opt):
    ''' Returns SGD Optimizer with given parameters '''
    #il faudra mettre les paramètres de opt, mais il faudra aussi configurer opt pour que ça marche
    if opt['optimizer'] == 'Adam': 
        return torch.optim.Adam(model.parameters(), lr=0.0001) #j'ai hardcodé le lr pour l'instant
    else: #idem on peut laisser les paramètres de SGD hardcodé pour le moment
        return SGD(
        model.parameters(),
        lr=0.1,
        momentum=0.9,
        nesterov=True,
        weight_decay=5e-4,
    )

def get_loss(opt):
    ''' Returns CrossEntropyLoss '''
    #même remarque #ici on peut laisser les paramètres par défaut pour l'instant
    return nn.CrossEntropyLoss()



def train(model, train_loader, valid_loader, optim, loss_fn, opt, epochs=1):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    best_loss = float('inf')
    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        for data, target in tqdm.tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs} - Training"):
            data, target = data.to(device), target.to(device)
            optim.zero_grad()
            output = model(data)
            loss = loss_fn(output, target)
            loss.backward()
            optim.step()
            train_loss += loss.item() * data.size(0)
            
        train_loss /= len(train_loader.dataset)
        train_losses.append(train_loss)
        
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for data, target in tqdm.tqdm(valid_loader, desc=f"Epoch {epoch + 1}/{epochs} - Validation"):
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = loss_fn(output, target)
                val_loss += loss.item() * data.size(0)
        
        val_loss /= len(valid_loader.dataset)
        val_losses.append(val_loss)
        
        if val_loss < best_loss:
            best_loss = val_loss
            #join the epoch number to the model_save_path
            os.makedirs(opt['model']['model_save_path'], exist_ok=True)
            path = os.path.join(opt['model']['model_save_path'], f"_epoch_{epoch + 1}.pth")
            torch.save(model.state_dict(), path)
            print(f"Model saved with Val Loss: {best_loss:.4f}")
    

        # Extract directory path (without the filename)
        if opt['model']['type'] == "ResNet18":
            checkpoint_dir = f"checkpoints/ResNet18"
        else:
            checkpoint_dir = f"checkpoints/{opt['model']['type']}_layer{opt['model']['num_couche']}_feat"
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch + 1}.pt")
    
        print(f"checkpoint_path: {checkpoint_path}")
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optim.state_dict(),
            'train_loss': train_loss,
            'valid_loss': val_loss,
        }, checkpoint_path)
        print(f"Saved checkpoint: {checkpoint_path}")

        
        
        ## Save checkpoint after every epoch
        #checkpoint_path = f"checkpoints/checkpoint_epoch_{epoch + 1}.pt"
        #torch.save({
        #    'epoch': epoch + 1,
        #    'model_state_dict': model.state_dict(),
        #    'optimizer_state_dict': optim.state_dict(),
        #    'train_loss': train_loss,
        #    'valid_loss': val_loss,
        #}, checkpoint_path)
        #print(f"Saved checkpoint: {checkpoint_path}")
        
        
    return train_losses, val_losses

def test(model, test_loader, loss_fn, opt):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()  # Set the model to evaluation mode

    test_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in tqdm.tqdm(test_loader, desc="Testing"):
            data, target = data.to(device), target.to(device)
            output = model(data)

            # Compute loss
            loss = loss_fn(output, target)
            test_loss += loss.item() * data.size(0)

            # Compute accuracy
            _, predicted = torch.max(output, dim=1)
            correct += (predicted == target).sum().item()
            total += target.size(0)

    # Calculate average loss and accuracy
    test_loss /= len(test_loader.dataset)
    accuracy = 100.0 * correct / total

    # Save test results
    #create the path to save the results
    os.makedirs(opt['model']['results_save_path'], exist_ok=True)
    results_path = os.path.join(opt['model']['results_save_path'], "results.txt")
    with open(results_path, "w") as f:
        f.write(f"Test Loss: {test_loss:.4f}\n")
        f.write(f"Test Accuracy: {accuracy:.2f}%\n")

    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {accuracy:.2f}%")

    return test_loss, accuracy


def plot_losses(train_losses, valid_losses):
    ''' Plots the training and validation losses per epoch '''
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label="Training Loss", marker='o')
    plt.plot(valid_losses, label="Validation Loss", marker='o')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss Per Epoch")
    plt.legend()
    plt.grid()
    plt.show()

