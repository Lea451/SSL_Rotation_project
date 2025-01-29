# SSL_Rotation_project

**Reimplementation of the paper "Unsupervised representation learning by predicting image rotations"**

This project uses a **self-supervised learning** approach to train a model to predict the rotation angle applied to an image as part of unsupervised representation learning.

---

## Project Structure

| **Folder/File**             | **Description**                                                                                                                                                                   |
|-----------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `config/`                  | Contains configuration files defining hyperparameters, data paths, architectures... etc, for each experiment                                                                                                           |
| `models/`                  | Includes the architectures of the models used for predicting image rotations.                                                                                                     |
| `notebooks/`               | Jupyter notebooks for data analysis and result visualization.                                                                                                                     |
| `data.ipynb`               | Demo notebook for exploring and visualizing input data, and check the rotations transformations.                                                                                                                                |
| `dataloader.py`            | Script for loading and preprocessing training and testing data in dataloader format.                                                                                                                   |
| `train.py`                 | Implements the model training loop, including loss definition and optimization.                                                                                                   |
| `main.py`                  | Main entry point for running training or model evaluation.                                                                                                                        |
| `.gitignore`               | Files and directories excluded from Git tracking.                                                                                                                                |
| `README.md`                | Document explaining the project and its main functionalities.                                                                                                                     |

---

## Installation and Usage

### Data Preparation
1. No specific data preparation is needed, CIFAR10 and OXFORD101 will be downloaded if not already.

   
### Training
1. Configure training parameters in `config/`.
2. Start training with:
   ```bash
   Train a RotNet on CIFAR10: main.py --exp CIFAR10_Rotnet_Resnet18
   Train a RotNet on Oxford Flowers 102 : main.py --exp Flowers_Rotnet_Resnet18

   Train a ConvClassifier on top of a pre-trained RotNet on CIFAR10: main.py --exp CIFAR10_LinearClassifier_on_Rotnet_Resnet18_layer[layer number]_feat --checkpoint [checkpoint number of the RotNet]


### Testing
Start : python main.py --evaluate = True --config=/...
