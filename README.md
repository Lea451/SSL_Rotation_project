# SSL_Rotation_project

**Reimplementation of the paper "Unsupervised representation learning by predicting image rotations" by Gidaris et al.**

Self-supervised learning (SSL) is where a model learns useful representations without requiring labeled data. 
In this project, we use image rotation prediction as the pretext task: the model is trained to recognize whether an image has been rotated by 0, 90, 180, or 270 degrees. 
By solving this task, the model learns robust image features that can be transferred to downstream tasks.

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
   
   Train a Linear Classifier on top of a pre-trained RotNet on CIFAR10: main.py --exp CIFAR10_LinearClassifier_on_Rotnet_Resnet18_layer[layer number]_feat --checkpoint [checkpoint number of the RotNet]
   Train a ConvClassifier on top of a pre-trained RotNet on CIFAR10: main.py --exp CIFAR10_ConvClassifier_on_Rotnet_Resnet18_layer[layer number]_feat --checkpoint [checkpoint number of the RotNet]

   Train a Linear Classifier on top of a pre-trained RotNet on Flowers: main.py --exp Flowers_LinearClassifier_on_Rotnet_Resnet18_layer[layer number]_feat --checkpoint [checkpoint number of the RotNet]
   Train a ConvClassifier on top of a pre-trained RotNet on Flowers: main.py --exp Flowers_ConvClassifier_on_Rotnet_Resnet18_layer[layer number]_feat --checkpoint [checkpoint number of the RotNet]
   
**Note:**
- `layer number` refers to the ResNet-18 layer on top of which we add our classifier.
- `checkpoint number` is the saved model checkpoint index from a previous RotNet training run. 

### Testing
   
   Test a pre-trained RotNet + Linear Classifier on CIFAR10: main.py --exp CIFAR10_LinearClassifier_on_Rotnet_Resnet18_layer[layer number]_feat --evaluate True --checkpoint [checkpoint number of the RotNet]
   Test a pre-trained RotNet + ConvClassifier on CIFAR10: main.py --exp CIFAR10_ConvClassifier_on_Rotnet_Resnet18_layer[layer number]_feat --evaluate True --checkpoint [checkpoint number of the RotNet]

   Test a pre-trained RotNet + Linear Classifier on Flowers: main.py --exp Flowers_LinearClassifier_on_Rotnet_Resnet18_layer[layer number]_feat --evaluate True --checkpoint [checkpoint number of the RotNet]
   Test a pre-trained RotNet + ConvClassifier on Flowers: main.py --exp Flowers_ConvClassifier_on_Rotnet_Resnet18_layer[layer number]_feat --evaluate True --checkpoint [checkpoint number of the RotNet]

## Results

### Experimental Setup

- **RotNet Structure**:
  - ResNet-18
  - Linear or Convolutional Classifier

- **Optimizer**: Adam with default parameters  
- **Loss**: Cross-Entropy  
- **Epochs**: 100  
- **Batch Size**: 32  


### CIFAR-10 Dataset

#### Key Metrics:
- **Rotation prediction accuracy**: 95.39%
- **Downstream classification accuracy (Linear Classifier)**: 55.15%
- **Downstream classification accuracy (Convolutional Classifier)**: **95.49%**

#### Classification Evaluation

| **Layer (Numcouche)** | **Test Loss** | **Accuracy (%)** |
|----------------------|-------------|----------------|
| AvgPooling (0)       | 0.496       | 87.09          |
| Conv Layer 4 (1)     | 0.362       | 93.77          |
| Conv Layer 3 (2)     | 0.335       | 94.39          |
| Conv Layer 2 (3)     | **0.215**   | **95.49**      |
| Conv Layer 1 (4)     | 0.393       | 88.32          |

> **Note:** This table presents the classification evaluation when placing our **ConvClassifier** on top of different layers of our **ResNet18 model**.

### Oxford Flowers 102 Dataset

#### Classification Evaluation

| **Task** | **Accuracy (%)** |
|----------------------|-------------|
| Rotation Prediction     | 43.89     |
| Classification Prediction  | 23.94       | 

We confirm that with a mainly rotation-invariant dataset, the model does not
learn anything. 

## Conclusion and Future Work

### **Conclusion**
- RotNet effectively learns transferable features.
- A simple yet powerful method for self-supervised learning.

### **Future Work**
- Explore larger datasets (e.g., ImageNet).
- Extend to other geometric transformations.

---

## References
1. Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. *Deep Residual Learning for Image Recognition*. CVPR 2016.





