import torch

config = {}
opt = {}

# Training Configuration
train_config = {
    "train": {
        "batch_size": 64,
        "num_workers": 0, #num_workers=0 pour le moment pour éviter les problèmes de multiprocessing
        "shuffle": True,
        "optimizer":'Adam',
        "learning_rate": 0.0001, #0.1 for SGD, 0.0001 for Adam
        "momentum": 0.9,
        "weight_decay": 0.0005,
        "nesterov": True,
        "epochs": 10,
    },
    "data": {
        "dataset_name": "CIFAR10",
        "dataset_path": "./data", #root where we will put downloaded not trained dataset 
        "download": True,
        "unsupervised": False, #for supervised classification task
    },
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}
config['train_config'] = train_config

# Model Configuration
model_config = {
    "model": {
        "type": "ClassifierModified",  # Options: "ResNet18" or "Classifier"
        "num_classes": 10,    # Number of output classes : here normal classification task
        "pool_size": 1,      # Pooling size for Classifier
        "pool_type": "avg",  # Options: "max" or "avg"
        "num_couche": 1,     # Number of conv layer to get rid of, here = 1 so get rid of layer4
    }
}
config['model_config'] = model_config

opt = {**config['train_config'], **config['model_config']}