import torch

config = {}

# Training Configuration
train_config = {
    "train": {
        "batch_size": 64,
        "num_workers": 4,
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
        "type": "Classifier",  # Options: "ResNet18" or "Classifier"
        "num_classes": 10,    # Number of output classes : here normal classification task
        "pool_size": 1,      # Pooling size for Classifier
        "pool_type": "avg",  # Options: "max" or "avg"
    }
}
config['model_config'] = model_config