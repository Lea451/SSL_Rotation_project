import torch

# Training Configuration
train_config = {
    "train": {
        "batch_size": 64,
        "num_workers": 4,
        "shuffle": True,
        "learning_rate": 0.1,
        "momentum": 0.9,
        "weight_decay": 0.0005,
        "nesterov": True,
        "epochs": 10,
    },
    "data": {
        "dataset_name": "CIFAR10",
        "dataset_path": "./data",
        "unsupervised": True,
    },
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}

# Model Configuration
model_config = {
    "model": {
        "type": "ResNet18",  # Options: "ResNet18" or "Classifier"
        "num_classes": 4,    # Number of output classes
        "pool_size": 1,      # Pooling size for Classifier
        "pool_type": "avg",  # Options: "max" or "avg"
    }
}
