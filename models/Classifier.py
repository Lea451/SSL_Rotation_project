import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, feat):
        return feat.view(feat.size(0), -1)

class Classifier(nn.Module):
    def __init__(self, opt): #the argument is opt["model"]
        super(Classifier, self).__init__()
        num_couche = opt['num_couche']
        if num_couche == 0:
            nChannels = 512
        elif num_couche == 1:
            nChannels = 256
        elif num_couche == 2:
            nChannels = 128
        elif num_couche == 3:
            nChannels = 64
        num_classes = opt['num_classes']
        pool_size = opt['pool_size']
        pool_type = opt['pool_type'] if ('pool_type' in opt) else 'max'
        nChannelsAll = nChannels * pool_size * pool_size

        # Pooling layer
        if pool_type == 'max':
            self.pool = nn.AdaptiveMaxPool2d((pool_size, pool_size))
        elif pool_type == 'avg':
            self.pool = nn.AdaptiveAvgPool2d((pool_size, pool_size))
        else:
            raise ValueError("Invalid pool_type. Use 'max' or 'avg'.")

        # Classifier layers
        self.batch_norm = nn.BatchNorm2d(nChannels, affine=False)
        self.flatten = Flatten()
        self.linear = nn.Linear(nChannelsAll, num_classes)
        self.initialize_weights()

    def forward(self, feat):
        feat = self.pool(feat)
        feat = self.batch_norm(feat)
        feat = self.flatten(feat)
        out = self.linear(feat)
        return out

    #initialization of linear layer
    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0.0, np.sqrt(2.0/m.out_features)) #Assign random values to the weights of the linear layer using a normal distribution using standard value for std (Kaiming Initialization for ReLu)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

def create_model(opt):
    return Classifier(opt)
