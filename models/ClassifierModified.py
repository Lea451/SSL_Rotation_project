import torch
import torch.nn as nn
from models.ResNet18 import ResNet18 
from models.Classifier import Classifier  
from models.ConvClassifier import ConvClassifier 


class ClassifierModified(nn.Module):
    def __init__(self, pretext_model, opt):
        super(ClassifierModified, self).__init__()
        
        # Use layers of pretext model up to (but excluding) layer4
        self.feature_extractor = nn.Sequential(*list(pretext_model.children())[:-(opt['num_couche']+2)])  # num_couche=nb of conv layer we get rid of
        
        # Replace layer4 with the custom classifier
        if opt['type_classifier'] == 'Classifier':
            self.classifier = Classifier(opt)
        elif opt['type_classifier'] == 'ConvClassifier':
            self.classifier = ConvClassifier(opt)

        # Freeze pretext model parameters
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

    def forward(self, x):
        # Extract features using the pretext model up to layer3
        features = self.feature_extractor(x)  # Shape: (batch_size, num_features, H, W)
        
        # Pass features through the classifier
        out = self.classifier(features)
        return out

    
def create_model(pretext_model, opt):
    return ClassifierModified(pretext_model, opt)