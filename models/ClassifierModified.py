import torch
import torch.nn as nn
from models.ResNet18 import ResNet18  # Assuming ResNet18 is in the models folder
from models.Classifier import Classifier  # Assuming Classifier is in the models folder

class ClassifierModified(nn.Module):
    def __init__(self, pretext_model, classifier_opt):
        super(ClassifierModified, self).__init__()
        
        # Load the pretext model (ResNet18)
        self.feature_extractor = pretext_model

        # Freeze pretext model parameters
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

        # Replace the ResNet's fully connected layer with the Classifier
        self.feature_extractor.fc_block = Classifier(classifier_opt)

    def forward(self, x):
        # Pass through the feature extractor (ResNet18 with the new classifier)
        out = self.feature_extractor(x)
        return out

    def initialize_weights(self):
        # Initialize weights for the classifier's linear layers
        self.feature_extractor.fc_block.initialize_weights()

