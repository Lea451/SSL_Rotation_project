import torch
import torch.nn as nn
from models.ResNet18 import ResNet18  # Assuming ResNet18 is in the models folder
from models.Classifier import Classifier  # Assuming Classifier is in the models folder

class ClassifierModified(nn.Module):
    def _init_(self, pretext_model, classifier_opt):
        super(ClassifierModified, self)._init_()
        
        # Use the pretext model as the feature extractor
        self.feature_extractor = pretext_model

        # Freeze pretext model parameters
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

        # Replace the ResNet's fully connected layer with the Classifier
        self.classifier = Classifier(classifier_opt)

    def forward(self, x):
        # Extract features using the pretext model
        features = self.feature_extractor(x, out_feat_keys=['avgpool'])  # Extract features before the replaced fc_block
        
        # Flatten the features (if necessary)
        if isinstance(features, list):  # Handle multiple outputs
            features = features[0]
        
        # Pass features through the classifier
        out = self.classifier(features)
        return out