import torch
import torch.nn as nn
from ..layers.flatten_layer import FlattenLayer
from ..layers.fc_layer import FCLayer  

class AuxiliaryClassifier(nn.Module):
    def __init__(self, in_channels, num_classes=1000):
        super(AuxiliaryClassifier, self).__init__()

        self.features = nn.Sequential(
            nn.AvgPool2d(kernel_size=5, stride=3),   
            nn.Conv2d(in_channels, 128, kernel_size=1),  
            nn.ReLU(inplace=True)
        )

        self.classifier = nn.Sequential(
            FlattenLayer(),
            nn.Linear(128 * 4 * 4, 1024),  
            nn.ReLU(inplace=True),
            nn.Dropout(0.7),
            nn.Linear(1024, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
