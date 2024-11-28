import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import AlexNet_Weights

class AlexNetPerso(nn.Module):
    def __init__(self, num_classes=4):
        super(AlexNetPerso, self).__init__()

        route_rgb=3

        self.alexnet = models.alexnet(weights=AlexNet_Weights)

        # Replace the last fully-connected layer
        self.alexnet.classifier[6] = nn.Linear(self.alexnet.classifier[6].in_features, num_classes)

        self.route_color = nn.Sequential(
            nn.Linear(route_rgb, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )

        self.classifier = nn.Sequential(
            nn.Linear(num_classes + 64, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x, route_color):
        image_features = self.alexnet(x)

        route_features = self.route_color(route_color)

        features = torch.cat((image_features, route_features), dim=1)

        return self.classifier(features)