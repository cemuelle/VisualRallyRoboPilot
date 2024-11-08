import torch
import torch.nn as nn
from torchvision import models

class AlexNetPerso(nn.Module):
    def __init__(self, num_classes=4, num_route=4):
        super(AlexNetPerso, self).__init__()

        self.alexnet = models.alexnet(pretrained=True)

        # Replace the last fully-connected layer
        self.alexnet.classifier[6] = nn.Linear(self.alexnet.classifier[6].in_features, num_classes)

        self.route_color = nn.Sequential(
            nn.Linear(num_route, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )

        self.classifier = nn.Sequential(
            nn.Linear(num_classes + num_route, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x, route):
        image_features = self.alexnet(x)

        route_features = self.route_color(route)

        features = torch.cat((image_features, route_features), dim=1)

        return self.classifier(features)