import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import AlexNet_Weights

class AlexNetPerso(nn.Module):
    def __init__(self, num_classes=4):
        super(AlexNetPerso, self).__init__()

        self.alexnet = models.alexnet(weights=AlexNet_Weights.DEFAULT)

        route_rgb=3
        alexNet_output_size = 4
        route_color_output_size = 64
        speed_output_size = 64

        dropout = 0.1

        # Replace the last fully-connected layer
        self.alexnet.classifier[6] = nn.Linear(self.alexnet.classifier[6].in_features, alexNet_output_size)

        self.route_color = nn.Sequential(
            nn.Linear(route_rgb, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, route_color_output_size),
            nn.ReLU()
        )

        self.speed = nn.Sequential(
            nn.Linear(1, 48),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(48, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, speed_output_size),
            nn.ReLU()
        )

        self.classifier = nn.Sequential(
            nn.Linear(alexNet_output_size + route_color_output_size + speed_output_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, num_classes)
        )

    def forward(self, x, route_color, speed):
        image_features = self.alexnet(x)

        route_features = self.route_color(route_color)

        print("Speed forward: ", speed)

        speed_features = self.speed(speed)

        if len(speed_features.shape) == 1:
            speed_features = speed_features.unsqueeze(0)


         # Print statements to help with debugging
        print("Image features shape: ", image_features.shape)
        print("Route features shape: ", route_features.shape)
        print("Speed features shape: ", speed_features.shape)

        # Concatenate the features along the feature dimension (dim=1)
        features = torch.cat((image_features, route_features, speed_features), dim=1)

        print("Concatenated features shape: ", features.shape)

        return self.classifier(features)