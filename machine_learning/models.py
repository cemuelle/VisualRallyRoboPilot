import torch
import torch.nn as nn

class AlexNetPerso(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        dropout = 0.5

        # Convolutional layers
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 12, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(12, 24, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(24, 36, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.Conv2d(36, 48, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv2d(48, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))

        # Fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64*6*6, 100),
            nn.Dropout(p=dropout),
            nn.ReLU(),
            nn.Linear(100, 10),
            nn.Dropout(p=dropout),
            nn.ReLU(),
            nn.Linear(10, num_classes),
            nn.Dropout(p=dropout),
        )

    def forward(self, image, route_color, speed):
        image = self.conv_layers(image)
        image = self.avgpool(image)
        image = self.fc_layers(image)
        
        return image