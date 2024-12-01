import torch
import torch.nn as nn

class AlexNetPerso(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()

        route_color = 3
        speed = 1
        image_features_size = 256 * 6 * 6

        dropout = 0.5

        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        # TODO : Add the route color
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(image_features_size + speed, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )


    def forward(self, image, route_color, speed):
        image_features = self.features(image)
        image_features = self.avgpool(image_features)
        image_features = torch.flatten(image_features, 1)
        if len(speed.shape) == 1:
            speed = speed.unsqueeze(0)
        # TODO : Add the route color
        features = torch.cat((image_features, speed), dim=1)
        return self.classifier(features)