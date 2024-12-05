import torch
import torch.nn as nn

class AlexNetPerso(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        dropout = 0.5

        self.use_speed = False

        self.use_color = True
        self.use_grayscale = False

        if self.use_speed:
            self.speed_size = 1
        else:
            self.speed_size = 0

        if self.use_grayscale:
            self.color_size = 1
        else:
            self.color_size = 3

        if self.use_color:
            self.color_size += 1

        # Convolutional layers
        self.conv_layers = nn.Sequential(
            nn.Conv2d(self.color_size, 12, kernel_size=5, stride=2), # kernel_size is the size of the matrix of the filter, stride is the step of the filter moving between each step
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # use to reduce the dimension of the image, by taking the max value of the kernel size matrix, and moving the kernel by the stride value
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

        self.avgpool = nn.Sequential(
            nn.AdaptiveAvgPool2d((6, 6)), # use to reduce the dimension of the image to a fixed size
            nn.Flatten(), # use to flatten the image to a 1D tensor to be passed to fc layers
        )

        # Fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Linear(64*6*6 + self.speed_size, 100),
            nn.Dropout(p=dropout),
            nn.ReLU(),
            nn.Linear(100, 10),
            nn.Dropout(p=dropout),
            nn.ReLU(),
            nn.Linear(10, num_classes),
            nn.Dropout(p=dropout),
        )

    def forward(self, image, speed):
        image = self.conv_layers(image)
        image = self.avgpool(image)

        if self.use_speed:
            if len(speed.shape) == 1:
                speed = speed.unsqueeze(0)
            features = torch.cat((image, speed), dim=1)
        else:
            features = image
        image = self.fc_layers(features)
        
        return image