import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    """Minimum model: 3 conv layers (8, 16, 32 filters), 2 pooling layers, 1 FC layer.

    Layer names match the original so its saved weights load. Expects 224x224 input.
    """

    def __init__(self, num_classes=4):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 8, 3)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(8, 16, 3)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(16, 32, 3)
        self.fc1 = nn.Linear(32 * 52 * 52, num_classes)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        return x
