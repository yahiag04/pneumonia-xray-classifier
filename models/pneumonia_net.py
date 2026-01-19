import torch.nn as nn
import torch.nn.functional as F


class PneumoniaNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.bn1   = nn.BatchNorm2d(16)

        self.conv2a = nn.Conv2d(16, 32, 3, padding=1)
        self.bn2a   = nn.BatchNorm2d(32)
        self.conv2b = nn.Conv2d(32, 32, 3, padding=1)
        self.bn2b   = nn.BatchNorm2d(32)

        self.conv3a = nn.Conv2d(32, 64, 3, padding=1)
        self.bn3a   = nn.BatchNorm2d(64)
        self.conv3b = nn.Conv2d(64, 64, 3, padding=1)
        self.bn3b   = nn.BatchNorm2d(64)

        self.conv4a = nn.Conv2d(64, 128, 3, padding=1)
        self.bn4a   = nn.BatchNorm2d(128)
        self.conv4b = nn.Conv2d(128, 128, 3, padding=1)
        self.bn4b   = nn.BatchNorm2d(128)

        self.pool = nn.MaxPool2d(2)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc1 = nn.Linear(128, 64)
        self.drop = nn.Dropout(0.5)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))

        x = F.relu(self.bn2a(self.conv2a(x)))
        x = F.relu(self.bn2b(self.conv2b(x)))
        x = self.pool(x)

        x = F.relu(self.bn3a(self.conv3a(x)))
        x = F.relu(self.bn3b(self.conv3b(x)))
        x = self.pool(x)

        x = F.relu(self.bn4a(self.conv4a(x)))
        x = F.relu(self.bn4b(self.conv4b(x)))
        x = self.pool(x)

        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.drop(x)
        x = self.fc2(x)

        return x