import torch.nn as nn
import torch.nn.functional as F


class PneumoniaNet(nn.Module):
    def __init__(self, num_classes=1, width=1.0):
        super().__init__()
        c1, c2, c3, c4 = [_scale_channels(channels, width) for channels in (16, 32, 64, 128)]
        hidden = _scale_channels(64, width)

        self.conv1 = nn.Conv2d(1, c1, 3, padding=1)
        self.bn1   = nn.BatchNorm2d(c1)

        self.conv2a = nn.Conv2d(c1, c2, 3, padding=1)
        self.bn2a   = nn.BatchNorm2d(c2)
        self.conv2b = nn.Conv2d(c2, c2, 3, padding=1)
        self.bn2b   = nn.BatchNorm2d(c2)

        self.conv3a = nn.Conv2d(c2, c3, 3, padding=1)
        self.bn3a   = nn.BatchNorm2d(c3)
        self.conv3b = nn.Conv2d(c3, c3, 3, padding=1)
        self.bn3b   = nn.BatchNorm2d(c3)

        self.conv4a = nn.Conv2d(c3, c4, 3, padding=1)
        self.bn4a   = nn.BatchNorm2d(c4)
        self.conv4b = nn.Conv2d(c4, c4, 3, padding=1)
        self.bn4b   = nn.BatchNorm2d(c4)

        self.pool = nn.MaxPool2d(2)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc1 = nn.Linear(c4, hidden)
        self.drop = nn.Dropout(0.5)
        self.fc2 = nn.Linear(hidden, num_classes)

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


def _scale_channels(channels: int, width: float) -> int:
    if width <= 0:
        raise ValueError("width must be positive")
    return max(1, int(round(channels * width)))
