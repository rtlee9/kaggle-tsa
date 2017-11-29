"""Convolutional neural net (TSA net)."""

import torch.nn as nn


class TsaNet(nn.Module):
    """3D convolutional neural net to predict threat probability by threat zone."""

    def __init__(self, num_classes=1):
        """Initialize TsaNet structure."""
        super(TsaNet, self).__init__()
        self.features = nn.Sequential(  # initial torch.Size([1, 1, 32, 32, 32])

            nn.Conv3d(1, 32, kernel_size=3, stride=2, padding=1),  # 16
            nn.ReLU(inplace=True),
            nn.Conv3d(32, 32, kernel_size=3, stride=2, padding=1),  # 8
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2),  # 4

            nn.Conv3d(32, 32, kernel_size=3, stride=2, padding=1),  # 2
            nn.ReLU(inplace=True),

        )

        self.classifier = nn.Sequential(
            # nn.Dropout(),
            nn.Linear(32 * 2 ** 3, num_classes),
            nn.Sigmoid(),
        )

    def forward(self, x):
        """Net forward pass."""
        x = self.features(x)
        x = x.view(x.size(0), 32 * 2 ** 3)
        x = self.classifier(x)
        return x.squeeze()
