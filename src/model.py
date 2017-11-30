"""Convolutional neural net (TSA net)."""

import torch.nn as nn


class TsaNet(nn.Module):
    """3D convolutional neural net to predict threat probability by threat zone."""

    def __init__(self, num_classes=1):
        """Initialize TsaNet structure."""
        super(TsaNet, self).__init__()
        self.features = nn.Sequential(  # initial torch.Size([1, 1, 32, 32, 32])

            nn.Conv3d(1, 16, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2),

            nn.Conv3d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2),

        )

        self.classifier = nn.Sequential(
            # nn.Dropout(),
            nn.Linear(32 * 8 ** 3, num_classes),
            nn.Sigmoid(),
        )

    def forward(self, x):
        """Net forward pass."""
        x = self.features(x)
        x = x.view(x.size(0), 32 * 8 ** 3)
        x = self.classifier(x)
        return x.squeeze()
