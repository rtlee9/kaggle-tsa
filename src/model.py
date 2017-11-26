"""Convolutional neural net (TSA net)."""

import torch.nn as nn


class TsaNet(nn.Module):
    """3D convolutional neural net to predict threat probability by threat zone."""

    def __init__(self, num_classes=1):
        """Initialize TsaNet structure."""
        super(TsaNet, self).__init__()
        self.features = nn.Sequential(  # initial torch.Size([1, 1, 64, 32, 32])

            nn.Conv3d(1, 64, kernel_size=5, stride=2, padding=2),  # torch.Size([1, 64, 16, 16, 16])
            nn.ReLU(inplace=True),

            nn.Conv3d(64, 128, kernel_size=5, stride=2, padding=2),  # torch.Size([1, 128, 8, 8, 8])
            nn.ReLU(inplace=True),
            nn.Conv3d(128, 128, kernel_size=3, stride=1, padding=1),  # torch.Size([1, 128, 8, 8, 8])
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2),  # torch.Size([1, 128, 4, 4, 4])

            nn.Conv3d(128, 256, kernel_size=5, stride=2, padding=2),  # torch.Size([1, 256, 2, 2, 2])
            nn.ReLU(inplace=True),
            nn.Conv3d(256, 256, kernel_size=3, stride=1, padding=1),  # torch.Size([1, 256, 2, 2, 2])
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2),  # torch.Size([1, 256, 1, 1, 1])
        )

        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(128, num_classes),
            nn.Sigmoid(),
        )

    def forward(self, x):
        """Net forward pass."""
        x = self.features(x)
        x = x.view(x.size(0), 256)
        x = self.classifier(x)
        return x.squeeze()
