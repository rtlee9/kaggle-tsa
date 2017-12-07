"""Convolutional neural net (TSA net)."""
import torch.nn as nn
from torchvision import models


class TsaNet(nn.Module):
    """3D convolutional neural net to predict threat probability by threat zone."""

    def __init__(self, num_classes=1, transfer=None):
        """Initialize TsaNet structure."""
        super(TsaNet, self).__init__()
        self.features = nn.Sequential(  # initial torch.Size([1, 1, 32, 32, 32])

            nn.Conv3d(1, 64, kernel_size=3, stride=1, padding=2),
            nn.BatchNorm3d(64),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2),

            nn.Conv3d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2),

        )

        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(64 * 8 ** 3, num_classes),
            nn.Sigmoid(),
        )

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                gain = nn.init.calculate_gain('leaky_relu')
                nn.init.xavier_normal(m.weight, gain=gain)
                nn.init.constant(m.bias, 0.1)
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        # transfer weights
        if transfer:
            try:
                vgg = models.vgg16(pretrained=True)
                w1_vgg = vgg.features[0].weight
                d = self.state_dict()
                d['features.0.weight'] = w1_vgg.unsqueeze(1).data
                self.load_state_dict(d)
            except RuntimeError:
                print('Incorrect dimensions for weight transfer')

    def forward(self, x):
        """Net forward pass."""
        x = self.features(x)
        x = x.view(x.size(0), x.size(1) * x.size(2) * x.size(3) * x.size(4))
        x = self.classifier(x)
        return x.squeeze()
