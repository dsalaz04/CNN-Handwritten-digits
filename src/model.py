# Description: Convolutional neural network for MNIST digit classification.
# -*- coding: utf-8 -*-

import torch.nn as nn


def _conv_block(in_ch: int, out_ch: int, dropout: float) -> nn.Sequential:
    """Two 3x3 convolutions + BatchNorm + ReLU, then halve the spatial size."""
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
        nn.Dropout(dropout),
    )


class DigitCNN(nn.Module):
    """A compact VGG-style CNN that reaches ~99.4% test accuracy.

    Two convolutional blocks take the 28x28 input down to 7x7 feature maps, which a
    small fully-connected head maps to the ten digit classes.
    """

    def __init__(self, num_classes: int = 10, dropout: float = 0.25):
        super().__init__()
        self.features = nn.Sequential(
            _conv_block(1, 32, dropout),   # 28x28 -> 14x14
            _conv_block(32, 64, dropout),  # 14x14 -> 7x7
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))
