import numpy as np
import torch
from torch import nn
from torch.nn import functional

class YoloMS(nn.Module):
    def __init__(self, S, B, num_classes):
        super(YoloMS, self).__init__()
        self.S = S
        self.B = B
        self.num_classes = num_classes

        # convolution
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, 5, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),

            nn.Conv2d(32, 64, 5, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),

            nn.Conv2d(64, 128, 5, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),

            nn.Conv2d(128, 256, 5, stride=1, padding=1),
            nn.SiLU(inplace=True),
            nn.MaxPool2d(2, stride=2),

            nn.Conv2d(256, 512, 5, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(inplace=True),

            nn.Conv2d(512, 128, 5, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.SiLU(inplace=True),
            nn.MaxPool2d(2, stride=2),

            nn.AdaptiveAvgPool2d((7,7))
        )

        # fully connected
        self.fc_layers = nn.Sequential(
            nn.Linear(6272, 1920),
            nn.SiLU(inplace=True),
            nn.Dropout(p=0.25),
            nn.Linear(1920, self.S * self.S * (self.B * 5 + self.num_classes)),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        out = self.conv_layers(x)
        out = out.view(out.size()[0], -1)
        out = self.fc_layers(out)
        out = out.reshape(-1, self.S, self.S, self.B * 5 + self.num_classes)
        return out
