import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

import enhance
from utils import RGB2HSV, AffineLayer


class TNIS(nn.Module):
    def __init__(self):
        super().__init__()
        self.fe = resnet18(weights=ResNet18_Weights.DEFAULT)
        num_filters = self.fe.fc.in_features
        self.fe.fc = nn.Linear(num_filters, 1024)
        self.act = nn.LeakyReLU()
        self.output = nn.Linear(1024, 50)
        self.rgb_hsv = RGB2HSV(eps=1e-8)
        self.affine = AffineLayer()
        self.curve = enhance.HSVCurve()

    def forward(self, x):
        hsv_x = self.rgb_hsv.rgb_to_hsv(x)
        y = self.fe(x)
        y = self.act(y)
        y = self.output(y)

        # Affine
        hsv_x = self.affine(hsv_x, torch.reshape(y[:, 36:36 + 6], (-1, 2, 3)))
        # hsv_x = hsv_x * y[:, 0:3].view(-1, 3, 1, 1)
        # HSV Adjust
        hsv_x= self.curve(hsv_x, y[:, 0:36])
        # hsv_x[:, 0, :, :] = self.curve(hsv_x[:, 0, :, :], y[:, 0:12])
        # hsv_x[:, 1, :, :] = self.curve(hsv_x[:, 1, :, :], y[:, 12:24])
        # hsv_x[:, 2, :, :] = self.curve(hsv_x[:, 2, :, :], y[:, 24:36])

        return self.rgb_hsv.hsv_to_rgb(hsv_x), y


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.fe = resnet18(weights=None)
        num_filters = self.fe.fc.in_features
        self.fe.fc = nn.Linear(num_filters, 1)
        self.act = nn.Tanh()

    def forward(self, x):
        x = self.fe(x)
        return self.act(x)
