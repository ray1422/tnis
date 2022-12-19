import numpy as np
import torch
from torch import nn


class HSVCurve(nn.Module):
    def forward(self, x: torch.Tensor, pts: torch.Tensor, cycle=True) -> torch.Tensor:
        """
        :param x: image shaped (batch, channels, height, width). values are normalized to [0, 1]
        :param pts: coefficients shaped (batch, channels, 12)
        :param cycle: apply arcsin(sin(x))
        :return: adjusted image with original shape
        """
        y = torch.zeros_like(x).to(x.device)
        n, c, h, w = np.shape(x)

        for i in range(pts.size()[1] // c):
            y += x * (pts[:, i * c:(i + 1) * c] ** i).view(-1, c, 1, 1)
        if cycle:
            y = (torch.sin(y) + 1) / 2.

        return y
