import torch
from torch import nn
import torch.nn.functional as F


class RGB2HSV(nn.Module):
    """
    Pytorch implementation of RGB convert to HSV, and HSV convert to RGB,
    RGB or HSV's shape: (B * C * H * W)
    RGB or HSV's range: [0, 1)

    版权声明：本文为CSDN博主「Brikie」的原创文章，遵循CC 4.0 BY-SA版权协议，转载请附上原文出处链接及本声明。
    原文链接：https://blog.csdn.net/Brikie/article/details/115086835
    """

    def __init__(self, eps=1e-8):
        super(RGB2HSV, self).__init__()
        self.eps = eps

    def rgb_to_hsv(self, img):
        hue = torch.Tensor(img.shape[0], img.shape[2], img.shape[3]).to(img.device)

        hue[img[:, 2] == img.max(1)[0]] = 4.0 + ((img[:, 0] - img[:, 1]) / (img.max(1)[0] - img.min(1)[0] + self.eps))[
            img[:, 2] == img.max(1)[0]]
        hue[img[:, 1] == img.max(1)[0]] = 2.0 + ((img[:, 2] - img[:, 0]) / (img.max(1)[0] - img.min(1)[0] + self.eps))[
            img[:, 1] == img.max(1)[0]]
        hue[img[:, 0] == img.max(1)[0]] = (0.0 + ((img[:, 1] - img[:, 2]) / (img.max(1)[0] - img.min(1)[0] + self.eps))[
            img[:, 0] == img.max(1)[0]]) % 6

        hue[img.min(1)[0] == img.max(1)[0]] = 0.0
        hue = hue / 6

        saturation = (img.max(1)[0] - img.min(1)[0]) / (img.max(1)[0] + self.eps)
        saturation[img.max(1)[0] == 0] = 0

        value = img.max(1)[0]

        hue = hue.unsqueeze(1)
        saturation = saturation.unsqueeze(1)
        value = value.unsqueeze(1)
        hsv = torch.cat([hue, saturation, value], dim=1)
        return hsv

    def hsv_to_rgb(self, hsv):
        h, s, v = hsv[:, 0, :, :], hsv[:, 1, :, :], hsv[:, 2, :, :]
        # 对出界值的处理
        h = h % 1
        s = torch.clamp(s, 0, 1)
        v = torch.clamp(v, 0, 1)

        r = torch.zeros_like(h)
        g = torch.zeros_like(h)
        b = torch.zeros_like(h)

        hi = torch.floor(h * 6)
        f = h * 6 - hi
        p = v * (1 - s)
        q = v * (1 - (f * s))
        t = v * (1 - ((1 - f) * s))

        hi0 = hi == 0
        hi1 = hi == 1
        hi2 = hi == 2
        hi3 = hi == 3
        hi4 = hi == 4
        hi5 = hi == 5

        r[hi0] = v[hi0]
        g[hi0] = t[hi0]
        b[hi0] = p[hi0]

        r[hi1] = q[hi1]
        g[hi1] = v[hi1]
        b[hi1] = p[hi1]

        r[hi2] = p[hi2]
        g[hi2] = v[hi2]
        b[hi2] = t[hi2]

        r[hi3] = p[hi3]
        g[hi3] = q[hi3]
        b[hi3] = v[hi3]

        r[hi4] = t[hi4]
        g[hi4] = p[hi4]
        b[hi4] = v[hi4]

        r[hi5] = v[hi5]
        g[hi5] = p[hi5]
        b[hi5] = q[hi5]

        r = r.unsqueeze(1)
        g = g.unsqueeze(1)
        b = b.unsqueeze(1)
        rgb = torch.cat([r, g, b], dim=1)
        return rgb


class AffineLayer(nn.Module):
    """
    code by chatGPT, reviewed by human
    """
    def forward(self, original_image, transformation_matrix):
        """
        :param original_image is N x C x H x W
        :param transformation_matrix is an N x 2 x 3 Tensor
        :return:
        """
        grid = F.affine_grid(transformation_matrix, original_image.size(), align_corners=False)
        # 對原圖進行轉換，生成新圖
        transformed_image = F.grid_sample(original_image, grid, align_corners=False)
        return transformed_image
