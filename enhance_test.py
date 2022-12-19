import unittest
import numpy as np
import torch
import enhance
from matplotlib import pyplot as plt


class TestStringMethods(unittest.TestCase):
    def test_affine(self):
        import torch
        import torch.nn.functional as F

        # 假設我們有一張原圖，大小為 3x3
        original_image = torch.tensor([[[
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9]
        ]]], dtype=torch.float32)

        # 定義轉換矩陣
        transformation_matrix = torch.tensor([[
            [1, 0, 1],
            [0, 1, 1]
        ]], dtype=torch.float32)

        # 生成網格
        grid = F.affine_grid(transformation_matrix, original_image.size(), align_corners=False)

        # 對原圖進行轉換，生成新圖
        transformed_image = F.grid_sample(original_image, grid, align_corners=False)

        print(transformed_image)

    def test_curve(self):
        plt.subplot(1, 2, 1)
        x = torch.rand((1, 3, 32, 32)).detach()
        y = enhance.HSVCurve()(x, torch.rand((1, 12))).detach()
        plt.imshow(np.transpose(x[0], (1, 2, 0)))
        plt.subplot(1, 2, 2)
        plt.imshow(np.transpose(y[0, :, :], (1, 2, 0)))
        plt.show()

    def test_backward(self):
        x = np.random.rand(1, 3, 32, 32)
        x = torch.Tensor(x)
        x.requires_grad = True
        y = torch.mean(enhance.HSVCurve()(x, torch.rand((1, 12)), cycle=True))
        y.backward()


if __name__ == '__main__':
    unittest.main()
