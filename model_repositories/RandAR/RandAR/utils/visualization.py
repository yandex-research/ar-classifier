import torch
import torchvision
import cv2
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp


def make_grid(imgs: np.ndarray, scale=0.5, row_first=True):
    """
    Args:
        imgs: [B, H, W, C] in [0, 1]
    Output:
        x row of images, and 2x column of images
        which means 2 x ^ 2 <= B

        img_grid: np.ndarray, [H', W', C]
    """

    B, H, W, C = imgs.shape
    imgs = torch.tensor(imgs)
    imgs = imgs.permute(0, 3, 1, 2).contiguous()

    num_row = int(np.sqrt(B / 2))
    if num_row < 1:
        num_row = 1
    num_col = int(np.ceil(B / num_row))

    if row_first:
        img_grid = torchvision.utils.make_grid(imgs, nrow=num_col, padding=0)
    else:
        img_grid = torchvision.utils.make_grid(imgs, nrow=num_row, padding=0)

    img_grid = img_grid.permute(1, 2, 0).cpu().numpy()

    # resize by scale
    img_grid = cv2.resize(img_grid, None, fx=scale, fy=scale)
    return img_grid