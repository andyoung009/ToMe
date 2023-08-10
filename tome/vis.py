# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------

import random
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

try:
    from scipy.ndimage import binary_erosion
except ImportError:
    pass  # Don't fail if scipy is not installed. It's only necessary for this one file.


def generate_colormap(N: int, seed: int = 0) -> List[Tuple[float, float, float]]:
    """Generates a equidistant colormap with N elements."""
    random.seed(seed)

    # generate_color()`函数使用Python中的`random`模块来生成每个颜色通道（红色、绿色和蓝色）的随机值，这些值的范围在0到1之间。
    def generate_color():
        return (random.random(), random.random(), random.random())

    return [generate_color() for _ in range(N)]

# 使用PyTorch和PIL库创建可视化函数，接受一个图像和一个PyTorch张量作为输入，并生成一个与输入图像大小相同的PIL图像。
def make_visualization(
    img: Image, source: torch.Tensor, patch_size: int = 16, class_token: bool = True
) -> Image:
    """
    Create a visualization like in the paper.

    Args:
     -

    Returns:
     - A PIL image the same size as the input.
    """

    # 函数的参数包括：
    # 1. img：一个PIL图像对象。
    # 2. source：一个PyTorch张量，表示输入图像的特征表示。
    # 3. patch_size：一个可选的整数，表示用于划分输入图像的补丁大小。默认为16。
    # 4. class_token：一个可选的布尔值，指示特征表示是否包含类令牌。默认为True。

    img = np.array(img.convert("RGB")) / 255.0
    source = source.detach().cpu()

    h, w, _ = img.shape
    ph = h // patch_size
    pw = w // patch_size

    if class_token:
        source = source[:, :, 1:]

    vis = source.argmax(dim=1)
    num_groups = vis.max().item() + 1

    cmap = generate_colormap(num_groups)
    vis_img = 0

    # 对于每个类别，该函数使用F.interpolate()函数将掩码大小调整为与输入图像相同，并使用二元腐蚀算法生成一个新的掩码(mask_eroded)和一个边缘掩码(mask_edge)。
    for i in range(num_groups):
        mask = (vis == i).float().view(1, 1, ph, pw)
        mask = F.interpolate(mask, size=(h, w), mode="nearest")
        mask = mask.view(h, w, 1).numpy()

        color = (mask * img).sum(axis=(0, 1)) / mask.sum()
        mask_eroded = binary_erosion(mask[..., 0])[..., None]
        mask_edge = mask - mask_eroded

        if not np.isfinite(color).all():
            color = np.zeros(3)

        vis_img = vis_img + mask_eroded * color.reshape(1, 1, 3)
        vis_img = vis_img + mask_edge * np.array(cmap[i]).reshape(1, 1, 3)

    # Convert back into a PIL image
    vis_img = Image.fromarray(np.uint8(vis_img * 255))

    return vis_img
