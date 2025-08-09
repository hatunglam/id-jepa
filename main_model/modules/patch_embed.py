from typing import Tuple
import torch
import torch.nn as nn
from einops import rearrange
from utils.types import ensure_tuple

class PatchEmbed2D(nn.Module):
    def __init__(self,
                 img_size: int | Tuple[int, int] = 224,
                 patch_size: int | Tuple[int, int] = 16,
                 in_chans: int = 3,
                 embed_dim: int = 64,):
        super().__init__()
        img_size: Tuple[int, int] = ensure_tuple(img_size)
        patch_size: Tuple[int, int] = ensure_tuple(patch_size)

        # Calculate the number of patches in each dimension
        self.patch_shape: Tuple[int, int] = (
            img_size[0] // patch_size[0],
            img_size[1] // patch_size[1],
        )

        # Convolutional layer to convert the image into patches
        self.conv = nn.Conv2d(
            in_channels=in_chans,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size,  # Same stride as the patch_size as to extract non-overlapping patches
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = rearrange(x, "b e h w -> b (h w) e")

        return x