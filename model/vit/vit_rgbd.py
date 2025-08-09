from typing import Any, Optional, Tuple, Union
import torch
import torch.nn as nn
from x_transformers import Encoder
from utils.types import ensure_tuple
from .patch_embed import PatchEmbed2D, PatchEmbed3D
import numpy as np
import math
from einops import rearrange

class PositionEmbeddingSine(nn.Module):
    def __init__(self,
                 dim=64,
                 base=10000,
                 normalize=False,
                 scale=None,):
        super().__init__()
        self.dim = dim
        self.base = base
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale
    
    def forward(self, x: torch.Tensor, **kwargs):
        return self._fwd(x, **kwargs)

    def _forward_2d(self, x: torch.Tensor, **kwargs):
        mask = kwargs.get("mask", torch.zeros((x.shape[0], x.shape[-2], x.shape[-1]),
                                               device=x.device, dtype=torch.bool))
        assert mask is not None
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32) - 1.0
        x_embed = not_mask.cumsum(2, dtype=torch.float32) - 1.0
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale
        dim = torch.arange(self.dim // 2, dtype=torch.float32, device=x.device)
        dim /= (self.dim / 2.0)
        dim = 1.0 / (self.base ** dim)
        pos_x = x_embed[:, :, :, None] * dim[None, None, None, :]
        pos_y = y_embed[:, :, :, None] * dim[None, None, None, :]
        pos_x = torch.cat([pos_x.sin(), pos_x.cos()], dim=-1)
        pos_y = torch.cat([pos_y.sin(), pos_y.cos()], dim=-1)
        pos = torch.cat((pos_x, pos_y), dim=3)
        pos = rearrange(pos, "B H W C -> B (H W) C")
        return pos

class RGBDVisionTransformer(nn.Module):
    def __init__(
        self,
        img_size: Union[int, Tuple[int, int]] = 224,
        patch_size: Union[int, Tuple[int, int]] = 16,
        num_frames: int = 1,
        tubelet_size: int = 2,
        in_chans_rgb: int = 3,
        in_chans_dep: int = 1,
        embed_dim: int = 64,
        enc_depth: int = 8,
        num_heads: int = 8,
        post_emb_norm: bool = True,
        post_enc_norm: bool = True,
        layer_dropout: float = 0.1,
        **kwargs: Any,
    ):
        super().__init__()
        self.img_size = ensure_tuple(img_size)
        self.patch_size = ensure_tuple(patch_size)

        self.num_frames = num_frames
        self.is_video = num_frames > 1
        self.tubelet_size = tubelet_size

        self.embed_dim = embed_dim
        self.num_heads = num_heads

        self.patch_embed_rgb: nn.Module = (
            PatchEmbed2D(
                img_size=img_size,
                patch_size=patch_size,
                in_chans=in_chans_rgb,
                embed_dim=embed_dim,
            )
            if not self.is_video
            else PatchEmbed3D(
                img_size=img_size,
                num_frames=num_frames,
                patch_size=patch_size,
                tubelet_size=tubelet_size,
                in_chans=in_chans_rgb,
                embed_dim=embed_dim,
            )
        )

        self.patch_embed_dep: nn.Module = (
            PatchEmbed2D(
                img_size=img_size,
                patch_size=patch_size,
                in_chans=in_chans_dep,
                embed_dim=embed_dim,
            )
            if not self.is_video
            else PatchEmbed3D(
                img_size=img_size,
                num_frames=num_frames,
                patch_size=patch_size,
                tubelet_size=tubelet_size,
                in_chans=in_chans_dep,
                embed_dim=embed_dim,
            )
        )

        # self.num_patches: int = int(
        #     torch.prod(torch.Tensor(self.patch_embed_rgb.patch_shape)).item()
        # )
        self.pos_embed = PositionEmbeddingSine(dim=embed_dim//2)

        self.num_patches = self.patch_embed_rgb.patch_shape[0] * self.patch_embed_rgb.patch_shape[1]

        self.post_emb_norm = post_emb_norm
        self.post_emb_norm_vit = (
            nn.LayerNorm(embed_dim) if self.post_emb_norm else nn.Identity()
        )

        self.layer_dropout = layer_dropout

        self.encoder = Encoder(  # student encoder
            dim=embed_dim,
            heads=num_heads,
            depth=enc_depth,
            layer_dropout=self.layer_dropout,
        )

        self.post_enc_norm = post_enc_norm
        self.post_enc_norm_vit = (
            nn.LayerNorm(embed_dim) if self.post_enc_norm else nn.Identity()
        )  # student encoder

    def forward_vit(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        patch_embed_only: bool = False,
    ) -> torch.Tensor:
        
        x = self.patch_embed_rgb(x)
        self.pos_embedding = self.pos_embed(x)
        x = rearrange(x, "b e h w -> b (h w) e")
        x = x + self.pos_embedding
        x = self.post_emb_norm_vit(x)
        if patch_embed_only:
            return x
        # Inference
        x = self.encoder(x, attn_mask=attention_mask)
        x = self.post_enc_norm_vit(x)
        return x
    
    def forward_skip(self, x: torch.Tensor):
        x = self.encoder(x, attn_mask=None)
        x = self.post_enc_norm_vit(x)
        return x

def vit_nano(img_size, patch_size=16, num_frames=1, tubelet_size=2, **kwargs):
    return RGBDVisionTransformer(
        img_size=img_size,
        patch_size=patch_size,
        num_frames=num_frames,
        tubelet_size=tubelet_size,
        in_chans_rgb=3,
        in_chans_dep=1,
        embed_dim=64,
        enc_depth=8,
        num_heads=8,
        **kwargs,
    )


def vit_tiny(img_size, patch_size=16, num_frames=1, tubelet_size=2, **kwargs):
    return RGBDVisionTransformer(
        img_size=img_size,
        patch_size=patch_size,
        num_frames=num_frames,
        tubelet_size=tubelet_size,
        in_chans_rgb=3,
        in_chans_dep=1,
        embed_dim=192,
        enc_depth=12,
        num_heads=8,
        **kwargs,
    )


def vit_small(img_size, patch_size=16, num_frames=1, tubelet_size=2, **kwargs):
    return RGBDVisionTransformer(
        img_size=img_size,
        patch_size=patch_size,
        num_frames=num_frames,
        tubelet_size=tubelet_size,
        in_chans_rgb=3,
        in_chans_dep=1,
        embed_dim=384,
        enc_depth=18,
        num_heads=8,
        **kwargs,
    )


def vit_base(img_size, patch_size=16, num_frames=1, tubelet_size=2, **kwargs):
    return RGBDVisionTransformer(
        img_size=img_size,
        patch_size=patch_size,
        num_frames=num_frames,
        tubelet_size=tubelet_size,
        in_chans_rgb=3,
        in_chans_dep=1,
        embed_dim=768,
        enc_depth=18,
        num_heads=12,
        **kwargs,
    )


def vit_large(img_size, patch_size=16, num_frames=1, tubelet_size=2, **kwargs):
    return RGBDVisionTransformer(
        img_size=img_size,
        patch_size=patch_size,
        num_frames=num_frames,
        tubelet_size=tubelet_size,
        in_chans_rgb=3,
        in_chans_dep=1,
        embed_dim=1024,
        enc_depth=24,
        num_heads=16,
        **kwargs,
    )


def vit_huge(img_size, patch_size=16, num_frames=1, tubelet_size=2, **kwargs):
    return RGBDVisionTransformer(
        img_size=img_size,
        patch_size=patch_size,
        num_frames=num_frames,
        tubelet_size=tubelet_size,
        in_chans_rgb=3,
        in_chans_dep=1,
        embed_dim=1280,
        enc_depth=32,
        num_heads=16,
        **kwargs,
    )