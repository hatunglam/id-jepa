from typing import Any, Optional, Tuple, Union
import torch
import torch.nn as nn
from x_transformers import Encoder
from utils.types import ensure_tuple
from .patch_embed import PatchEmbed2D, PatchEmbed3D
import numpy as np
import math

def get_2d_sincos_pos_embed(embed_dim, grid_size_h, grid_size_w, cls_token=False):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size_h, dtype=float)
    grid_w = np.arange(grid_size_w, dtype=float)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size_h, grid_size_w])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    grid_size: int of the grid length
    return:
    pos_embed: [grid_size, embed_dim] or [1+grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid = np.arange(grid_size, dtype=float)
    pos_embed = get_1d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=float)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega   # (D/2,)

    pos = pos.reshape(-1)   # (M,)
    out = np.einsum('m,d->md', pos, omega)   # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


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

        self.num_patches = self.patch_embed_rgb.patch_shape[0] * self.patch_embed_rgb.patch_shape[1]

        self.pos_embedding = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim), requires_grad=False)
        pos_embed = get_2d_sincos_pos_embed(self.pos_embedding.shape[-1],
                                            patch_shape[0],
                                            patch_shape[1]
                                            cls_token=False)
        self.pos_embedding.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

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
        pos_embedding = self.interpolate_pos_encoding(x, self.pos_embedding)
        x = x + pos_embedding
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
    
    def interpolate_pos_encoding(self, x, pos_embed):
        npatch = x.shape[1] - 1
        N = pos_embed.shape[1] - 1
        if npatch == N:
            return pos_embed
        class_emb = pos_embed[:, 0]
        pos_embed = pos_embed[:, 1:]
        dim = x.shape[-1]
        pos_embed = nn.functional.interpolate(
            pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
            scale_factor=math.sqrt(npatch / N),
            mode='bicubic',
        )
        pos_embed = pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_emb.unsqueeze(0), pos_embed), dim=1)

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