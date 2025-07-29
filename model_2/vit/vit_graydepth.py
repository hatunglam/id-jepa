import torch
import torch.nn as nn
from x_transformers import Encoder

from utils.types import ensure_tuple

from .patch_embed import PatchEmbed2D

class GrayImageDepthTransformer(nn.Module):
    def __init__(
            self,
            img_size=224,
            patch_size=16,
            num_frames=1,
            tubelet_size=2,
            in_chans=1,
            embed_dim=64,
            enc_depth=8,
            num_heads=8,
            post_emb_norm=True,
            post_enc_norm=True,
            layer_dropout=0.1,
            **kwargs 
            ):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads

        self.img_size = ensure_tuple(img_size)
        self.patch_size = ensure_tuple(patch_size)

        self.patch_embed = PatchEmbed2D(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim
        )

        self.num_patches = self.patch_embed.patch_shape[0] * self.patch_embed.patch_shape[1]
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches, embed_dim))
        self.post_emb_norm = post_emb_norm
        self.post_emb_norm_vit = (
            nn.LayerNorm(embed_dim) if self.post_emb_norm else nn.Identity()
        )

        self.layer_dropout = layer_dropout

        self.student_encoder = Encoder(  # student encoder
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
            x_img,
            x_dep,
            attention_mask=None,
            patch_embed_only=False
    ):
        x_img = self.patch_embed(x_img)
        x_img = x_img + self.pos_embedding
        x_img = self.post_emb_norm_vit(x_img)

        if x_dep is not None: # If in training
            x_dep = self.patch_embed(x_dep)
            x_dep = x_dep + self.pos_embedding
            x_dep = self.post_emb_norm_vit(x_dep)
        if patch_embed_only: # Training
            return x_img, x_dep # To put these into forward_base Encoder
        
        # Inference
        # x_img will be passed into an Encoder 
        x_img = self.student_encoder(x_img, attn_mask=attention_mask)
        x_img = self.post_enc_norm_vit(x_img)
        return x_img, None
    
    def vit_nano(img_size, patch_size=16, num_frames=1, tubelet_size=2, **kwargs):
        return GrayImageDepthTransformer(
            img_size=img_size,
            patch_size=patch_size,
            num_frames=num_frames,
            tubelet_size=tubelet_size,
            in_chans=1,
            embed_dim=64,
            enc_depth=8,
            num_heads=8,
            **kwargs,
        )


def vit_tiny(img_size, patch_size=16, num_frames=1, tubelet_size=2, **kwargs):
    return GrayImageDepthTransformer(
        img_size=img_size,
        patch_size=patch_size,
        num_frames=num_frames,
        tubelet_size=tubelet_size,
        in_chans=1,
        embed_dim=192,
        enc_depth=12,
        num_heads=8,
        **kwargs,
    )


def vit_small(img_size, patch_size=16, num_frames=1, tubelet_size=2, **kwargs):
    return GrayImageDepthTransformer(
        img_size=img_size,
        patch_size=patch_size,
        num_frames=num_frames,
        tubelet_size=tubelet_size,
        in_chans=1,
        embed_dim=384,
        enc_depth=18,
        num_heads=8,
        **kwargs,
    )


def vit_base(img_size, patch_size=16, num_frames=1, tubelet_size=2, **kwargs):
    return GrayImageDepthTransformer(
        img_size=img_size,
        patch_size=patch_size,
        num_frames=num_frames,
        tubelet_size=tubelet_size,
        in_chans=1,
        embed_dim=768,
        enc_depth=18,
        num_heads=12,
        **kwargs,
    )


def vit_large(img_size, patch_size=16, num_frames=1, tubelet_size=2, **kwargs):
    return GrayImageDepthTransformer(
        img_size=img_size,
        patch_size=patch_size,
        num_frames=num_frames,
        tubelet_size=tubelet_size,
        in_chans=1,
        embed_dim=1024,
        enc_depth=24,
        num_heads=16,
        **kwargs,
    )


def vit_huge(img_size, patch_size=16, num_frames=1, tubelet_size=2, **kwargs):
    return GrayImageDepthTransformer(
        img_size=img_size,
        patch_size=patch_size,
        num_frames=num_frames,
        tubelet_size=tubelet_size,
        in_chans=1,
        embed_dim=1280,
        enc_depth=32,
        num_heads=16,
        **kwargs,
    )
