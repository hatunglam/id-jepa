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
        x_img = self.encoder(x_img, attn_mask=attention_mask)
        x_img = self.post_enc_norm_vit(x_img)
        return x_img, None
