from typing import Optional

import torch
import torch.nn as nn
from x_transformers import Encoder


class Predictor(nn.Module):
 
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        depth: int,
        layer_dropout: float = 0.0,
        predictor_embed_dim: Optional[int] = None,
    ):
        super().__init__()
        # Initialize the transformer-based decoder
        self.encoder = Encoder(
            dim=embed_dim, depth=depth, heads=num_heads, layer_dropout=layer_dropout
        )

        self.predictor_embed = (
            nn.Linear(embed_dim, predictor_embed_dim, bias=True)
            if predictor_embed_dim
            else nn.Identity()
        )

        self.predictor_norm = (
            nn.LayerNorm(predictor_embed_dim) if predictor_embed_dim else nn.Identity()
        )
        self.predictor_proj = (
            nn.Linear(predictor_embed_dim, embed_dim, bias=True)
            if predictor_embed_dim
            else nn.Identity()
        )

    def forward(
        self, context_encoding: torch.Tensor, target_masks: torch.Tensor
    ):
        # Concatenate the context encoding and the target masks
        x = torch.cat(
            (context_encoding, target_masks), dim=1
        )  # (batch_size, num_context_patches + num_target_patches, embed_dim)

        # Map context tokens to the predictor dimension
        x = self.predictor_embed(x)

        # Pass the concatenated tensor through the transformer decoder
        x = self.encoder(x)  # (batch_size, predictor_embed_dim, embed_dim)

        # Normalise and project predictor ouputs back to the input dimension
        x = self.predictor_proj(
            self.predictor_norm(x)
        )  # (batch_size, num_context_patches + num_target_patches, embed_dim)

        # Return the output corresponding to target tokens, i.e., the last len(target_masks) tokens
        prediction = x[
            :, -target_masks.shape[1] :, :  # Include entire batch
        ]  # (batch_size, num_target_patches, embed_dim)

        return prediction
