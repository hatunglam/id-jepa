from typing import Optional

import torch
import torch.nn as nn
from x_transformers import Decoder
import torch.nn.functional as F


class Predictor(nn.Module):

    def __init__(
        self,
        embed_dim,
        hidden_size,
        z_dim,
        num_heads,
        depth,
        layer_dropout = 0.0,
        predictor_embed_dim = None,
    ):
        super().__init__()
        
        self.decoder = Decoder(
            dim=embed_dim, depth=depth, heads=num_heads, layer_dropout=layer_dropout
        )
        # Encoder:
        # First FC Layer to project down (Input -> Hidden)
        self.fc1 = nn.Linear(embed_dim, hidden_size)
        # mu layer (Hidden -> Latent)
        self.mu = nn.Linear(hidden_size, z_dim)
        # logvar layer (Hidden -> Latent)
        self.logvar = nn.Linear(hidden_size, z_dim)

        # Decoder:
        # Second layer (Z -> Hidden)
        self.fc2 = nn.Linear(z_dim, hidden_size)
        # Decoder layer (Hidden -> Hidden)
        self.decoder = Decoder(
            dim=hidden_size, depth=depth, heads=num_heads, layer_dropout=layer_dropout
        )
        # Output layer (Hidden -> Input)
        self.output = nn.Linear(hidden_size, embed_dim)
    
    def encoder(self, x):
        h = self.fc1(x)
        mu = self.mu(h)
        logvar = self.logvar(h)
        return mu, logvar

    def decoder(self, x):
        h = self.fc2(x)
        decoded = self.decoder(h)
        return self.output(decoded)
    
    def reparameterize(self, mu, logvar):
        #logvar = log(std**2) -> std = sqrt(exp(logvar)) 
        # logvar = torch.clamp(logvar, min=-4, max=4)
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + std*eps

    def forward(
        self, context_encoding, target_masks
    ):
        
        # Concatenate the context encoding and the target masks
        x = torch.cat(
            (context_encoding, target_masks), dim=1
        )  # (batch_size, num_context_patches + num_target_patches, embed_dim)

        
        # Return the output corresponding to target tokens, i.e., the last len(target_masks) tokens
        prediction = x[
            :, -target_masks.shape[1] :, :  # Include entire batch
        ]  # (batch_size, num_target_patches, embed_dim)

        return prediction
