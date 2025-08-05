from typing import Optional

import torch
import torch.nn as nn
from x_transformers import Decoder
import torch.nn.functional as F


class PredictorVAE(nn.Module):

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
        # Flow: Input -> Hidden -> Latent (mu and logvar) -> Hidden -> Decoder -> Output

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
        self.decode = Decoder(
            dim=hidden_size, depth=depth, heads=num_heads, layer_dropout=layer_dropout
        )
        # Output layer (Hidden -> Input)
        self.output = nn.Linear(hidden_size, embed_dim)
        self.relu = nn.ReLU()
    
    def encoder(self, x):
        h = self.relu(self.fc1(x))
        mu = self.mu(h)
        logvar = self.logvar(h)
        return mu, logvar

    def decoder(self, x):
        h = self.fc2(x)
        decoded = self.decode(h)
        return self.output(decoded)
    
    def reparameterize(self, mu, logvar):
        #logvar = log(std**2) -> std = sqrt(exp(logvar)) 
        # logvar = torch.clamp(logvar, min=-4, max=4)
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + std*eps

    def forward(self, context_encoding, target_masks):
        """
        context_encoding: (B, num_context_patches, embed_dim)
        target_masks:     (B, num_target_patches, embed_dim)
        """
        # Concatenate context + target mask
        x = torch.cat((context_encoding, target_masks), dim=1)  # (B, total_patches, embed_dim)

        # Encode to get mu and logvar
        mu, logvar = self.encoder(x)

        # Sample z using reparameterization trick
        z = self.reparameterize(mu, logvar)

        # Decode from latent
        decoded = self.decoder(z)  # (B, total_patches, embed_dim)

        # Keep only target output
        prediction = decoded[:, -target_masks.shape[1]:, :]  # (B, num_target_patches, embed_dim)

        return prediction, mu, logvar

def loss_vae(x, x_pred, mu, logvar):
    # To make it reconstruct better
    recon_loss = F.mse_loss(x_pred, x, reduction="sum")

    # KL(q(z|x)||p(z)) = 1/2 SUM(mu^2 + var - log(var) -1)
    #  = -1/2 SUM(log(var) + 1 - mu^2 - var)
    # where: var = exp(log(var))
    # To make more standard Gaussian
    kl = -0.5 * torch.sum(logvar + 1 - mu.pow(2) - logvar.exp())
    batch_size = x.size(0)

    return  (recon_loss + kl)/batch_size