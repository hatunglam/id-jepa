from .base_latent import JEPAVAriationalLatent
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_latent import CrossAttentionFusion


def compute_kl_divergence(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """
    Compute the KL divergence between N(mu, sigma^2) and N(0, 1) for each latent dimension.

    Args:
        mu (torch.Tensor): Mean of the approximate posterior q(z|x) — shape (B, L, D)
        logvar (torch.Tensor): Log-variance of q(z|x) — shape (B, L, D)

    Returns:
        torch.Tensor: KL divergence per batch element — shape (B,)
    """
    kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())  # shape (B, L, D)

    return kl.sum(dim=[1, 2])  # Sum over latent dimensions and sequence length → (B,)

def compute_latent_jepa_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    kl_weight: float = 1.0,
):
    """
    Computes masked MSE + KL-divergence (ELBO) loss for multimodal latent JEPA.

    Returns
    -------
    torch.Tensor
        Scalar loss
    """
    recon_loss = F.mse_loss(predictions, targets, reduction="mean")
    kl_loss = compute_kl_divergence(mu, logvar).mean()

    return recon_loss + (kl_weight * kl_loss)



class VariationalLatentIDJEPA(JEPAVAriationalLatent, pl.LightningModule):
    def __init__(self,
                 image_encoder,
                 depth_encoder,
                 decoder_depth,
                 n_heads,
                 latent_num_heads=8,
                 fusion_module = CrossAttentionFusion,
                 latent_dim=512,
                 latent_dropout_prob=0.1,
                 predictor_embed_dim=None,
                 mode="train",
                 context_ratio_range=(0.85, 0.95),
                 target_mask_range=(0.15, 0.25),
                 kl_anneal_start=0,
                 kl_anneal_end=10000,
                 kl_anneal_max=1.0,
                 lr=1e-3,
                 weight_decay=0.05):
        pl.LightningModule.__init__(self)
        JEPAVAriationalLatent.__init__(self,
                           image_encoder=image_encoder,
                           depth_encoder=depth_encoder,
                           decoder_depth=decoder_depth,
                           n_heads=n_heads,
                           latent_num_heads=latent_num_heads,
                           fusion_module = fusion_module,
                           latent_dim=latent_dim,
                           latent_dropout_prob=latent_dropout_prob,
                           predictor_embed_dim=predictor_embed_dim,
                           mode=mode,
                           context_ratio_range=context_ratio_range,
                           target_mask_range=target_mask_range,
                           kl_anneal_start=kl_anneal_start,
                           kl_anneal_end=kl_anneal_end,
                           kl_anneal_max=kl_anneal_max
                           )

        self.lr = lr 
        self.weight_decay = weight_decay

    def forward(self, x_img, x_dep):
        return self.forward_base(
            image=x_img,
            depth=x_dep
        ) 
    
    def training_step(self,
                      batch,
                      batch_idx
                      ):
        self.mode = "train"
        x_img, x_dep = batch["student_input"].to(self.device), batch["teacher_input"].to(self.device)

        prediction, target_block, mu, logvar = self(x_img=x_img,
                                                    x_dep=x_dep)
        kl_weight = self.get_kl_weight()

        kl_loss = compute_kl_divergence(mu, logvar).mean()
        recon_loss = F.mse_loss(prediction, target_block, reduction="mean")
        
        loss = compute_latent_jepa_loss(predictions=prediction,
                                        targets=target_block,
                                        mu=mu,
                                        logvar=logvar,
                                        kl_weight=kl_weight)
        
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_kl_divergence", kl_loss, prog_bar=True)
        self.log("train_reconstruction_loss", recon_loss)
        self.log("train_kl_weight", kl_weight)

        return loss

    def validation_step(self,
                      batch,
                      batch_idx
                      ):
        self.mode = "train"
        x_img, x_dep = batch["student_input"].to(self.device), batch["teacher_input"].to(self.device)

        prediction, target_block, mu, logvar = self(x_img=x_img,
                                                    x_dep=x_dep)
        kl_weight = self.get_kl_weight()

        kl_loss = compute_kl_divergence(mu, logvar).mean()
        recon_loss = F.mse_loss(prediction, target_block, reduction="mean")
        
        loss = compute_latent_jepa_loss(predictions=prediction,
                                        targets=target_block,
                                        mu=mu,
                                        logvar=logvar,
                                        kl_weight=kl_weight)
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_kl_divergence", kl_loss, prog_bar=True)
        self.log("val_reconstruction_loss", recon_loss)
        self.log("val_kl_weight", kl_weight)

        return loss

    def predict_step(self,
                      batch,
                      batch_idx
                      ):
        self.mode = "test"
        x_img, x_dep = batch["student_input"].to(self.device), batch["teacher_input"].to(self.device)
        
        x_img = x_img.to(self.device)
        x_dep = x_dep.to(self.device)
                
        return self(x_img=x_img,
                    x_dep=x_dep
                    )
    
    def configure_optimizers(self,):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.lr,
            total_steps=self.trainer.estimated_stepping_batches,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }
       
