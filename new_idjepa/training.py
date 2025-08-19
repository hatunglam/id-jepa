from .base_model import JEPA_base
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from .variational_predictor import loss_vae

class IDJEPA(JEPA_base, pl.LightningModule):
    def __init__(self,
                 image_encoder,
                 depth_encoder,
                 decoder_depth,
                 n_heads,
                 predictor_embed_dim,
                 post_enc_norm=False,
                 mode="train",
                 context_ratio_range=(0.85, 0.95),
                 target_mask_range=(0.15, 0.25),
                 variational_predictor=False,
                 lr=1e-3,
                 weight_decay=0.05):
        pl.LightningModule.__init__(self)
        JEPA_base.__init__(self,
                           image_encoder=image_encoder,
                           depth_encoder=depth_encoder,
                           decoder_depth=decoder_depth,
                           n_heads=n_heads,
                           predictor_embed_dim=predictor_embed_dim,
                           post_enc_norm=post_enc_norm,
                           mode=mode,
                           context_ratio_range=context_ratio_range,
                           target_mask_range=target_mask_range,
                           variational_predictor=variational_predictor)

        self.lr = lr 
        self.weight_decay = weight_decay

        self.criterion = nn.MSELoss()

    def forward(self, x_img, x_dep):
        return self.forward_base(
            image=x_img,
            depth=x_dep
        ) # (prediction, actual)
    
    def training_step(self,
                      batch,
                      batch_idx
                      ):
        self.mode = "train"
        x_img, x_dep = batch["student_input"].to(self.device), batch["teacher_input"].to(self.device)

        y_predicted, y_teacher = self(x_img=x_img,
                                      x_dep=x_dep)
        
        if self.variational_predictor:
            y_student, mu, logvar = y_predicted
            loss = loss_vae(y_teacher, y_student, mu, logvar)
            self.log("train_loss", loss)
        else:
            loss = self.criterion(y_predicted, y_teacher)
            self.log("train_loss", loss)

        return loss

    def validation_step(self,
                      batch,
                      batch_idx
                      ):
        self.mode = "test"
        x_img, x_dep = batch["student_input"].to(self.device), batch["teacher_input"].to(self.device)

        y_predicted, y_teacher = self(x_img=x_img,
                                      x_dep=x_dep)

        if self.variational_predictor:
            y_student, mu, logvar = y_predicted
            loss = loss_vae(y_teacher, y_student, mu, logvar)
            self.log("val_loss", loss)
        else:
            loss = self.criterion(y_predicted, y_teacher)
            self.log("val_loss", loss)

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
       
