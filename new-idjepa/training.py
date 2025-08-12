from base_model import JEPA_base
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn

class IDJEPA(JEPA_base, pl.LightningModule):
    def __init__(self,
                 image_encoder,
                 image_preprocessor,
                 depth_encoder,
                 depth_preprocessor,
                 decoder_depth,
                 n_heads,
                 predictor_embed_dim,
                 post_enc_norm=False,
                 mode="train",
                 context_ratio_range=(0.85, 0.95),
                 target_mask_range=(0.15, 0.25),
                 freeze="depth",
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
            freeze=freeze,
            )

        self.image_preprocessor = image_preprocessor
        self.depth_preprocessor = depth_preprocessor

        self.lr = lr 
        self.weight_decay = weight_decay

        self.criterion = nn.MSELoss()

    def forward(self, data):
        x_img, x_dep = data["image"], data["depth"]
        return self.forward_base(
            image=x_img,
            depth=x_dep
        )
    
    def training_step(self,
                      batch,
                      batch_idx
                      ):
        self.mode = "train"
        x_img, x_dep = batch["image"], batch["depth"]
        
        x_img = self.image_preprocessor(x_img)
        x_dep = self.depth_preprocessor(x_dep)
                          
        y_predicted, y_teacher = self(x_img=x_img,
                                      x_dep=x_dep
                                      )
        loss = self.criterion(y_predicted, y_teacher)
        self.log("train_loss", loss)

        return loss

    def validation_step(self,
                      batch,
                      batch_idx
                      ):
        self.mode = "train"
        x_img, x_dep = batch["image"], batch["depth"]
        y_predicted, y_teacher = self(x_img=x_img,
                                      x_dep=x_dep
                                      )
        loss = self.criterion(y_predicted, y_teacher)
        self.log("validation_loss", loss)

        return loss

    def predict_step(self,
                      batch,
                      batch_idx
                      ):
        self.mode = "test"
        x_img, x_dep = batch["image"], batch["depth"]
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
       
