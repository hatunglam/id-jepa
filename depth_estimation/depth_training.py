import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from .depth_estimator import DepthEstimatorBase
import torch.nn.functional as F

# def silog_loss(pred, target, mask=None, eps=1e-6):
#     pred = pred.squeeze(1)
#     target = target.squeeze(1)

#     if mask is None:
#         mask = (target > 0).float()
#     else:
#         mask = mask.squeeze(1)

#     log_diff = torch.log(pred + eps) - torch.log(target + eps)
#     log_diff = log_diff * mask

#     n_valid = mask.sum()
#     silog1 = (log_diff ** 2).sum() / n_valid
#     silog2 = (log_diff.sum() ** 2) / (n_valid ** 2)

#     return silog1 - 0.85 * silog2

def si_log_loss(pred, target, mask=None, eps=1e-8, lambd=0.5):
    """https://github.com/DepthAnything/Depth-Anything-V2/blob/main/metric_depth/util/loss.py#L5"""
    # Scale-invariant logarithmic loss
    log_pred = torch.log(pred + eps)
    log_target = torch.log(target + eps)
    
    if mask is not None:
        log_pred = log_pred[mask]
        log_target = log_target[mask]
    
    d = log_pred - log_target
    return torch.mean(d**2) - lambd * (torch.mean(d)**2)

class DepthEstimator(DepthEstimatorBase, pl.LightningModule):
    def __init__(self,
                 image_encoder,
                 depth_estimator,
                 lr=1e-4,
                 weight_decay=1e-5):
        pl.LightningModule.__init__(self)
        DepthEstimatorBase.__init__(self,
                                    image_encoder=image_encoder,
                                    depth_estimator=depth_estimator)

        self.lr = lr
        self.weight_decay = weight_decay
        self.save_hyperparameters()
        
    def forward(self, pixel_value):
        return self.forward_base(pixel_value)
        
    def training_step(self, batch, batch_idx):
        x_img, x_dep = batch["image_input"].to(self.device), batch["depth_input"].to(self.device)
        pred_depth = self(x_img)
        if pred_depth.shape[-2:] != x_dep.shape[-2:]:
            pred_depth = post_process_depth_estimation(
                outputs=pred_depth,
                target_sizes=[tuple(x_img.shape[-2:])] * x_img.shape[0],
            )
            
        loss = si_log_loss(pred_depth, x_dep)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x_img, x_dep = batch["image_input"].to(self.device), batch["depth_input"].to(self.device)
        pred_depth = self(x_img)
        if pred_depth.shape[-2:] != x_dep.shape[-2:]:
            x_dep = F.interpolate(x_dep.unsqueeze(1) if len(x_dep.shape)==3 else x_dep, 
                                size=pred_depth.shape[-2:], mode='nearest').squeeze(1) if len(pred_depth.shape)==3 else x_dep
        loss = si_log_loss(pred_depth, x_dep)
        self.log("val_loss", loss)
        return loss
    
    def predict_step(self, batch, batch_idx):
        x_img = batch["image_input"].to(self.device)
        pred_depth = self(x_img) 
        return pred_depth

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
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

def post_process_depth_estimation(
    outputs: "DepthEstimatorOutput",
    target_sizes: Optional[Union[torch.Tensor, list[tuple[int, int]], None]] = None,
) -> list[dict[str, torch.Tensor]]:
    """
    https://github.com/huggingface/transformers/blob/main/src/transformers/models/dpt/image_processing_dpt.py#L637
    
    Converts the raw output of [`DepthEstimatorOutput`] into final depth predictions and depth PIL images.
    Only supports PyTorch.

    Args:
        outputs ([`DepthEstimatorOutput`]):
            Raw outputs of the model.
        target_sizes (`torch.Tensor` or `list[tuple[int, int]]`, *optional*):
            Tensor of shape `(batch_size, 2)` or list of tuples (`tuple[int, int]`) containing the target size
            (height, width) of each image in the batch. If left to None, predictions will not be resized.

    Returns:
        `list[dict[str, torch.Tensor]]`: A list of dictionaries of tensors representing the processed depth
        predictions.
    """

    predicted_depth = outputs.predicted_depth

    if (target_sizes is not None) and (len(predicted_depth) != len(target_sizes)):
        raise ValueError(
            "Make sure that you pass in as many target sizes as the batch dimension of the predicted depth"
        )

    results = []
    target_sizes = [None] * len(predicted_depth) if target_sizes is None else target_sizes
    for depth, target_size in zip(predicted_depth, target_sizes):
        if target_size is not None:
            depth = torch.nn.functional.interpolate(
                depth.unsqueeze(0).unsqueeze(1), size=target_size, mode="bicubic", align_corners=False
            ).squeeze()

        results.append({"predicted_depth": depth})

    return results
