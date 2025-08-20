import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from .depth_estimator import DepthEstimatorBase
import torch.nn.functional as F
from typing import Union, Optional

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
        x_img, x_dep = batch["image_input"].to(self.device), batch["metric_depth"].to(self.device)
        pred_depth = self(x_img)
        if pred_depth.shape[-2:] != x_dep.shape[-2:]:
            pred_depth = torch.stack(post_process_depth_estimation(
                outputs=pred_depth,
                target_sizes=[tuple(x_img.shape[-2:])] * x_img.shape[0],
            ))
        valid_mask = torch.ones_like(x_dep).bool()
        pred_depth = pred_depth[valid_mask]
        x_dep = x_dep[valid_mask]

        loss = si_log_loss(pred_depth, x_dep)
        self.log("train_loss", loss)
        return loss
    
    def on_validation_start(self):
        self.results = {'d1': torch.tensor([0.0]).cuda(), 'd2': torch.tensor([0.0]).cuda(), 'd3': torch.tensor([0.0]).cuda(), 
                   'abs_rel': torch.tensor([0.0]).cuda(), 'sq_rel': torch.tensor([0.0]).cuda(), 'rmse': torch.tensor([0.0]).cuda(), 
                   'rmse_log': torch.tensor([0.0]).cuda(), 'log10': torch.tensor([0.0]).cuda(), 'silog': torch.tensor([0.0]).cuda()}

    def validation_step(self, batch, batch_idx):
        x_img, x_dep = batch["image_input"].to(self.device), batch["metric_depth"].to(self.device)
        pred_depth = self(x_img)
        if pred_depth.shape[-2:] != x_dep.shape[-2:]:
            pred_depth = torch.stack(post_process_depth_estimation(
                outputs=pred_depth,
                target_sizes=[tuple(x_img.shape[-2:])] * x_img.shape[0],
            ))
        valid_mask = torch.ones_like(x_dep).bool()
        pred_depth = pred_depth[valid_mask]
        x_dep = x_dep[valid_mask]
        evaluation_result = eval_depth(pred_depth, x_dep)
        for k in self.results.keys():
            self.results[k] += evaluation_result[k]
        loss = si_log_loss(pred_depth, x_dep)
        self.log("val_loss", loss)
        return loss
    
    def on_validation_end(self):
        for k, v in self.results.items():
            print(k, v)
            # self.logger.experiment(k, v, prog_bar=True, on_epoch=True, on_step=False)

    
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
    outputs,
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

    predicted_depth = outputs

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

        results.append(depth)

    return results

def eval_depth(pred, target):
    assert pred.shape == target.shape

    thresh = torch.max((target / pred), (pred / target))

    d1 = torch.sum(thresh < 1.25).float() / len(thresh)
    d2 = torch.sum(thresh < 1.25 ** 2).float() / len(thresh)
    d3 = torch.sum(thresh < 1.25 ** 3).float() / len(thresh)

    diff = pred - target
    diff_log = torch.log(pred) - torch.log(target)

    abs_rel = torch.mean(torch.abs(diff) / target)
    sq_rel = torch.mean(torch.pow(diff, 2) / target)

    rmse = torch.sqrt(torch.mean(torch.pow(diff, 2)))
    rmse_log = torch.sqrt(torch.mean(torch.pow(diff_log , 2)))

    log10 = torch.mean(torch.abs(torch.log10(pred) - torch.log10(target)))
    silog = torch.sqrt(torch.pow(diff_log, 2).mean() - 0.5 * torch.pow(diff_log.mean(), 2))

    return {'d1': d1.item(), 'd2': d2.item(), 'd3': d3.item(), 'abs_rel': abs_rel.item(), 'sq_rel': sq_rel.item(), 
            'rmse': rmse.item(), 'rmse_log': rmse_log.item(), 'log10':log10.item(), 'silog':silog.item()}

