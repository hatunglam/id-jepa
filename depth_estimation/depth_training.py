import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from .depth_estimator import DepthEstimatorBase

class DepthEstimator(DepthEstimatorBase, pl.LightningModule):
    def __init__(self,
                 image_encoder,
                 depth_estimator):
        pl.LightningModule.__init__(self)
        DepthEstimatorBase.__init__(self,
                                    image_encoder=image_encoder,
                                    depth_estimator=depth_estimator)
        
    def forward(self, pixel_value):
        return self.forward_base(pixel_value)
        
    def training_step(self, batch, batch_idx):
        pass