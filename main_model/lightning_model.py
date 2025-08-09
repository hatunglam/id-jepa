from typing import Callable, Dict, List, Optional, Set, Tuple, Union
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from .model import ID_JEPA
from utils.types import Number

class ID_JEPA_Lightning(pl.LightningModule):
    def __init__(self,
                 experiment_config,
                 runtime_config,
                 tracking_config,
                 model_config,
                 dataset_config,
                 testing_purposes_only=False):
        super().__init__()
        if not testing_purposes_only:
            self.save_hyperparameters()
        self.experiment_config = experiment_config
        self.runtime_config = runtime_config
        self.tracking_config = tracking_config
        self.model_config = model_config
        self.dataset_config = dataset_config
        self.model = ID_JEPA()
        self.criterion = nn.MSELoss()

    @staticmethod
    def randomly_select_starting_patch_for_block(patch_dim: Tuple[int, int],
                                                 block_dim: Tuple[int, int],
                                                 seed: Optional[int] = None,) -> int:
        if seed is not None:
            torch.manual_seed(seed)

        def random_int(limit: int) -> int:
            return torch.randint(0, limit, (1,)).item()

        num_patches_h, num_patches_w = (patch_dim)
        num_blocks_h, num_blocks_w = (block_dim)

        max_start_index_h: int = num_patches_h - num_blocks_h + 1
        max_start_index_w: int = num_patches_w - num_blocks_w + 1
        assert all((num_blocks_h <= num_patches_h,
                    num_blocks_w <= num_patches_w,)),\
                    f"Blocks cannot be smaller than patches along any dimension, but there were more blocks than patches along at least one dimension ({patch_dim=}, {block_dim=})"

        start_index_h: int = random_int(max_start_index_h)
        start_index_w: int = random_int(max_start_index_w)

        start_index: int = (start_index_h * num_patches_w) + start_index_w

        return start_index

    @staticmethod
    def generate_target_patches(patch_dim: Tuple[int, int],
                                aspect_ratio: Number,
                                scale: Number,
                                num_target_blocks: int,
                                seed: Optional[int] = None,) -> Tuple[List[List[int]], Set[int]]:
        num_patches_h, num_patches_w = patch_dim
        num_patches_block: int = int(num_patches_h * num_patches_w * scale)
        num_blocks_h: int = int(torch.sqrt(torch.tensor(num_patches_block / aspect_ratio)))
        num_blocks_w: int = int(aspect_ratio * num_blocks_h)
        block_dim: Tuple[int, int] = num_blocks_h, num_blocks_w
        target_patches: List[List[int]] = []
        all_patches: Set[int] = set()
        _target_patches: List[List[int]] = []
        _all_patches: Set[int] = set()

        for target_block_idx in range(num_target_blocks):
            start_patch: int = ID_JEPA_Lightning.randomly_select_starting_patch_for_block(patch_dim=patch_dim,
                                                                                          block_dim=block_dim,
                                                                                          seed=target_block_idx * seed if seed is not None else None,)
            patches: List[int] = []
            for h in range(num_blocks_h):
                for w in range(num_blocks_w):
                    patch_start_position: int = start_patch + h * num_patches_w + w
                    patches.append(patch_start_position)
                    all_patches.add(patch_start_position)

            target_patches.append(patches)

            h = torch.arange(num_blocks_h)
            w = torch.arange(num_blocks_w)
            hw_grid = torch.cartesian_prod(h, w)

            block_patch_indices = start_patch + (hw_grid[:, 0] * num_patches_w + hw_grid[:, 1])

            _target_patches.append(block_patch_indices.tolist())
            _all_patches.update(block_patch_indices.tolist())

        assert len(target_patches) == len(_target_patches)
        assert len(all_patches) == len(_all_patches)
        assert target_patches == _target_patches
        assert all_patches == _all_patches
        return target_patches, all_patches

    @staticmethod
    def generate_context_patches(patch_dim: Tuple[int, int],
                                 aspect_ratio: Number,
                                 scale: Number,
                                 target_patches_to_exclude: Set[int],
                                 seed: Optional[int] = None,) -> List[int]:
        num_patches_h, num_patches_w = patch_dim
        num_patches_block: int = int(num_patches_h * num_patches_w * scale)
        num_blocks_h: int = int(torch.sqrt(torch.tensor(num_patches_block / aspect_ratio)))
        num_blocks_w: int = int(aspect_ratio * num_blocks_h)
        block_dim: Tuple[int, int] = num_blocks_h, num_blocks_w
        start_patch: int = ID_JEPA_Lightning.randomly_select_starting_patch_for_block(patch_dim=patch_dim,
                                                                                      block_dim=block_dim,
                                                                                      seed=seed,)
        context_patches_set: Set[int] = set()
        for h in range(num_blocks_h):
            for w in range(num_blocks_w):
                patch_start_position: int = start_patch + h * num_patches_w + w
                context_patches_set.add(patch_start_position)
        context_patches: List[int] = list(context_patches_set.difference(target_patches_to_exclude))

        h = torch.arange(num_blocks_h)
        w = torch.arange(num_blocks_w)
        hw_grid = torch.cartesian_prod(h, w)

        _context_patches_tensor: torch.Tensor = start_patch + (
            +hw_grid[:, 0] * num_patches_w + hw_grid[:, 1]
        )
        _context_patches_set = set(_context_patches_tensor.tolist())
        _context_patches: List[int] = list(
            _context_patches_set.difference(target_patches_to_exclude)
        )
        assert len(context_patches) == len(_context_patches)
        assert context_patches == _context_patches
        return context_patches

    def _forward_step(self,x: torch.Tensor,
                      target_aspect_ratio: float,
                      target_scale: float,
                      context_aspect_ratio: Number,
                      context_scale: float,) -> Tuple[torch.Tensor, torch.Tensor]:
        target_patches: List[List[int]]
        all_unique_target_patches: Set[int]
        target_patches, all_unique_target_patches = ID_JEPA_Lightning.generate_target_patches(patch_dim=self.patch_embed.patch_shape,
                                                                                              aspect_ratio=target_aspect_ratio,
                                                                                              scale=target_scale,
                                                                                              num_target_blocks=self.num_target_blocks,)
        context_patches: List[int] = ID_JEPA_Lightning.generate_context_patches(patch_dim=self.patch_embed.patch_shape,
                                                                                aspect_ratio=context_aspect_ratio,
                                                                                scale=context_scale,
                                                                                target_patches_to_exclude=all_unique_target_patches,)

        return self.model(x=x, target_patches=target_patches, context_patches=context_patches,)

    def update_momentum(self, m: float) -> None:
        student_model: nn.Module = self.model.student.eval()
        teacher_model: nn.Module = self.model.teacher.eval()
        with torch.no_grad():
            for student_param, teacher_param in zip(student_model.parameters(), 
                                                    teacher_model.parameters()):
                teacher_param.data.mul_(other=m).add_(other=student_param.data, alpha=1 - m)

    def training_step(self,
                      batch: torch.Tensor,
                      batch_idx: int,
                      dataloader_idx: int = 0,):
        target_aspect_ratio: float = np.random.uniform(self.target_aspect_ratio[0], self.target_aspect_ratio[1])
        target_scale: float = np.random.uniform(low=self.target_scale_interval[0], high=self.target_scale_interval[1])
        context_scale: float = np.random.uniform(self.context_scale[0], self.context_scale[1])

        y_student, y_teacher = self._forward_step(x=batch,
                                                  target_aspect_ratio=target_aspect_ratio,
                                                  target_scale=target_scale,
                                                  context_aspect_ratio=self.context_aspect_ratio,
                                                  context_scale=context_scale,)
        
        loss: torch.Tensor = self.criterion(y_student, y_teacher)
        self.log("train_loss", loss)
        return loss

    def validation_step(self,
                        batch: torch.Tensor,
                        batch_idx: int,
                        dataloader_idx: int = 0,):
        target_aspect_ratio: float = np.random.uniform(self.target_aspect_ratio[0], self.target_aspect_ratio[1])
        target_scale: float = np.random.uniform(low=self.target_scale_interval[0], high=self.target_scale_interval[1])
        context_scale: float = np.random.uniform(self.context_scale[0], self.context_scale[1])

        y_student, y_teacher = self._forward_step(x=batch,
                                                  target_aspect_ratio=target_aspect_ratio,
                                                  target_scale=target_scale,
                                                  context_aspect_ratio=self.context_aspect_ratio,
                                                  context_scale=context_scale,)
        
        loss: torch.Tensor = self.criterion(y_student, y_teacher)
        self.log("val_loss", loss)
        return loss

    def predict_step(self,
                     batch: torch.Tensor,
                     batch_idx: int,
                     dataloader_idx: int = 0,) -> torch.Tensor:
        self.mode = "test"
        return self.model(x=batch, target_patches=None, context_patches=None)

    def on_after_backward(self) -> None:
        self.update_momentum(self.m)
        self.m += (self.momentum_limits[1] - self.momentum_limits[0]) / self.trainer.estimated_stepping_batches

    def configure_optimizers(self,
    ) -> Dict[str, Union[Callable, Dict[str, Union[str, Callable]]]]:
        optimizer: Callable = torch.optim.AdamW(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        scheduler: Callable = torch.optim.lr_scheduler.OneCycleLR(
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