from pathlib import Path
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from .dataset import RGBDImageDataset
from .data_transform import rgb_transform, depth_transform


class RGBDDataModule(pl.LightningDataModule):
    def __init__(
        self,
        dataset_dir: Path,
        batch_size: int = 32,
        num_workers: int = 4,
        pin_memory: bool = True,
    ):
        super().__init__()
        self.dataset_dir = Path(dataset_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

    def setup(self, stage=None):
        self.train_dataset = RGBDImageDataset(
            dataset_dir=self.dataset_dir / "train",
            transform_rgb=rgb_transform(),
            transform_depth=depth_transform()
        )
        self.val_dataset = RGBDImageDataset(
            dataset_dir=self.dataset_dir / "val",
            transform_rgb=rgb_transform(),
            transform_depth=depth_transform()
        )
        self.test_dataset = RGBDImageDataset(
            dataset_dir=self.dataset_dir / "test",
            transform_rgb=rgb_transform(),
            transform_depth=depth_transform()
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, batch_size=self.batch_size,
            shuffle=True, num_workers=self.num_workers,
            pin_memory=self.pin_memory
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, batch_size=self.batch_size,
            shuffle=False, num_workers=self.num_workers,
            pin_memory=self.pin_memory
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset, batch_size=self.batch_size,
            shuffle=False, num_workers=self.num_workers,
            pin_memory=self.pin_memory
        )
