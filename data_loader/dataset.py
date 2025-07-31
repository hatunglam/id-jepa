import os
from pathlib import Path
from torch.utils.data import Dataset
from PIL import Image


class RGBDImageDataset(Dataset):
    def __init__(self, dataset_dir, transform_pair=None, transform_rgb=None, transform_depth=None):
        """
        Args:
            dataset_dir (Path or str): Directory path containing 'rgb/' and 'depth/' folders.
        """
        self.rgb_dir = Path(dataset_dir) / "rgb"
        self.depth_dir = Path(dataset_dir) / "depth"

        self.rgb_files = sorted(list(self.rgb_dir.glob("*")))
        self.depth_files = sorted(list(self.depth_dir.glob("*")))

        assert len(self.rgb_files) == len(self.depth_files), "Mismatch in dataset size."

        self.transform_pair = transform_pair
        self.transform_rgb = transform_rgb
        self.transform_depth = transform_depth

    def __len__(self):
        return len(self.rgb_files)

    def __getitem__(self, idx):
        rgb_path = self.rgb_files[idx]
        depth_path = self.depth_files[idx]

        rgb_image = Image.open(rgb_path).convert("RGB")
        depth_image = Image.open(depth_path)

        if self.transform_pair:
            rgb_image, depth_image = self.transform_pair(rgb_image, depth_image)

        if self.transform_rgb:
            rgb_image = self.transform_rgb(rgb_image)

        if self.transform_depth:
            depth_image = self.transform_depth(depth_image)

        return rgb_image, depth_image
    


