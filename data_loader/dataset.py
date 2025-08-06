from natsort import natsorted
from pathlib import Path
import cv2
import torch
from torch.utils.data import Dataset

class RGBDImageDataset(Dataset):
    def __init__(self, dataset_dir, transform_pair=None, transform_rgb=None, transform_depth=None, img_size=None):
        self.rgb_dir = Path(dataset_dir) / "rgb"
        self.depth_dir = Path(dataset_dir) / "depth"

        # Natural sorting
        rgb_paths = natsorted(list(self.rgb_dir.iterdir()))
        depth_paths = natsorted(list(self.depth_dir.iterdir()))

        # Get names without extension for alignment check
        rgb_names = [p.stem for p in rgb_paths]
        depth_names = [p.stem for p in depth_paths]
        assert rgb_names == depth_names, "RGB and depth file names do not match!"

        self.rgb_files = [str(p) for p in rgb_paths]
        self.depth_files = [str(p) for p in depth_paths]

        self.transform_pair = transform_pair
        self.transform_rgb = transform_rgb
        self.transform_depth = transform_depth
        self.img_size = img_size  

    def __len__(self):
        return len(self.rgb_files)

    def __getitem__(self, idx):
        rgb_path = self.rgb_files[idx]
        depth_path = self.depth_files[idx]

        # Load RGB as grayscale 
        rgb_image = cv2.imread(str(rgb_path), cv2.IMREAD_GRAYSCALE)  # Shape H, W
        rgb_image = torch.from_numpy(rgb_image).float()[None] / 255.0  # Shape 1, H, W

        # Load depth
        depth_image = cv2.imread(str(depth_path), cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        depth_image = torch.from_numpy(depth_image).float()  
        depth_image = depth_image.unsqueeze(0)

        # Filer invalid value 
        depth_image[(depth_image <= 0) | (depth_image >= 65535)] = 0
        
        # Convert from mm to m
        depth_image = depth_image / 1000
        
        # Normalize depth image
        max_dep_val = depth_image.max()
        depth_image = depth_image / max_dep_val

        # Optional resize if no random crop 
        if self.img_size and self.transform_pair is None:
            # Resize RGB 
            rgb_np = rgb_image.squeeze(0).numpy()
            rgb_resized = cv2.resize(rgb_np, dsize=(self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
            rgb_image = torch.from_numpy(rgb_resized).float()[None] / 255.0

            # Resize depth 
            depth_np = depth_image.numpy()
            depth_resized = cv2.resize(depth_np, dsize=(self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)
            depth_image = torch.from_numpy(depth_resized).float()

        # Paired transform 
        if self.transform_pair:
            rgb_image, depth_image = self.transform_pair(rgb_image, depth_image)

        #  Modality-specific transforms 
        if self.transform_rgb:
            rgb_image = self.transform_rgb(rgb_image)
        if self.transform_depth:
            depth_image = self.transform_depth(depth_image)

        return rgb_image, depth_image

    


