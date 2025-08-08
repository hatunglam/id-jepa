from natsort import natsorted
from pathlib import Path
import cv2
import torch
from torch.utils.data import Dataset
import numpy as np

class RGBDImageDataset(Dataset):
    def __init__(self, dataset_dir, transform_pair=False, transform_pair_fn=None,
                 transform_rgb=None, transform_depth=None, img_resize=None):
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

        self.max_depth_m = 10.0

        self.transform_pair = transform_pair
        self.transform_pair_fn = transform_pair_fn
        self.transform_rgb = transform_rgb
        self.transform_depth = transform_depth
        self.img_resize = img_resize  

    def __len__(self):
        return len(self.rgb_files)

    def __getitem__(self, idx):

        rgb_path = self.rgb_files[idx]
        depth_path = self.depth_files[idx]

        rgb_image = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
        rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)

        depth_image = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        # depth_image = depth_image.byteswap().astype("float32")
        depth_image = depth_image.astype("float32")
        depth_image[(depth_image <= 0.0) | (depth_image >= 65503.0)] = 0.0
        depth_image = depth_image / (1000.0) 
        depth_image = np.clip(depth_image, 0, self.max_depth_m) / self.max_depth_m

        if self.img_resize and self.transform_pair is None:
            rgb_image = cv2.resize(rgb_image, (self.img_resize, self.img_resize), interpolation=cv2.INTER_LINEAR)
            depth_image = cv2.resize(depth_image, (self.img_resize, self.img_resize), interpolation=cv2.INTER_NEAREST)

        rgb_image = torch.from_numpy(rgb_image).permute(2,0,1).float() / 255.0
        depth_image = torch.from_numpy(depth_image).unsqueeze(0).float()

        return rgb_image, depth_image

    


