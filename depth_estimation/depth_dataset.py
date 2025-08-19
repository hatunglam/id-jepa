from transformers import AutoImageProcessor
from torch.utils.data import Dataset
import torch
import numpy as np
import csv
from PIL import Image
from sklearn.utils import shuffle

class DepthDataset(Dataset):
    def __init__(self,
                 data_dir: str, 
                 mode: str,
                 encoder_mode="depthanything-dpt",
                 crop_size=224,
                 max_depth=1000.0):
        self.data_dir = data_dir
        self.mode = mode
        self.max_depth = max_depth
        self.crop_size = crop_size
        self.encoder_mode = encoder_mode.lower()

        if self.encoder_mode == "dino-dpt":
            # image processor when using dino-dpt
            self.image_processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
            self.image_processor.do_center_crop = True
            self.image_processor.crop_size = {'height': crop_size[0], 'width': crop_size[1]}
            self.image_processor.do_resize = False
        elif self.encoder_mode == "depthanything-dpt":
            # image processor when using depthanything-dpt
            self.image_processor = AutoImageProcessor.from_pretrained("depth-anything/Depth-Anything-V2-Base-hf")
            self.image_processor.do_resize = False

        # depth image processor 
        self.depth_processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
        self.depth_processor.do_center_crop = True
        if isinstance(crop_size, int): crop_size = (crop_size, crop_size)
        self.depth_processor.crop_size = {'height': crop_size[0], 'width': crop_size[1]}
        self.depth_processor.do_convert_rgb = False
        self.depth_processor.do_normalize = False
        self.depth_processor.do_rescale = False
        self.depth_processor.do_resize = False

        self._init_dataset()

    def _init_dataset(self):
        with open(f'{self.data_dir}nyu2_{self.mode}.csv', mode='r') as file:
            data_csv = csv.reader(file)
            nyu_data = list(data_csv)
        if self.mode == 'train':
            shuffle(nyu_data, random_state=0)
        self.nyu_data = nyu_data

    def __len__(self):
        return len(self.nyu_data)

    def __getitem__(self, idx):
        image_path = self.data_dir + self.nyu_data[idx][0].removeprefix("data/")
        depth_path = self.data_dir + self.nyu_data[idx][1].removeprefix("data/")

        image = Image.open(image_path).convert("RGB")
        image_input = self.image_processor(images=image,
                                               return_tensors="pt").pixel_values.squeeze(0)  

        depth = np.asarray(Image.open(depth_path), dtype=np.float32).copy() / 10.0  
        depth = np.clip(depth / self.max_depth, 0.0, 1.0)  
        depth_tensor = torch.tensor(depth, dtype=torch.float32).unsqueeze(0) 
        depth_input = self.depth_processor(images=depth_tensor,
                                               return_tensors="pt").pixel_values.squeeze(0) 

        return {
            'image_input': image_input, 
            'depth_input': depth_input   
        }