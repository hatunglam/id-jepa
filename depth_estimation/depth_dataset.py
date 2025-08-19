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
                 crop_size=224,
                 max_depth=1000.0):
        self.data_dir = data_dir
        self.mode = mode
        self.max_depth = max_depth
        self.crop_size = crop_size

        # Student image processor 
        self.student_processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
        self.student_processor.do_center_crop = True
        self.student_processor.crop_size = {'height': crop_size[0], 'width': crop_size[1]}
        self.student_processor.do_resize = False

        # Teacher image processor
        self.teacher_processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
        self.teacher_processor.do_center_crop = True
        if isinstance(crop_size, int): crop_size = (crop_size, crop_size)
        self.teacher_processor.crop_size = {'height': crop_size[0], 'width': crop_size[1]}
        self.teacher_processor.do_convert_rgb = False
        self.teacher_processor.do_normalize = False
        self.teacher_processor.do_rescale = False
        self.teacher_processor.do_resize = False

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
        student_input = self.student_processor(images=image,
                                               return_tensors="pt").pixel_values.squeeze(0)  

        depth = np.asarray(Image.open(depth_path), dtype=np.float32).copy() / 10.0  
        depth = np.clip(depth / self.max_depth, 0.0, 1.0)  
        depth_tensor = torch.tensor(depth, dtype=torch.float32).unsqueeze(0) 
        teacher_input = self.teacher_processor(images=depth_tensor,
                                               return_tensors="pt").pixel_values.squeeze(0) 

        return {
            'student_input': student_input, 
            'teacher_input': teacher_input   
        }