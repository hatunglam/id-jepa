import torch
from torch.utils.data import Dataset
import csv
from PIL import Image
from sklearn.utils import shuffle
from typing import Union
import numpy as np
from transformers import AutoImageProcessor

class NYUDataset(Dataset):
    def __init__(self,
                 data_dir: str, 
                 mode: str,
                 crop_size: Union[int, list] = 224,
                 teacher_model: str = "DepthAnything",
                 max_depth: int = 1000.0):
        self.data_dir = data_dir
        self.mode = mode
        self.max_depth = max_depth
        self.teacher_model = teacher_model.lower()
        if isinstance(crop_size, int): crop_size = (crop_size, crop_size)
        self.crop_size = crop_size

        # Student (DINOv2) processor setup
        # Everything is Default except for do_resize we set it as False
        self.dino_processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
        self.dino_processor.do_center_crop = True
        self.dino_processor.crop_size = {'height': crop_size[0], 'width': crop_size[1]}
        self.dino_processor.do_resize = False
        
        # Teacher model setup
        if self.teacher_model == "depthanything":
            # Using DepthAnything as teacher, so use Images directly and use DepthAnything processor
            print("Using DepthAnything Processor for teacher")
            self.teacher_processor = AutoImageProcessor.from_pretrained("depth-anything/Depth-Anything-V2-Base-hf")
            self.teacher_processor.do_resize = False
        else:
            # Using Dinov2 as teacher, so use depth maps, stack 3 times to match RGB channels, then use Dinov2 processor
            # Set everything as False except center crop as True
            print("Using Dinov2 Processor for teacher")
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
    
    def center_crop(self, image):
        h, w = image.shape[-2:]
        crop_h, crop_w = self.crop_size
        top = (h - crop_h) // 2
        left = (w - crop_w) // 2
        cropped_image = image[:, top : top + crop_h, left: left + crop_w]
        return cropped_image
    
    def __getitem__(self, idx):
        image = self.data_dir + self.nyu_data[idx][0].removeprefix("data/")
        depth = self.data_dir + self.nyu_data[idx][1].removeprefix("data/")
        image = Image.open(image)
        student_input =  self.dino_processor(images=image,
                                             return_tensors="pt").pixel_values.squeeze(0)
        
        if self.teacher_model == "depthanything":
            # For DepthAnything, we use the image directly
            teacher_input = self.teacher_processor(images=image,
                                                   return_tensors="pt").pixel_values.squeeze(0)
            teacher_input = self.center_crop(teacher_input)
        else:
            # For Dinov2, we use the depth map and stack it to match RGB channels
            if self.mode == 'train':
                depth = np.clip(np.asarray(Image.open(depth))/255*self.max_depth,
                                0,
                                self.max_depth) # 0.0 to 1000 cm
            else:
                depth = np.asarray(Image.open(depth), dtype=np.float32).copy().astype(float) / 10.0
            depth = depth / self.max_depth
            teacher_input = torch.as_tensor(depth, dtype=torch.float32).unsqueeze(0)
            teacher_input = teacher_input.repeat(3, 1, 1)  # Repeat to match RGB channels
            teacher_input = self.teacher_processor(images=teacher_input,
                                                   return_tensors="pt").pixel_values.squeeze(0)
        
        return {'student_input': student_input,
            'teacher_input': teacher_input}


if __name__ == "__main__":
    # Example usage
    dataset = NYUDataset(data_dir='/home/ec2-user/data/data/',
                         mode='train',
                         crop_size=224,
                         teacher_model='DepthAnything')
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)
    for batch in dataloader:
        student_input = batch['student_input']
        teacher_input = batch['teacher_input']
        print(f"Student input shape: {student_input.shape}")
        print(f"Teacher input shape: {teacher_input.shape}")
        student_input = student_input.squeeze().permute(1,2,0)
        teacher_input = teacher_input.squeeze().permute(1,2,0)
        import matplotlib.pyplot as plt
        fig, axs = plt.subplots(1, 2, figsize=(18, 6))

        axs[0].imshow(student_input)
        axs[0].set_title("student_input")
        axs[0].axis("off")
        
        axs[0].imshow(teacher_input)
        axs[0].set_title("teacher_input")
        axs[0].axis("off")

        plt.tight_layout()
        plt.savefig("nyu_sample.png", dpi=300)
        print("Figure saved as 'nyu_sample.png'")
        break