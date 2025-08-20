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
                 crop_size=448,
                 max_depth=1000.0,
                ):
        self.data_dir = data_dir
        self.mode = mode
        self.max_depth = max_depth
        self.encoder_mode = encoder_mode.lower()
        if isinstance(crop_size, int): crop_size = (crop_size, crop_size)
        self.crop_size = crop_size

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

        # Raw depth map with center crop
        

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
        image_path = self.data_dir + self.nyu_data[idx][0].removeprefix("data/")
        depth_path = self.data_dir + self.nyu_data[idx][1].removeprefix("data/")

        image = Image.open(image_path)

        image_input = self.image_processor(images=image,
                                            return_tensors="pt").pixel_values.squeeze(0)
        if self.encoder_mode == "depthanything-dpt":
            image_input = self.center_crop(image_input)
              

        if self.mode == 'train':
            depth = np.clip(np.asarray(Image.open(depth_path))/255*self.max_depth,
                            0,
                            self.max_depth) # 0.0 to 1000 cm
        else:
            depth = np.asarray(Image.open(depth_path), dtype=np.float32).copy().astype(float) / 10.0  
        depth = depth / self.max_depth  
        depth_tensor = torch.tensor(depth, dtype=torch.float32).unsqueeze(0) 
        depth_input =  self.center_crop(depth_tensor)
        metric_depth = depth_input * 10.0

        return {
            'image_input': image_input, 
            'depth_input': depth_input.squeeze(0),  
            'metric_depth': metric_depth.squeeze(0) 
        }
    
if __name__ == "__main__":
    dataset = DepthDataset(data_dir='/home/ec2-user/data/data/',
                         mode='train',
                         crop_size=224,
                         )
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)
    for batch in dataloader:
        print(batch["image_input"].shape)
        print(batch["depth_input"].shape, batch["depth_input"].min(), batch["depth_input"].max())
        print(batch["metric_depth"].shape, batch["metric_depth"].min(), batch["metric_depth"].max()) 
      
        break