import torch
from torch.utils.data import Dataset
import numpy as np
import csv
from PIL import Image
from skimage.transform import resize
from sklearn.utils import shuffle
from typing import Union
import albumentations as A

class NYUDataset(Dataset):
    def __init__(self,
                 data_dir: str, 
                 mode: str,
                 max_depth: int = 1000.0,
                 img_resize: Union[int, bool] = False,
                 return_metric_depth: bool = False,
                 apply_transforms: bool = False):
        self.data_dir = data_dir
        self.mode = mode
        self.max_depth = max_depth
        self.img_resize = img_resize
        self.return_metric_depth = return_metric_depth
        self.apply_transforms = apply_transforms
        if apply_transforms:
            self._init_transforms()
        self._init_dataset()
    
    def _init_dataset(self):
        with open(f'{self.data_dir}nyu2_{self.mode}.csv', mode='r') as file:
            data_csv = csv.reader(file)
            nyu_data = list(data_csv)
        if self.mode == 'train':
            shuffle(nyu_data, random_state=0)
        self.nyu_data = nyu_data
    
    def _init_transforms(self):
        self.spatial_transforms = A.Compose([A.HorizontalFlip(p=0.5),
                                             A.VerticalFlip(p=0.2)])
        self.image_only_transforms = A.OneOf([A.ChannelShuffle(p=0.5),
                                              A.OneOf([A.ISONoise(color_shift=(0.0, 0.01), intensity=(0.15, 0.30), p=1.0),
                                                       A.GaussNoise(var_limit=(10.0, 50.0), p=1.0)], p=0.5),
                                              A.RandomGamma(gamma_limit=(80, 125), p=0.5)], 
                                              p=1.0)
    
    def __len__(self):
        return len(self.nyu_data)
    
    def nyu_resize(self, image, img_resize):
        return resize(image, (img_resize, int(img_resize*4/3)),
                      preserve_range=True, mode='reflect', anti_aliasing=True)
    
    def __getitem__(self, idx):
        image = self.data_dir + self.nyu_data[idx][0].removeprefix("data/")
        depth = self.data_dir + self.nyu_data[idx][1].removeprefix("data/")
        
        image = np.asarray(Image.open(image))
        
        if self.mode == 'train':
            depth = np.clip(np.asarray(Image.open(depth))/255*self.max_depth,
                            0,
                            self.max_depth) # 0.0 to 1000 cm
            if self.apply_transforms:
                transformed = self.spatial_transforms(image=image, mask=depth)
                image, depth = transformed['image'], transformed['mask']
                image = self.image_only_transforms(image=image)['image']
        
        if self.mode == 'test':
            depth = np.asarray(Image.open(depth), dtype=np.float32).copy().astype(float) / 10.0 # 0.0 to 1000 cm    
        
        image = np.clip(image / 255., 0.0, 1.0)
        depth = depth / self.max_depth # 0.0 to 1 (Normalized)

        if self.img_resize:
            image = self.nyu_resize(image, self.img_resize)
            depth = self.nyu_resize(depth, self.img_resize)

        data = {'image': torch.as_tensor(image, dtype=torch.float32).permute(2,0,1),
                'depth': torch.as_tensor(depth, dtype=torch.float32).unsqueeze(0)}
        if self.return_metric_depth:
            metric_depth = depth * 10.0 # 0.0 to 10.0 m
            data['metric_depth'] = torch.as_tensor(metric_depth, dtype=torch.float32).unsqueeze(0)
        return data

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    dataset = NYUDataset(data_dir="/home/ec2-user/data/", 
                         mode="train",
                         max_depth=1000.0,
                         img_resize=False,
                         return_metric_depth=True,
                         apply_transforms=True)

    from torch.utils.data import DataLoader
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    for i, batch in enumerate(dataloader):
        if i ==0:
            continue
        image = batch['image']
        depth = batch['depth']
        metric_depth = batch['metric_depth']
        print("Image Shape: ", image.shape)
        print("Depth Shape: ", depth.shape)
        print("Metric Depth Shape: ", metric_depth.shape)
        image = image.squeeze().permute(1,2,0)
        depth = depth.squeeze()
        metric_depth = metric_depth.squeeze()
        print("Depth Min: ", depth.min(), " Depth Max:", depth.max(), " Depth Mean:", depth.mean())
        print("Metric Depth Min: ", metric_depth.min(), " Metric Depth Max:", metric_depth.max(), " Metric Depth Mean:", metric_depth.mean())

        fig, axs = plt.subplots(1, 3, figsize=(18, 6))

        axs[0].imshow(image)
        axs[0].set_title("RGB Image")
        axs[0].axis("off")

        depth_plot = axs[1].imshow(depth, cmap='inferno')
        axs[1].set_title("Normalized Depth (0-1)")
        axs[1].axis("off")
        fig.colorbar(depth_plot, ax=axs[1], fraction=0.046, pad=0.04)

        metric_plot = axs[2].imshow(metric_depth, cmap='inferno')
        axs[2].set_title("Metric Depth (meters)")
        axs[2].axis("off")
        fig.colorbar(metric_plot, ax=axs[2], fraction=0.046, pad=0.04)

        plt.tight_layout()
        plt.savefig("nyu_sample.png", dpi=300)
        print("Figure saved as 'nyu_sample.png'")
        break