from torchvision import transforms
import torchvision.transforms.functional as F
import random

class PairedRandomCrop:
    def __init__(self, size: int | tuple):
        if isinstance(size, int):
            size = (size, size)
        self.size = size  # (h, w)

    def __call__(self, img1, img2):
        # Ensure both images are the same size
        assert img1.size == img2.size, "RGB and Depth must have the same dimensions before cropping"
        
        i, j, h, w = transforms.RandomCrop.get_params(img1, output_size=self.size)
        img1 = F.crop(img1, i, j, h, w)
        img2 = F.crop(img2, i, j, h, w)
        return img1, img2

def rgb_transform():
    return transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

def depth_transform():
    return transforms.Compose([
    transforms.ToTensor()
])