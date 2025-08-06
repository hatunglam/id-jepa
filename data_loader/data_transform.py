from torchvision import transforms
import torchvision.transforms.functional as F
import random
import torch

class PairedRandomCrop:
    def __init__(self, size):
        if isinstance(size, int):
            size = (size, size)
        self.size = size 

    def __call__(self, img1, img2):
        # Ensure channel dimension
        if img1.ndim == 2:
            img1 = img1.unsqueeze(0)
        if img2.ndim == 2:
            img2 = img2.unsqueeze(0)

        # Check that dimensions match
        assert img1.shape[1:] == img2.shape[1:], \
            f"RGB and Depth must have the same dimensions before cropping, got {img1.shape} vs {img2.shape}"

        _, h, w = img1.shape
        i, j, th, tw = transforms.RandomCrop.get_params(torch.zeros(1, h, w), output_size=self.size)

        img1 = img1[:, i:i+th, j:j+tw]
        img2 = img2[:, i:i+th, j:j+tw]
        return img1, img2
    
def rgb_transform():
    return transforms.Compose([
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

def depth_transform():
    return transforms.Compose([

    ])