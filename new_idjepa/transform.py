import torch
import torchvision
import torchvision.transforms as transforms 
import numpy as np

def center_crop(x, size=(224,224)):
    crop = transforms.CenterCrop(size)
    x_crop = crop(x)
    return x_crop

def normalize(img):
    # img_n = ((img.T - img.mean((1,2))) / img.std((1,2))).T
    img_n = (img - img.mean((1,2), keepdim=True)) / img.std((1,2), keepdim=True)
    return img_n

def image_preprocessor(image):
    image = center_crop(image)
    image_norm = normalize(image)
    return image_norm

def depth_preprocessor(depth):
    # Crop and normalize
    depth = center_crop(depth)
    depth_norm = normalize(depth)

    # Stacking to match 3 channels
    depth_tensor = depth_norm.repeat(3, 1, 1)
    return depth_tensor













