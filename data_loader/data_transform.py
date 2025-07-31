from torchvision import transforms

def rgb_transform(image_size=224):
    return transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

def depth_transform():
    """
    Transform for Depth image:
    - No resize (to preserve perspective)
    - Convert to tensor
    - Normalize to [-1, 1]
    """
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])