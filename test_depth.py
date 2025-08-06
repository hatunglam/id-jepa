import cv2
import torch
import random
from pathlib import Path

# Path to depth images
depth_dir = Path(r"D:\Documents_D\processed_data\train\depth")

# Get all PNG files
depth_files = list(depth_dir.glob("*.png"))

# Pick 3 random depth images
sample_files = random.sample(depth_files, 3)

for file_path in sample_files:
    # Load with OpenCV in original depth format
    depth_img = cv2.imread(str(file_path), cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)

    # Convert to torch tensor for easier handling
    depth_tensor = torch.from_numpy(depth_img).float()

    # Show stats
    print(f"\nFile: {file_path.name}")
    print(f"Shape: {depth_tensor.shape}")
    print(f"Data type: {depth_tensor.dtype}")
    print(f"Min value: {depth_tensor.min().item()}")
    print(f"Max value: {depth_tensor.max().item()}")
    print(f"Unique sample values: {depth_tensor.view(-1)[:10].tolist()}")