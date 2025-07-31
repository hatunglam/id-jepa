import os
import torch

from datamodule import RGBDDataModule

# Set the path to your processed data
DATASET_DIR = "D:/Documents_D/processed_data"

# Instantiate the datamodule
datamodule = RGBDDataModule(
    dataset_dir=DATASET_DIR,
    batch_size=4,  # Small batch for testing
    num_workers=0,  # Set to 0 for simple CPU-safe test
    pin_memory=False
)

# Prepare datasets
datamodule.setup()

# Get one batch from the training dataloader
train_loader = datamodule.train_dataloader()
batch = next(iter(train_loader))

x_rgb, x_depth = batch

print("✅ RGB shape:   ", x_rgb.shape)
print("✅ Depth shape: ", x_depth.shape)
print("Data type:      ", type(x_rgb), type(x_depth))
print("RGB range:      ", f"[{x_rgb.min().item():.2f}, {x_rgb.max().item():.2f}]")
print("Depth range:    ", f"[{x_depth.min().item():.2f}, {x_depth.max().item():.2f}]")

