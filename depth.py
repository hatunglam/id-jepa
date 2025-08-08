
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt

# Path to your depth image
depth_path = r"D:/Documents_D/test_data/train/depth/00005.png"
# depth_path = r"D:/Documents_D/processed_data/train/depth/00025.png"
# depth_path = r"D:/Documents_D/nyu_data/living_room_0003/d-1294887164.578458-4010400095.pgm"


# Example paths


# ---- Load Depth (16-bit) ----
depth_image = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED).astype("float32")
print("Depth min/max (meters) imread", depth_image .min(), depth_image .max())
# Mask invalid depths
depth_image[(depth_image <= 0.0) | (depth_image >= 65535.0)] = 0.0
# Convert mm -> meters
depth_image = depth_image / 1000.0  # meters 0->10m
print("Depth min/max (meters) before norm:", depth_image .min(), depth_image .max())
# Normalize for training (optional)
max_depth_m = 10.0 # example max depth in meters

depth_image = np.clip(depth_image, 0, max_depth_m) 

depth_image /= max_depth_m
print("Depth min/max (meters):", depth_image .min(), depth_image .max())
# ---- Torch conversion (for your dataset) ----
depth_image = torch.from_numpy(depth_image).unsqueeze(0).float()



# ---- Plotting ----
plt.figure(figsize=(12, 5))
depth_vis = depth_image .squeeze().cpu().numpy()
depth_vis[depth_vis == 0] = np.nan  # Hide invalid pixels

plt.imshow(depth_vis, cmap='grey')
plt.colorbar(label='Depth (meters)')
plt.title("Depth Image")
plt.axis("off")

plt.show()