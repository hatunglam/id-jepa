from pathlib import Path
from random import shuffle
import cv2
from tqdm import tqdm

# --- Set up paths ---
RAW_DATA_DIR = Path("D:/Documents_D/nyu_data")
OUTPUT_DIR = RAW_DATA_DIR.parent / "processed_data"

def extract_timestamp(file):
    return float(file.stem.split("-")[1])

def align_rgb_depth(scene_dir, max_gap=0.05):
    rgb_files = sorted(scene_dir.glob("r-*.ppm"))
    depth_files = sorted(scene_dir.glob("d-*.pgm"))

    rgb_dict = {extract_timestamp(f): f for f in rgb_files}
    depth_dict = {extract_timestamp(f): f for f in depth_files}

    aligned = []
    for r_ts, r_file in rgb_dict.items():
        closest_d_ts = min(depth_dict.keys(), key=lambda d: abs(d - r_ts))
        if abs(closest_d_ts - r_ts) <= max_gap:
            d_file = depth_dict[closest_d_ts]
            aligned.append((r_file, d_file))
    return aligned

def save_pairs(pairs, split):
    rgb_dir = OUTPUT_DIR / split / "rgb"
    depth_dir = OUTPUT_DIR / split / "depth"
    rgb_dir.mkdir(parents=True, exist_ok=True)
    depth_dir.mkdir(parents=True, exist_ok=True)

    i = 0
    for rgb_path, depth_path in tqdm(pairs, desc=f"Saving aligned pairs: {split}"):
        rgb_img = cv2.imread(str(rgb_path))
        depth_img = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)

        # Skip corrupted or unreadable images
        if rgb_img is None or depth_img is None:
            continue

        cv2.imwrite(str(rgb_dir / f"{i:05d}.png"), rgb_img)
        cv2.imwrite(str(depth_dir / f"{i:05d}.png"), depth_img)
        i += 1

def main():
    all_pairs = []
    scene_dirs = sorted([p for p in RAW_DATA_DIR.iterdir() if p.is_dir()])

    for scene in tqdm(scene_dirs, desc="Aligning all scenes"):
        pairs = align_rgb_depth(scene)
        all_pairs.extend(pairs)

    shuffle(all_pairs)

    # Split
    n = len(all_pairs)
    train_pairs = all_pairs[:int(n * 0.8)]
    val_pairs = all_pairs[int(n * 0.8):int(n * 0.9)]
    test_pairs = all_pairs[int(n * 0.9):]

    print(f"Total aligned pairs: {n}")
    print(f"Train: {len(train_pairs)} | Val: {len(val_pairs)} | Test: {len(test_pairs)}")

    save_pairs(train_pairs, "train")
    save_pairs(val_pairs, "val")
    save_pairs(test_pairs, "test")

if __name__ == "__main__":
    main()