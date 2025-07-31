import os
import requests
from bs4 import BeautifulSoup
from zipfile import ZipFile
from tqdm import tqdm
from collections import defaultdict

# Target webpage
URL = "https://cs.nyu.edu/~fergus/datasets/nyu_depth_v2.html"

# Destination directory on D: drive
SAVE_DIR = r"D:\Documents_D\nyu_data"
os.makedirs(SAVE_DIR, exist_ok=True)

# Step 1: Scrape HTML content
response = requests.get(URL)
soup = BeautifulSoup(response.content, "html.parser")

# Step 2: Group .zip links by room type
room_links = defaultdict(list)

for td in soup.find_all("td"):
    room_name = td.get_text().split("(")[0].strip().lower()
    for a in td.find_all("a", href=True):
        href = a["href"]
        if href.endswith(".zip"):
            full_url = requests.compat.urljoin(URL, href)
            room_links[room_name].append(full_url)

# Step 3: Define rooms you want to download
target_rooms = ["bedrooms", "kitchens", "bathrooms", "living rooms"]

# Step 4: Download one file per target room type
for room in target_rooms:

    # Use the first zip
    selected_zip_url = room_links[room][0]
    zip_filename = os.path.basename(selected_zip_url)
    zip_path = os.path.join(SAVE_DIR, zip_filename)

    print(f"\n Downloading {room.capitalize()} → {zip_filename}")
    with requests.get(selected_zip_url, stream=True) as r:
        r.raise_for_status()
        total_size = int(r.headers.get('content-length', 0))
        with open(zip_path, 'wb') as f, tqdm(
            desc=zip_filename,
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024
        ) as bar:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
                bar.update(len(chunk))

    print(f" Unzipping to {SAVE_DIR}...")
    with ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(SAVE_DIR)

    os.remove(zip_path)  # delete zip after extracting

print("\n Download Completed.")