import os
import requests
import random
from bs4 import BeautifulSoup
from zipfile import ZipFile
from tqdm import tqdm
from collections import defaultdict

URL = "https://cs.nyu.edu/~fergus/datasets/nyu_depth_v2.html"
SAVE_DIR = "./raw_data"
os.makedirs(SAVE_DIR, exist_ok=True)

# Step 1: Scrape HTML
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

# Step 3: Pick 5 distinct room types randomly
available_room_types = list(room_links.keys())
random.shuffle(available_room_types)
selected_room_types = available_room_types[:5]

print("Selected room types:", selected_room_types)

# Step 4: Download one zip from each selected room type
for room in selected_room_types:
    selected_zip_url = random.choice(room_links[room])
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

print("\n All 5 room types downloaded and extracted.")