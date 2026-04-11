import json
import os
import shutil
import numpy as np

base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'train_sample_videos'))
dataset_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'prepared_dataset'))

print('📁 Creating dataset folder:', dataset_path)
os.makedirs(dataset_path, exist_ok=True)

tmp_fake_path = os.path.join(dataset_path, 'tmp_fake_frames')
os.makedirs(tmp_fake_path, exist_ok=True)

real_path = os.path.join(dataset_path, 'real')
fake_path = os.path.join(dataset_path, 'fake')

os.makedirs(real_path, exist_ok=True)
os.makedirs(fake_path, exist_ok=True)

def get_filename_only(file_path):
    return os.path.basename(file_path).split('.')[0]

# Load metadata
with open(os.path.join(base_path, 'metadata.json')) as f:
    metadata = json.load(f)

real_count = 0
fake_count = 0

print("\n🚀 Processing frames...\n")

for filename in metadata.keys():
    label = metadata[filename]['label']
    folder_name = get_filename_only(filename)

    frame_folder = os.path.join(base_path, folder_name)

    if not os.path.exists(frame_folder):
        print("❌ Missing folder:", frame_folder)
        continue

    images = [f for f in os.listdir(frame_folder) if f.endswith('.jpg')]

    if len(images) == 0:
        print("⚠️ Empty folder:", frame_folder)
        continue

    for img in images:
        src = os.path.join(frame_folder, img)

        if label == 'REAL':
            dst = os.path.join(real_path, f"real_{real_count}.jpg")
            shutil.copyfile(src, dst)
            real_count += 1

        elif label == 'FAKE':
            dst = os.path.join(tmp_fake_path, f"fake_{fake_count}.jpg")
            shutil.copyfile(src, dst)
            fake_count += 1

print("\n✅ Copy completed!")
print("Real frames:", real_count)
print("Fake frames:", fake_count)

# Balance dataset
all_real = os.listdir(real_path)
all_fake = os.listdir(tmp_fake_path)

count = min(len(all_real), len(all_fake))
selected_fake = np.random.choice(all_fake, count, replace=False)

for i, fname in enumerate(selected_fake):
    src = os.path.join(tmp_fake_path, fname)
    dst = os.path.join(fake_path, f"fake_{i}.jpg")
    shutil.copyfile(src, dst)

print("\n🎯 Dataset balanced successfully!")