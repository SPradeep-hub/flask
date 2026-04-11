import os
import shutil
from sklearn.model_selection import train_test_split

base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'prepared_dataset'))
output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'split_dataset'))

for category in ['real', 'fake']:
    path = os.path.join(base_dir, category)
    images = os.listdir(path)

    train, temp = train_test_split(images, test_size=0.3, random_state=42)
    val, test = train_test_split(temp, test_size=0.5, random_state=42)

    for split_name, split_data in zip(['train', 'val', 'test'], [train, val, test]):
        split_path = os.path.join(output_dir, split_name, category)
        os.makedirs(split_path, exist_ok=True)

        for img in split_data:
            shutil.copyfile(os.path.join(path, img), os.path.join(split_path, img))

print("Dataset split complete!")