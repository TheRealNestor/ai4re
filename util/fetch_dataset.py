import kagglehub
import os
import shutil

path = kagglehub.dataset_download("iamvaibhav100/software-requirements-dataset")

dest_dir = os.path.join(os.path.dirname(__file__), "..", "data", "software-requirements-dataset")
os.makedirs(dest_dir, exist_ok=True)

# Copy all files from the downloaded dataset to the destination directory
for item in os.listdir(path):
    s = os.path.join(path, item)
    d = os.path.join(dest_dir, item)
    if os.path.isdir(s):
        shutil.copytree(s, d, dirs_exist_ok=True)
    else:
        shutil.copy2(s, d)


print("Dataset files copied to (relative):", os.path.relpath(dest_dir, os.path.dirname(__file__)))
