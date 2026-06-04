from pathlib import Path
import torch
from .utils import get_base_dir, local_run

base_dir = get_base_dir()
print(f"Base directory: {base_dir}")

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f"Using device: {device}.")

CHECKPOINT_PATH = f"{base_dir}/models"
Path(CHECKPOINT_PATH).mkdir(parents=True, exist_ok=True)

paths = {
    "RGB": base_dir / "data" / "RGB" /"raw",
    "AM4": base_dir / "data" / "AM4" / "raw",
    "AT4": base_dir / "data" / "AT4" / "raw",
}

downloads = {
    "RGB": ["https://doi.org/10.5281/zenodo.3233081", "Amazon Forest Dataset.rar"],
    "AM4": ["https://doi.org/10.5281/zenodo.4498086","AMAZON.rar"],
    "AT4": ["https://doi.org/10.5281/zenodo.4498086","ATLANTIC FOREST.rar"]
}

for path in paths.values():
    path.mkdir(parents=True, exist_ok=True)

train_dataset_dir = paths["AM4"] / "amazon" / "Training"
test_dataset_dir = paths["AM4"] / "amazon" / "Test"
val_dataset_dir = paths["AM4"] / "amazon" / "Validation"

train_dataset_dir_imgs = train_dataset_dir / "image"
test_dataset_dir_imgs = test_dataset_dir / "image"
val_dataset_dir_imgs = val_dataset_dir / "images"

train_dataset_dir_masks = train_dataset_dir / "label"
test_dataset_dir_masks = test_dataset_dir / "mask"
val_dataset_dir_masks = val_dataset_dir / "masks"

img_dirs = [train_dataset_dir_imgs, test_dataset_dir_imgs, val_dataset_dir_imgs]
mask_dirs = [train_dataset_dir_masks, test_dataset_dir_masks, val_dataset_dir_masks]

if local_run():
    batch_size = 4
    num_workers = 0
else:
    batch_size = 32
    num_workers = 4
