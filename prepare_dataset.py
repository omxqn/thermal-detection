import random
import shutil
from pathlib import Path

DATASET_ROOT = Path("datasets")
IMAGES_TRAIN = DATASET_ROOT / "images" / "train"
LABELS_TRAIN = DATASET_ROOT / "labels" / "train"

IMAGES_VAL = DATASET_ROOT / "images" / "val"
LABELS_VAL = DATASET_ROOT / "labels" / "val"

IMAGES_VAL.mkdir(parents=True, exist_ok=True)
LABELS_VAL.mkdir(parents=True, exist_ok=True)

SPLIT_RATIO = 0.20
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def training_images():
    return sorted(path for path in IMAGES_TRAIN.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)


def matched_pairs():
    pairs = []
    for img_path in training_images():
        lbl_path = LABELS_TRAIN / f"{img_path.stem}.txt"
        if lbl_path.exists():
            pairs.append((img_path, lbl_path))
    return pairs

def split_dataset():
    if not IMAGES_TRAIN.exists() or not LABELS_TRAIN.exists():
        print("Error: Training image/label folders do not exist.")
        return

    pairs = matched_pairs()
    if not pairs:
        print("Error: No matching image/label pairs were found in the training split.")
        return

    if len(pairs) < 5:
        print(f"Error: Only {len(pairs)} matched pairs were found. Add more labeled data before splitting.")
        return

    existing_val_pairs = sum(1 for path in IMAGES_VAL.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)
    if existing_val_pairs:
        print(f"Validation split already contains {existing_val_pairs} image files. Refusing to reshuffle existing data.")
        return

    if len(pairs) != len(training_images()):
        print(f"Found {len(pairs)} matched pairs out of {len(training_images())} training images.")
        print("Only matched pairs will be moved into validation.")

    random.seed(42)
    random.shuffle(pairs)

    n_val = max(1, int(len(pairs) * SPLIT_RATIO))
    if n_val >= len(pairs):
        n_val = len(pairs) - 1

    if n_val <= 0:
        print("Error: No images found in training folder.")
        return

    val_pairs = pairs[:n_val]

    print(f"Moving {len(val_pairs)} matched image/label pairs from train to val...")

    moved_count = 0
    for img_path, lbl_path in val_pairs:
        shutil.move(str(img_path), str(IMAGES_VAL / img_path.name))
        shutil.move(str(lbl_path), str(LABELS_VAL / lbl_path.name))
        moved_count += 1

    print(f"Pipeline sync successful: {moved_count} pairs relocated to validation.")

if __name__ == "__main__":
    split_dataset()
