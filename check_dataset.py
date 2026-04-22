from pathlib import Path

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def image_files(image_dir):
    if not image_dir.exists():
        return []
    return sorted(path for path in image_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)


def check_split(dataset_root, split):
    img_dir = dataset_root / "images" / split
    lbl_dir = dataset_root / "labels" / split

    images = {path.stem for path in image_files(img_dir)}
    labels = {path.stem for path in lbl_dir.glob("*.txt")} if lbl_dir.exists() else set()
    overlap = images & labels

    print(f"\n[{split.upper()}]")
    print(f"  Images:         {len(images)}")
    print(f"  Labels:         {len(labels)}")
    print(f"  Matching pairs: {len(overlap)}")
    print(f"  Missing labels: {len(images - labels)}")
    print(f"  Orphan labels:  {len(labels - images)}")

    if images and labels and not overlap:
        print("  [CRITICAL] The filenames do not overlap at all.")
        print(f"  Image range: {min(images)} -> {max(images)}")
        print(f"  Label range: {min(labels)} -> {max(labels)}")
    elif overlap:
        preview = ", ".join(sorted(overlap)[:5])
        print(f"  Sample matches: {preview}")


def check_dataset_alignment():
    print("\n" + "=" * 60)
    print("  DATASET ALIGNMENT DIAGNOSTIC")
    print("=" * 60)

    dataset_root = Path("datasets")
    if not dataset_root.exists():
        print(f"Error: Dataset root does not exist: {dataset_root}")
        return

    for split in ("train", "val", "test"):
        check_split(dataset_root, split)

    print("=" * 60 + "\n")

if __name__ == "__main__":
    check_dataset_alignment()
