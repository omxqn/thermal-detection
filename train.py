"""
train.py — Full YOLO Training Script for Crack Detection
=========================================================
Uses Ultralytics YOLO (YOLOv11 / latest available).
Run from the project root:
    python scripts/train.py
"""

import os
import sys
import torch
import yaml
from pathlib import Path
from ultralytics import YOLO

# ── Project root (one level up from scripts/) ──────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
DATA_YAML = "data.yaml"
WEIGHTS_DIR = Path("weights")
RUNS_DIR = Path("runs")
WEIGHTS_DIR.mkdir(exist_ok=True)
RUNS_DIR.mkdir(exist_ok=True)

# ══════════════════════════════════════════════════════════════════════════════
#  CONFIGURATION — Edit these values before training
# ══════════════════════════════════════════════════════════════════════════════
CONFIG = {
    # Model to use (YOLOv12 Series):
    #   "yolov12n.pt" — nano (1.6ms latency)
    #   "yolov12s.pt" — small
    #   "yolov12m.pt" — medium ← RECOMMENDED for graduation project
    #   "yolov12l.pt" — large
    #   "yolov12x.pt" — extra-large
    "model":       "yolov12n-seg.pt",

    # Path to your data.yaml
    "data":        str(DATA_YAML),

    # Number of training epochs
    # Start with 100; increase to 200-300 if accuracy plateaus
    "epochs":      150,

    # Image size (pixels). Thermal cameras often output 320×240 or 640×512.
    # Use 640 for best accuracy; 320 if GPU memory is limited.
    "imgsz":       640,

    # Batch size. Reduce if you get CUDA out-of-memory errors.
    # -1 = auto-select based on GPU memory (recommended)
    "batch":       16,

    # Number of dataloader workers (threads). Use 4-8 on Windows.
    "workers":     4,

    # Device: "cuda" for GPU, "cpu" for CPU-only, "0" for first GPU
    "device":      "0",  # Use RTX 3070 (first GPU)

    # Output folder name inside runs/
    "project":     str(RUNS_DIR),
    "name":        "crack_detection",

    # Resume training from last checkpoint if interrupted
    "resume":      False,

    # Save best model checkpoint
    "save":        True,

    # Show plots during training
    "plots":       True,

    # ── Crack-Specific Hyperparameters ────────────────────────────────────
    # Learning rate (initial). Low LR helps detect fine cracks.
    "lr0":         0.001,

    # Final learning rate (fraction of lr0)
    "lrf":         0.01,

    # Momentum for SGD optimizer
    "momentum":    0.937,

    # Weight decay (regularization)
    "weight_decay": 0.0005,

    # Warmup epochs — gradually increase LR at the start
    "warmup_epochs": 5,

    # IoU threshold for NMS (Non-Maximum Suppression)
    "iou":         0.5,

    # Confidence threshold for detections
    "conf":        0.25,

    # ── Augmentation Settings (Pushed for Generalization) ───────────────
    # Horizontal flip probability
    "fliplr":      0.5,

    # Vertical flip probability
    "flipud":      0.5,

    # Random rotation (degrees) - Important for cracks at different angles
    "degrees":     15.0,

    # Random translation
    "translate":   0.1,

    # Random scale
    "scale":       0.5,

    # Random shear
    "shear":       2.0,

    # Perspective distortion - Helps with different camera angles
    "perspective": 0.001,

    # Mosaic augmentation (combines 4 images)
    "mosaic":      1.0,

    # Mix-up augmentation probability
    "mixup":       0.15,

    # Copy-paste augmentation (Excellent for segmentation)
    "copy_paste":  0.4,

    # Close mosaic — disable mosaic for final N epochs for stability
    "close_mosaic": 20,

    # ── Loss Weights ──────────────────────────────────────────────────────
    # Box regression loss weight
    "box":         7.5,

    # Classification loss weight
    "cls":         0.5,

    # DFL (Distribution Focal Loss) weight
    "dfl":         1.5,

    # ── Early Stopping ───────────────────────────────────────────────────
    # Stop training if no improvement for N epochs
    "patience":    50,
}


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN TRAINING FUNCTION
# ══════════════════════════════════════════════════════════════════════════════
def check_environment():
    """Print environment info before training starts."""
    print("\n" + "=" * 60)
    print("  CRACK DETECTION — YOLO Training Pipeline")
    print("=" * 60)
    print(f"  Python:     {sys.version.split()[0]}")
    print(f"  PyTorch:    {torch.__version__}")
    print(f"  CUDA avail: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  GPU:        {torch.cuda.get_device_name(0)}")
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  VRAM:       {vram:.1f} GB")
    print(f"  Device:     {CONFIG['device']}")
    print(f"  Data YAML:  {CONFIG['data']}")
    print("=" * 60 + "\n")


def validate_dataset():
    """Check that data.yaml exists and dataset folders are not empty."""
    yaml_path = Path(CONFIG["data"])
    if not yaml_path.exists():
        raise FileNotFoundError(f"data.yaml not found at: {yaml_path}")

    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)

    base_path = Path(data.get("path", ROOT))
    for split in ["train", "val"]:
        img_dir = base_path / data[split]
        if not img_dir.exists():
            print(f"  [WARNING] Image folder missing: {img_dir}")
        else:
            n_imgs = len(list(img_dir.glob("*.[jJpP][pPnN][gGgG]")))
            print(f"  [{split.upper()}] {n_imgs} images found in {img_dir}")

    print()


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def resolve_dataset_root(data, yaml_path):
    dataset_root = data.get("path")
    if not dataset_root:
        return yaml_path.parent.resolve()

    dataset_root = Path(dataset_root)
    if dataset_root.is_absolute():
        return dataset_root
    return (yaml_path.parent / dataset_root).resolve()


def gather_images(image_dir):
    if not image_dir.exists():
        return []
    return sorted(path for path in image_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)


def label_path_for_image(image_path):
    parts = list(image_path.parts)
    try:
        images_index = parts.index("images")
    except ValueError as exc:
        raise RuntimeError(f"Image path is not under an 'images' directory: {image_path}") from exc

    parts[images_index] = "labels"
    return Path(*parts).with_suffix(".txt")


def label_dir_for_image_dir(image_dir):
    parts = list(image_dir.parts)
    try:
        images_index = parts.index("images")
    except ValueError as exc:
        raise RuntimeError(f"Image directory is not under an 'images' directory: {image_dir}") from exc

    parts[images_index] = "labels"
    return Path(*parts)


def detect_label_type(label_files):
    for label_file in label_files:
        if not label_file.exists():
            continue
        for raw_line in label_file.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = raw_line.strip()
            if not line:
                continue
            columns = line.split()
            if len(columns) < 5:
                return "invalid"
            return "segment" if len(columns) > 5 else "detect"
    return "unknown"


def validate_dataset_pairs():
    yaml_path = Path(CONFIG["data"]).resolve()
    if not yaml_path.exists():
        raise FileNotFoundError(f"data.yaml not found at: {yaml_path}")

    with open(yaml_path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}

    dataset_root = resolve_dataset_root(data, yaml_path)
    print("Dataset summary")
    print("-" * 60)
    print(f"YAML:         {yaml_path}")
    print(f"Dataset root: {dataset_root}")

    problems = []
    train_label_type = "unknown"

    for split in ["train", "val"]:
        split_path = data.get(split)
        if not split_path:
            problems.append(f"'{split}' is missing from {yaml_path.name}.")
            continue

        img_dir = (dataset_root / split_path).resolve()
        lbl_dir = label_dir_for_image_dir(img_dir)
        images = gather_images(img_dir)
        labels_for_images = [label_path_for_image(image_path) for image_path in images]
        matched = [image_path for image_path, label_path in zip(images, labels_for_images) if label_path.exists()]
        missing_labels = [image_path for image_path, label_path in zip(images, labels_for_images) if not label_path.exists()]

        orphan_labels = []
        if lbl_dir.exists():
            image_stems = {image_path.stem for image_path in images}
            orphan_labels = [label_path for label_path in lbl_dir.glob("*.txt") if label_path.stem not in image_stems]

        if split == "train":
            train_label_type = detect_label_type([label_path_for_image(image_path) for image_path in matched[:10]])

        print(
            f"{split.upper():<5} images={len(images):<4} "
            f"matched={len(matched):<4} "
            f"missing_labels={len(missing_labels):<4} "
            f"orphan_labels={len(orphan_labels):<4}"
        )

        if not images:
            problems.append(f"No images were found for split '{split}' in {img_dir}.")
        if not matched:
            problems.append(f"No matching image/label pairs were found for split '{split}' in {img_dir}.")

    model_name = Path(CONFIG["model"]).name.lower()
    model_is_segmentation = "-seg" in model_name
    if train_label_type == "detect" and model_is_segmentation:
        problems.append(
            f"Model '{CONFIG['model']}' expects segmentation labels, but the training labels look like detection boxes."
        )
    if train_label_type == "segment" and not model_is_segmentation:
        problems.append(
            f"Model '{CONFIG['model']}' is a detection model, but the training labels look like segmentation polygons."
        )
    if train_label_type == "invalid":
        problems.append("At least one label file is malformed and does not follow YOLO format.")

    print("-" * 60)
    if problems:
        raise RuntimeError(
            "Dataset validation failed:\n- "
            + "\n- ".join(problems)
            + "\n\nFix the dataset pairing or split generation before training."
        )
    print()


def train():
    """Load model and run training with crack-optimized settings."""
    if CONFIG["device"] != "cpu" and not torch.cuda.is_available():
        print("  CUDA is not available in this environment. Falling back to CPU.")
        CONFIG["device"] = "cpu"

    check_environment()
    validate_dataset_pairs()

    # Load pretrained YOLO model (downloads automatically on first run)
    model = YOLO(CONFIG["model"])
    print(f"  Loaded model: {CONFIG['model']}")
    print(f"  Starting training for {CONFIG['epochs']} epochs...\n")

    train_args = {k: v for k, v in CONFIG.items()}

    results = model.train(**train_args)

    print("\n" + "=" * 60)
    print("  TRAINING COMPLETE")
    print("=" * 60)
    print(f"  Best weights saved to: {RUNS_DIR}/crack_detection/weights/best.pt")
    print(f"  Last weights saved to: {RUNS_DIR}/crack_detection/weights/last.pt")
    print("\n  Run evaluation with:")
    print("    python scripts/evaluate.py")
    print("=" * 60 + "\n")

    return results


if __name__ == "__main__":
    train()
