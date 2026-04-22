"""
inference.py — Crack Detection Inference Script
================================================
Runs YOLO detection on a single image, a folder of images,
or a video file. Saves annotated outputs.

Usage examples:
    # Single image
    python scripts/inference.py --source datasets/images/test/img001.jpg

    # Folder of images
    python scripts/inference.py --source datasets/images/test/

    # Video file
    python scripts/inference.py --source path/to/video.mp4

    # Custom confidence threshold
    python scripts/inference.py --source test/ --conf 0.4

    # Use specific weights
    python scripts/inference.py --source test/ --weights runs/crack_detection/weights/best.pt
"""

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

# ── Project root ────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = ROOT / "runs"
DEFAULT_WEIGHTS = RUNS_DIR / "crack_detection" / "weights" / "best.pt"
OUTPUT_DIR = ROOT / "runs" / "inference_results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
#  COLOR SCHEME — Thermal heatmap-style annotation colors
# ══════════════════════════════════════════════════════════════════════════════
CRACK_COLOR   = (0, 60, 255)    # Deep red — stands out on thermal images
TEXT_COLOR    = (255, 255, 255) # White text
BOX_THICKNESS = 2
FONT          = cv2.FONT_HERSHEY_DUPLEX
FONT_SCALE    = 0.6


# ══════════════════════════════════════════════════════════════════════════════
#  ANNOTATE FRAME — Draw bounding boxes with confidence scores
# ══════════════════════════════════════════════════════════════════════════════
def annotate_frame(frame: np.ndarray, result) -> np.ndarray:
    """
    Draw detection bounding boxes and segmentation masks on a frame.

    Args:
        frame:  Original image as numpy array (BGR)
        result: Single YOLO result object

    Returns:
        Annotated image
    """
    annotated = frame.copy()
    boxes = result.boxes
    masks = result.masks

    if boxes is None or len(boxes) == 0:
        # No detections — add "No Crack Detected" overlay
        h, w = annotated.shape[:2]
        cv2.putText(annotated, "No Crack Detected", (10, 30),
                    FONT, 0.8, (0, 200, 0), 2)
        return annotated

    # 1. ── Draw Segmentation Masks ──────────────────────────────────────────
    if masks is not None:
        for mask_coords in masks.xy:
            if len(mask_coords) == 0:
                continue
            polygon = np.array(mask_coords, dtype=np.int32)
            
            # Create a semi-transparent overlay for the mask
            overlay = annotated.copy()
            cv2.fillPoly(overlay, [polygon], CRACK_COLOR)
            cv2.addWeighted(overlay, 0.35, annotated, 0.65, 0, annotated)
            
            # Draw a subtle outline for the polygon
            cv2.polylines(annotated, [polygon], True, CRACK_COLOR, 1, cv2.LINE_AA)

    # 2. ── Draw Bounding Boxes and Labels ──────────────────────────────────
    for box in boxes:
        # Extract box coordinates (xyxy format)
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        conf = float(box.conf[0])

        # Draw bounding box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), CRACK_COLOR, BOX_THICKNESS)

        # Draw label background (filled rectangle)
        label = f"CRACK {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, FONT, FONT_SCALE, 1)
        cv2.rectangle(annotated, (x1, y1 - th - 8), (x1 + tw + 6, y1), CRACK_COLOR, -1)

        # Draw label text
        cv2.putText(annotated, label, (x1 + 3, y1 - 5),
                    FONT, FONT_SCALE, TEXT_COLOR, 1, cv2.LINE_AA)

    # 3. ── Overlay: detection count ─────────────────────────────────────────
    summary = f"Cracks detected: {len(boxes)}"
    h, w = annotated.shape[:2]
    cv2.putText(annotated, summary, (10, h - 15),
                FONT, 0.65, (0, 255, 255), 2, cv2.LINE_AA)

    return annotated


# ══════════════════════════════════════════════════════════════════════════════
#  IMAGE INFERENCE
# ══════════════════════════════════════════════════════════════════════════════
def run_on_images(model: YOLO, source: Path, conf: float, iou: float):
    """Run inference on one or more image files."""
    # Collect images
    if source.is_file():
        images = [source]
    else:
        images = sorted(
            list(source.glob("*.jpg")) +
            list(source.glob("*.jpeg")) +
            list(source.glob("*.png")) +
            list(source.glob("*.bmp")) +
            list(source.glob("*.tif")) +
            list(source.glob("*.tiff"))
        )

    if not images:
        print(f"  [ERROR] No images found in: {source}")
        return

    print(f"  Found {len(images)} image(s). Running inference...")

    total_cracks = 0
    start_time = time.time()

    for idx, img_path in enumerate(images, 1):
        # Read image
        frame = cv2.imread(str(img_path))
        if frame is None:
            print(f"  [SKIP] Cannot read: {img_path.name}")
            continue

        # Run YOLO inference
        results = model.predict(
            source=frame,
            conf=conf,
            iou=iou,
            imgsz=640,
            device="cuda",     # Change to "cpu" if no GPU
            verbose=False,
        )

        result = results[0]
        n_detections = len(result.boxes) if result.boxes else 0
        total_cracks += n_detections

        # Annotate image
        annotated = annotate_frame(frame, result)

        # Save output
        out_path = OUTPUT_DIR / f"detected_{img_path.name}"
        cv2.imwrite(str(out_path), annotated)

        print(f"  [{idx:03d}/{len(images):03d}] {img_path.name} "
              f"→ {n_detections} crack(s) detected → saved to {out_path.name}")

    elapsed = time.time() - start_time
    fps = len(images) / elapsed if elapsed > 0 else 0

    print("\n" + "=" * 60)
    print(f"  INFERENCE COMPLETE")
    print(f"  Images processed: {len(images)}")
    print(f"  Total cracks:     {total_cracks}")
    print(f"  Time elapsed:     {elapsed:.2f}s  ({fps:.1f} FPS)")
    print(f"  Output folder:    {OUTPUT_DIR}")
    print("=" * 60)


# ══════════════════════════════════════════════════════════════════════════════
#  VIDEO INFERENCE
# ══════════════════════════════════════════════════════════════════════════════
def run_on_video(model: YOLO, source: Path, conf: float, iou: float):
    """Run inference on a video file and save annotated video."""
    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        print(f"  [ERROR] Cannot open video: {source}")
        return

    # Video writer setup
    fps    = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out_video = OUTPUT_DIR / f"detected_{source.stem}.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_video), fourcc, fps, (width, height))

    print(f"  Video: {source.name} | {width}x{height} @ {fps:.1f} FPS | {total} frames")
    print("  Processing frames...\n")

    frame_id = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(
            source=frame,
            conf=conf,
            iou=iou,
            imgsz=640,
            device="cuda",
            verbose=False,
        )
        annotated = annotate_frame(frame, results[0])
        writer.write(annotated)

        frame_id += 1
        if frame_id % 30 == 0:
            pct = (frame_id / total * 100) if total > 0 else 0
            print(f"  Progress: {frame_id}/{total} frames ({pct:.1f}%)")

    cap.release()
    writer.release()
    print(f"\n  Output video saved to: {out_video}")


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN / CLI
# ══════════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="Crack Detection — YOLO Inference")
    parser.add_argument(
        "--source", type=str, required=True,
        help="Input: image file, folder of images, or video file"
    )
    parser.add_argument(
        "--weights", type=str, default=str(DEFAULT_WEIGHTS),
        help="Path to trained weights (.pt file)"
    )
    parser.add_argument(
        "--conf", type=float, default=0.35,
        help="Confidence threshold (0.0–1.0). Lower = more detections."
    )
    parser.add_argument(
        "--iou", type=float, default=0.45,
        help="IoU threshold for NMS (Non-Maximum Suppression)"
    )
    args = parser.parse_args()

    weights_path = Path(args.weights)
    source_path  = Path(args.source)

    # Check weights
    if not weights_path.exists():
        raise FileNotFoundError(
            f"Weights not found: {weights_path}\n"
            "Run training first: python scripts/train.py"
        )

    # Check source
    if not source_path.exists():
        raise FileNotFoundError(f"Source not found: {source_path}")

    print("\n" + "=" * 60)
    print("  CRACK DETECTION — Inference")
    print("=" * 60)
    print(f"  Weights:    {weights_path}")
    print(f"  Source:     {source_path}")
    print(f"  Conf:       {args.conf}")
    print(f"  IoU:        {args.iou}")
    print(f"  Output dir: {OUTPUT_DIR}")
    print("=" * 60 + "\n")

    # Load model
    model = YOLO(str(weights_path))
    print(f"  Model loaded: {weights_path.name}\n")

    # Determine input type
    VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".wmv"}
    if source_path.suffix.lower() in VIDEO_EXTS:
        run_on_video(model, source_path, args.conf, args.iou)
    else:
        run_on_images(model, source_path, args.conf, args.iou)


if __name__ == "__main__":
    main()
