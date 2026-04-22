"""
realtime.py — Live Inference Script
====================================
Runs crack detection on a live webcam or connected thermal camera stream.
Press 'Q' to exit.
"""

import argparse
import cv2
import numpy as np
from ultralytics import YOLO

# ── Visualization Settings ──────────────────────────────────────────────────
CRACK_COLOR   = (0, 60, 255)    # Deep red
TEXT_COLOR    = (255, 255, 255) # White
BOX_THICKNESS = 2
FONT          = cv2.FONT_HERSHEY_DUPLEX
FONT_SCALE    = 0.6

def annotate_frame(frame: np.ndarray, result) -> np.ndarray:
    """Semi-transparent mask and bounding box annotation."""
    annotated = frame.copy()
    boxes = result.boxes
    masks = result.masks

    if boxes is None or len(boxes) == 0:
        return annotated

    # 1. Draw Masks
    if masks is not None:
        for mask_coords in masks.xy:
            if len(mask_coords) == 0: continue
            polygon = np.array(mask_coords, dtype=np.int32)
            overlay = annotated.copy()
            cv2.fillPoly(overlay, [polygon], CRACK_COLOR)
            cv2.addWeighted(overlay, 0.4, annotated, 0.6, 0, annotated)
            cv2.polylines(annotated, [polygon], True, CRACK_COLOR, 1, cv2.LINE_AA)

    # 2. Draw Boxes
    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        conf = float(box.conf[0])
        cv2.rectangle(annotated, (x1, y1), (x2, y2), CRACK_COLOR, BOX_THICKNESS)
        label = f"CRACK {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, FONT, FONT_SCALE, 1)
        cv2.rectangle(annotated, (x1, y1 - th - 8), (x1 + tw + 6, y1), CRACK_COLOR, -1)
        cv2.putText(annotated, label, (x1 + 3, y1 - 5), FONT, FONT_SCALE, TEXT_COLOR, 1)

    return annotated

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, default="runs/crack_detection/weights/best.pt")
    parser.add_argument("--source", type=str, default="0", help="Camera index or URL")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    args = parser.parse_args()

    try:
        model = YOLO(args.weights)
    except Exception as e:
        print(f"Error loading model: {e}. Did you train it yet?")
        return

    # Initialize Camera
    # Use index 0 for default webcam, or try 1, 2 for external USB cameras
    source = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        print(f"Error: Cannot open camera source {args.source}")
        return

    print("--- LIVE DETECTION STARTED ---")
    print("Press 'Q' to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Run Inference
        results = model.predict(frame, conf=args.conf, verbose=False)
        annotated_frame = annotate_frame(frame, results[0])

        # Display results
        cv2.putText(annotated_frame, "LIVE TEST", (10, 30), FONT, 1.0, (255, 255, 255), 2)
        cv2.imshow("Crack Detection - Live Stream", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
