"""
export.py — Export YOLOv12 model to ONNX, TensorRT, etc.
=========================================================
Exports the trained PyTorch (.pt) weights to deployment formats.
Recommended for high-performance inference.

Usage:
    python scripts/export.py --weights runs/crack_detection/weights/best.pt --format onnx
    python scripts/export.py --weights runs/crack_detection/weights/best.pt --format engine (TensorRT)
"""

import argparse
from pathlib import Path
from ultralytics import YOLO

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, required=True, help="Path to best.pt")
    parser.add_argument("--format", type=str, default="onnx", 
                        choices=["onnx", "engine", "openvino", "tflite", "coreml"],
                        help="Target format")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--half", action="store_true", help="FP16 quantization (faster)")
    args = parser.parse_args()

    weights_path = Path(args.weights)
    if not weights_path.exists():
        print(f"Error: Weights not found at {weights_path}")
        return

    print(f"Loading model: {weights_path}")
    model = YOLO(str(weights_path))

    print(f"Exporting to {args.format} format...")
    # 'half=True' uses FP16, which is much faster on RTX GPUs
    path = model.export(format=args.format, imgsz=args.imgsz, half=args.half)
    
    print("\n" + "="*30)
    print("Export Complete!")
    print(f"File saved to: {path}")
    print("="*30)

if __name__ == "__main__":
    main()
