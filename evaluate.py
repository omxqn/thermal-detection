"""
evaluate.py — Model Evaluation Script for Crack Detection
==========================================================
Computes mAP50, mAP50-95, Precision, Recall, F1.
Generates confusion matrix and per-class metrics.

Run from project root:
    python scripts/evaluate.py
    python scripts/evaluate.py --weights runs/crack_detection/weights/best.pt
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from ultralytics import YOLO

# ── Project root ────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
DATA_YAML = ROOT / "data.yaml"
RUNS_DIR = ROOT / "runs"
LOGS_DIR = ROOT / "logs"
LOGS_DIR.mkdir(exist_ok=True)

DEFAULT_WEIGHTS = RUNS_DIR / "segment" / "runs" / "crack_detection3" / "weights" / "best.pt"


# ══════════════════════════════════════════════════════════════════════════════
#  EVALUATION FUNCTION
# ══════════════════════════════════════════════════════════════════════════════
def evaluate(weights: str, split: str = "val", conf: float = 0.25, iou: float = 0.5):
    """
    Evaluate a trained model on the validation or test set.

    Args:
        weights: Path to model weights (.pt file)
        split:   Dataset split to evaluate — 'val' or 'test'
        conf:    Confidence threshold
        iou:     IoU threshold for NMS
    """
    weights = Path(weights)
    if not weights.exists():
        raise FileNotFoundError(
            f"Weights not found: {weights}\n"
            "Train the model first: python scripts/train.py"
        )

    print("\n" + "=" * 60)
    print("  CRACK DETECTION — Model Evaluation")
    print("=" * 60)
    print(f"  Weights:  {weights}")
    print(f"  Dataset:  {DATA_YAML}")
    print(f"  Split:    {split}")
    print(f"  Conf:     {conf}")
    print(f"  IoU:      {iou}")
    print("=" * 60 + "\n")

    # Load trained model
    model = YOLO(str(weights))

    # Run validation
    metrics = model.val(
        data=str(DATA_YAML),
        split=split,
        conf=conf,
        iou=iou,
        imgsz=640,
        batch=16,
        device="cuda",          # Change to "cpu" if no GPU
        plots=True,             # Saves confusion matrix, PR curve, etc.
        save_json=True,         # COCO-format JSON results
        verbose=True,
    )

    # ── Extract and Display Metrics ─────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  EVALUATION RESULTS")
    print("=" * 60)

    # ── Extraction (using Box metrics for detection, Seg for segmentation)
    map50_box = metrics.box.map50
    map50_seg = metrics.seg.map50
    precision = metrics.seg.mp         # Mask-based Precision
    recall    = metrics.seg.mr         # Mask-based Recall

    # F1 score (harmonic mean of Precision and Recall)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-8)

    print(f"  Box mAP50:    {map50_box:.4f}  ({map50_box*100:.2f}%)")
    print(f"  Mask mAP50:   {map50_seg:.4f}  ({map50_seg*100:.2f}%)  ← PRIMARY")
    print(f"  Precision:    {precision:.4f}  ({precision*100:.2f}%)")
    print(f"  Recall:       {recall:.4f}  ({recall*100:.2f}%)")
    print(f"  F1 Score:     {f1:.4f}  ({f1*100:.2f}%)")
    print("=" * 60)

    # Academic grading guidance
    print("\n  📊 Academic Performance Guide:")
    if map50_seg >= 0.85:
        grade = "Excellent ✅ — Publication-level Segmentation!"
    elif map50_seg >= 0.70:
        grade = "Very Good ✅ — Strong graduation project"
    elif map50_seg >= 0.50:
        grade = "Good ⚠️  — Acceptable; segmentation is harder than boxes"
    else:
        grade = "Needs Improvement ❌ — Try more augmentation"
    print(f"  Mask mAP50 = {map50_seg*100:.1f}% → {grade}")

    # ── Save Results to JSON ────────────────────────────────────────────────
    results_dict = {
        "weights": str(weights),
        "split": split,
        "conf_threshold": conf,
        "iou_threshold": iou,
        "box_mAP50": round(float(map50_box), 6),
        "mask_mAP50": round(float(map50_seg), 6),
        "precision": round(float(precision), 6),
        "recall": round(float(recall), 6),
        "f1_score": round(float(f1), 6),
    }

    out_json = LOGS_DIR / f"eval_results_{split}.json"
    with open(out_json, "w") as f:
        json.dump(results_dict, f, indent=4)
    print(f"\n  Results saved to: {out_json}")

    # ── Plot PR Curve ───────────────────────────────────────────────────────
    plot_pr_curve(metrics, split)

    return metrics


def plot_pr_curve(metrics, split: str = "val"):
    """
    Plot and save the Precision-Recall curve.
    The built-in YOLO validator already saves this, but we also save
    a clean custom version for use in reports.
    """
    try:
        # metrics.box.p = precision values per confidence threshold
        # metrics.box.r = recall values per confidence threshold
        p_curve = metrics.seg.p      # shape: (num_classes, num_thresholds)
        r_curve = metrics.seg.r

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(r_curve[0], p_curve[0], color="#00A8E8", linewidth=2,
                label=f"crack (Mask mAP50={metrics.seg.map50:.3f})")
        ax.set_xlabel("Recall", fontsize=13)
        ax.set_ylabel("Precision", fontsize=13)
        ax.set_title("Precision-Recall Curve — Crack Detection", fontsize=14, fontweight="bold")
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        plt.tight_layout()

        pr_path = LOGS_DIR / f"pr_curve_{split}.png"
        plt.savefig(pr_path, dpi=150)
        plt.close()
        print(f"  PR curve saved to: {pr_path}")
    except Exception as e:
        print(f"  [INFO] Could not plot PR curve: {e}")


# ══════════════════════════════════════════════════════════════════════════════
#  CLI ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate crack detection model")
    parser.add_argument(
        "--weights", type=str, default=str(DEFAULT_WEIGHTS),
        help="Path to model weights (.pt)"
    )
    parser.add_argument(
        "--split", type=str, default="val", choices=["val", "test"],
        help="Dataset split to evaluate on"
    )
    parser.add_argument(
        "--conf", type=float, default=0.25,
        help="Confidence threshold (0.0–1.0)"
    )
    parser.add_argument(
        "--iou", type=float, default=0.5,
        help="IoU threshold for NMS"
    )
    args = parser.parse_args()

    evaluate(
        weights=args.weights,
        split=args.split,
        conf=args.conf,
        iou=args.iou,
    )
