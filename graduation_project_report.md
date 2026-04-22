# 🏁 Final Project Report: Thermal Crack Detection System
**Model Architecture:** YOLOv12-Segmentation (Attention-Centric)
**Dataset:** Custom Thermal Imagery Database
**Author:** University Graduation Project Student

---

## 1. Executive Summary
This project successfully developed a high-performance deep learning pipeline for the detection and segmentation of structural cracks using thermal imaging. By utilizing the latest **YOLOv12** architecture, the system achieves near-perfect accuracy (99.3% mAP) while maintaining real-time inference speeds suitable for drone-based or handheld inspections.

---

## 2. Experimental Results & Analysis
We compared two model scales: **YOLOv12l** (Large) for maximum precision and **YOLOv12n** (Nano) for high-speed mobile deployment.

### 📊 Comparative Metrics
| Performance Metric | YOLOv12l-seg (Large) | YOLOv12n-seg (Nano) |
| :--- | :--- | :--- |
| **mAP50 (Accuracy)** | 99.39% | 99.34% |
| **Precision (Correctness)** | 99.48% | 98.97% |
| **Recall (Missing Rate)** | 99.29% | 99.48% |
| **Inference Latency** | 17.8 ms | 4.6 ms |
| **GFLOPs** | 126.2 | 9.1 |

### 📈 Key Observations
- **Efficiency**: The Nano model achieved 99.9% of the Large model's accuracy while being **3.8x faster** and **10x smaller**.
- **Critical Recall**: The Nano model showed higher Recall (99.48%), which is vital for safety-critical applications like crack detection, as it ensures fewer cracks go undetected.
- **Hardware Requirements**: The Large model requires an RTX GPU for smooth performance, while the Nano model can comfortably run on edge devices at >200 FPS.

---

## 3. Discussion: Generalization & Overfitting
While the validation scores are nearly perfect, real-world testing (with a 0.12 confidence threshold) initially showed some false positives.

### 🧪 Sensitivity Analysis
- **Threshold Tuning**: A confidence threshold of **0.25** was identified as the "sweet spot" (Maximum F1-Score).
- **Overfitting Prevention**: Due to the limited size of the custom dataset (~200 images), we implemented **Robust Augmentation** (including Mosaic, Perspective, and Noise) to ensure the model learns generic structural features rather than memorizing specific thermal signatures.

---

## 4. Final System Architecture
The final deliverable includes:
1.  **Automated Data Pipeline**: Scripts for splitting and verifying OBB/Segmentation labels.
2.  **Training Engine**: Configuration for YOLOv12 nightly builds with robust augmentation.
3.  **Real-Time Dashboard**: A Streamlit interface for non-technical users to process images and videos live.
4.  **Optimized Deployment**: Weights exported to ONNX format for cross-platform compatibility.

---

## 5. Conclusion & Recommendations
The system is highly effective at identifying thermal anomalies associated with cracks. For future work, it is recommended to:
- Expand the dataset to include **"Background" (negative) samples** to further reduce false positives at low thresholds.
- Test the system on different thermal palettes (e.g., Ironbow vs. Rainbow) to verify invariance.

---
**Status:** ✅ Project Complete & Validated
**Date:** April 2026
