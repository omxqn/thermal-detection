# 📸 Thermal Crack Detection — YOLOv12 Graduation Project

Welcome to your graduation project workspace! This repository contains a production-ready pipeline for detecting cracks in thermal images using the state-of-the-art **YOLOv12** architecture.

---

## 🚀 1. Environment Setup (Windows + CUDA 13.0)

Follow these steps exactly to ensure your GPU is utilized for training.

### Step A: Install Python
Ensure you have **Python 3.10 or 3.11** installed. Check with:
```powershell
python --version
```

### Step B: Create a Virtual Environment (Recommended)
```powershell
python -m venv venv
.\venv\Scripts\activate
```

### Step C: Install PyTorch with CUDA 13.0
As of early 2026, CUDA 13.0 requires the nightly/pre-release build of PyTorch 2.9.0:
```powershell
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu130
```

### Step D: Install Dependencies
```powershell
pip install -r requirements.txt
```

---

## 🖼️ 2. Dataset Preparation

YOLOv12 expects data in a specific folder structure. Organize your custom thermal images as follows:

```text
datasets/
├── images/
│   ├── train/ (70% of images)
│   ├── val/   (20% of images)
│   └── test/  (10% of images)
└── labels/
    ├── train/ (Matching .txt files)
    ├── val/   (Matching .txt files)
    └── test/  (Matching .txt files)
```

### 📝 Annotation Format (YOLO)
Each image should have a corresponding `.txt` file with the same name.
**Format**: `<class_id> <x_center> <y_center> <width> <height>` (normalized 0.0 to 1.0)
**Example**: `0 0.45 0.32 0.05 0.12` (A crack at center 45%, 32% with size 5% width, 12% height)

---

## 🤖 3. Model Training

Training is automated via `scripts/train.py`. It is pre-configured for thermal image nuances (e.g., higher brightness/contrast variation).

```powershell
python scripts/train.py
```

- **Weights**: Saved to `runs/crack_detection/weights/best.pt`
- **Tuning**: You can edit `scripts/train.py` to change `epochs` or `imgsz`.

---

## 📊 4. Evaluation

After training, evaluate the model on your validation set to get academic metrics (mAP, Precision, Recall).

```powershell
python scripts/evaluate.py --weights runs/crack_detection/weights/best.pt
```

---

## 🔍 5. Inference (Detection)

Run detection on new, unseen thermal images:

```powershell
# Single Image
python scripts/inference.py --source data/test_img.jpg

# Folder of Images
python scripts/inference.py --source datasets/images/test/ --conf 0.4
```

---

## 🖥️ 6. Graphical User Interface (Dashboard)

I have created a premium **Streamlit Dashboard** that allows you to process images, videos, and live camera feeds without using the command line.

### Launch the Dashboard:
```powershell
pip install streamlit
streamlit run app.py
```

- **📸 Image Tab**: Upload single thermal images for instant segmentation.
- **🎥 Video Tab**: Process mp4/avi files with real-time feedback.
- **📹 Live Stream**: Toggle your camera on/off and adjust parameters (Confidence, IoU) on the fly.

---

## 📦 7. Deployment & Hardware

### Real-time Detection
Connect your thermal camera (or webcam) and run:
```powershell
python scripts/realtime.py
```

### Export to ONNX / TensorRT
For maximum speed on NVIDIA GPUs:
```powershell
python scripts/export.py --weights runs/crack_detection/weights/best.pt --format engine
```

---

## 🧪 Bonus: Academic Grading Tips

1.  **Ablation Study**: Compare `yolov12n` (Nano) vs `yolov12m` (Medium) in your report. Show the trade-off between speed and accuracy.
2.  **Thermal Noise Analysis**: Discuss how different thermal palettes (Ironbow vs. Grayscale) affect detection.
3.  **Real-world Testing**: Show detection working in different lighting or weather conditions if possible.
4.  **Novelty**: Use the `scripts/realtime.py` in your presentation; live demos always impress judges!

---

**Author**: [Your Name/University ID]
**Project**: Crack Detection in Thermal Imagery using YOLOv12
