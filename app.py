import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import tempfile
import os
import time

# ==========================================================
#  PAGE CONFIGURATION & STYLING
# ==========================================================
st.set_page_config(
    page_title="Crack Detection Dashboard",
    page_icon="🔍",
    layout="wide",
)

st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
    }
    .stApp {
        background: linear-gradient(135deg, #0e1117 0%, #1c2331 100%);
        color: #e0e0e0;
    }
    .stSidebar {
        background-color: #161b22 !important;
    }
    h1 {
        color: #00d2ff;
        font-family: 'Outfit', sans-serif;
        font-weight: 700;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    .stMetric {
        background-color: #21262d;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #30363d;
    }
    .stButton>button {
        background-color: #00d2ff;
        color: white;
        border-radius: 5px;
        border: none;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #007bb5;
        box-shadow: 0px 4px 15px rgba(0, 210, 255, 0.4);
    }
    </style>
""", unsafe_allow_html=True)

# ==========================================================
#  SIDEBAR - SETTINGS
# ==========================================================
st.sidebar.title("🛠️ Configuration")

# Dynamic weights discovery
known_weights = [
    "runs/segment/runs/crack_detection4/weights/best.pt",
    "runs/segment/runs/crack_detection3/weights/best.pt",
    "weights/best.pt",
    "yolov12l-seg.pt"
]
default_path = next((p for p in known_weights if os.path.exists(p)), "yolov12l-seg.pt")

model_path = st.sidebar.text_input("Model Weights Path", default_path)

# Load Model (Cached)
@st.cache_resource
def load_model(path):
    if not os.path.exists(path):
        return None, f"⚠️ File not found: {path}"
    try:
        m = YOLO(path)
        return m, f"✅ Loaded: {os.path.basename(path)}"
    except Exception as e:
        return None, f"❌ Error: {e}"

model, model_status = load_model(model_path)
st.sidebar.info(model_status)

if "yolov12" in model_path.lower() and "crack" not in model_path.lower():
    st.sidebar.warning("📝 Using pretrained COCO model. It cannot detect 'cracks' specifically!")

st.sidebar.divider()
conf_threshold = st.sidebar.number_input("Confidence Threshold", min_value=0.0, max_value=1.0, value=0.01, step=0.01)
iou_threshold = st.sidebar.number_input("IoU Threshold", min_value=0.0, max_value=1.0, value=0.45, step=0.01)
camera_index = st.sidebar.number_input("Camera Index", 0, 5, 0)
show_masks = st.sidebar.checkbox("Show Segmentation Masks", True)

# ==========================================================
#  MAIN APP HEADER
# ==========================================================
st.title("🏗️ Crack Detection System — YOLOv12")
st.write("Professional Thermal Imagery Analysis Pipeline")

# ==========================================================
#  TABS: IMAGE | VIDEO | CAMERA
# ==========================================================
tab1, tab2, tab3 = st.tabs(["🖼️ Process Image", "🎥 Process Video", "📹 Live Camera"])

# ----------------------------------------------------------
# TAB 1: IMAGE PROCESSING
# ----------------------------------------------------------
with tab1:
    st.subheader("Single Image Analysis")
    uploaded_file = st.file_uploader("Choose a thermal image...", type=["jpg", "jpeg", "png", "tif"])
    
    if uploaded_file is not None:
        img = Image.open(uploaded_file)
        img_array = np.array(img)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.image(img, caption="Original Image", width="stretch")
            
        with col2:
            if model:
                with st.spinner('Analyzing...'):
                    results = model.predict(img_array, conf=conf_threshold, iou=iou_threshold)
                    annotated_img = results[0].plot(masks=show_masks)
                    st.image(annotated_img, caption="Detection Results", width="stretch")
                    
                    # Metrics
                    n_cracks = len(results[0].boxes)
                    st.success(f"Detections complete! Found **{n_cracks}** crack instances.")
            else:
                st.error("Model not loaded. check weights path.")

# ----------------------------------------------------------
# TAB 2: VIDEO PROCESSING
# ----------------------------------------------------------
with tab2:
    st.subheader("Video Sequence Analysis")
    uploaded_video = st.file_uploader("Upload a thermal video...", type=["mp4", "avi", "mov"])
    
    if uploaded_video is not None:
        # Save to temp file
        tfile = tempfile.NamedTemporaryFile(delete=False) 
        tfile.write(uploaded_video.read())
        
        cap = cv2.VideoCapture(tfile.name)
        st_frame = st.empty()
        
        stop_video = st.button("Stop Video Processing")
        
        progress_bar = st.progress(0)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        frame_idx = 0
        while cap.isOpened() and not stop_video:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Run YOLO
            results = model.predict(frame, conf=conf_threshold, iou=iou_threshold, verbose=False)
            annotated_frame = results[0].plot(masks=show_masks)
            
            # Convert BGR to RGB
            annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            st_frame.image(annotated_frame, width="stretch")
            
            frame_idx += 1
            progress_bar.progress(frame_idx / total_frames)
            
        cap.release()
        st.success("Video processing finished.")

# ----------------------------------------------------------
# TAB 3: LIVE CAMERA
# ----------------------------------------------------------
with tab3:
    st.subheader("Real-Time Infrastructure Monitoring")
    start_cam = st.toggle("Start Camera Stream")
    
    if start_cam:
        if model:
            cap = cv2.VideoCapture(camera_index)
            if not cap.isOpened():
                st.error(f"Cannot open camera at index {camera_index}. Try another index in the sidebar.")
            else:
                st_frame = st.empty()
                st_info = st.empty()
                
                while start_cam:
                    ret, frame = cap.read()
                    if not ret:
                        st.error("Lost camera connection.")
                        break
                    
                    start_time = time.time()
                    results = model.predict(frame, conf=conf_threshold, iou=iou_threshold, verbose=False)
                    end_time = time.time()
                    
                    annotated_frame = results[0].plot(masks=show_masks)
                    annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                    
                    st_frame.image(annotated_frame, width="stretch")
                    
                    # Latency calculation
                    fps = 1 / (end_time - start_time)
                    st_info.info(f"⚡ In-Stream Stats: {fps:.1f} FPS | Detections: {len(results[0].boxes)}")
                    
                    if not start_cam: # Double check toggle
                        break
                
                cap.release()
        else:
            st.error("Model weights missing.")
    else:
        st.info("Toggle the switch above to enable live monitoring.")

# ==========================================================
#  FOOTER
# ==========================================================
st.divider()
st.caption("Crack detection pipeline for thermal imagery — University Graduation Project © 2026")
