"""
Face Mask Compliance Monitoring - Streamlit Frontend
Run:  streamlit run app.py
Needs: pip install streamlit ultralytics opencv-python pillow numpy
"""
import os
import tempfile
import collections
from pathlib import Path
try:
    import cv2  # type: ignore[reportMissingImports]
except ImportError as exc:
    raise RuntimeError(
        "OpenCV is required. Install it with: pip install opencv-python"
    ) from exc
import numpy as np
import streamlit as st
from PIL import Image
from ultralytics import YOLO

# ---------------------------------------------------------------- config ----
st.set_page_config(page_title="Face Mask Compliance Monitor", page_icon="😷", layout="wide")

PROJECT_DIR = Path(__file__).resolve().parent
configured_weights = Path(os.getenv("MODEL_WEIGHTS", "best_yolov8l_weights.pt"))
DEFAULT_WEIGHTS = str(
    configured_weights
    if configured_weights.is_absolute()
    else PROJECT_DIR / configured_weights
)
CLASS_LABELS = {
    "with_mask": "With mask",
    "without_mask": "Without mask",
    "incorrectly_worn_mask": "Incorrectly worn",
}


# ---------------------------------------------------------------- helpers ---
@st.cache_resource(show_spinner="Loading model...")
def load_model(path: str) -> YOLO:
    return YOLO(path)


def run_detection(model: YOLO, bgr_image: np.ndarray, conf: float, iou: float):
    """Returns (annotated RGB image, Counter of class names)."""
    result = model.predict(bgr_image, conf=conf, iou=iou, verbose=False)[0]
    annotated = cv2.cvtColor(result.plot(), cv2.COLOR_BGR2RGB)
    names = [model.names[int(c)] for c in result.boxes.cls.tolist()]
    return annotated, collections.Counter(names)


def show_metrics(counts: collections.Counter):
    total = sum(counts.values())
    compliant = counts.get("with_mask", 0)
    rate = (compliant / total * 100) if total else 0.0

    cols = st.columns(4)
    cols[0].metric("Faces detected", total)
    for col, key in zip(cols[1:], CLASS_LABELS):
        col.metric(CLASS_LABELS[key], counts.get(key, 0))

    st.progress(min(int(rate), 100), text=f"Compliance rate: {rate:.1f}%")
    if total == 0:
        st.info("No faces detected. Try lowering the confidence threshold.")
    elif rate < 100:
        st.warning("Non-compliance detected.")
    else:
        st.success("All detected faces are wearing masks correctly.")


def image_to_png_bytes(rgb: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    return buf.tobytes() if ok else b""


# ---------------------------------------------------------------- sidebar ---
st.sidebar.title("Settings")
weights = st.sidebar.text_input("Model weights (.pt)", DEFAULT_WEIGHTS)
conf = st.sidebar.slider("Confidence threshold", 0.05, 0.95, 0.25, 0.05)
iou = st.sidebar.slider("IoU threshold (NMS)", 0.10, 0.95, 0.45, 0.05)
st.sidebar.caption("Classes: with_mask, without_mask, incorrectly_worn_mask")

# ------------------------------------------------------------------ main ----
st.title("😷 Face Mask Compliance Monitoring")
st.caption("YOLO-based deep learning detector · upload an image, take a photo, or analyse a video")

if not os.path.exists(weights):
    st.error(f"Weights file not found: `{weights}`. Update the path in the sidebar.")
    st.stop()

model = load_model(weights)
tab_img, tab_cam, tab_vid = st.tabs(["🖼️ Image", "📷 Camera", "🎞️ Video"])

# ---- Image upload
with tab_img:
    file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "bmp"], key="img")
    if file:
        pil = Image.open(file).convert("RGB")
        bgr = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
        annotated, counts = run_detection(model, bgr, conf, iou)

        left, right = st.columns(2)
        left.image(pil, caption="Original", use_container_width=True)
        right.image(annotated, caption="Detections", use_container_width=True)
        show_metrics(counts)
        st.download_button("Download result", image_to_png_bytes(annotated),
                           "result.png", "image/png")

# ---- Camera snapshot
with tab_cam:
    shot = st.camera_input("Take a photo")
    if shot:
        pil = Image.open(shot).convert("RGB")
        bgr = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
        annotated, counts = run_detection(model, bgr, conf, iou)
        st.image(annotated, caption="Detections", use_container_width=True)
        show_metrics(counts)

# ---- Video upload
with tab_vid:
    vid = st.file_uploader("Upload a video", type=["mp4", "avi", "mov", "mkv"], key="vid")
    skip = st.slider("Process every Nth frame (higher = faster)", 1, 10, 2)
    if vid and st.button("Analyse video"):
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        cap = None
        try:
            tmp.write(vid.read())
            tmp.close()

            cap = cv2.VideoCapture(tmp.name)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
            frame_box = st.empty()
            bar = st.progress(0)
            totals, idx = collections.Counter(), 0

            while cap.isOpened():
                ok, frame = cap.read()
                if not ok:
                    break
                if idx % skip == 0:
                    annotated, counts = run_detection(model, frame, conf, iou)
                    totals.update(counts)
                    frame_box.image(annotated, use_container_width=True)
                idx += 1
                bar.progress(min(idx / total_frames, 1.0))
        finally:
            if cap is not None:
                cap.release()
            if not tmp.closed:
                tmp.close()
            if os.path.exists(tmp.name):
                os.unlink(tmp.name)

        st.subheader("Video summary (total detections across processed frames)")
        show_metrics(totals)

st.divider()
st.caption("Microproject: Deep Learning (ADCOR1PC301) · TY-AIDS · SAKEC")