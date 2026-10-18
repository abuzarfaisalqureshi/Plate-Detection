"""
License Plate Detection - Streamlit App
Model : YOLO11n (Ultralytics), class: License-Plate
Run   : streamlit run app.py
"""

import io
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image, ImageOps
from ultralytics import YOLO

MODEL_PATH = Path(__file__).parent / "best.pt"
IMG_SIZE = 640

st.set_page_config(page_title="License Plate Detection", page_icon="🚘", layout="wide")


def find_model_file():
    """Find a valid best.pt (next to app.py, in the repo root, or in any subfolder)."""
    here = Path(__file__).parent
    roots = [here, here.parent, Path.cwd()]
    for root in roots:
        p = root / "best.pt"
        if p.exists() and p.stat().st_size > 1_000_000:
            return p
    for root in roots:
        for p in sorted(root.rglob("*.pt")):
            if p.stat().st_size > 1_000_000:
                return p
    return None


@st.cache_resource(show_spinner="Loading model...")
def load_model(path: str):
    return YOLO(path)


model_file = find_model_file()
if model_file is None:
    base = Path(__file__).parent
    st.error("A valid best.pt (about 5.4 MB) was not found in the repository.")
    st.write("Files found (name -> size in bytes). best.pt must be ~5,469,530 bytes:")
    st.code("\n".join(sorted(f"{p.relative_to(base)} -> {p.stat().st_size}" for p in base.rglob("*") if p.is_file() and ".git" not in p.parts)))
    st.stop()

model = load_model(str(model_file))

# ----------------------------- Sidebar ----------------------------------- #
st.sidebar.header("⚙️ Settings")
conf = st.sidebar.slider("Confidence threshold", 0.05, 0.95, 0.25, 0.05)
iou = st.sidebar.slider("IoU threshold (NMS)", 0.10, 0.90, 0.45, 0.05)
source = st.sidebar.radio("Image source", ["Upload image", "Use camera"])
st.sidebar.info("Model: YOLO11n\n\nClass: License-Plate\n\nInput size: 640")

# ------------------------------ Main ------------------------------------- #
st.title("🚘 License Plate Detection")
st.write("Upload a vehicle image and the YOLO11 model will find the number plate(s).")

if source == "Upload image":
    file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png", "bmp", "webp"])
else:
    file = st.camera_input("Take a picture")

if file is not None:
    image = ImageOps.exif_transpose(Image.open(file)).convert("RGB")
    rgb = np.array(image)

    with st.spinner("Detecting..."):
        # Ultralytics expects BGR numpy arrays
        result = model.predict(rgb[:, :, ::-1], conf=conf, iou=iou, imgsz=IMG_SIZE, verbose=False)[0]
        annotated = result.plot()[:, :, ::-1]  # BGR -> RGB

    col1, col2 = st.columns(2)
    col1.subheader("Original")
    col1.image(image, use_container_width=True)
    col2.subheader("Result")
    col2.image(annotated, use_container_width=True)

    rows = []
    for i, box in enumerate(result.boxes, start=1):
        x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())
        rows.append(
            {
                "#": i,
                "Class": result.names[int(box.cls[0])],
                "Confidence": round(float(box.conf[0]), 3),
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
            }
        )

    if rows:
        st.success(f"{len(rows)} license plate(s) detected.")
        st.dataframe(rows, use_container_width=True, hide_index=True)
    else:
        st.warning("No license plate detected. Try lowering the confidence threshold.")

    buf = io.BytesIO()
    Image.fromarray(annotated).save(buf, format="PNG")
    st.download_button("⬇️ Download result", buf.getvalue(), "result.png", "image/png")
