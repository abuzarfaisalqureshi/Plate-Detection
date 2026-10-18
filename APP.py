"""
License Plate Detection - Gradio Web App
Model : YOLO11n (Ultralytics) trained on a custom number-plate dataset
Class : License-Plate
Run   : python app.py
"""

import os
import tempfile
from pathlib import Path

import cv2
import gradio as gr
from ultralytics import YOLO

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
MODEL_PATH = Path(__file__).parent / "best.pt"
IMG_SIZE = 640  # image size used during training
DEFAULT_CONF = 0.25
DEFAULT_IOU = 0.45

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model file not found: {MODEL_PATH}\n"
        "Place 'best.pt' in the same folder as app.py."
    )

model = YOLO(str(MODEL_PATH))


# --------------------------------------------------------------------------- #
# Image detection
# --------------------------------------------------------------------------- #
def detect_image(image, conf, iou):
    """Detect license plates in a single image (numpy RGB from Gradio)."""
    if image is None:
        return None, [], "Please upload an image first."

    # Ultralytics treats numpy arrays as BGR, Gradio gives RGB -> convert
    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    result = model.predict(bgr, conf=conf, iou=iou, imgsz=IMG_SIZE, verbose=False)[0]

    annotated = cv2.cvtColor(result.plot(), cv2.COLOR_BGR2RGB)

    rows = []
    for i, box in enumerate(result.boxes, start=1):
        x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())
        rows.append([i, result.names[int(box.cls[0])], round(float(box.conf[0]), 3), x1, y1, x2, y2])

    status = f"{len(rows)} license plate(s) detected." if rows else "No license plate detected."
    return annotated, rows, status


# --------------------------------------------------------------------------- #
# Video detection
# --------------------------------------------------------------------------- #
def detect_video(video_path, conf, iou, progress=gr.Progress()):
    """Detect license plates in every frame of a video and return annotated video."""
    if video_path is None:
        return None, "Please upload a video first."

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None, "Could not open the video file."

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1

    out_path = os.path.join(tempfile.mkdtemp(), "output.mp4")
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    frame_idx, detections = 0, 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        result = model.predict(frame, conf=conf, iou=iou, imgsz=IMG_SIZE, verbose=False)[0]
        detections += len(result.boxes)
        writer.write(result.plot())
        frame_idx += 1
        progress(frame_idx / total, desc=f"Processing frame {frame_idx}/{total}")

    cap.release()
    writer.release()
    return out_path, f"Processed {frame_idx} frames, {detections} total detections."


# --------------------------------------------------------------------------- #
# UI
# --------------------------------------------------------------------------- #
with gr.Blocks(title="License Plate Detection") as demo:
    gr.Markdown(
        "# 🚘 License Plate Detection\n"
        "YOLO11n model trained to detect vehicle number plates. "
        "Upload an image or a video, adjust the thresholds and run detection."
    )

    with gr.Tab("🖼️ Image"):
        with gr.Row():
            with gr.Column():
                img_in = gr.Image(label="Input image", type="numpy", sources=["upload", "webcam", "clipboard"])
                img_conf = gr.Slider(0.05, 0.95, value=DEFAULT_CONF, step=0.05, label="Confidence threshold")
                img_iou = gr.Slider(0.1, 0.9, value=DEFAULT_IOU, step=0.05, label="IoU threshold (NMS)")
                img_btn = gr.Button("Detect", variant="primary")
            with gr.Column():
                img_out = gr.Image(label="Result")
                img_status = gr.Textbox(label="Status", interactive=False)
                img_table = gr.Dataframe(
                    headers=["#", "Class", "Confidence", "x1", "y1", "x2", "y2"],
                    label="Detections",
                )
        img_btn.click(detect_image, [img_in, img_conf, img_iou], [img_out, img_table, img_status])

    with gr.Tab("🎬 Video"):
        with gr.Row():
            with gr.Column():
                vid_in = gr.Video(label="Input video")
                vid_conf = gr.Slider(0.05, 0.95, value=DEFAULT_CONF, step=0.05, label="Confidence threshold")
                vid_iou = gr.Slider(0.1, 0.9, value=DEFAULT_IOU, step=0.05, label="IoU threshold (NMS)")
                vid_btn = gr.Button("Detect", variant="primary")
            with gr.Column():
                vid_out = gr.Video(label="Result")
                vid_status = gr.Textbox(label="Status", interactive=False)
        vid_btn.click(detect_video, [vid_in, vid_conf, vid_iou], [vid_out, vid_status])

    gr.Markdown("Built with [Ultralytics YOLO11](https://docs.ultralytics.com) and [Gradio](https://gradio.app).")


if __name__ == "__main__":
    demo.launch()
