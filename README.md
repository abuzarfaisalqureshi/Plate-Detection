# Plate-Detection
# 🚘 License Plate Detection (YOLO11)

A deep-learning project that detects **vehicle license plates** in images and videos.
It uses a custom-trained **YOLO11n** model (`best.pt`) from [Ultralytics](https://docs.ultralytics.com) and ships with a simple **Gradio** web app.

---

## ✨ Features

- Detects license plates in **images** (upload, webcam or clipboard)
- Detects license plates in **videos** (frame-by-frame, saved as annotated MP4)
- Adjustable **confidence** and **IoU** thresholds
- Shows bounding-box coordinates and confidence scores in a table
- Lightweight model (~5 MB) – runs on CPU, faster with GPU

## 🧠 Model Details

| Item            | Value                         |
|-----------------|-------------------------------|
| Architecture    | YOLO11n (nano)                |
| Task            | Object detection              |
| Classes         | 1 → `License-Plate`           |
| Input size      | 640 × 640                     |
| Epochs          | 25                            |
| Batch size      | 32                            |
| Framework       | Ultralytics 8.4.x / PyTorch   |

## 📁 Project Structure

```
.
├── app.py             # Gradio web application
├── best.pt            # Trained YOLO11n weights
├── requirements.txt   # Python dependencies
└── README.md
```

## ⚙️ Installation

1. **Clone / download** this project and open the folder.

2. **Create a virtual environment** (recommended):

   ```bash
   python -m venv venv
   # Windows
   venv\Scripts\activate
   # Linux / macOS
   source venv/bin/activate
   ```

3. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

   > For GPU support, install PyTorch with CUDA first from
   > [pytorch.org](https://pytorch.org/get-started/locally/), then run the command above.

4. Make sure `best.pt` is in the **same folder** as `app.py`.

## 🚀 Run the App

```bash
python app.py
```

Then open the local URL shown in the terminal (usually <http://127.0.0.1:7860>).

### How to use

1. Choose the **Image** or **Video** tab.
2. Upload a file (or use the webcam for images).
3. Adjust the thresholds if needed:
   - **Confidence** – higher = fewer but more certain detections.
   - **IoU** – controls how overlapping boxes are merged.
4. Click **Detect**.

## 🐍 Use the Model in Your Own Code

```python
from ultralytics import YOLO

model = YOLO("best.pt")
results = model.predict("car.jpg", conf=0.25, imgsz=640, save=True)

for box in results[0].boxes:
    print(box.xyxy[0].tolist(), float(box.conf[0]))
```

Or via the command line:

```bash
yolo detect predict model=best.pt source=car.jpg conf=0.25
```

## 💡 Tips

- If detections are missed, lower the confidence threshold (e.g. `0.15`).
- If you get false positives, raise it (e.g. `0.5`).
- For video output to play in the browser, having **FFmpeg** installed helps.
- Best results are on clear, well-lit images where the plate is reasonably large.

## 🛠️ Troubleshooting

| Problem | Fix |
|---------|-----|
| `FileNotFoundError: best.pt` | Put `best.pt` next to `app.py` |
| Slow on video | Use a GPU or a shorter / lower-resolution clip |
| Output video won't play | Install FFmpeg, or open the file from the temp folder |
| `ModuleNotFoundError` | Re-run `pip install -r requirements.txt` |

## 📜 License

Ultralytics YOLO is released under the **AGPL-3.0** license, so projects built on it
should comply with it (or use an [Ultralytics Enterprise License](https://ultralytics.com/license)).

## 🙏 Acknowledgements

- [Ultralytics YOLO11](https://docs.ultralytics.com)
- [Gradio](https://gradio.app)
