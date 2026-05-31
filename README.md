# UYIR — Road Accident Detection

Real-time accident detection for Indian traffic using **YOLOv8 + ByteTrack** and a **5-factor scoring engine** (IOU overlap, speed drop, trajectory deviation, optical flow, consecutive-frame confirmation).

**Repository:** [github.com/max-mani/last_hope_in_ai](https://github.com/max-mani/last_hope_in_ai)

---

## Project structure

```
last_hope_in_ai/
├── models/                 # YOLO weights (auto-download on first run)
├── data_logs/              # CSV output from data_logger.py
├── dataset/                # Local video clips (not in git — see dataset/README.md)
├── outputs/                # Threshold analysis plots
├── snapshots/              # Accident frame captures (runtime)
├── paths.py                # Central paths (models, logs, outputs)
├── test_setup.py           # Run first — verify install + webcam
├── vehicle_tracker.py      # YOLOv8 + ByteTrack tracking
├── accident_detector.py    # 5-factor accident logic
├── stream_processor.py     # Live pipeline (webcam / file / RTSP)
├── data_logger.py          # Log factors to CSV for tuning
├── threshold_analyzer.py   # Plot distributions, suggest thresholds
├── firebase_uploader.py    # Optional Firebase upload
└── requirements.txt
```

---

## YOLOv8 vs YOLOv8n — which one?

Use **YOLOv8n** by default. The **n** means **nano** — smallest and fastest.

| Model   | Speed   | Accuracy   | Use case                          |
|---------|---------|------------|-----------------------------------|
| YOLOv8n | Fastest | Good enough| **Pi 5 + laptop (default)** ✅    |
| YOLOv8s | Fast    | Better     | Laptop only if nano misses cars   |
| YOLOv8m | Medium  | Best       | Too heavy for Pi 5 ❌             |

You do **not** download weights manually. On first run, Ultralytics downloads the model (e.g. `yolov8n.pt`, ~6 MB) into `models/` automatically.

**Laptop testing — more detections:** If nano misses vehicles, switch to small + lower confidence:

```bash
python stream_processor.py --source 0 --model models/yolov8s.pt
```

In code, use `confidence=0.30` instead of `0.45` in `vehicle_tracker.py` / `stream_processor.py`.

**Raspberry Pi 5 deployment:** Stay on **yolov8n** for FPS. When the custom Indian-road model is ready, use `models/best.pt`.

---

## Setup (laptop)

Open a terminal in the project folder:

```bash
# Clone
git clone https://github.com/max-mani/last_hope_in_ai.git
cd last_hope_in_ai

# Virtual environment
python -m venv uyir_env

# Activate
# Windows:
uyir_env\Scripts\activate
# Mac/Linux:
source uyir_env/bin/activate

# Dependencies
pip install -r requirements.txt
```

Copy your video dataset into `dataset/` locally (clips are not stored on GitHub due to size limits).

---

## What each file does

| File | Purpose |
|------|---------|
| `test_setup.py` | Run **first** — checks imports, webcam, YOLO |
| `vehicle_tracker.py` | YOLOv8n + ByteTrack — detect & track vehicles |
| `accident_detector.py` | 5-factor scoring, cooldown, accident events |
| `stream_processor.py` | Video source → tracker → detector → alerts |
| `data_logger.py` | Log raw CSV for threshold analysis |
| `threshold_analyzer.py` | Plots + recommended thresholds from CSV |
| `firebase_uploader.py` | Upload events to Firebase (optional) |

---

## Run — step by step

### 1. Test setup

```bash
python test_setup.py
```

Choose **1** (quick webcam + YOLO). `yolov8n.pt` downloads on first run. You should see bounding boxes on vehicles.

### 2. Full pipeline (webcam)

```bash
python stream_processor.py --source 0
```

Press **Q** to quit.

Other sources:

```bash
python stream_processor.py --source path/to/video.mp4
python stream_processor.py --source rtsp://192.168.1.5:8080/h264_ulaw.sdp
```

### 3. Log data from clips

```bash
python data_logger.py --video "dataset/uyir sample/sampleA.mp4" --label accident
python data_logger.py --video path/to/normal_clip.mp4 --label normal
```

Output: `data_logs/uyir_data_log.csv`

### 4. Analyze thresholds

```bash
python threshold_analyzer.py
# or: python threshold_analyzer.py --csv data_logs/uyir_data_log.csv
```

Plots are saved under `outputs/`.

### 5. Firebase (optional)

1. Download service account key as `firebase_key.json` in the project root (gitignored).
2. Update `FIREBASE_BUCKET` in `firebase_uploader.py`.
3. Run with:

```bash
python stream_processor.py --source 0 --firebase
```

---

## Pre-trained YOLO limits (Indian roads)

Current default uses **COCO pre-trained** weights. It detects **car, motorcycle, bus, truck, person** — not auto-rickshaws.

| Issue | Fix |
|-------|-----|
| Too few boxes | Lower `confidence` to `0.30`; try `yolov8s.pt` on laptop |
| Autos missing | Needs custom model (`models/best.pt`) when training is done |
| Top-down CCTV angle | Custom model on Indian footage |

When Kishore’s trained model is ready, change the default in `paths.py`:

```python
DEFAULT_YOLO_MODEL = MODELS_DIR / "best.pt"
```

---

## Model size reference

```
yolov8n  →  nano    →  fastest, least accurate   (default)
yolov8s  →  small   →  better on laptop
yolov8m  →  medium  →  avoid on Pi 5
yolov8l  →  large   →  too slow for edge devices
yolov8x  →  xlarge  →  not practical for real-time
```

---

## License & team

UYIR — Coimbatore road safety project.  
For issues and updates, use the [GitHub repo](https://github.com/max-mani/last_hope_in_ai).
