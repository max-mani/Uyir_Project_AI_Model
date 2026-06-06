# Multi-Stage Spatio-Temporal Hybrid Accident Detection System 🚨

A production-ready Intelligent Transportation System (ITS) pipeline built in Python to detect road accidents in real-time. The system processes video and image inputs, fusing spatial kinematic reasoning, dense motion algorithms, and temporal deep learning sequences into a unified decision engine.

---

## 🔷 System Architecture & Features

This system integrates a hybrid multi-layered pipeline to ensure high accuracy and resilience to edge-case failures (such as brief vehicle occlusions, skidding, or sudden camera jitter):

1. **Object Detection**: YOLOv8 vehicle extraction (`car`, `bike`, `bus`, `truck`).
2. **Multi-Object Tracking (MOT)**: Python-native IoU-based vehicle tracker keeping track of unique IDs, frame-by-frame velocity vectors, and 30-frame historical trajectories.
3. **Phase A — Proximity Filtering**: Euclidean distance centroid calculations. Only selects vehicle pairs with distance $< 120$ pixels, reducing downstream CPU load.
4. **Phase B — Trajectory Conflict**:
   * **Segment Intersection**: Checks if historical paths crossed.
   * **Kinetic Energy Drop**: Measures sudden energy collapse ($KE = \text{area} \times v^2 > 80\%$ drop in 3 frames).
   * **Heading Instability (Spin)**: Computes circular variance of travel angles over 5 frames to flag skids or rollovers.
   * **BBox Merge**: Overlap checks (IoU $> 0.60$) to catch crashes where two vehicles merge into one.
5. **Phase C — Anomaly Confirmation**:
   * **Optical Flow Spike**: Identifies dense Farneback flow spikes ($> 2.5\text{x}$ historical average) within vehicle bounding boxes.
   * **BBox Deformation**: Detects sudden aspect ratio ($> 30\%$) or area ($> 40\%$) changes representing crash impact.
   * **Flow Angular Dispersion**: Calculates flow angle standard deviation. Values $> 45^\circ$ signify chaotic radial scatter (crash debris) rather than parallel traffic.
   * **Multi-Frame Consistency**: Requires anomalies to persist for $\ge 3$ consecutive frames to filter out noise.
6. **CNN-LSTM DL Module**: Caches ResNet18 CNN features and feeds a sliding 16-frame feature buffer to the LSTM network. Runs **16x faster** than raw frame inputs.
7. **Score Fusion Engine**: Blends all 11 indicators using a 9-weight formula:
   $$\text{Final Score} = w_1 S_{\text{prox}} + w_2 (0.6 S_{\text{traj}} + 0.4 S_{\text{spin}}) + w_3 S_{\text{flow}} + w_4 P_{\text{lstm\_peak}} + w_5 S_{\text{occlusion}} + w_6 S_{\text{merge}} + w_7 S_{\text{energy}} + w_8 S_{\text{scene}} + w_9 \max(S_{\text{diff}}, S_{\text{dispersion}})$$
8. **Intersection Risk Multiplier**: Applies a `1.2x` score multiplier for candidate crashes occurring in the center 50% "intersection zone" of the camera view.
9. **Visual Telemetry Overlay**: Annotates vehicle bounding boxes, trails, yellow proximity links, red collision targets, and draws a 240px overlay panel showing real-time metrics on the output frames (encoded in native H.264 `avc1` for browser playback).

---

## 📁 Project Directory Structure

```
├── app.py                     # Entry point FastAPI application
├── model.py                   # PyTorch CNN-LSTM model loader
├── model_output/
│   └── accident_model.pth     # Trained ResNet18 + LSTM model checkpoint
├── detection/
│   ├── __init__.py
│   └── yolo_module.py         # YOLOv8 vehicle detector wrapper
├── tracking/
│   ├── __init__.py
│   └── deepsort_module.py     # Custom IoU Multi-Object Tracker
├── phases/
│   ├── __init__.py
│   ├── phase_a_proximity.py   # Centroid proximity filtering
│   ├── phase_b_trajectory.py  # Path intersections, KE drops, skidding, merges
│   └── phase_c_anomaly.py     # Flow magnitude spikes, deformations, dispersions
├── fusion/
│   ├── __init__.py
│   └── scoring.py             # 9-weight multi-signal fusion formula
├── utils/
│   ├── __init__.py
│   ├── geometry.py            # Euclidean distance, line intersection, angle variance
│   └── optical_flow.py        # Farneback flow, angular dispersion, frame diff
├── templates/
│   └── index.html             # Responsive HTML5 web dashboard
└── static/
    └── uploads/               # Stores raw uploads and processed annotated videos/images
```

---

## 🛠️ Installation & Setup

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/Kishorp28/accident-system.git
   cd accident-system
   ```

2. **Set Up Python Virtual Environment**:
   ```bash
   python -m venv venv
   venv\Scripts\activate      # On Windows
   source venv/bin/activate    # On Unix/macOS
   ```

3. **Install Dependencies**:
   Ensure you have PyTorch, OpenCV, FastAPI, and Ultralytics installed.
   ```bash
   pip install torch torchvision numpy opencv-python fastapi uvicorn ultralytics jinja2 python-multipart pillow
   ```

---

## 🚀 Running the Application

Start the FastAPI application locally:
```bash
python app.py
```
Open a browser and navigate to `http://127.0.0.1:8000`.

### Dashboard Features:
* **Image Upload**: Detects vehicles, draws bounding boxes, calculates proximity lines, and shows fused scorecards.
* **Video Upload**: Processes videos frame-by-frame, writing H.264-encoded annotated videos featuring centroid trajectory trails, collision targets, flashing banners, and a live telemetry card.
* **Score Gauges**: Renders real-time meters for Proximity, Trajectory, Anomaly, CNN-LSTM, Occlusion, Merge, Kinetic Energy, and Scene Interruption.

---

## 📡 API Endpoints

### 1. Upload Video
* **Endpoint**: `POST /predict-video`
* **Request**: Multipart Form-Data (file: video file)
* **Response**:
  ```json
  {
    "class": "ACCIDENT",
    "confidence": 84.50,
    "trigger_phase": "Phase A (Proximity) & Phase B (Trajectory/Spin) & CNN-LSTM Peak",
    "processed_video_url": "/static/uploads/processed_abc123.mp4",
    "details": {
      "proximity_score": 0.85,
      "trajectory_score": 0.70,
      "flow_score": 0.60,
      "lstm_peak": 0.85,
      "occlusion_score": 0.30,
      "merge_score": 0.00,
      "energy_drop": 0.95,
      "spin_score": 0.40,
      "scene_interruption": 0.20,
      "diff_burst": 0.50,
      "flow_dispersion": 0.45
    }
  }
  ```

---

## 📤 Pushing to GitHub

To push your local repository changes to your remote GitHub repository, execute the following commands in your terminal:

```bash
# Initialize local git repository (if not already initialized)
git init

# Add remote origin
git remote add origin https://github.com/Kishorp28/accident-system.git

# Track files, commit, and push
git add .
git commit -m "Initialize accident system with multi-stage hybrid detection, H.264 web streaming, and 11 advanced physical signals"
git branch -M main
git push -u origin main
```
