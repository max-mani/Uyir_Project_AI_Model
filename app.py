import os
import json
import base64
import asyncio
import threading
from collections import defaultdict
import uuid
import math
import time
import cv2
import torch
import numpy as np
import csv
from pathlib import Path
from PIL import Image

from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Import custom modules
import config
from detection.yolo_module import YOLODetector
from tracking.deepsort_module import VehicleTracker
from phases.phase_a_proximity import proximity_filter
from phases.phase_b_trajectory import analyze_trajectory_conflict, is_stationary, compute_iou
from phases.phase_c_anomaly import analyze_anomaly
from utils.optical_flow import compute_optical_flow, calculate_frame_diff_ratio
from utils.geometry import calculate_bbox_containment_ratio
from fusion.scoring import fuse_scores
from model import model, transform, DEVICE, predict_image, SEQUENCE_LEN
from llm_vision_module import analyze_frame_with_llm
from utils.incident_clip import create_video_writer, transcode_video_for_browser, extract_clip_from_file
from utils import incident_store
from firebase_uploader import FirebaseUploader

app = FastAPI()
_firebase = FirebaseUploader()
_firebase.retry_local_events()

# ================= XGBOOST MODEL INTEGRATION =================
CSV_FILE = "accident_features.csv"
XGB_MODEL_PATH = "model_output/accident_xgboost.json"
xgb_clf = None

def init_csv_file():
    if not os.path.exists(CSV_FILE):
        with open(CSV_FILE, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "proximity",
                "trajectory",
                "anomaly",
                "cnn",
                "occlusion",
                "merge",
                "kinetic",
                "density",
                "avg_speed",
                "stopped_ratio",
                "label"
            ])

XGB_MIN_ROWS_PER_CLASS = 50

def load_xgboost_model():
    global xgb_clf
    if os.path.exists(XGB_MODEL_PATH):
        try:
            # Check dataset size/balance before trusting XGBoost over fusion logic
            if os.path.exists(CSV_FILE):
                import pandas as pd
                df = pd.read_csv(CSV_FILE)
                if "label" in df.columns:
                    counts = df["label"].value_counts()
                    n0 = int(counts.get(0, 0))
                    n1 = int(counts.get(1, 0))
                    if n0 < XGB_MIN_ROWS_PER_CLASS or n1 < XGB_MIN_ROWS_PER_CLASS:
                        print(
                            f"[INFO] XGBoost model found but dataset too small/imbalanced "
                            f"(label0={n0}, label1={n1}, need >= {XGB_MIN_ROWS_PER_CLASS} each). "
                            f"Falling back to rule-based fusion scoring."
                        )
                        xgb_clf = None
                        return

            from xgboost import XGBClassifier
            XGBClassifier._estimator_type = "classifier"
            xgb_clf = XGBClassifier()
            xgb_clf.load_model(XGB_MODEL_PATH)
            print("[SUCCESS] Loaded XGBoost classifier from", XGB_MODEL_PATH)
        except Exception as e:
            print("[ERROR] Failed to load XGBoost model:", e)
            xgb_clf = None
    else:
        xgb_clf = None

# Initialize files and models on startup
init_csv_file()
load_xgboost_model()

class FeatureLog(BaseModel):
    proximity: float
    trajectory: float
    anomaly: float
    cnn: float
    occlusion: float
    merge: float
    kinetic: float
    density: float
    avg_speed: float
    stopped_ratio: float
    label: int

def _phases_from_details(details):
    phases = []
    # Use the explicit confirmed flags from Option 2 pipeline when available
    if details.get("phase_a_confirmed") or (details.get("proximity_score", 0) or details.get("ttc_score", 0)) > 0.3:
        phases.append("phase_a")
    if details.get("phase_b_confirmed") or (details.get("trajectory_score", 0) or details.get("trajectory_stop_score", 0)) > 0.3:
        phases.append("phase_b")
    if details.get("phase_c_confirmed") or details.get("flow_score", 0) > 0.3:
        phases.append("phase_c")
    return phases or ["phase_a"]


def _save_incident_records(stubs, input_path, source_fps, total_frames, location, source="web"):
    """Build clip/snapshot assets, persist locally, and upload to Firebase."""
    saved_incidents = []
    for stub in stubs:
        incident_id = str(uuid.uuid4())
        clip_fs, snap_fs, clip_url, snap_url = incident_store.build_incident_paths(incident_id)
        cv2.imwrite(snap_fs, stub["snapshot_frame"])
        clip_ok = extract_clip_from_file(
            input_path,
            stub["source_frame_idx"],
            source_fps,
            clip_fs,
            total_frames=total_frames,
        )
        details = stub["details"]
        details.setdefault("lstm_peak", stub["dl_confidence"])
        status = stub.get("status", "confirmed")
        llm_text = analyze_frame_with_llm(snap_fs, details) if status == "confirmed" else None
        record = incident_store.save_incident({
            "id": incident_id,
            "source": source,
            "camera_id": "UPLOAD",
            "location": location,
            "frame_number": stub["source_frame_idx"],
            "time_in_video_sec": round(stub["source_frame_idx"] / source_fps, 2),
            "confidence": stub["confidence"],
            "dl_confidence": stub["dl_confidence"],
            "trigger_phase": stub["trigger_phase"],
            "phases_triggered": _phases_from_details(details),
            "clip_url": clip_url if clip_ok else None,
            "snapshot_url": snap_url,
            "llm_analysis": llm_text,
            "details": details,
            "status": status,
        })
        _firebase.upload_incident_record_async(record)
        saved_incidents.append(record)
    return saved_incidents


# ================= BASE PATH =================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ================= DIRECTORIES =================
UPLOAD_DIR = os.path.join(BASE_DIR, "static", "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

# ================= STATIC FILES =================
app.mount(
    "/static",
    StaticFiles(directory=os.path.join(BASE_DIR, "static")),
    name="static"
)

# ================= TEMPLATES =================
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))


# ================= STREAMING STATE =================
# Stores per-job streaming data: frames, metrics, status
_stream_jobs = {}          # job_id -> dict
_stream_lock = threading.Lock()

def _new_job(job_id):
    with _stream_lock:
        _stream_jobs[job_id] = {
            "frames": [],           # list of (jpeg_bytes, metrics_dict)
            "done": False,
            "error": None,
            "result": None,         # final JSON result
            "incidents": [],        # confirmed/suspicious/suppressed stubs
        }

def _push_frame(job_id, jpeg_bytes, metrics):
    with _stream_lock:
        if job_id in _stream_jobs:
            _stream_jobs[job_id]["frames"].append((jpeg_bytes, metrics))

def _finish_job(job_id, result=None, error=None):
    with _stream_lock:
        if job_id in _stream_jobs:
            _stream_jobs[job_id]["done"] = True
            _stream_jobs[job_id]["result"] = result
            _stream_jobs[job_id]["error"] = error

def _push_incident(job_id, incident):
    with _stream_lock:
        if job_id in _stream_jobs:
            _stream_jobs[job_id]["incidents"].append(incident)


# ================= SSE STREAM ENDPOINT =================
@app.get("/stream/{job_id}")
async def stream_job(job_id: str):
    """Server-Sent Events endpoint — sends annotated JPEG frames + live metrics."""
    async def event_generator():
        sent = 0
        while True:
            with _stream_lock:
                job = _stream_jobs.get(job_id)

            if job is None:
                yield "data: {\"error\": \"Job not found\"}\n\n"
                return

            frames = job["frames"]
            while sent < len(frames):
                jpeg_bytes, metrics = frames[sent]
                b64 = base64.b64encode(jpeg_bytes).decode("utf-8")
                payload = json.dumps({"frame": b64, "metrics": metrics})
                yield f"data: {payload}\n\n"
                sent += 1

            # Push any new incidents
            incidents = job.get("incidents", [])
            if incidents:
                with _stream_lock:
                    new_incidents = _stream_jobs[job_id]["incidents"]
                    # Only send ones we haven't sent yet (track by index)
                if new_incidents:
                    for inc in new_incidents:
                        if not inc.get("_sent"):
                            inc["_sent"] = True
                            payload = json.dumps({"incident": inc})
                            yield f"data: {payload}\n\n"

            if job["done"]:
                result = job.get("result") or {}
                payload = json.dumps({"done": True, "result": result})
                yield f"data: {payload}\n\n"
                # Clean up job after a delay
                await asyncio.sleep(30)
                with _stream_lock:
                    _stream_jobs.pop(job_id, None)
                return

            await asyncio.sleep(0.04)  # ~25 fps max polling rate

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        }
    )


# ================= START STREAMING VIDEO JOB =================
@app.post("/start-stream")
async def start_stream(file: UploadFile = File(...), threshold: float = 0.55):
    """Upload a video and start background processing. Returns a job_id for SSE."""
    import asyncio

    if file.content_type not in ["video/mp4", "video/avi", "video/mov", "video/quicktime"]:
        return JSONResponse(status_code=400, content={"error": "Invalid video format"})

    file_ext = os.path.splitext(file.filename)[1]
    raw_filename = f"{uuid.uuid4()}{file_ext}"
    input_path = os.path.join(UPLOAD_DIR, raw_filename)

    with open(input_path, "wb") as f:
        f.write(await file.read())

    job_id = str(uuid.uuid4())
    _new_job(job_id)

    # Run processing in a background thread so SSE can stream concurrently
    thread = threading.Thread(
        target=_process_video_streaming,
        args=(job_id, input_path, file.filename or "video", threshold),
        daemon=True,
    )
    thread.start()

    return {"job_id": job_id}


def _encode_frame_jpeg(frame, quality=70):
    """Encode a BGR frame as JPEG bytes."""
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return buf.tobytes()


def _process_video_streaming(job_id, input_path, filename, threshold):
    """Background thread: processes video frame-by-frame and pushes to SSE."""
    try:
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            _finish_job(job_id, error="Cannot open video file")
            return

        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        MAX_WIDTH = 800
        if width > MAX_WIDTH:
            scale  = MAX_WIDTH / float(width)
            width  = MAX_WIDTH
            height = int(height * scale)

        PROCESS_EVERY_N_FRAMES = 2
        output_fps = fps / PROCESS_EVERY_N_FRAMES

        processed_filename = f"processed_{uuid.uuid4()}.mp4"
        output_path = os.path.join(UPLOAD_DIR, processed_filename)
        try:
            from utils.incident_clip import create_video_writer
            out_writer = create_video_writer(output_path, output_fps, width, height)
        except RuntimeError as e:
            _finish_job(job_id, error=str(e))
            return

        tracker = VehicleTracker()
        from phases.phase_b_trajectory import is_stationary as _is_stationary
        from utils.optical_flow import compute_optical_flow, calculate_frame_diff_ratio

        prev_gray = None
        features_buffer = []
        lstm_scores_history = []
        scene_speed_history = []

        source_frame_idx   = -1
        frame_idx          = 0
        accident_detected_globally = False
        max_accident_score = 0.0
        triggering_phase_globally = "None"
        accident_details   = {}
        max_accident_frame_image = None
        max_accident_frame_idx = 0
        max_peak_confidence = 0.0
        max_peak_dl        = 0.0
        max_peak_trigger   = "None"
        max_peak_details   = {}
        last_alert_source_frame = -999999
        last_alt_alert_frame = -999999
        last_alt_status = None
        consec_count       = 0
        incident_stubs     = []
        source_fps         = fps
        cooldown_frames    = max(1, int(config.COOLDOWN_SECONDS * source_fps))

        zone_x_min = int(width * 0.25); zone_x_max = int(width * 0.75)
        zone_y_min = int(height * 0.25); zone_y_max = int(height * 0.75)

        DL_GATE_THRESHOLD  = 0.55
        PHASE_SIGNAL_MIN   = 0.30
        t_start            = time.time()
        frames_processed   = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            source_frame_idx += 1
            if source_frame_idx % PROCESS_EVERY_N_FRAMES != 0:
                continue

            if frame.shape[1] > MAX_WIDTH:
                frame = cv2.resize(frame, (width, height))

            frames_processed += 1
            elapsed = time.time() - t_start
            proc_fps = frames_processed / elapsed if elapsed > 0.5 else 0.0

            frame_rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # DL feature extraction
            pil_img    = Image.fromarray(frame_rgb)
            frame_feat = transform(pil_img).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                feat = model.cnn(frame_feat)
            features_buffer.append(feat)
            if len(features_buffer) > SEQUENCE_LEN:
                features_buffer.pop(0)
            padded = [features_buffer[0]] * (SEQUENCE_LEN - len(features_buffer)) + features_buffer if len(features_buffer) < SEQUENCE_LEN else features_buffer
            features_tensor = torch.stack(padded, dim=1)
            with torch.no_grad():
                lstm_out, _ = model.bilstm(features_tensor)
                context     = model.attention(lstm_out)
                logits      = model.classifier(context)
                probs       = torch.softmax(logits, dim=1)
                cnn_lstm_prob = float(probs[0, 1].item())

            lstm_scores_history.append(cnn_lstm_prob)
            if len(lstm_scores_history) > 30:
                lstm_scores_history.pop(0)
            lstm_peak = max(lstm_scores_history) if len(features_buffer) >= 16 else cnn_lstm_prob

            # YOLO + tracker
            active_tracks = tracker.update(frame=frame)

            # Optical flow
            flow = compute_optical_flow(prev_gray, frame_gray) if prev_gray is not None else None

            # Scene speed
            mean_current_speed = 0.0
            if active_tracks:
                speeds = [math.sqrt(t.velocities[-1][0]**2 + t.velocities[-1][1]**2) for t in active_tracks]
                mean_current_speed = sum(speeds) / len(speeds)
            scene_speed_history.append(mean_current_speed)
            if len(scene_speed_history) > 120:
                scene_speed_history.pop(0)

            scene_interruption_score = 0.0
            if len(scene_speed_history) >= 20:
                bw = max(5, len(scene_speed_history) // 4)
                base_spd  = sum(scene_speed_history[:bw]) / bw
                recent_spd = sum(scene_speed_history[-5:]) / 5
                if base_spd > 1.0:
                    drop = (base_spd - recent_spd) / base_spd
                    scene_interruption_score = min(1.0, max(0.0, drop / 0.5)) if drop > 0.2 else 0.0

            # Phase signals
            candidate_pairs = proximity_filter(active_tracks)
            ttc_score = trajectory_stop_score = emergency_stop_score = 0.0
            relative_velocity_score = trajectory_score = anomaly_score = 0.0
            occlusion_score = merge_score = energy_drop_score = spin_score = 0.0
            flow_dispersion_score = diff_burst_score = 0.0
            frame_collision_pairs = []
            is_in_intersection_zone = False

            for track in active_tracks:
                anom_res = analyze_anomaly(track, flow)
                anomaly_score = max(anomaly_score, anom_res["anomaly_score"])
                flow_dispersion_score = max(flow_dispersion_score, anom_res["dispersion_val"])
                diff_ratio = calculate_frame_diff_ratio(prev_gray, frame_gray, track.bbox)
                diff_burst_score = max(diff_burst_score, 1.0 if diff_ratio > 0.15 else diff_ratio / 0.15)

            for t1, t2, dist, pair_ttc in candidate_pairs:
                if t1.age < 2 or t2.age < 2:
                    continue
                if _is_stationary(t1) and _is_stationary(t2):
                    continue
                pair_threshold = config.PROXIMITY_PERSON_THRESHOLD if (t1.label == config.PERSON_CLASS or t2.label == config.PERSON_CLASS) else config.PROXIMITY_THRESHOLD
                prox_s = max(0.0, 1.0 - (dist / pair_threshold), pair_ttc)
                ttc_score = max(ttc_score, prox_s, pair_ttc)
                traj_res = analyze_trajectory_conflict(t1, t2)
                trajectory_score = max(trajectory_score, traj_res["score"], traj_res["trajectory_stop_score"], traj_res["emergency_stop_score"], traj_res["relative_velocity_score"])
                trajectory_stop_score = max(trajectory_stop_score, traj_res["trajectory_stop_score"])
                emergency_stop_score  = max(emergency_stop_score,  traj_res["emergency_stop_score"])
                relative_velocity_score = max(relative_velocity_score, traj_res["relative_velocity_score"])
                occlusion_score = max(occlusion_score, traj_res["containment"] if traj_res["occluded"] else traj_res["containment"] * 0.5)
                if traj_res["merged"]: merge_score = 1.0
                energy_drop_score = max(energy_drop_score, traj_res["max_ke_drop"], traj_res["emergency_stop_score"])
                spin_score = max(spin_score, traj_res["max_spin_var"])
                c1, c2 = t1.get_centroid(), t2.get_centroid()
                cx, cy = int((c1[0]+c2[0])/2), int((c1[1]+c2[1])/2)
                if zone_x_min <= cx <= zone_x_max and zone_y_min <= cy <= zone_y_max:
                    is_in_intersection_zone = True
                if traj_res["class"] == "Collision" or traj_res["merged"] or traj_res["occluded"]:
                    frame_collision_pairs.append((t1, t2, dist, traj_res["class"]))

            prev_gray = frame_gray

            stopped = sum(1 for t in active_tracks if math.sqrt(t.velocities[-1][0]**2 + t.velocities[-1][1]**2) < 2.0) if active_tracks else 0
            stopped_ratio  = stopped / len(active_tracks) if active_tracks else 0.0
            traffic_density = min(len(active_tracks) / 20.0, 1.0)

            # Option 2 pipeline
            phase_a_signal = ttc_score
            phase_b_signal = max(trajectory_stop_score, emergency_stop_score, relative_velocity_score)
            phase_c_signal = anomaly_score
            dl_confirmed   = lstm_peak >= DL_GATE_THRESHOLD

            phases_signalling = 0
            phases_detail = {}
            if dl_confirmed:
                if phase_a_signal >= PHASE_SIGNAL_MIN: phases_signalling += 1; phases_detail["phase_a"] = True
                if phase_b_signal >= PHASE_SIGNAL_MIN: phases_signalling += 1; phases_detail["phase_b"] = True
                if phase_c_signal >= PHASE_SIGNAL_MIN: phases_signalling += 1; phases_detail["phase_c"] = True

            # Determine detection status and score
            if not dl_confirmed:
                frame_score = float(lstm_peak)
                frame_accident = False
                frame_trigger = f"DL Gate Not Cleared ({lstm_peak:.2f})"
                detection_status = "scanning"
            elif phases_signalling >= 2:
                fuse_res = fuse_scores(trajectory_stop=trajectory_stop_score, ttc_critical=ttc_score, emergency_stop=emergency_stop_score, cnn_lstm=lstm_peak, optical_flow=anomaly_score, flow_dispersion=flow_dispersion_score, scene_density=len(active_tracks), avg_scene_speed=mean_current_speed, stopped_ratio=stopped_ratio, threshold=threshold)
                frame_score = max(fuse_res["score"], lstm_peak * 0.8)
                frame_accident = True
                frame_trigger = f"DL + {' & '.join(p.replace('phase_','Phase ').upper() for p in phases_detail)} Verified"
                detection_status = "confirmed"
            elif phases_signalling == 1:
                fuse_res = fuse_scores(trajectory_stop=trajectory_stop_score, ttc_critical=ttc_score, emergency_stop=emergency_stop_score, cnn_lstm=lstm_peak, optical_flow=anomaly_score, flow_dispersion=flow_dispersion_score, scene_density=len(active_tracks), avg_scene_speed=mean_current_speed, stopped_ratio=stopped_ratio, threshold=threshold)
                frame_score = fuse_res["score"] * 0.7
                frame_accident = False
                frame_trigger = f"DL Confirmed, 1 Phase Only"
                detection_status = "suspicious"
            else:
                fuse_res = fuse_scores(trajectory_stop=trajectory_stop_score, ttc_critical=ttc_score, emergency_stop=emergency_stop_score, cnn_lstm=lstm_peak, optical_flow=anomaly_score, flow_dispersion=flow_dispersion_score, scene_density=len(active_tracks), avg_scene_speed=mean_current_speed, stopped_ratio=stopped_ratio, threshold=threshold)
                frame_score = fuse_res["score"] * 0.5
                frame_accident = False
                frame_trigger = "DL Confirmed, No Phase Signal"
                detection_status = "suppressed"

            # XGBoost refinement
            if dl_confirmed and phases_signalling >= 2 and xgb_clf is not None:
                try:
                    feats = [[float(ttc_score), float(trajectory_stop_score), float(anomaly_score), float(lstm_peak), float(occlusion_score), float(merge_score), float(emergency_stop_score), float(traffic_density), float(mean_current_speed), float(stopped_ratio)]]
                    xgb_prob = float(xgb_clf.predict_proba(feats)[0][1])
                    if xgb_prob < 0.35: frame_score *= 0.7; frame_trigger += " (XGB↓)"
                    else: frame_score = max(frame_score, xgb_prob); frame_trigger += " & XGB"
                    if frame_score < threshold: frame_accident = False
                except Exception: pass

            if frame_accident and is_in_intersection_zone:
                frame_score = min(1.0, frame_score * 1.15)
                frame_trigger += " & Risk Zone"

            frame_details = {
                "proximity_score": float(phase_a_signal), "trajectory_score": float(phase_b_signal),
                "flow_score": float(phase_c_signal), "ttc_score": float(ttc_score),
                "trajectory_stop_score": float(trajectory_stop_score), "emergency_stop_score": float(emergency_stop_score),
                "relative_velocity_score": float(relative_velocity_score), "energy_drop": float(energy_drop_score),
                "occlusion_score": float(occlusion_score), "merge_score": float(merge_score),
                "spin_score": float(spin_score), "lstm_peak": float(lstm_peak), "cnn_lstm_prob": float(cnn_lstm_prob),
                "traffic_density": float(traffic_density), "avg_speed": float(mean_current_speed),
                "stopped_ratio": float(stopped_ratio), "scene_interruption": float(scene_interruption_score),
                "dl_confirmed": bool(dl_confirmed), "phases_signalling": int(phases_signalling),
                "phase_a_confirmed": bool(phases_detail.get("phase_a", False)),
                "phase_b_confirmed": bool(phases_detail.get("phase_b", False)),
                "phase_c_confirmed": bool(phases_detail.get("phase_c", False)),
                "post_intersect_static": False,
            }

            # Consecutive gate
            cooldown_ok = (source_frame_idx - last_alert_source_frame) > cooldown_frames
            if frame_accident and cooldown_ok:
                consec_count += 1
            else:
                consec_count = max(0, consec_count - 1) if not frame_accident else consec_count

            confirmed_accident = frame_accident and consec_count >= config.CONSECUTIVE_FRAMES and cooldown_ok
            display_accident = frame_accident

            is_max_frame = False
            if frame_score > max_accident_score:
                max_accident_score = frame_score
                triggering_phase_globally = frame_trigger
                accident_details = dict(frame_details)
                is_max_frame = True

            if confirmed_accident:
                accident_detected_globally = True
                last_alert_source_frame = source_frame_idx
                consec_count = 0
                # Create a stub immediately for the live dashboard
                stub = {
                    "source_frame_idx": source_frame_idx,
                    "confidence": float(frame_score),
                    "dl_confidence": float(lstm_peak),
                    "trigger_phase": frame_trigger,
                    "details": dict(frame_details),
                    "snapshot_frame": frame.copy(),
                    "status": detection_status,
                }
                incident_stubs.append(stub)
                # Push immediate notification (without clip — clip built at end)
                _push_incident(job_id, {
                    "status": detection_status,
                    "confidence": float(frame_score),
                    "dl_confidence": float(lstm_peak),
                    "trigger_phase": frame_trigger,
                    "frame": source_frame_idx,
                    "time_sec": round(source_frame_idx / source_fps, 2),
                    "details": dict(frame_details),
                })

            # DL gate triggered but not confirmed — record suspicious/suppressed (deduped)
            elif dl_confirmed and not frame_accident and cooldown_ok and detection_status in ("suspicious", "suppressed"):
                alt_cooldown_ok = (source_frame_idx - last_alt_alert_frame) > cooldown_frames
                status_changed = detection_status != last_alt_status
                if alt_cooldown_ok or status_changed:
                    last_alt_alert_frame = source_frame_idx
                    last_alt_status = detection_status
                    stub = {
                        "source_frame_idx": source_frame_idx,
                        "confidence": float(frame_score),
                        "dl_confidence": float(lstm_peak),
                        "trigger_phase": frame_trigger,
                        "details": dict(frame_details),
                        "snapshot_frame": frame.copy(),
                        "status": detection_status,
                    }
                    incident_stubs.append(stub)
                    _push_incident(job_id, {
                        "status": detection_status,
                        "confidence": float(frame_score),
                        "dl_confidence": float(lstm_peak),
                        "trigger_phase": frame_trigger,
                        "frame": source_frame_idx,
                        "time_sec": round(source_frame_idx / source_fps, 2),
                        "details": dict(frame_details),
                    })

            if is_max_frame:
                max_accident_frame_image = frame.copy()
                max_accident_frame_idx = source_frame_idx
                max_peak_confidence = float(frame_score)
                max_peak_dl = float(lstm_peak)
                max_peak_trigger = frame_trigger
                max_peak_details = dict(frame_details)

            # Annotate frame
            cv2.rectangle(frame, (zone_x_min, zone_y_min), (zone_x_max, zone_y_max), (255,255,255), 1)
            cv2.putText(frame, "RISK ZONE", (zone_x_min+5, zone_y_min-5), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255,255,255), 1)

            for track in active_tracks:
                x1,y1,x2,y2 = map(int, track.bbox)
                color = (0,255,0)
                lbl   = f"ID{track.track_id}:{track.label[:3].upper()}"
                if hasattr(track, "anomaly_streak") and track.anomaly_streak >= 3:
                    color = (0,0,255); lbl += " ANOM"
                elif hasattr(track, "anomaly_streak") and track.anomaly_streak > 0:
                    color = (0,165,255)
                cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
                cv2.putText(frame, lbl, (x1, max(y1-6,0)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

            if display_accident:
                for t1, t2, _, status in frame_collision_pairs:
                    c1,c2 = t1.get_centroid(), t2.get_centroid()
                    pt1 = (int(c1[0]),int(c1[1])); pt2 = (int(c2[0]),int(c2[1]))
                    cx,cy = (pt1[0]+pt2[0])//2, (pt1[1]+pt2[1])//2
                    cv2.line(frame, pt1, pt2, (0,255,255), 2)
                    cv2.circle(frame, (cx,cy), 10, (0,0,255), -1)
                    cv2.putText(frame, f"IMPACT", (cx-30, cy-14), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
                overlay = frame.copy()
                cv2.rectangle(overlay, (0,0), (width,55), (0,0,255), -1)
                cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)
                cv2.putText(frame, f"ACCIDENT DETECTED  {frame_score*100:.1f}%", (16,36), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255,255,255), 2)

            # Mini HUD top-left
            hud_lines = [
                f"DL:{lstm_peak:.2f}{'✓' if dl_confirmed else '✗'}  Votes:{phases_signalling}/3",
                f"A:{phase_a_signal:.2f}  B:{phase_b_signal:.2f}  C:{phase_c_signal:.2f}",
                f"Score:{frame_score:.2f}  FPS:{proc_fps:.1f}",
                f"Tracks:{len(active_tracks)}  Speed:{mean_current_speed:.1f}",
            ]
            hud_bg = frame.copy()
            cv2.rectangle(hud_bg, (0, 56), (260, 56+len(hud_lines)*18+6), (10,10,30), -1)
            cv2.addWeighted(hud_bg, 0.7, frame, 0.3, 0, frame)
            for i, line in enumerate(hud_lines):
                cv2.putText(frame, line, (8, 72+i*18), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200,230,255), 1)

            out_writer.write(frame)
            frame_idx += 1

            # Push JPEG frame to SSE (every frame)
            jpeg = _encode_frame_jpeg(frame, quality=65)
            metrics = {
                "frame": source_frame_idx, "total_frames": total_frames,
                "progress": min(100, int(source_frame_idx / max(total_frames,1) * 100)),
                "fps": round(proc_fps, 1),
                "dl": round(lstm_peak, 3), "dl_confirmed": bool(dl_confirmed),
                "phase_a": round(phase_a_signal, 3), "phase_b": round(phase_b_signal, 3),
                "phase_c": round(phase_c_signal, 3), "votes": phases_signalling,
                "score": round(frame_score, 3), "status": detection_status,
                "tracks": len(active_tracks), "speed": round(mean_current_speed, 2),
                "density": round(traffic_density, 2), "stopped": round(stopped_ratio, 2),
                "trigger": frame_trigger, "elapsed": round(elapsed, 1),
            }
            _push_frame(job_id, jpeg, metrics)

        cap.release()
        out_writer.release()

        from utils.incident_clip import transcode_video_for_browser
        transcode_video_for_browser(output_path)

        saved_incidents = _save_incident_records(
            incident_stubs, input_path, source_fps, total_frames, filename,
        )

        if os.path.exists(input_path):
            try: os.remove(input_path)
            except: pass

        result = {
            "class": "ACCIDENT" if accident_detected_globally else "NO ACCIDENT",
            "confidence": float(max_accident_score * 100),
            "trigger_phase": triggering_phase_globally,
            "processed_video_url": f"/static/uploads/{processed_filename}",
            "details": accident_details,
            "incidents": saved_incidents,
            "incident_count": len(saved_incidents),
        }
        _finish_job(job_id, result=result)

    except Exception as e:
        import traceback; traceback.print_exc()
        _finish_job(job_id, error=str(e))
@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse(request, "index.html")


@app.get("/api/incidents")
def api_incidents(limit: int = 50):
    return {"incidents": incident_store.list_incidents(limit=limit)}


@app.get("/api/firebase/status")
def api_firebase_status():
    return {"connected": _firebase.enabled}


@app.delete("/api/incidents/{incident_id}")
def api_delete_incident(incident_id: str):
    if not incident_store.delete_incident(incident_id):
        return JSONResponse(status_code=404, content={"error": "Incident not found"})
    return {"ok": True}


# ================= IMAGE PREDICTION =================
@app.post("/predict-image")
async def predict_image_api(file: UploadFile = File(...), threshold: float = 0.50):
    try:
        # Validate file type
        if file.content_type not in ["image/jpeg", "image/png", "image/jpg", "image/webp"]:
            return JSONResponse(status_code=400, content={"error": "Invalid image format"})

        file_ext = os.path.splitext(file.filename)[1]
        raw_filename = f"{uuid.uuid4()}{file_ext}"
        file_path = os.path.join(UPLOAD_DIR, raw_filename)

        with open(file_path, "wb") as f:
            f.write(await file.read())

        print("Processing image:", file_path)

        # 1. Run CNN-LSTM via model.py predict_image helper
        dl_label, dl_conf_pct = predict_image(file_path)
        cnn_lstm_prob = dl_conf_pct / 100.0 if dl_label == "ACCIDENT" else (100.0 - dl_conf_pct) / 100.0

        # 2. Run YOLO Object Detection
        detector = YOLODetector()
        frame = cv2.imread(file_path)
        if frame is None:
            return JSONResponse(status_code=400, content={"error": "Could not read image file."})
            
        detections = detector.detect(frame)

        # 3. Calculate Proximity and Occlusion/Containment
        from tracking.deepsort_module import Track
        tracks = []
        for idx, det in enumerate(detections):
            track = Track(idx + 1, det["bbox"], det["label"], det["confidence"])
            tracks.append(track)

        proximity_score = 0.0
        occlusion_score = 0.0
        
        candidate_pairs = proximity_filter(tracks)
        
        for t1, t2, dist, ttc_s in candidate_pairs:
            pair_threshold = config.PROXIMITY_PERSON_THRESHOLD if (
                t1.label == config.PERSON_CLASS or t2.label == config.PERSON_CLASS
            ) else config.PROXIMITY_THRESHOLD
            prox_s = max(0.0, 1.0 - (dist / pair_threshold), ttc_s)
            proximity_score = max(proximity_score, prox_s)
            
            # Static containment check for occlusion
            area1 = (t1.bbox[2] - t1.bbox[0]) * (t1.bbox[3] - t1.bbox[1])
            area2 = (t2.bbox[2] - t2.bbox[0]) * (t2.bbox[3] - t2.bbox[1])
            if area1 < area2:
                containment = calculate_bbox_containment_ratio(t1.bbox, t2.bbox)
            else:
                containment = calculate_bbox_containment_ratio(t2.bbox, t1.bbox)
            occlusion_score = max(occlusion_score, containment)

        # Apply suppression for standing/queuing vehicles in static images
        max_iou = 0.0
        if len(tracks) >= 2:
            for i in range(len(tracks)):
                for j in range(i + 1, len(tracks)):
                    iou = compute_iou(tracks[i].bbox, tracks[j].bbox)
                    max_iou = max(max_iou, iou)

        if len(tracks) >= 2:
            # If the overlap is not extreme (not a physical crash geometry), suppress to prevent false alerts on standing traffic
            if max_iou < 0.40 and occlusion_score < 0.70:
                print(f"Standing traffic/parking detected ({len(tracks)} vehicles, IoU={max_iou:.2f}, Occlusion={occlusion_score:.2f}). Suppressing proximity and occlusion scores.")
                proximity_score = proximity_score * 0.15
                occlusion_score = occlusion_score * 0.15
            elif len(tracks) > 4:
                # Dense queue / parking lot
                print(f"Dense image queue detected ({len(tracks)} vehicles). Suppressing scores.")
                proximity_score = proximity_score * 0.25
                occlusion_score = occlusion_score * 0.25

        # Calculate traffic features for static image
        vehicle_count = len(tracks)
        traffic_density = min(vehicle_count / 20.0, 1.0)
        avg_speed = 0.0
        stopped_ratio = 1.0 if vehicle_count > 0 else 0.0

        # Stage 3: Phase C + CNN Gate (Phase C flow is 0.0 for static images)
        if cnn_lstm_prob < 0.40:
            final_score = 0.0
            is_accident = False
            final_class = "NO ACCIDENT"
        else:
            # 4. Score Fusion for Static Image (Fallback Heuristic)
            final_score = 0.2 * proximity_score + 0.2 * occlusion_score + 0.6 * cnn_lstm_prob
            
            # Use XGBoost model if available
            if xgb_clf is not None:
                feats = [[
                    float(proximity_score),
                    0.0, # no trajectory in static image
                    0.0, # no flow in static image
                    float(cnn_lstm_prob),
                    float(occlusion_score),
                    0.0, # no merge in static image
                    0.0, # no energy drop in static image
                    float(traffic_density),
                    float(avg_speed),
                    float(stopped_ratio)
                ]]
                try:
                    final_score = float(xgb_clf.predict_proba(feats)[0][1])
                except Exception as e:
                    print("XGBoost image predict error, falling back:", e)
            
            is_accident = final_score >= threshold  
            final_class = "ACCIDENT" if is_accident else "NO ACCIDENT"

        # 5. Annotate Image
        for track in tracks:
            x1, y1, x2, y2 = map(int, track.bbox)
            label = f"{track.label.upper()} {track.confidence:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        for t1, t2, dist, _ in candidate_pairs:
            c1 = t1.get_centroid()
            c2 = t2.get_centroid()
            pt1 = (int(c1[0]), int(c1[1]))
            pt2 = (int(c2[0]), int(c2[1]))
            cv2.line(frame, pt1, pt2, (0, 255, 255), 2)
            
            if is_accident:
                cx = int((c1[0] + c2[0]) / 2)
                cy = int((c1[1] + c2[1]) / 2)
                cv2.circle(frame, (cx, cy), 10, (0, 0, 255), -1)

        if is_accident:
            h, w = frame.shape[:2]
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (w, 50), (0, 0, 255), -1)
            cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)
            cv2.putText(frame, f"🚨 ACCIDENT SUSPECTED ({final_score*100:.1f}%)", (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        processed_filename = f"processed_{uuid.uuid4()}.jpg"
        processed_path = os.path.join(UPLOAD_DIR, processed_filename)
        cv2.imwrite(processed_path, frame)

        # Remove raw image
        if os.path.exists(file_path):
            os.remove(file_path)

        return {
            "class": final_class,
            "confidence": float(final_score * 100),
            "trigger_phase": "XGBoost Classifier Model" if xgb_clf is not None else ("Phase A (Proximity) & Occlusion-Containment & CNN-LSTM DL Module" if is_accident else "None"),
            "processed_image_url": f"/static/uploads/{processed_filename}",
            "details": {
                "proximity_score": float(proximity_score),
                "occlusion_score": float(occlusion_score),
                "trajectory_score": 0.0,
                "anomaly_score": 0.0,
                "cnn_lstm_prob": float(cnn_lstm_prob),
                "traffic_density": float(traffic_density),
                "avg_speed": float(avg_speed),
                "stopped_ratio": float(stopped_ratio)
            }
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(e)})


# ================= VIDEO PREDICTION =================
@app.post("/predict-video")
async def predict_video_api(file: UploadFile = File(...), threshold: float = 0.55):
    try:
        # Validate file type
        if file.content_type not in ["video/mp4", "video/avi", "video/mov", "video/quicktime"]:
            return JSONResponse(status_code=400, content={"error": "Invalid video format"})

        file_ext = os.path.splitext(file.filename)[1]
        raw_filename = f"{uuid.uuid4()}{file_ext}"
        input_path = os.path.join(UPLOAD_DIR, raw_filename)

        with open(input_path, "wb") as f:
            f.write(await file.read())

        print("Processing video:", input_path)

        # Open Video Capture
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            return JSONResponse(status_code=400, content={"error": "Failed to open video file."})

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0 or fps is None or math.isnan(fps):
            fps = 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # --- OPTIMIZATION: Frame Resizing ---
        MAX_WIDTH = 800
        if width > MAX_WIDTH:
            scale = MAX_WIDTH / float(width)
            width = MAX_WIDTH
            height = int(height * scale)
            
        # --- OPTIMIZATION: Frame Skipping ---
        PROCESS_EVERY_N_FRAMES = 2  # Process every 2nd frame (cuts time by 50%)
        output_fps = fps / PROCESS_EVERY_N_FRAMES

        processed_filename = f"processed_{uuid.uuid4()}.mp4"
        output_path = os.path.join(UPLOAD_DIR, processed_filename)
        try:
            out_writer = create_video_writer(output_path, output_fps, width, height)
        except RuntimeError as e:
            return JSONResponse(status_code=500, content={"error": str(e)})

        # Instantiate modules
        tracker = VehicleTracker()

        prev_gray = None
        features_buffer = []
        lstm_scores_history = []
        scene_speed_history = []

        frame_idx = 0
        accident_detected_globally = False
        max_accident_score = 0.0
        triggering_phase_globally = "None"
        accident_details = {}
        max_accident_frame_image = None
        max_accident_frame_idx = 0
        max_peak_confidence = 0.0
        max_peak_dl = 0.0
        max_peak_trigger = "None"
        max_peak_details = {}
        last_alert_source_frame = -999999
        last_alt_alert_frame = -999999
        last_alt_status = None
        consec_count = 0
        incident_stubs = []
        source_fps = fps
        cooldown_frames = max(1, int(config.COOLDOWN_SECONDS * source_fps))

        # High-risk intersection zone dimensions (center 50% area)
        zone_x_min, zone_x_max = int(width * 0.25), int(width * 0.75)
        zone_y_min, zone_y_max = int(height * 0.25), int(height * 0.75)

        # Main frame loop
        source_frame_idx = -1
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            source_frame_idx += 1

            if source_frame_idx % PROCESS_EVERY_N_FRAMES != 0:
                continue

            if frame.shape[1] > MAX_WIDTH:
                frame = cv2.resize(frame, (width, height))

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # --- CNN-BiLSTM-Attention Feature Extraction ---
            pil_img = Image.fromarray(frame_rgb)
            frame_feat = transform(pil_img).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                feat = model.cnn(frame_feat)  # shape: (1, 1280)

            features_buffer.append(feat)
            if len(features_buffer) > SEQUENCE_LEN:
                features_buffer.pop(0)

            # Pad features if buffer is not full
            if len(features_buffer) < SEQUENCE_LEN:
                padded_buffer = [features_buffer[0]] * (SEQUENCE_LEN - len(features_buffer)) + features_buffer
            else:
                padded_buffer = features_buffer

            features_tensor = torch.stack(padded_buffer, dim=1)  # shape: (1, SEQUENCE_LEN, 1280)
            with torch.no_grad():
                lstm_out, _ = model.bilstm(features_tensor)        # (1, SEQUENCE_LEN, 512)
                context = model.attention(lstm_out)                # (1, 512)
                logits = model.classifier(context)
                probs = torch.softmax(logits, dim=1)
                cnn_lstm_prob = float(probs[0, 1].item())

            # LSTM Peak Memory (rolling 30-frame maximum)
            # When buffer is still building (< 16 frames), use current prob directly
            # so the DL model contributes signal from the start of short videos.
            lstm_scores_history.append(cnn_lstm_prob)
            if len(lstm_scores_history) > 30:
                lstm_scores_history.pop(0)
            if len(features_buffer) >= 16:
                lstm_peak = max(lstm_scores_history)
            else:
                # Buffer still warming up - use current probability directly
                lstm_peak = cnn_lstm_prob

            # --- YOLO + ByteTrack ---
            active_tracks = tracker.update(frame=frame)

            # --- Optical Flow ---
            flow = None
            if prev_gray is not None:
                flow = compute_optical_flow(prev_gray, frame_gray)

            # --- Scene-Level Speed calculation (Traffic Interruption) ---
            mean_current_speed = 0.0
            if active_tracks:
                speeds = []
                for track in active_tracks:
                    vx, vy = track.velocities[-1]
                    speeds.append(math.sqrt(vx**2 + vy**2))
                mean_current_speed = sum(speeds) / len(speeds)
                scene_speed_history.append(mean_current_speed)
            else:
                scene_speed_history.append(0.0)

            if len(scene_speed_history) > 120:
                scene_speed_history.pop(0)
            
            scene_interruption_score = 0.0
            if len(scene_speed_history) >= 20:
                # Use first 25% of history as "before accident" baseline
                baseline_window = max(5, len(scene_speed_history) // 4)
                baseline_speed = sum(scene_speed_history[:baseline_window]) / baseline_window
                # Compare against recent 5 frames
                recent_speed = sum(scene_speed_history[-5:]) / 5
                if baseline_speed > 1.0:
                    speed_drop = (baseline_speed - recent_speed) / baseline_speed
                    if speed_drop > 0.50:
                        scene_interruption_score = 1.0
                    elif speed_drop > 0.20:
                        scene_interruption_score = float(speed_drop / 0.50)

            # --- Phase A: Proximity + TTC Filtering ---
            candidate_pairs = proximity_filter(active_tracks)

            # --- Score Accumulators ---
            ttc_score = 0.0
            trajectory_stop_score = 0.0
            emergency_stop_score = 0.0
            relative_velocity_score = 0.0
            trajectory_score = 0.0
            anomaly_score = 0.0
            occlusion_score = 0.0
            merge_score = 0.0
            energy_drop_score = 0.0
            spin_score = 0.0
            diff_burst_score = 0.0
            flow_dispersion_score = 0.0
            post_intersect_static_score = False

            frame_collision_pairs = []
            is_in_intersection_zone = False

            # Evaluate Track Anomaly (Phase C & Diff Burst)
            for track in active_tracks:
                anom_res = analyze_anomaly(track, flow)
                anomaly_score = max(anomaly_score, anom_res["anomaly_score"])
                flow_dispersion_score = max(flow_dispersion_score, anom_res["dispersion_val"])
                
                # Frame Difference Shock Burst
                diff_ratio = calculate_frame_diff_ratio(prev_gray, frame_gray, track.bbox)
                diff_burst = 1.0 if diff_ratio > 0.15 else float(diff_ratio / 0.15)
                diff_burst_score = max(diff_burst_score, diff_burst)

            # Evaluate Candidate Trajectory Conflicts (Phase B)
            for t1, t2, dist, pair_ttc in candidate_pairs:
                if t1.age < 2 or t2.age < 2:
                    continue

                if is_stationary(t1) and is_stationary(t2):
                    continue

                pair_threshold = config.PROXIMITY_PERSON_THRESHOLD if (
                    t1.label == config.PERSON_CLASS or t2.label == config.PERSON_CLASS
                ) else config.PROXIMITY_THRESHOLD
                prox_s = max(0.0, 1.0 - (dist / pair_threshold), pair_ttc)
                ttc_score = max(ttc_score, prox_s, pair_ttc)

                traj_res = analyze_trajectory_conflict(t1, t2)
                trajectory_score = max(
                    trajectory_score,
                    traj_res["score"],
                    traj_res["trajectory_stop_score"],
                    traj_res["emergency_stop_score"],
                    traj_res["relative_velocity_score"],
                )
                trajectory_stop_score = max(trajectory_stop_score, traj_res["trajectory_stop_score"])
                emergency_stop_score = max(emergency_stop_score, traj_res["emergency_stop_score"])
                relative_velocity_score = max(relative_velocity_score, traj_res["relative_velocity_score"])

                if traj_res["occluded"]:
                    occlusion_score = max(occlusion_score, traj_res["containment"])
                else:
                    occlusion_score = max(occlusion_score, traj_res["containment"] * 0.5)

                if traj_res["merged"]:
                    merge_score = 1.0

                energy_drop_score = max(energy_drop_score, traj_res["max_ke_drop"], traj_res["emergency_stop_score"])
                spin_score = max(spin_score, traj_res["max_spin_var"])

                if traj_res.get("post_intersect_static", False):
                    post_intersect_static_score = True

                # Determine if collision center point lies inside high-risk zone
                c1, c2 = t1.get_centroid(), t2.get_centroid()
                cx, cy = int((c1[0] + c2[0]) / 2), int((c1[1] + c2[1]) / 2)
                if zone_x_min <= cx <= zone_x_max and zone_y_min <= cy <= zone_y_max:
                    is_in_intersection_zone = True

                if traj_res["class"] == "Collision" or traj_res["merged"] or traj_res["occluded"]:
                    frame_collision_pairs.append((t1, t2, dist, traj_res["class"]))

            # Store gray frame reference
            prev_gray = frame_gray

            # Calculate stopped ratio for the frame
            stopped = 0
            if active_tracks:
                for track in active_tracks:
                    vx, vy = track.velocities[-1]
                    track_speed = math.sqrt(vx**2 + vy**2)
                    if track_speed < 2.0: # stopped threshold = 2.0
                        stopped += 1
                stopped_ratio = stopped / len(active_tracks)
            else:
                stopped_ratio = 0.0
            
            traffic_density = min(len(active_tracks) / 20.0, 1.0)

            # ================================================================
            # OPTION 2 DETECTION PIPELINE
            # Step 1: DL model as hard gate (must be >= 0.55 to proceed)
            # Step 2: Run all 3 phases in parallel
            # Step 3: Count phases with signal >= 0.3 (need at least 2 of 3)
            # Step 4: If DL confirmed AND 2+ phases signal → accident
            # ================================================================

            DL_GATE_THRESHOLD = 0.55
            PHASE_SIGNAL_MIN   = 0.30

            # -- Phase signals (already computed above) --
            phase_a_signal = ttc_score                                          # Phase A: proximity/TTC
            phase_b_signal = max(trajectory_stop_score,
                                 emergency_stop_score,
                                 relative_velocity_score)                       # Phase B: trajectory
            phase_c_signal = anomaly_score                                      # Phase C: optical flow/anomaly

            # -- DL gate --
            dl_confirmed = lstm_peak >= DL_GATE_THRESHOLD

            # -- Phase vote (only evaluated when DL confirms) --
            phases_signalling = 0
            phases_detail = {}
            if dl_confirmed:
                if phase_a_signal >= PHASE_SIGNAL_MIN:
                    phases_signalling += 1
                    phases_detail["phase_a"] = True
                if phase_b_signal >= PHASE_SIGNAL_MIN:
                    phases_signalling += 1
                    phases_detail["phase_b"] = True
                if phase_c_signal >= PHASE_SIGNAL_MIN:
                    phases_signalling += 1
                    phases_detail["phase_c"] = True

            # -- Final decision --
            if not dl_confirmed:
                frame_score = float(lstm_peak)
                frame_accident = False
                frame_trigger = f"DL Gate Not Cleared ({lstm_peak:.2f} < {DL_GATE_THRESHOLD})"
                detection_status = "scanning"
            elif phases_signalling >= 2:
                # DL confirmed + 2 of 3 phases agree → confirmed accident
                # Use fusion score for the final confidence value
                fuse_res = fuse_scores(
                    trajectory_stop=trajectory_stop_score,
                    ttc_critical=ttc_score,
                    emergency_stop=emergency_stop_score,
                    cnn_lstm=lstm_peak,
                    optical_flow=anomaly_score,
                    flow_dispersion=flow_dispersion_score,
                    scene_density=len(active_tracks),
                    avg_scene_speed=mean_current_speed,
                    stopped_ratio=stopped_ratio,
                    threshold=threshold,
                )
                frame_score = max(fuse_res["score"], lstm_peak * 0.8)
                frame_accident = True
                confirmed_phases = [k for k, v in phases_detail.items() if v]
                frame_trigger = f"DL + {' & '.join(p.replace('phase_', 'Phase ').upper() for p in confirmed_phases)} Verified"
                detection_status = "confirmed"
                fuse_res_details = fuse_res["details"]
            elif phases_signalling == 1:
                # DL confirmed but only 1 phase agrees — suspicious but not confirmed
                fuse_res = fuse_scores(
                    trajectory_stop=trajectory_stop_score,
                    ttc_critical=ttc_score,
                    emergency_stop=emergency_stop_score,
                    cnn_lstm=lstm_peak,
                    optical_flow=anomaly_score,
                    flow_dispersion=flow_dispersion_score,
                    scene_density=len(active_tracks),
                    avg_scene_speed=mean_current_speed,
                    stopped_ratio=stopped_ratio,
                    threshold=threshold,
                )
                frame_score = fuse_res["score"] * 0.7   # reduce score to stay below threshold
                frame_accident = False
                frame_trigger = f"DL Confirmed but Only 1 Phase ({list(phases_detail.keys())[0] if phases_detail else 'none'})"
                detection_status = "suspicious"
                fuse_res_details = fuse_res["details"]
            else:
                # DL confirmed but no phases signal — likely DL false positive
                fuse_res = fuse_scores(
                    trajectory_stop=trajectory_stop_score,
                    ttc_critical=ttc_score,
                    emergency_stop=emergency_stop_score,
                    cnn_lstm=lstm_peak,
                    optical_flow=anomaly_score,
                    flow_dispersion=flow_dispersion_score,
                    scene_density=len(active_tracks),
                    avg_scene_speed=mean_current_speed,
                    stopped_ratio=stopped_ratio,
                    threshold=threshold,
                )
                frame_score = fuse_res["score"] * 0.5   # suppress heavily
                frame_accident = False
                frame_trigger = "DL Confirmed but No Phase Signal (Suppressed)"
                detection_status = "suppressed"
                fuse_res_details = fuse_res["details"]

            # -- XGBoost refinement (only when DL+2 phases already confirmed) --
            if dl_confirmed and phases_signalling >= 2 and xgb_clf is not None:
                feats = [[
                    float(ttc_score),
                    float(trajectory_stop_score),
                    float(anomaly_score),
                    float(lstm_peak),
                    float(occlusion_score),
                    float(merge_score),
                    float(emergency_stop_score),
                    float(traffic_density),
                    float(mean_current_speed),
                    float(stopped_ratio),
                ]]
                try:
                    xgb_prob = float(xgb_clf.predict_proba(feats)[0][1])
                    # XGBoost can only refine downward if it strongly disagrees,
                    # not override a confirmed multi-phase detection upward.
                    if xgb_prob < 0.35:
                        frame_score *= 0.7
                        frame_trigger += " (XGB Low Confidence)"
                        if frame_score < threshold:
                            frame_accident = False
                    else:
                        frame_score = max(frame_score, xgb_prob)
                        frame_trigger += " & XGBoost"
                except Exception as e:
                    print("XGBoost refinement error:", e)

            # -- Zone risk multiplier (only amplifies already-confirmed frames) --
            if frame_accident and is_in_intersection_zone:
                frame_score = min(1.0, frame_score * 1.15)
                if "Risk Zone" not in frame_trigger:
                    frame_trigger += " & Risk Zone"

            # -- Build frame_details for dashboard display --
            frame_details = {
                # Phase scores for display
                "proximity_score":       float(phase_a_signal),
                "trajectory_score":      float(phase_b_signal),
                "flow_score":            float(phase_c_signal),
                # Sub-signals
                "ttc_score":             float(ttc_score),
                "trajectory_stop_score": float(trajectory_stop_score),
                "emergency_stop_score":  float(emergency_stop_score),
                "relative_velocity_score": float(relative_velocity_score),
                "energy_drop":           float(energy_drop_score),
                "occlusion_score":       float(occlusion_score),
                "merge_score":           float(merge_score),
                "spin_score":            float(spin_score),
                # DL
                "lstm_peak":             float(lstm_peak),
                "cnn_lstm_prob":         float(cnn_lstm_prob),
                # Scene
                "traffic_density":       float(traffic_density),
                "avg_speed":             float(mean_current_speed),
                "stopped_ratio":         float(stopped_ratio),
                "scene_interruption":    float(scene_interruption_score),
                # Pipeline status
                "dl_confirmed":          bool(dl_confirmed),
                "phases_signalling":     int(phases_signalling),
                "phase_a_confirmed":     bool(phases_detail.get("phase_a", False)),
                "phase_b_confirmed":     bool(phases_detail.get("phase_b", False)),
                "phase_c_confirmed":     bool(phases_detail.get("phase_c", False)),
                # Post intersect
                "post_intersect_static": bool(post_intersect_static_score),
            }

            # Consecutive frame gate + cooldown
            if frame_accident and frame_score >= threshold:
                consec_count += 1
            else:
                consec_count = 0

            confirmed_accident = (
                consec_count >= config.CONSECUTIVE_FRAMES
                and (source_frame_idx - last_alert_source_frame) >= cooldown_frames
            )

            if confirmed_accident:
                last_alert_source_frame = source_frame_idx
                consec_count = 0
                display_accident = True
                incident_stubs.append({
                    "source_frame_idx": source_frame_idx,
                    "confidence": float(frame_score),
                    "dl_confidence": float(lstm_peak),
                    "trigger_phase": frame_trigger,
                    "details": dict(frame_details),
                    "snapshot_frame": frame.copy(),
                    "status": "confirmed",
                })
            elif dl_confirmed and not frame_accident and detection_status in ("suspicious", "suppressed"):
                cooldown_ok = (source_frame_idx - last_alert_source_frame) > cooldown_frames
                if cooldown_ok:
                    alt_cooldown_ok = (source_frame_idx - last_alt_alert_frame) > cooldown_frames
                    status_changed = detection_status != last_alt_status
                    if alt_cooldown_ok or status_changed:
                        last_alt_alert_frame = source_frame_idx
                        last_alt_status = detection_status
                        incident_stubs.append({
                            "source_frame_idx": source_frame_idx,
                            "confidence": float(frame_score),
                            "dl_confidence": float(lstm_peak),
                            "trigger_phase": frame_trigger,
                            "details": dict(frame_details),
                            "snapshot_frame": frame.copy(),
                            "status": detection_status,
                        })
            else:
                display_accident = False

            # Log max statistics
            is_max_frame = False
            if frame_score > max_accident_score:
                max_accident_score = frame_score
                triggering_phase_globally = frame_trigger
                accident_details = dict(frame_details)
                is_max_frame = True

            if confirmed_accident:
                accident_detected_globally = True

            # --- Annotations ---
            # High-Risk Intersection Zone Boundary
            cv2.rectangle(frame, (zone_x_min, zone_y_min), (zone_x_max, zone_y_max), (255, 255, 255), 1, lineType=cv2.LINE_AA)
            cv2.putText(frame, "INTERSECTION RISK ZONE", (zone_x_min + 5, zone_y_min - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

            # Draw tracks
            for track in active_tracks:
                x1, y1, x2, y2 = map(int, track.bbox)
                
                # Check if vehicle is stationary
                if is_stationary(track):
                    label = f"ID {track.track_id}: {track.label.upper()} (STATIONARY)"
                    color = (0, 180, 0) # slightly darker green for stationary/parked vehicles
                else:
                    label = f"ID {track.track_id}: {track.label.upper()} {track.confidence:.2f}"
                    color = (0, 255, 0) # bright green for moving vehicles

                # sus or confirmed anomaly highlighting
                if hasattr(track, "anomaly_streak") and track.anomaly_streak > 0:
                    if track.anomaly_streak >= 3:
                        color = (0, 0, 255)
                        cv2.putText(frame, "⚠️ ANOMALY IMPACT", (x1, y1 - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
                    else:
                        color = (0, 165, 255)

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

                for pt in track.history:
                    cv2.circle(frame, (int(pt[0]), int(pt[1])), 2, color, -1)

            # Draw collision points & lines — only on confirmed-accident frames,
            # so the "IMPACT" overlay matches the actual decision (avoids showing
            # IMPACT markers on frames the system correctly scored as non-accident
            # due to harmless bbox overlap/occlusion from camera perspective).
            if display_accident:
                for t1, t2, dist, status in frame_collision_pairs:
                    c1, c2 = t1.get_centroid(), t2.get_centroid()
                    pt1 = (int(c1[0]), int(c1[1]))
                    pt2 = (int(c2[0]), int(c2[1]))

                    cv2.line(frame, pt1, pt2, (0, 255, 255), 2)
                    center_x, center_y = int((c1[0] + c2[0]) / 2), int((c1[1] + c2[1]) / 2)
                    cv2.circle(frame, (center_x, center_y), 8, (0, 0, 255), -1)
                    cv2.circle(frame, (center_x, center_y), 15, (0, 0, 255), 2)
                    cv2.putText(frame, f"IMPACT ({status})", (center_x - 50, center_y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)


            if is_max_frame:
                max_accident_frame_image = frame.copy()
                max_accident_frame_idx = source_frame_idx
                max_peak_confidence = float(frame_score)
                max_peak_dl = float(lstm_peak)
                max_peak_trigger = frame_trigger
                max_peak_details = dict(frame_details)

            # Flashing banner alert
            if display_accident:
                overlay = frame.copy()
                cv2.rectangle(overlay, (0, 0), (width, 60), (0, 0, 255), -1)
                cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)
                cv2.putText(frame, f"🚨 ACCIDENT DETECTED ({frame_score*100:.1f}%)", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)
                cv2.putText(frame, f"Triggers: {frame_trigger}", (width - 480, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

            # ITS Telemetry Overlay Panel (expanded size for new factors)
            panel_w, panel_h = 320, 240
            panel_x, panel_y = width - panel_w - 10, 70

            panel_overlay = frame.copy()
            cv2.rectangle(panel_overlay, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), (30, 20, 10), -1)
            cv2.addWeighted(panel_overlay, 0.75, frame, 0.25, 0, frame)
            cv2.rectangle(frame, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), (255, 255, 255), 1)

            cv2.putText(frame, "ITS SYSTEM MONITOR v3", (panel_x + 10, panel_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            cv2.putText(frame, f"Frame: {source_frame_idx}/{total_frames} | Active: {len(active_tracks)}", (panel_x + 10, panel_y + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

            cv2.putText(frame, f"TTC Critical: {ttc_score:.2f}", (panel_x + 10, panel_y + 65), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255) if ttc_score > 0 else (200, 200, 200), 1)
            cv2.putText(frame, f"Trajectory Stop: {trajectory_stop_score:.2f}", (panel_x + 10, panel_y + 85), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 165, 255) if trajectory_stop_score > 0 else (200, 200, 200), 1)
            cv2.putText(frame, f"Emergency Stop: {emergency_stop_score:.2f}", (panel_x + 10, panel_y + 105), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255) if emergency_stop_score > 0 else (200, 200, 200), 1)
            cv2.putText(frame, f"Rel Velocity: {relative_velocity_score:.2f}", (panel_x + 10, panel_y + 125), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255) if relative_velocity_score > 0 else (200, 200, 200), 1)
            cv2.putText(frame, f"Optical Flow: {anomaly_score:.2f}", (panel_x + 10, panel_y + 150), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255) if anomaly_score > 0 else (200, 200, 200), 1)
            cv2.putText(frame, f"Flow Dispersion: {flow_dispersion_score:.2f}", (panel_x + 10, panel_y + 170), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 165, 255) if flow_dispersion_score > 0 else (200, 200, 200), 1)
            cv2.putText(frame, f"Spin: {spin_score:.2f} | Merge: {int(merge_score)}", (panel_x + 10, panel_y + 190), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255) if spin_score > 0.15 else (200, 200, 200), 1)
            cv2.putText(frame, f"Scene Interruption: {scene_interruption_score:.2f}", (panel_x + 10, panel_y + 210), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0) if scene_interruption_score > 0.5 else (200, 200, 200), 1)
            cv2.putText(frame, f"Consec: {consec_count}/{config.CONSECUTIVE_FRAMES} | Score: {frame_score:.2f}", (panel_x + 10, panel_y + 230), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 255, 0), 1)

            out_writer.write(frame)
            frame_idx += 1

        cap.release()
        out_writer.release()

        if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
            return JSONResponse(
                status_code=500,
                content={"error": "Failed to write processed video output."},
            )

        transcode_video_for_browser(output_path)

        # Peak-score fallback: UI treats max score >= threshold as incident even when
        # the consecutive-frame gate did not fire (common with XGBoost single-frame spikes).
        if not incident_stubs and max_accident_score >= threshold and max_accident_frame_image is not None:
            incident_stubs.append({
                "source_frame_idx": max_accident_frame_idx,
                "confidence": max_peak_confidence,
                "dl_confidence": max_peak_dl,
                "trigger_phase": max_peak_trigger,
                "details": max_peak_details,
                "snapshot_frame": max_accident_frame_image.copy(),
                "status": "confirmed",
            })

        if max_accident_score >= threshold:
            accident_detected_globally = True

        saved_incidents = _save_incident_records(
            incident_stubs,
            input_path,
            source_fps,
            total_frames,
            file.filename or "Uploaded video",
        )

        # Clean up raw video
        if os.path.exists(input_path):
            try:
                os.remove(input_path)
            except Exception as e:
                print("Could not clean up raw input file:", e)

        # Dashboard summary should match recorded incident confidence, not unrelated
        # later frame spikes (e.g. XGBoost or zone multipliers on non-alert frames).
        summary_confidence = float(max_accident_score)
        accident_frame_url = None
        llm_analysis_text = "No accident detected."

        if saved_incidents:
            best_incident = max(saved_incidents, key=lambda inc: float(inc.get("confidence", 0)))
            summary_confidence = float(best_incident["confidence"])
            accident_details = best_incident.get("details") or accident_details
            triggering_phase_globally = best_incident.get("trigger_phase") or triggering_phase_globally
            accident_frame_url = best_incident.get("snapshot_url")
            llm_analysis_text = best_incident.get("llm_analysis") or "No analysis provided."
        elif accident_detected_globally and max_accident_frame_image is not None:
            frame_filename = f"accident_frame_{uuid.uuid4()}.jpg"
            frame_path = os.path.join(UPLOAD_DIR, frame_filename)
            cv2.imwrite(frame_path, max_accident_frame_image)
            accident_frame_url = f"/static/uploads/{frame_filename}"
            llm_analysis_text = analyze_frame_with_llm(frame_path, accident_details)

        return {
            "class": "ACCIDENT" if accident_detected_globally else "NO ACCIDENT",
            "confidence": float(summary_confidence * 100),
            "trigger_phase": triggering_phase_globally,
            "processed_video_url": f"/static/uploads/{processed_filename}",
            "accident_frame_url": accident_frame_url,
            "llm_analysis": llm_analysis_text,
            "details": accident_details,
            "incidents": saved_incidents,
            "incident_count": len(saved_incidents),
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(e)})


# ================= LOG FEATURES ENDPOINT =================
@app.post("/log-feature")
async def log_feature(data: FeatureLog):
    try:
        import csv
        init_csv_file()
        with open(CSV_FILE, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                data.proximity,
                data.trajectory,
                data.anomaly,
                data.cnn,
                data.occlusion,
                data.merge,
                data.kinetic,
                data.density,
                data.avg_speed,
                data.stopped_ratio,
                data.label
            ])
        
        # Get total logged row count
        row_count = 0
        if os.path.exists(CSV_FILE):
            with open(CSV_FILE, "r") as f:
                row_count = sum(1 for line in f) - 1 # exclude header
                
        return {"success": True, "message": "Features logged successfully", "total_rows": row_count}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


# ================= TRAIN XGBOOST MODEL =================
@app.post("/train-model")
async def train_model():
    try:
        import pandas as pd
        import os
        from xgboost import XGBClassifier
        XGBClassifier._estimator_type = "classifier"
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score
        
        if not os.path.exists(CSV_FILE):
            return JSONResponse(status_code=400, content={"error": "Dataset file accident_features.csv does not exist. Log some samples first."})
            
        df = pd.read_csv(CSV_FILE)
        if len(df) < 5:
            return JSONResponse(status_code=400, content={"error": f"Insufficient data: only {len(df)} rows. Please log at least 5 rows to train."})
            
        X = df.drop("label", axis=1)
        y = df["label"]
        
        if len(y.unique()) < 2:
            return JSONResponse(status_code=400, content={"error": "Dataset must contain both classes (0 and 1) to train the classifier."})
            
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
            stratify=y
        )
        
        model = XGBClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            eval_metric="logloss"
        )
        
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        acc = accuracy_score(y_test, pred)
        
        # Save model
        os.makedirs(os.path.dirname(XGB_MODEL_PATH), exist_ok=True)
        model.save_model(XGB_MODEL_PATH)
        
        # Reload active classifier on server
        load_xgboost_model()
        
        return {
            "success": True,
            "accuracy": float(acc),
            "train_size": len(X_train),
            "test_size": len(X_test),
            "total_rows": len(df)
        }
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(e)})


# ================= DATASET STATUS =================
@app.get("/dataset-status")
def dataset_status():
    import os
    import pandas as pd
    
    total_rows = 0
    class_0_count = 0
    class_1_count = 0
    
    if os.path.exists(CSV_FILE):
        try:
            df = pd.read_csv(CSV_FILE)
            total_rows = len(df)
            if "label" in df.columns:
                counts = df["label"].value_counts()
                class_0_count = int(counts.get(0, 0))
                class_1_count = int(counts.get(1, 0))
        except Exception as e:
            print("Error reading CSV status:", e)
            
    return {
        "total_rows": total_rows,
        "class_0": class_0_count,
        "class_1": class_1_count,
        "xgboost_active": xgb_clf is not None
    }


# ================= RUN APP =================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)