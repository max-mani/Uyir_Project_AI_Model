import os
import uuid
import math
import cv2
import torch
import numpy as np
from pathlib import Path
from PIL import Image

from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

# Import custom modules
from detection.yolo_module import YOLODetector
from tracking.deepsort_module import VehicleTracker
from phases.phase_a_proximity import proximity_filter
from phases.phase_b_trajectory import analyze_trajectory_conflict
from phases.phase_c_anomaly import analyze_anomaly
from utils.optical_flow import compute_optical_flow, calculate_frame_diff_ratio
from utils.geometry import calculate_bbox_containment_ratio
from fusion.scoring import fuse_scores
from model import model, transform, DEVICE, predict_image

app = FastAPI()

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


# ================= HOME PAGE =================
@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


# ================= IMAGE PREDICTION =================
@app.post("/predict-image")
async def predict_image_api(file: UploadFile = File(...)):
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
        candidate_pairs = proximity_filter(tracks, threshold=120.0)
        
        for t1, t2, dist in candidate_pairs:
            prox_s = max(0.0, 1.0 - (dist / 120.0))
            proximity_score = max(proximity_score, prox_s)
            
            # Static containment check for occlusion
            area1 = (t1.bbox[2] - t1.bbox[0]) * (t1.bbox[3] - t1.bbox[1])
            area2 = (t2.bbox[2] - t2.bbox[0]) * (t2.bbox[3] - t2.bbox[1])
            if area1 < area2:
                containment = calculate_bbox_containment_ratio(t1.bbox, t2.bbox)
            else:
                containment = calculate_bbox_containment_ratio(t2.bbox, t1.bbox)
            occlusion_score = max(occlusion_score, containment)

        # 4. Score Fusion for Static Image
        # Blend Proximity, Occlusion, and CNN-LSTM
        final_score = 0.2 * proximity_score + 0.2 * occlusion_score + 0.6 * cnn_lstm_prob
        is_accident = final_score >= 0.50  
        final_class = "ACCIDENT" if is_accident else "NO ACCIDENT"

        # 5. Annotate Image
        for track in tracks:
            x1, y1, x2, y2 = map(int, track.bbox)
            label = f"{track.label.upper()} {track.confidence:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        for t1, t2, dist in candidate_pairs:
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
            "trigger_phase": "Phase A (Proximity) & Occlusion-Containment & CNN-LSTM DL Module" if is_accident else "None",
            "processed_image_url": f"/static/uploads/{processed_filename}",
            "details": {
                "proximity_score": float(proximity_score),
                "occlusion_score": float(occlusion_score),
                "trajectory_score": 0.0,
                "anomaly_score": 0.0,
                "cnn_lstm_prob": float(cnn_lstm_prob)
            }
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(e)})


# ================= VIDEO PREDICTION =================
@app.post("/predict-video")
async def predict_video_api(file: UploadFile = File(...)):
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

        # Create Video Writer (H.264 MSMF on Windows)
        processed_filename = f"processed_{uuid.uuid4()}.mp4"
        output_path = os.path.join(UPLOAD_DIR, processed_filename)
        fourcc = cv2.VideoWriter_fourcc(*'avc1')
        out_writer = cv2.VideoWriter(output_path, cv2.CAP_MSMF, fourcc, fps, (width, height))

        # Instantiate modules
        detector = YOLODetector()
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

        # High-risk intersection zone dimensions (center 50% area)
        zone_x_min, zone_x_max = int(width * 0.25), int(width * 0.75)
        zone_y_min, zone_y_max = int(height * 0.25), int(height * 0.75)

        # Main frame loop
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # --- CNN-LSTM Feature Extraction ---
            pil_img = Image.fromarray(frame_rgb)
            frame_feat = transform(pil_img).unsqueeze(0).to(DEVICE)
            
            with torch.no_grad():
                feat = model.cnn(frame_feat)

            features_buffer.append(feat)
            if len(features_buffer) > 16:
                features_buffer.pop(0)

            # Pad features if buffer is not full
            if len(features_buffer) < 16:
                padded_buffer = [features_buffer[0]] * (16 - len(features_buffer)) + features_buffer
            else:
                padded_buffer = features_buffer

            features_tensor = torch.cat(padded_buffer, dim=0).unsqueeze(0)
            with torch.no_grad():
                lstm_out, _ = model.lstm(features_tensor)
                out_state = lstm_out[:, -1, :]
                logits = model.classifier(out_state)
                probs = torch.softmax(logits, dim=1)
                cnn_lstm_prob = float(probs[0, 1].item())

            # LSTM Peak Memory (rolling 30-frame maximum)
            lstm_scores_history.append(cnn_lstm_prob)
            if len(lstm_scores_history) > 30:
                lstm_scores_history.pop(0)
            lstm_peak = max(lstm_scores_history)

            # --- YOLO Detection & Tracking ---
            detections = detector.detect(frame)
            active_tracks = tracker.update(detections)

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
                baseline_speed = sum(scene_speed_history) / len(scene_speed_history)
                if baseline_speed > 1.0:
                    speed_drop = (baseline_speed - mean_current_speed) / baseline_speed
                    if speed_drop > 0.60:
                        scene_interruption_score = 1.0
                    else:
                        scene_interruption_score = max(0.0, float(speed_drop / 0.60))

            # --- Phase A: Proximity Filtering ---
            candidate_pairs = proximity_filter(active_tracks, threshold=120.0)

            # --- Score Accumulators ---
            proximity_score = 0.0
            trajectory_score = 0.0
            anomaly_score = 0.0
            occlusion_score = 0.0
            merge_score = 0.0
            energy_drop_score = 0.0
            spin_score = 0.0
            diff_burst_score = 0.0
            flow_dispersion_score = 0.0

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
            for t1, t2, dist in candidate_pairs:
                prox_s = max(0.0, 1.0 - (dist / 120.0))
                proximity_score = max(proximity_score, prox_s)

                traj_res = analyze_trajectory_conflict(t1, t2)
                trajectory_score = max(trajectory_score, traj_res["score"])
                
                # Advanced signals
                if traj_res["occluded"]:
                    occlusion_score = max(occlusion_score, traj_res["containment"])
                else:
                    occlusion_score = max(occlusion_score, traj_res["containment"] * 0.5)
                
                if traj_res["merged"]:
                    merge_score = 1.0
                
                energy_drop_score = max(energy_drop_score, traj_res["max_ke_drop"])
                spin_score = max(spin_score, traj_res["max_spin_var"])

                # Determine if collision center point lies inside high-risk zone
                c1, c2 = t1.get_centroid(), t2.get_centroid()
                cx, cy = int((c1[0] + c2[0]) / 2), int((c1[1] + c2[1]) / 2)
                if zone_x_min <= cx <= zone_x_max and zone_y_min <= cy <= zone_y_max:
                    is_in_intersection_zone = True

                if traj_res["class"] == "Collision" or traj_res["merged"] or traj_res["occluded"]:
                    frame_collision_pairs.append((t1, t2, dist, traj_res["class"]))

            # Store gray frame reference
            prev_gray = frame_gray

            # Fused Scores for this frame (threshold=0.55 for early warnings)
            fuse_res = fuse_scores(
                proximity=proximity_score,
                trajectory=trajectory_score,
                flow=anomaly_score,
                lstm_peak=lstm_peak,
                occlusion=occlusion_score,
                merge=merge_score,
                energy_drop=energy_drop_score,
                spin=spin_score,
                scene_interruption=scene_interruption_score,
                diff_burst=diff_burst_score,
                flow_dispersion=flow_dispersion_score,
                threshold=0.55
            )

            frame_accident = fuse_res["is_accident"]
            frame_score = fuse_res["score"]
            frame_trigger = fuse_res["trigger_phase"]

            # Zone Risk weighting multiplier (1.2x score bump in high-risk zones)
            if is_in_intersection_zone:
                multiplied_score = min(1.0, frame_score * 1.2)
                if multiplied_score >= 0.55:
                    frame_accident = True
                    frame_score = multiplied_score
                    frame_trigger += " & Risk Zone"

            # Log max statistics
            if frame_score > max_accident_score:
                max_accident_score = frame_score
                triggering_phase_globally = frame_trigger
                accident_details = fuse_res["details"]

            if frame_accident:
                accident_detected_globally = True

            # --- Annotations ---
            # High-Risk Intersection Zone Boundary
            cv2.rectangle(frame, (zone_x_min, zone_y_min), (zone_x_max, zone_y_max), (255, 255, 255), 1, lineType=cv2.LINE_AA)
            cv2.putText(frame, "INTERSECTION RISK ZONE", (zone_x_min + 5, zone_y_min - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

            # Draw tracks
            for track in active_tracks:
                x1, y1, x2, y2 = map(int, track.bbox)
                label = f"ID {track.track_id}: {track.label.upper()} {track.confidence:.2f}"
                color = (0, 255, 0)

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

            # Draw collision points & lines
            for t1, t2, dist, status in frame_collision_pairs:
                c1, c2 = t1.get_centroid(), t2.get_centroid()
                pt1 = (int(c1[0]), int(c1[1]))
                pt2 = (int(c2[0]), int(c2[1]))

                cv2.line(frame, pt1, pt2, (0, 255, 255), 2)
                center_x, center_y = int((c1[0] + c2[0]) / 2), int((c1[1] + c2[1]) / 2)
                cv2.circle(frame, (center_x, center_y), 8, (0, 0, 255), -1)
                cv2.circle(frame, (center_x, center_y), 15, (0, 0, 255), 2)
                cv2.putText(frame, f"IMPACT ({status})", (center_x - 50, center_y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            # Flashing banner alert
            if frame_accident:
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

            cv2.putText(frame, "ITS SYSTEM MONITOR v2", (panel_x + 10, panel_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            cv2.putText(frame, f"Frame: {frame_idx}/{total_frames} | Active: {len(active_tracks)}", (panel_x + 10, panel_y + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
            
            # Draw scores list
            cv2.putText(frame, f"Phase A Proximity: {proximity_score:.2f}", (panel_x + 10, panel_y + 65), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255) if proximity_score > 0 else (200, 200, 200), 1)
            cv2.putText(frame, f"Phase B Trajectory: {trajectory_score:.2f}", (panel_x + 10, panel_y + 85), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 165, 255) if trajectory_score > 0 else (200, 200, 200), 1)
            cv2.putText(frame, f"Phase C Anomaly: {anomaly_score:.2f}", (panel_x + 10, panel_y + 105), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255) if anomaly_score > 0 else (200, 200, 200), 1)
            cv2.putText(frame, f"CNN-LSTM DL Peak: {lstm_peak:.2f}", (panel_x + 10, panel_y + 125), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255) if lstm_peak > 0.5 else (200, 200, 200), 1)
            
            # New Advanced Indicators
            cv2.putText(frame, f"Occlusion Contain: {occlusion_score:.2f}", (panel_x + 10, panel_y + 150), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255) if occlusion_score > 0.6 else (200, 200, 200), 1)
            cv2.putText(frame, f"Kinetic Energy Drop: {energy_drop_score:.2f}", (panel_x + 10, panel_y + 170), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 165, 255) if energy_drop_score > 0.8 else (200, 200, 200), 1)
            cv2.putText(frame, f"Spin (Var): {spin_score:.2f} | Merge: {int(merge_score)}", (panel_x + 10, panel_y + 190), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255) if spin_score > 0.15 else (200, 200, 200), 1)
            cv2.putText(frame, f"Scene Traffic Interruption: {scene_interruption_score:.2f}", (panel_x + 10, panel_y + 210), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0) if scene_interruption_score > 0.5 else (200, 200, 200), 1)
            cv2.putText(frame, f"Shock Burst: {diff_burst_score:.2f} | Scatter: {flow_dispersion_score:.2f}", (panel_x + 10, panel_y + 230), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 255, 0), 1)

            out_writer.write(frame)
            frame_idx += 1

        cap.release()
        out_writer.release()

        # Clean up raw video
        if os.path.exists(input_path):
            try:
                os.remove(input_path)
            except Exception as e:
                print("Could not clean up raw input file:", e)

        return {
            "class": "ACCIDENT" if accident_detected_globally else "NO ACCIDENT",
            "confidence": float(max_accident_score * 100),
            "trigger_phase": triggering_phase_globally,
            "processed_video_url": f"/static/uploads/{processed_filename}",
            "details": accident_details
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(e)})


# ================= RUN APP =================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)