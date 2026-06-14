"""
UYIR Accident Detector — uses shared phase modules and fusion engine.
Stage 1 YOLO accident model is optional (skipped if model file missing).
"""

import logging
import os
import time
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np
import torch
from PIL import Image

import config
from fusion.scoring import fuse_scores
from model import DEVICE, SEQUENCE_LEN, model, transform
from phases.phase_a_proximity import proximity_filter
from phases.phase_b_trajectory import analyze_trajectory_conflict
from phases.phase_c_anomaly import analyze_anomaly
from tracking.deepsort_module import Track
from utils.optical_flow import compute_optical_flow

logger = logging.getLogger("AccidentDetector")


@dataclass
class AccidentEvent:
    camera_id: str
    location: str
    timestamp: float
    frame_num: int
    confidence_score: float
    stage1_confidence: float
    phases_triggered: list
    involved_vehicle_ids: list
    snapshot_frame: object
    trigger_phase: str = "Weighted Fusion"
    fusion_details: dict = field(default_factory=dict)
    cnn_lstm_confidence: float = 0.0
    clip_path: Optional[str] = None


def _tracked_vehicle_to_track(v) -> Track:
    """Convert TrackedVehicle to Track for shared phase modules."""
    track = Track(v.id, v.bbox, v.class_name, 1.0)
    track.history = list(v.centroid_history)
    track.bbox_history = list(v.bbox_history)
    track.speed_history = list(v.speed_history)
    track.velocities = []
    for i in range(1, len(v.centroid_history)):
        p0 = v.centroid_history[i - 1]
        p1 = v.centroid_history[i]
        track.velocities.append((p1[0] - p0[0], p1[1] - p0[1]))
    if not track.velocities:
        track.velocities = [(0.0, 0.0)]
    track.age = len(v.centroid_history)
    return track


class AccidentDetector:
    def __init__(self, camera_id=None, location=None):
        self.camera_id = camera_id or config.CAMERA_ID
        self.location = location or config.CAMERA_LOCATION
        self.accident_model = None
        self._consec_count = 0
        self._last_alert_time = 0.0
        self._prev_gray = None
        self._features_buffer = []

        if os.path.exists(config.ACCIDENT_MODEL_PATH):
            try:
                from ultralytics import YOLO
                self.accident_model = YOLO(config.ACCIDENT_MODEL_PATH)
                logger.info("[Detector] Stage-1 accident model loaded.")
            except Exception as e:
                logger.warning(f"[Detector] Stage-1 model failed to load: {e}")
        else:
            logger.info("[Detector] No Stage-1 model — running 3-phase verification only.")

    def _cnn_lstm_prob(self, frame: np.ndarray) -> float:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(frame_rgb)
        frame_feat = transform(pil_img).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            feat = model.cnn(frame_feat)

        self._features_buffer.append(feat)
        if len(self._features_buffer) > SEQUENCE_LEN:
            self._features_buffer.pop(0)

        if len(self._features_buffer) < SEQUENCE_LEN:
            padded_buffer = [self._features_buffer[0]] * (SEQUENCE_LEN - len(self._features_buffer)) + self._features_buffer
        else:
            padded_buffer = self._features_buffer

        features_tensor = torch.stack(padded_buffer, dim=1)
        with torch.no_grad():
            lstm_out, _ = model.bilstm(features_tensor)
            context = model.attention(lstm_out)
            logits = model.classifier(context)
            probs = torch.softmax(logits, dim=1)
            return float(probs[0, 1].item())

    def analyze(self, frame: np.ndarray, vehicles: dict, frame_num: int) -> Optional[AccidentEvent]:
        now = time.time()
        cnn_lstm_prob = self._cnn_lstm_prob(frame)

        if now - self._last_alert_time < config.COOLDOWN_SECONDS:
            self._update_flow(frame)
            return None

        stage1_conf = 0.0
        if self.accident_model is not None:
            stage1_conf, _ = self._run_stage1(frame)
            if stage1_conf < config.STAGE1_GATE_CONFIDENCE:
                self._consec_count = 0
                self._update_flow(frame)
                return None

        tracks = [_tracked_vehicle_to_track(v) for v in vehicles.values()]
        if len(tracks) < 2:
            self._update_flow(frame)
            return None

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        flow = compute_optical_flow(self._prev_gray, gray) if self._prev_gray is not None else None
        self._prev_gray = gray.copy()

        candidate_pairs = proximity_filter(tracks)
        if not candidate_pairs:
            self._consec_count = 0
            return None

        ttc_score = 0.0
        trajectory_stop_score = 0.0
        emergency_stop_score = 0.0
        optical_flow = 0.0
        flow_dispersion = 0.0
        occlusion_score = 0.0
        merge_score = 0.0
        spin_score = 0.0
        phases = ["phase_a"]
        involved_ids = []

        for t1, t2, dist, pair_ttc in candidate_pairs:
            ttc_score = max(ttc_score, pair_ttc)
            involved_ids.extend([t1.track_id, t2.track_id])
            traj = analyze_trajectory_conflict(t1, t2)
            trajectory_stop_score = max(trajectory_stop_score, traj["trajectory_stop_score"])
            emergency_stop_score = max(emergency_stop_score, traj["emergency_stop_score"])
            spin_score = max(spin_score, traj.get("max_spin_var", 0.0))
            if traj.get("occluded"):
                occlusion_score = max(occlusion_score, traj.get("containment", 0.0))
            else:
                occlusion_score = max(occlusion_score, traj.get("containment", 0.0) * 0.5)
            if traj.get("merged"):
                merge_score = 1.0
            if traj["class"] == "Collision":
                phases.append("phase_b")

        stopped = sum(
            1 for track in tracks
            if track.velocities and (track.velocities[-1][0] ** 2 + track.velocities[-1][1] ** 2) ** 0.5 < 2.0
        )
        stopped_ratio = stopped / len(tracks) if tracks else 0.0
        speeds = []
        for track in tracks:
            if track.velocities:
                vx, vy = track.velocities[-1]
                speeds.append((vx ** 2 + vy ** 2) ** 0.5)
        avg_speed = sum(speeds) / len(speeds) if speeds else 0.0

        for track in tracks:
            anom = analyze_anomaly(track, flow)
            optical_flow = max(optical_flow, anom["anomaly_score"])
            flow_dispersion = max(flow_dispersion, anom["dispersion_val"])
            if anom["anomaly_confirmed"]:
                phases.append("phase_c")
                break

        fuse_res = fuse_scores(
            trajectory_stop=trajectory_stop_score,
            ttc_critical=ttc_score,
            emergency_stop=emergency_stop_score,
            cnn_lstm=cnn_lstm_prob,
            optical_flow=optical_flow,
            flow_dispersion=flow_dispersion,
            scene_density=len(tracks),
            avg_scene_speed=avg_speed,
            stopped_ratio=stopped_ratio,
        )

        fusion_details = dict(fuse_res["details"])
        fusion_details.update({
            "lstm_peak": cnn_lstm_prob,
            "occlusion_score": float(occlusion_score),
            "merge_score": float(merge_score),
            "spin_score": float(spin_score),
            "scene_interruption": 0.0,
        })

        if fuse_res["score"] >= config.FUSION_THRESHOLD:
            self._consec_count += 1
        else:
            self._consec_count = 0

        if self._consec_count >= config.CONSECUTIVE_FRAMES:
            self._consec_count = 0
            self._last_alert_time = now
            confidence = min(1.0, fuse_res["score"])
            return AccidentEvent(
                camera_id=self.camera_id,
                location=self.location,
                timestamp=now,
                frame_num=frame_num,
                confidence_score=confidence,
                stage1_confidence=stage1_conf,
                phases_triggered=list(set(phases)),
                involved_vehicle_ids=list(set(involved_ids)),
                snapshot_frame=frame.copy(),
                trigger_phase=fuse_res["trigger_phase"],
                fusion_details=fusion_details,
                cnn_lstm_confidence=cnn_lstm_prob,
            )

        return None

    def _run_stage1(self, frame):
        results = self.accident_model.predict(frame, conf=config.ACCIDENT_CONF_THRESHOLD, verbose=False)
        if not results or results[0].boxes is None or len(results[0].boxes) == 0:
            return 0.0, []
        return float(results[0].boxes.conf.max().cpu().numpy()), []

    def _update_flow(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self._prev_gray = gray.copy()
