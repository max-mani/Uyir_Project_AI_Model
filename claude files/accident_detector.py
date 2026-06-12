# ============================================================
# UYIR — Accident Detector
# Manikandan — Task M5
#
# Stage 1: Runs the accident detection model on each frame.
#           If it flags an accident with confidence >= gate,
#           passes to Stage 2.
#
# Stage 2: Three-phase verification using the vehicle cache.
#   Phase A — Proximity (Euclidean distance)
#   Phase B — Trajectory conflict (intersection + stop)
#   Phase C — Anomaly confirmation (optical flow + bbox deform)
#   Gate    — 3 consecutive frames must agree
#
# Research sources:
#   Phase A: NJIT 2022
#   Phase B: IITH 2018 + NJIT 2022
#   Phase C: HFG 2010 + Fuzzy 2023
# ============================================================

import cv2
import time
import logging
import numpy as np
from dataclasses import dataclass, field
from typing import Optional
from itertools import combinations
from ultralytics import YOLO
import config

logger = logging.getLogger("AccidentDetector")


# ── Data classes ─────────────────────────────────────────────

@dataclass
class FactorResult:
    name      : str
    triggered : bool
    value     : float
    threshold : float
    score     : int

@dataclass
class AccidentEvent:
    """
    A confirmed accident event.
    This gets sent to firebase_uploader.py for upload.
    """
    camera_id           : str
    location            : str
    timestamp           : float
    frame_num           : int
    confidence_score    : float
    stage1_confidence   : float
    phases_triggered    : list
    involved_vehicle_ids: list
    snapshot_frame      : object   # numpy array — the frame at detection time


# ── Main Detector ─────────────────────────────────────────────

class AccidentDetector:
    """
    Full two-stage accident detection engine.

    Usage:
        detector = AccidentDetector()
        event = detector.analyze(frame, vehicles, frame_num)
        if event:
            # confirmed accident — upload to Firebase
    """

    def __init__(self,
                 camera_id : str = config.CAMERA_ID,
                 location  : str = config.CAMERA_LOCATION):

        self.camera_id = camera_id
        self.location  = location

        # Load accident detection model (Stage 1)
        logger.info(f"[Detector] Loading accident model: {config.ACCIDENT_MODEL_PATH}")
        self.accident_model = YOLO(config.ACCIDENT_MODEL_PATH)
        logger.info("[Detector] Accident model loaded.")

        # Consecutive frame counter and cooldown
        self._consec_count    = 0
        self._last_alert_time = 0.0

        # Optical flow state
        self._prev_gray    = None
        self._flow_history = []   # rolling magnitude history

    # ── Public method ─────────────────────────────────────────

    def analyze(self,
                frame   : np.ndarray,
                vehicles: dict,
                frame_num: int) -> Optional[AccidentEvent]:
        """
        Call this every frame with the current tracked vehicles.
        Returns AccidentEvent if accident is fully confirmed, else None.
        """
        now = time.time()

        # ── Cooldown check ────────────────────────────────────
        if now - self._last_alert_time < config.COOLDOWN_SECONDS:
            self._update_flow(frame)
            return None

        # ── Stage 1: Run accident detection model ─────────────
        stage1_conf, stage1_ids = self._run_stage1(frame, vehicles)

        if stage1_conf < config.STAGE1_GATE_CONFIDENCE:
            # Model not confident enough — reset counter, skip
            self._consec_count = 0
            self._update_flow(frame)
            return None

        # ── Stage 2: Three-phase verification ─────────────────
        total_score, phases, involved_ids = self._run_stage2(
            frame, vehicles, stage1_ids
        )

        if total_score >= config.MIN_SCORE_TO_PASS:
            self._consec_count += 1
        else:
            self._consec_count = 0

        # ── Consecutive frame gate ────────────────────────────
        if self._consec_count >= config.CONSECUTIVE_FRAMES:
            self._consec_count    = 0
            self._last_alert_time = now

            confidence = min(1.0, round(
                (stage1_conf * 0.5) + (total_score / 8.0 * 0.5), 3
            ))

            event = AccidentEvent(
                camera_id            = self.camera_id,
                location             = self.location,
                timestamp            = now,
                frame_num            = frame_num,
                confidence_score     = confidence,
                stage1_confidence    = round(stage1_conf, 3),
                phases_triggered     = phases,
                involved_vehicle_ids = list(set(involved_ids)),
                snapshot_frame       = frame.copy(),
            )

            logger.warning(
                f"ACCIDENT CONFIRMED | cam={self.camera_id} "
                f"frame={frame_num} conf={confidence:.2f} "
                f"phases={phases} vehicles={involved_ids}"
            )
            return event

        return None

    # ── Stage 1 — Accident Model ─────────────────────────────

    def _run_stage1(self, frame: np.ndarray,
                    vehicles: dict) -> tuple:
        """
        Run the accident detection model on the frame.
        Returns (max_confidence, list_of_nearby_vehicle_ids).
        """
        results = self.accident_model.predict(
            frame,
            conf    = config.ACCIDENT_CONF_THRESHOLD,
            verbose = False,
        )

        if not results or results[0].boxes is None:
            return 0.0, []

        boxes  = results[0].boxes
        if len(boxes) == 0:
            return 0.0, []

        max_conf     = float(boxes.conf.max().cpu().numpy())
        accident_ids = []

        # Find which tracked vehicles are inside/near the flagged region
        for box in boxes.xyxy.cpu().numpy():
            ax1, ay1, ax2, ay2 = box
            for vid, v in vehicles.items():
                vx1, vy1, vx2, vy2 = v.bbox
                # Check if vehicle centroid is inside accident box
                cx, cy = v.centroid
                if ax1 <= cx <= ax2 and ay1 <= cy <= ay2:
                    accident_ids.append(vid)

        return max_conf, accident_ids

    # ── Stage 2 — Three-Phase Verification ───────────────────

    def _run_stage2(self, frame: np.ndarray,
                    vehicles: dict,
                    stage1_ids: list) -> tuple:
        """
        Run three-phase verification on the flagged vehicles.
        Returns (total_score, triggered_phase_names, involved_ids).
        """
        vlist = list(vehicles.values())
        if len(vlist) < 2:
            self._update_flow(frame)
            return 0, [], []

        total_score   = 0
        phases        = []
        involved_ids  = []

        # ── Phase A ───────────────────────────────────────────
        close_pairs = self._phase_a(vlist)
        if close_pairs:
            total_score += config.SCORE_PHASE_A
            phases.append("phase_a")
            for a, b in close_pairs:
                involved_ids.extend([a.id, b.id])
        else:
            self._update_flow(frame)
            return 0, [], []   # Phase A failed — stop here

        # ── Phase B ───────────────────────────────────────────
        b_score, b_ids = self._phase_b(close_pairs)
        if b_score:
            total_score += config.SCORE_PHASE_B
            phases.append("phase_b")
            involved_ids.extend(b_ids)

        # ── Phase C ───────────────────────────────────────────
        c_score = self._phase_c(frame, close_pairs)
        if c_score:
            total_score += config.SCORE_PHASE_C
            phases.append("phase_c")

        return total_score, phases, involved_ids

    # ── Phase A: Proximity ────────────────────────────────────

    def _phase_a(self, vehicles: list) -> list:
        """
        Returns list of (vehicle_A, vehicle_B) pairs closer than threshold.
        NJIT 2022.
        """
        close_pairs = []
        for va, vb in combinations(vehicles, 2):
            cx1, cy1 = va.centroid
            cx2, cy2 = vb.centroid
            dist = float(np.sqrt((cx1-cx2)**2 + (cy1-cy2)**2))
            if dist < config.PROXIMITY_THRESHOLD:
                close_pairs.append((va, vb))
        return close_pairs

    # ── Phase B: Trajectory Conflict ─────────────────────────

    def _phase_b(self, pairs: list) -> tuple:
        """
        Check velocity/angle conflict AND trajectory intersection + stop.
        IITH 2018 + NJIT 2022.
        Returns (triggered: bool, involved_ids: list).
        """
        for va, vb in pairs:
            # Check 1: velocity sum + angle divergence
            vel_sum    = va.get_avg_speed(5) + vb.get_avg_speed(5)
            angle_diff = abs(va.direction - vb.direction)
            vel_conflict = (
                vel_sum < config.VELOCITY_SUM_STOP and
                angle_diff > config.ANGLE_DIVERGENCE_DEG
            )

            # Check 2: trajectory intersection + stop (IITH paper)
            # If trajectories crossed AND one vehicle stopped = collision
            intersects = self._trajectories_intersect(
                va.centroid_history, vb.centroid_history
            )
            a_stopped = va.get_speed_drop_percent() > config.SPEED_DROP_PERCENT
            b_stopped = vb.get_speed_drop_percent() > config.SPEED_DROP_PERCENT
            traj_stop = intersects and (a_stopped or b_stopped)

            if vel_conflict or traj_stop:
                return True, [va.id, vb.id]

        return False, []

    def _trajectories_intersect(self,
                                 hist_a: list,
                                 hist_b: list) -> bool:
        """
        Check if the centroid paths of two vehicles crossed
        in the last N frames using line segment intersection.
        """
        if len(hist_a) < 2 or len(hist_b) < 2:
            return False

        # Use the last 10 frames for intersection check
        path_a = hist_a[-10:]
        path_b = hist_b[-10:]

        for i in range(len(path_a) - 1):
            for j in range(len(path_b) - 1):
                if self._segments_intersect(
                    path_a[i], path_a[i+1],
                    path_b[j], path_b[j+1]
                ):
                    return True
        return False

    @staticmethod
    def _segments_intersect(p1, p2, p3, p4) -> bool:
        """Returns True if line segment p1-p2 intersects p3-p4."""
        def cross(o, a, b):
            return (a[0]-o[0]) * (b[1]-o[1]) - (a[1]-o[1]) * (b[0]-o[0])

        d1 = cross(p3, p4, p1)
        d2 = cross(p3, p4, p2)
        d3 = cross(p1, p2, p3)
        d4 = cross(p1, p2, p4)

        if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and \
           ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
            return True
        return False

    # ── Phase C: Anomaly Confirmation ────────────────────────

    def _phase_c(self, frame: np.ndarray, pairs: list) -> bool:
        """
        Optical flow magnitude spike AND bbox deformation.
        HFG 2010 + Fuzzy 2023.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Signal 1 — Optical flow spike
        flow_flag = False
        if self._prev_gray is not None:
            try:
                flow = cv2.calcOpticalFlowFarneback(
                    self._prev_gray, gray, None,
                    pyr_scale=0.5, levels=3, winsize=15,
                    iterations=3, poly_n=5, poly_sigma=1.2, flags=0
                )
                mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                magnitude = float(np.mean(mag))
            except Exception:
                magnitude = 0.0

            self._flow_history.append(magnitude)
            if len(self._flow_history) > config.FLOW_HISTORY_FRAMES:
                self._flow_history.pop(0)

            avg = max(0.1, float(np.mean(self._flow_history)))
            flow_flag = (magnitude / avg) > config.OPTICAL_FLOW_SPIKE

        self._prev_gray = gray.copy()

        # Signal 2 — Bounding box deformation
        deform_flag = False
        for va, vb in pairs:
            if va.get_bbox_area_change() > config.BBOX_DEFORM_RATIO:
                deform_flag = True
                break
            if vb.get_bbox_area_change() > config.BBOX_DEFORM_RATIO:
                deform_flag = True
                break

        return flow_flag or deform_flag

    def _update_flow(self, frame: np.ndarray):
        """Keep optical flow state fresh even when skipping analysis."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self._prev_gray is not None:
            try:
                flow = cv2.calcOpticalFlowFarneback(
                    self._prev_gray, gray, None,
                    0.5, 3, 15, 3, 5, 1.2, 0
                )
                mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                self._flow_history.append(float(np.mean(mag)))
                if len(self._flow_history) > config.FLOW_HISTORY_FRAMES:
                    self._flow_history.pop(0)
            except Exception:
                pass
        self._prev_gray = gray.copy()
