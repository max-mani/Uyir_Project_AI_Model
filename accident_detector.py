"""
UYIR — Accident Detection Logic Engine
Manikandan's core module.

Uses 5 factors to score each frame.
Accident is confirmed only when score >= 4 across 3 consecutive frames.

Factors:
  1. IOU overlap between two vehicles        (weight: 3)
  2. Sudden speed drop                       (weight: 2)
  3. Trajectory deviation                    (weight: 2)
  4. Optical flow magnitude spike            (weight: 1)
  5. Consecutive frame confirmation gate     (gate: 3 frames)
"""

import cv2
import numpy as np
import time
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from vehicle_tracker import TrackedVehicle

logging.basicConfig(level=logging.INFO,
                    format="[%(levelname)s] %(message)s")
logger = logging.getLogger("AccidentDetector")


# ── Thresholds (research-backed starting values) ───────────────────────────
# These are your starting points — you will tune these during threshold
# analysis by running data_logger.py and analyzing the CSV output.

class Thresholds:
    IOU_OVERLAP          = 0.5    # IOU between two vehicle bboxes
    SPEED_DROP_PERCENT   = 70.0   # % speed drop vs recent average
    TRAJECTORY_DEVIATION = 40.0   # pixels deviation from expected path
    OPTICAL_FLOW_SPIKE   = 2.5    # multiplier over rolling average
    MIN_SCORE_TO_CONFIRM = 4      # minimum score to consider an accident
    CONSECUTIVE_FRAMES   = 3      # frames that must agree before alert
    COOLDOWN_SECONDS     = 20.0   # seconds before same camera alerts again
    YOLO_CONFIDENCE      = 0.5    # minimum YOLO detection confidence


# ── Data classes ────────────────────────────────────────────────────────────

@dataclass
class FactorResult:
    """Result from evaluating one factor."""
    name:        str
    triggered:   bool
    value:       float
    threshold:   float
    score:       int

@dataclass
class FrameAnalysis:
    """Full analysis result for one frame."""
    frame_num:       int
    timestamp:       float
    factors:         List[FactorResult]
    total_score:     int
    accident_score:  bool           # score >= threshold
    vehicles_count:  int
    involved_ids:    List[int]      # vehicle IDs that triggered factors

@dataclass
class AccidentEvent:
    """Confirmed accident event — this gets uploaded to Firebase."""
    camera_id:         str
    timestamp:         float
    frame_num:         int
    confidence_score:  float
    factors_triggered: List[str]
    involved_vehicle_ids: List[int]
    snapshot_frame:    object       # numpy frame image


# ── Main Detector Class ─────────────────────────────────────────────────────

class AccidentDetector:
    """
    5-factor accident detection engine.

    Usage:
        detector = AccidentDetector(camera_id="CAM_001")
        event = detector.analyze(frame, tracked_vehicles, frame_num)
        if event:
            # accident confirmed — upload to Firebase
    """

    def __init__(self, camera_id: str,
                 thresholds: Thresholds = None):
        self.camera_id    = camera_id
        self.T            = thresholds or Thresholds()

        # Consecutive frame counter
        self._consec_count      = 0
        self._last_frame_scores = []

        # Cooldown tracker
        self._last_alert_time   = 0.0

        # Optical flow state
        self._prev_gray          = None
        self._flow_history       = []   # rolling magnitude history

        logger.info(f"AccidentDetector ready — camera: {camera_id}")

    # ── Public method ────────────────────────────────────────────────────────

    def analyze(self,
                frame: np.ndarray,
                vehicles: Dict[int, TrackedVehicle],
                frame_num: int) -> Optional[AccidentEvent]:
        """
        Call this on every frame with current tracked vehicles.
        Returns AccidentEvent if accident is confirmed, else None.
        """
        now = time.time()

        # Skip if in cooldown
        if now - self._last_alert_time < self.T.COOLDOWN_SECONDS:
            remaining = self.T.COOLDOWN_SECONDS - (now - self._last_alert_time)
            self._update_optical_flow(frame)
            return None

        # Run all 5 factor checks
        analysis = self._analyze_frame(frame, vehicles, frame_num)

        # Update consecutive counter
        if analysis.accident_score:
            self._consec_count += 1
        else:
            self._consec_count = 0

        # Confirm accident only after N consecutive positive frames
        if self._consec_count >= self.T.CONSECUTIVE_FRAMES:
            self._consec_count  = 0
            self._last_alert_time = now

            triggered_names = [
                f.name for f in analysis.factors if f.triggered
            ]
            confidence = min(1.0, analysis.total_score / 8.0)

            event = AccidentEvent(
                camera_id            = self.camera_id,
                timestamp            = now,
                frame_num            = frame_num,
                confidence_score     = round(confidence, 3),
                factors_triggered    = triggered_names,
                involved_vehicle_ids = analysis.involved_ids,
                snapshot_frame       = frame.copy()
            )
            logger.warning(
                f"ACCIDENT DETECTED | camera={self.camera_id} "
                f"frame={frame_num} "
                f"confidence={confidence:.2f} "
                f"factors={triggered_names}"
            )
            return event

        return None

    # ── Frame analysis ───────────────────────────────────────────────────────

    def _analyze_frame(self,
                       frame: np.ndarray,
                       vehicles: Dict[int, TrackedVehicle],
                       frame_num: int) -> FrameAnalysis:

        vehicle_list = list(vehicles.values())
        total_score  = 0
        all_factors  = []
        involved_ids = []

        # ── Factor 1: IOU Overlap ─────────────────────────────────────────
        iou_result, iou_ids = self._check_iou_overlap(vehicle_list)
        all_factors.append(iou_result)
        total_score += iou_result.score
        if iou_result.triggered:
            involved_ids.extend(iou_ids)

        # ── Factor 2: Speed Drop ──────────────────────────────────────────
        speed_result, speed_ids = self._check_speed_drop(vehicle_list)
        all_factors.append(speed_result)
        total_score += speed_result.score
        if speed_result.triggered:
            involved_ids.extend(speed_ids)

        # ── Factor 3: Trajectory Deviation ───────────────────────────────
        traj_result, traj_ids = self._check_trajectory(vehicle_list)
        all_factors.append(traj_result)
        total_score += traj_result.score
        if traj_result.triggered:
            involved_ids.extend(traj_ids)

        # ── Factor 4: Optical Flow Spike ──────────────────────────────────
        flow_result = self._check_optical_flow(frame)
        all_factors.append(flow_result)
        total_score += flow_result.score

        accident_score = total_score >= self.T.MIN_SCORE_TO_CONFIRM

        return FrameAnalysis(
            frame_num      = frame_num,
            timestamp      = time.time(),
            factors        = all_factors,
            total_score    = total_score,
            accident_score = accident_score,
            vehicles_count = len(vehicle_list),
            involved_ids   = list(set(involved_ids))
        )

    # ── Factor 1 — IOU Overlap ───────────────────────────────────────────────

    def _check_iou_overlap(self,
                            vehicles: List[TrackedVehicle]
                            ) -> Tuple[FactorResult, List[int]]:
        """
        Check if any two vehicles have overlapping bounding boxes.
        High IOU = vehicles physically in the same space = collision.
        """
        max_iou    = 0.0
        pair_ids   = []

        for i in range(len(vehicles)):
            for j in range(i + 1, len(vehicles)):
                iou = self._compute_iou(vehicles[i].bbox, vehicles[j].bbox)
                if iou > max_iou:
                    max_iou  = iou
                    pair_ids = [vehicles[i].id, vehicles[j].id]

        triggered = max_iou > self.T.IOU_OVERLAP
        return FactorResult(
            name      = "iou_overlap",
            triggered = triggered,
            value     = round(max_iou, 4),
            threshold = self.T.IOU_OVERLAP,
            score     = 3 if triggered else 0
        ), pair_ids if triggered else []

    def _compute_iou(self, bbox1: list, bbox2: list) -> float:
        """Compute Intersection over Union between two bboxes."""
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])

        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        if intersection == 0:
            return 0.0

        area1 = (bbox1[2]-bbox1[0]) * (bbox1[3]-bbox1[1])
        area2 = (bbox2[2]-bbox2[0]) * (bbox2[3]-bbox2[1])
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0.0

    # ── Factor 2 — Speed Drop ────────────────────────────────────────────────

    def _check_speed_drop(self,
                           vehicles: List[TrackedVehicle]
                           ) -> Tuple[FactorResult, List[int]]:
        """
        Check if any vehicle suddenly decelerated.
        Measures % speed drop compared to recent average.
        """
        max_drop  = 0.0
        drop_ids  = []

        for v in vehicles:
            drop = v.get_speed_drop_percent(last_n=5)
            if drop > max_drop:
                max_drop = drop
                drop_ids = [v.id]

        triggered = max_drop > self.T.SPEED_DROP_PERCENT
        return FactorResult(
            name      = "speed_drop",
            triggered = triggered,
            value     = round(max_drop, 2),
            threshold = self.T.SPEED_DROP_PERCENT,
            score     = 2 if triggered else 0
        ), drop_ids if triggered else []

    # ── Factor 3 — Trajectory Deviation ─────────────────────────────────────

    def _check_trajectory(self,
                           vehicles: List[TrackedVehicle]
                           ) -> Tuple[FactorResult, List[int]]:
        """
        Check if any vehicle deviated sharply from its expected path.
        Expected path = linear extrapolation from last two positions.
        High deviation = sudden swerve, impact deflection.
        """
        max_dev  = 0.0
        dev_ids  = []

        for v in vehicles:
            if len(v.centroid_history) < 3:
                continue

            # Expected position based on last movement vector
            prev2 = v.centroid_history[-3]
            prev1 = v.centroid_history[-2]
            curr  = v.centroid_history[-1]

            expected_x = prev1[0] + (prev1[0] - prev2[0])
            expected_y = prev1[1] + (prev1[1] - prev2[1])

            deviation = float(np.sqrt(
                (curr[0] - expected_x)**2 +
                (curr[1] - expected_y)**2
            ))

            if deviation > max_dev:
                max_dev = deviation
                dev_ids = [v.id]

        triggered = max_dev > self.T.TRAJECTORY_DEVIATION
        return FactorResult(
            name      = "trajectory_deviation",
            triggered = triggered,
            value     = round(max_dev, 2),
            threshold = self.T.TRAJECTORY_DEVIATION,
            score     = 2 if triggered else 0
        ), dev_ids if triggered else []

    # ── Factor 4 — Optical Flow Spike ────────────────────────────────────────

    def _check_optical_flow(self, frame: np.ndarray) -> FactorResult:
        """
        Compute optical flow magnitude and check if it spikes
        above 2.5x the rolling average of the last 10 frames.
        Catches accidents even when vehicles are partially occluded.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        magnitude = 0.0

        if self._prev_gray is not None:
            try:
                flow = cv2.calcOpticalFlowFarneback(
                    self._prev_gray, gray,
                    None,
                    pyr_scale=0.5,
                    levels=3,
                    winsize=15,
                    iterations=3,
                    poly_n=5,
                    poly_sigma=1.2,
                    flags=0
                )
                mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                magnitude = float(np.mean(mag))
            except Exception:
                magnitude = 0.0

        self._prev_gray = gray.copy()

        # Update rolling history
        self._flow_history.append(magnitude)
        if len(self._flow_history) > 10:
            self._flow_history.pop(0)

        rolling_avg = float(np.mean(self._flow_history)) if self._flow_history else 1.0
        if rolling_avg < 0.1:
            rolling_avg = 0.1

        ratio     = magnitude / rolling_avg
        triggered = ratio > self.T.OPTICAL_FLOW_SPIKE

        return FactorResult(
            name      = "optical_flow_spike",
            triggered = triggered,
            value     = round(ratio, 3),
            threshold = self.T.OPTICAL_FLOW_SPIKE,
            score     = 1 if triggered else 0
        )

    def _update_optical_flow(self, frame: np.ndarray):
        """Update optical flow state during cooldown (keeps history fresh)."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self._prev_gray = gray.copy()
