# ============================================================
# UYIR — Vehicle Tracker + Cache
# Manikandan — Task M2
#
# Uses the vehicle detection model + ByteTrack to assign
# persistent IDs to every vehicle and store their history.
# This cache is what the three-phase logic reads from.
# ============================================================

import cv2
import numpy as np
import logging
from collections import defaultdict
from ultralytics import YOLO
import config

logger = logging.getLogger("VehicleTracker")

# COCO class IDs for pre-trained model fallback
# When using Kishore/Lalu's custom model, class names come from the model itself
COCO_VEHICLE_CLASSES = {
    0:  "person",
    2:  "car",
    3:  "motorcycle",
    5:  "bus",
    7:  "truck",
}


class TrackedVehicle:
    """
    Holds all data for one tracked vehicle across frames.
    The three-phase logic reads directly from this object.
    """

    def __init__(self, track_id: int, class_name: str,
                 bbox: list, frame_num: int):
        self.id           = track_id
        self.class_name   = class_name
        self.bbox         = bbox            # [x1, y1, x2, y2]
        self.frame_num    = frame_num       # last seen frame number

        # Computed every frame
        self.centroid     = self._centroid(bbox)
        self.speed        = 0.0             # pixels per frame
        self.direction    = 0.0             # heading angle in degrees

        # Rolling history — last N frames
        self.centroid_history : list = [self.centroid]
        self.bbox_history     : list = [bbox]
        self.speed_history    : list = [0.0]

        # Predicted next position (linear extrapolation)
        self.expected_next = self.centroid

    # ── Helpers ───────────────────────────────────────────────

    @staticmethod
    def _centroid(bbox: list) -> tuple:
        x1, y1, x2, y2 = bbox
        return (int((x1 + x2) / 2), int((y1 + y2) / 2))

    def bbox_area(self) -> float:
        x1, y1, x2, y2 = self.bbox
        return max(0.0, float((x2 - x1) * (y2 - y1)))

    # ── Update every frame ───────────────────────────────────

    def update(self, bbox: list, frame_num: int):
        prev = self.centroid
        self.bbox      = bbox
        self.centroid  = self._centroid(bbox)
        self.frame_num = frame_num

        dx = self.centroid[0] - prev[0]
        dy = self.centroid[1] - prev[1]
        self.speed     = float(np.sqrt(dx * dx + dy * dy))
        self.direction = float(np.degrees(np.arctan2(dy, dx)))

        # Predict next centroid by linear extrapolation
        self.expected_next = (
            self.centroid[0] + dx,
            self.centroid[1] + dy,
        )

        # Append to histories, cap at max length
        N = config.TRACK_HISTORY_FRAMES
        self.centroid_history.append(self.centroid)
        self.bbox_history.append(bbox)
        self.speed_history.append(self.speed)
        if len(self.centroid_history) > N:
            self.centroid_history.pop(0)
            self.bbox_history.pop(0)
            self.speed_history.pop(0)

    # ── Derived metrics ──────────────────────────────────────

    def get_avg_speed(self, last_n: int = 5) -> float:
        h = self.speed_history[-last_n:]
        return float(np.mean(h)) if h else 0.0

    def get_speed_drop_percent(self, last_n: int = 5) -> float:
        """
        How much has speed dropped compared to the N frames before now?
        Returns 0-100. High value = sudden stop.
        """
        if len(self.speed_history) < last_n + 3:
            return 0.0
        avg_before = float(np.mean(self.speed_history[-(last_n + 3):-3]))
        current    = self.speed
        if avg_before < 1.0:
            return 0.0
        drop = ((avg_before - current) / avg_before) * 100.0
        return max(0.0, drop)

    def get_trajectory_deviation(self) -> float:
        """
        How far did the vehicle deviate from its expected linear path?
        High value = sudden swerve or impact deflection.
        """
        if len(self.centroid_history) < 3:
            return 0.0
        p2 = self.centroid_history[-3]
        p1 = self.centroid_history[-2]
        p0 = self.centroid_history[-1]
        ex = p1[0] + (p1[0] - p2[0])
        ey = p1[1] + (p1[1] - p2[1])
        return float(np.sqrt((p0[0] - ex) ** 2 + (p0[1] - ey) ** 2))

    def get_bbox_area_change(self) -> float:
        """
        Fractional change in bounding box area vs 5 frames ago.
        High value = sudden deformation (crash impact).
        """
        if len(self.bbox_history) < 5:
            return 0.0
        old_bbox  = self.bbox_history[-5]
        old_area  = max(1.0,
            float((old_bbox[2]-old_bbox[0]) * (old_bbox[3]-old_bbox[1])))
        curr_area = self.bbox_area()
        return abs(curr_area - old_area) / old_area


class VehicleTracker:
    """
    Main tracker class.

    Runs the vehicle detection model + ByteTrack on every frame.
    Maintains a per-vehicle cache (TrackedVehicle objects).
    Call process_frame(frame) every frame.
    """

    def __init__(self,
                 model_path: str = config.VEHICLE_MODEL_PATH,
                 confidence: float = config.VEHICLE_CONF_THRESHOLD):

        logger.info(f"[Tracker] Loading vehicle model: {model_path}")
        self.model      = YOLO(model_path)
        self.confidence = confidence
        self.vehicles   : dict[int, TrackedVehicle] = {}
        self.frame_count = 0
        logger.info("[Tracker] Vehicle model loaded.")

    def process_frame(self, frame: np.ndarray) -> dict:
        """
        Detect + track vehicles in one frame.
        Returns dict { track_id: TrackedVehicle } for active vehicles.
        """
        self.frame_count += 1
        active_ids = set()

        results = self.model.track(
            frame,
            persist    = True,
            conf       = self.confidence,
            tracker    = "bytetrack.yaml",
            verbose    = False,
        )

        if results[0].boxes.id is not None:
            boxes   = results[0].boxes.xyxy.cpu().numpy()
            ids     = results[0].boxes.id.cpu().numpy().astype(int)
            classes = results[0].boxes.cls.cpu().numpy().astype(int)
            names   = results[0].names   # class index → name from model

            for bbox, tid, cls_id in zip(boxes, ids, classes):
                cls_name  = names.get(cls_id, "vehicle")
                bbox_list = bbox.tolist()

                if tid in self.vehicles:
                    self.vehicles[tid].update(bbox_list, self.frame_count)
                else:
                    self.vehicles[tid] = TrackedVehicle(
                        tid, cls_name, bbox_list, self.frame_count
                    )
                active_ids.add(int(tid))

        # Remove vehicles not seen for too long
        stale = [
            tid for tid, v in self.vehicles.items()
            if self.frame_count - v.frame_num > config.TRACK_LOST_TIMEOUT
        ]
        for tid in stale:
            del self.vehicles[tid]

        return {tid: v for tid, v in self.vehicles.items()
                if tid in active_ids}

    def draw_tracks(self, frame: np.ndarray,
                    vehicles: dict) -> np.ndarray:
        """Draw bounding boxes, IDs and speed on frame."""
        for vid, v in vehicles.items():
            x1, y1, x2, y2 = [int(c) for c in v.bbox]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 220, 0), 2)
            label = f"ID:{v.id} {v.class_name} {v.speed:.1f}px/f"
            cv2.putText(frame, label, (x1, max(y1 - 6, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 220, 0), 1)
        return frame
