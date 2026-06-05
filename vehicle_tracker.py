"""
UYIR — Vehicle Tracker
Wraps YOLOv8n + ByteTrack to give persistent vehicle IDs across frames.
Manikandan's module — feeds data into the accident detection logic.
"""

import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict

from paths import DEFAULT_YOLO_MODEL, ensure_dirs


# ── Class names matching pre-trained COCO classes ──────────────────────────
# When Kishore's custom model is ready, swap models/yolov8n.pt with models/best.pt
# These COCO IDs cover our needed vehicle types
VEHICLE_CLASSES = {
    0:  "person",
    2:  "car",
    3:  "motorcycle",   # covers bike/scooter
    5:  "bus",
    7:  "truck",
}
# Note: "auto" (auto-rickshaw) is not in COCO.
# Once Kishore's trained model is ready, this will include "auto" class.


class TrackedVehicle:
    """
    Holds all data for one tracked vehicle across frames.
    This is the core data structure the accident logic reads from.
    """

    def __init__(self, track_id, class_name, bbox, frame_num):
        self.id            = track_id
        self.class_name    = class_name
        self.bbox          = bbox           # [x1, y1, x2, y2]
        self.centroid      = self._get_centroid(bbox)
        self.frame_num     = frame_num

        # History — last 30 frames of positions
        self.centroid_history  = [self.centroid]
        self.bbox_history      = [bbox]
        self.speed_history     = [0.0]

        # Computed values — updated every frame
        self.speed             = 0.0        # pixels per frame
        self.direction         = 0.0        # angle in degrees
        self.expected_next_pos = self.centroid

    def _get_centroid(self, bbox):
        x1, y1, x2, y2 = bbox
        return (int((x1 + x2) / 2), int((y1 + y2) / 2))

    def update(self, bbox, frame_num):
        """Call this every frame the vehicle is detected."""
        prev_centroid      = self.centroid
        self.bbox          = bbox
        self.centroid      = self._get_centroid(bbox)
        self.frame_num     = frame_num

        # Speed = distance from previous centroid
        dx = self.centroid[0] - prev_centroid[0]
        dy = self.centroid[1] - prev_centroid[1]
        self.speed = float(np.sqrt(dx**2 + dy**2))

        # Direction angle
        self.direction = float(np.degrees(np.arctan2(dy, dx)))

        # Predict next position using linear extrapolation
        self.expected_next_pos = (
            self.centroid[0] + dx,
            self.centroid[1] + dy
        )

        # Keep history (max 30 frames)
        self.centroid_history.append(self.centroid)
        self.bbox_history.append(bbox)
        self.speed_history.append(self.speed)
        if len(self.centroid_history) > 30:
            self.centroid_history.pop(0)
            self.bbox_history.pop(0)
            self.speed_history.pop(0)

    def get_avg_speed(self, last_n=5):
        """Average speed over last N frames."""
        history = self.speed_history[-last_n:]
        return float(np.mean(history)) if history else 0.0

    def get_speed_drop_percent(self, last_n=5):
        """
        How much has speed dropped compared to recent average?
        Returns 0–100. High value = sudden stop.
        """
        if len(self.speed_history) < last_n + 1:
            return 0.0
        avg_before = float(np.mean(self.speed_history[-(last_n+3):-3]))
        current    = self.speed
        if avg_before < 1.0:
            return 0.0
        drop = ((avg_before - current) / avg_before) * 100
        return max(0.0, drop)


class VehicleTracker:
    """
    Main tracker class.
    Runs YOLOv8n + ByteTrack and returns TrackedVehicle objects.
    """

    def __init__(self, model_path=None, confidence=0.1):
        """
        model_path: Path under models/ (default: yolov8n.pt).
                    When Kishore finishes training, change to models/best.pt
        confidence: Detection threshold (0.45 default; use 0.30 for more boxes on laptop)
        """
        ensure_dirs()
        if model_path is None:
            model_path = str(DEFAULT_YOLO_MODEL)
        print(f"[Tracker] Loading model: {model_path}")
        self.model       = YOLO(model_path)
        self.confidence  = confidence
        self.vehicles    = {}           # track_id → TrackedVehicle
        self.frame_count = 0
        print("[Tracker] Model loaded successfully.")

    def process_frame(self, frame):
        """
        Main method — call this on every frame.
        Returns dict of {track_id: TrackedVehicle}
        """
        self.frame_count += 1

        # Run YOLO with ByteTrack tracking
        results = self.model.track(
            frame,
            persist=True,               # keeps track IDs across frames
            conf=self.confidence,
            imgsz=1024,
            classes=list(VEHICLE_CLASSES.keys()),
            tracker="bytetrack.yaml",
            verbose=False
        )

        active_ids = set()

        if results[0].boxes.id is not None:
            boxes    = results[0].boxes.xyxy.cpu().numpy()
            ids      = results[0].boxes.id.cpu().numpy().astype(int)
            classes  = results[0].boxes.cls.cpu().numpy().astype(int)

            for bbox, track_id, cls_id in zip(boxes, ids, classes):
                class_name = VEHICLE_CLASSES.get(cls_id, "vehicle")
                bbox_list  = bbox.tolist()

                if track_id in self.vehicles:
                    self.vehicles[track_id].update(bbox_list, self.frame_count)
                else:
                    self.vehicles[track_id] = TrackedVehicle(
                        track_id, class_name, bbox_list, self.frame_count
                    )
                active_ids.add(track_id)

        # Remove vehicles not seen for 30+ frames
        stale = [
            tid for tid, v in self.vehicles.items()
            if self.frame_count - v.frame_num > 30
        ]
        for tid in stale:
            del self.vehicles[tid]

        return {tid: v for tid, v in self.vehicles.items()
                if tid in active_ids}

    def draw_tracks(self, frame, vehicles):
        """Draw bounding boxes and IDs on frame for visualization."""
        for vid, v in vehicles.items():
            x1, y1, x2, y2 = [int(c) for c in v.bbox]
            color = (0, 255, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f"ID:{v.id} {v.class_name} spd:{v.speed:.1f}"
            cv2.putText(frame, label, (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        return frame
