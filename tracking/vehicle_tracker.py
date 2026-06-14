"""
UYIR Vehicle Tracker — ByteTrack wrapper for the stream pipeline.
Uses TrackedVehicle objects with speed/centroid history.
"""

import logging

import cv2
import numpy as np
from ultralytics import YOLO

import config
from detection.yolo_module import get_yolo_model

logger = logging.getLogger("VehicleTracker")


class TrackedVehicle:
    def __init__(self, track_id: int, class_name: str, bbox: list, frame_num: int):
        self.id = track_id
        self.class_name = class_name
        self.bbox = bbox
        self.frame_num = frame_num
        self.centroid = self._centroid(bbox)
        self.speed = 0.0
        self.direction = 0.0
        self.centroid_history = [self.centroid]
        self.bbox_history = [bbox]
        self.speed_history = [0.0]
        self.expected_next = self.centroid

    @staticmethod
    def _centroid(bbox: list) -> tuple:
        x1, y1, x2, y2 = bbox
        return (int((x1 + x2) / 2), int((y1 + y2) / 2))

    def bbox_area(self) -> float:
        x1, y1, x2, y2 = self.bbox
        return max(0.0, float((x2 - x1) * (y2 - y1)))

    def update(self, bbox: list, frame_num: int):
        prev = self.centroid
        self.bbox = bbox
        self.centroid = self._centroid(bbox)
        self.frame_num = frame_num

        dx = self.centroid[0] - prev[0]
        dy = self.centroid[1] - prev[1]
        self.speed = float(np.sqrt(dx * dx + dy * dy))
        self.direction = float(np.degrees(np.arctan2(dy, dx)))
        self.expected_next = (self.centroid[0] + dx, self.centroid[1] + dy)

        n = config.TRACK_HISTORY_FRAMES
        self.centroid_history.append(self.centroid)
        self.bbox_history.append(bbox)
        self.speed_history.append(self.speed)
        if len(self.centroid_history) > n:
            self.centroid_history.pop(0)
            self.bbox_history.pop(0)
            self.speed_history.pop(0)

    def get_avg_speed(self, last_n: int = 5) -> float:
        h = self.speed_history[-last_n:]
        return float(np.mean(h)) if h else 0.0

    def get_speed_drop_percent(self, last_n: int = 5) -> float:
        if len(self.speed_history) < last_n + 3:
            return 0.0
        avg_before = float(np.mean(self.speed_history[-(last_n + 3):-3]))
        if avg_before < 1.0:
            return 0.0
        drop = ((avg_before - self.speed) / avg_before) * 100.0
        return max(0.0, drop)

    def get_trajectory_deviation(self) -> float:
        if len(self.centroid_history) < 3:
            return 0.0
        p2 = self.centroid_history[-3]
        p1 = self.centroid_history[-2]
        p0 = self.centroid_history[-1]
        ex = p1[0] + (p1[0] - p2[0])
        ey = p1[1] + (p1[1] - p2[1])
        return float(np.sqrt((p0[0] - ex) ** 2 + (p0[1] - ey) ** 2))

    def get_bbox_area_change(self) -> float:
        if len(self.bbox_history) < 5:
            return 0.0
        old_bbox = self.bbox_history[-5]
        old_area = max(1.0, float((old_bbox[2] - old_bbox[0]) * (old_bbox[3] - old_bbox[1])))
        return abs(self.bbox_area() - old_area) / old_area


class VehicleTracker:
    def __init__(self, model_path=None, confidence=None):
        model_path = model_path or config.VEHICLE_MODEL_PATH
        self.model = get_yolo_model(model_path)
        self.confidence = confidence or config.VEHICLE_CONF_THRESHOLD
        self.vehicles = {}
        self.frame_count = 0
        logger.info("[Tracker] ByteTrack vehicle tracker ready.")

    def process_frame(self, frame: np.ndarray) -> dict:
        self.frame_count += 1
        active_ids = set()

        results = self.model.track(
            frame,
            persist=True,
            conf=self.confidence,
            tracker="bytetrack.yaml",
            verbose=False,
        )

        result = results[0]
        if result.boxes is not None and result.boxes.id is not None:
            boxes = result.boxes.xyxy.cpu().numpy()
            ids = result.boxes.id.cpu().numpy().astype(int)
            classes = result.boxes.cls.cpu().numpy().astype(int)

            for bbox, tid, cls_id in zip(boxes, ids, classes):
                if int(cls_id) not in config.TARGET_CLASSES:
                    continue
                cls_name = config.TARGET_CLASSES[int(cls_id)]
                bbox_list = bbox.tolist()
                if tid in self.vehicles:
                    self.vehicles[tid].update(bbox_list, self.frame_count)
                else:
                    self.vehicles[tid] = TrackedVehicle(tid, cls_name, bbox_list, self.frame_count)
                active_ids.add(int(tid))

        stale = [
            tid for tid, v in self.vehicles.items()
            if self.frame_count - v.frame_num > config.TRACK_LOST_TIMEOUT
        ]
        for tid in stale:
            del self.vehicles[tid]

        return {tid: self.vehicles[tid] for tid in active_ids if tid in self.vehicles}

    def draw_tracks(self, frame: np.ndarray, vehicles: dict) -> np.ndarray:
        for vid, v in vehicles.items():
            x1, y1, x2, y2 = [int(c) for c in v.bbox]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 220, 0), 2)
            label = f"ID:{v.id} {v.class_name} {v.speed:.1f}px/f"
            cv2.putText(frame, label, (x1, max(y1 - 6, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 220, 0), 1)
        return frame
