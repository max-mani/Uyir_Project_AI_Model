import math

import config


class Track:
    def __init__(self, track_id, bbox, label, conf):
        self.track_id = track_id
        self.bbox = bbox
        self.label = label
        self.confidence = conf
        self.history = [self.get_centroid()]
        self.bbox_history = [bbox]
        self.velocities = [(0.0, 0.0)]
        self.speed_history = [0.0]
        self.age = 1
        self.missing_count = 0
        self.last_seen_frame = 0

    def get_centroid(self):
        x1, y1, x2, y2 = self.bbox
        return (float((x1 + x2) / 2.0), float((y1 + y2) / 2.0))

    def update(self, bbox, conf):
        prev_centroid = self.get_centroid()
        self.bbox = bbox
        self.confidence = conf
        curr_centroid = self.get_centroid()

        vx = curr_centroid[0] - prev_centroid[0]
        vy = curr_centroid[1] - prev_centroid[1]
        speed = math.sqrt(vx * vx + vy * vy)

        self.velocities.append((vx, vy))
        self.speed_history.append(speed)

        max_len = config.TRACK_HISTORY_FRAMES
        if len(self.velocities) > max_len:
            self.velocities.pop(0)
        if len(self.speed_history) > max_len:
            self.speed_history.pop(0)

        self.history.append(curr_centroid)
        if len(self.history) > max_len:
            self.history.pop(0)

        self.bbox_history.append(bbox)
        if len(self.bbox_history) > max_len:
            self.bbox_history.pop(0)

        self.missing_count = 0
        self.age += 1
        self.last_seen_frame = getattr(self, "_current_frame", self.last_seen_frame)


class ByteTrackTracker:
    """
    YOLOv8 + ByteTrack tracker. Maintains Track objects compatible
    with the existing phase modules and app.py pipeline.
    """

    def __init__(self, model_name=None, conf_threshold=None):
        from detection.yolo_module import get_yolo_model
        model_path = model_name or config.VEHICLE_MODEL_PATH
        self.model = get_yolo_model(model_path)
        self.conf_threshold = conf_threshold or config.VEHICLE_CONF_THRESHOLD
        self.tracks = {}
        self.frame_count = 0
        self.max_missing = config.TRACK_LOST_TIMEOUT

    def update(self, detections=None, frame=None):
        """
        Update tracks from a frame using ByteTrack, or from raw detections
        (legacy path for static image processing without tracking IDs).
        """
        if frame is not None:
            return self._update_from_frame(frame)

        return self._update_from_detections(detections or [])

    def _update_from_frame(self, frame):
        self.frame_count += 1
        active_ids = set()

        results = self.model.track(
            frame,
            persist=True,
            conf=self.conf_threshold,
            tracker="bytetrack.yaml",
            verbose=False,
        )

        result = results[0]
        if result.boxes is not None and result.boxes.id is not None:
            boxes = result.boxes.xyxy.cpu().numpy()
            ids = result.boxes.id.cpu().numpy().astype(int)
            classes = result.boxes.cls.cpu().numpy().astype(int)
            confs = result.boxes.conf.cpu().numpy()

            for bbox, tid, cls_id, conf in zip(boxes, ids, classes, confs):
                if int(cls_id) not in config.TARGET_CLASSES:
                    continue

                bbox_list = bbox.tolist()
                label = config.TARGET_CLASSES[int(cls_id)]

                if tid in self.tracks:
                    track = self.tracks[tid]
                    track._current_frame = self.frame_count
                    track.update(bbox_list, float(conf))
                    track.last_seen_frame = self.frame_count
                else:
                    track = Track(tid, bbox_list, label, float(conf))
                    track.last_seen_frame = self.frame_count
                    self.tracks[tid] = track

                active_ids.add(int(tid))

        stale = [
            tid for tid, track in self.tracks.items()
            if self.frame_count - track.last_seen_frame > self.max_missing
        ]
        for tid in stale:
            del self.tracks[tid]

        return [self.tracks[tid] for tid in active_ids if tid in self.tracks]

    def _update_from_detections(self, detections):
        """Fallback for single-frame image processing without ByteTrack IDs."""
        active_tracks = []
        for idx, det in enumerate(detections):
            track_id = idx + 1
            if track_id in self.tracks:
                self.tracks[track_id].update(det["bbox"], det["confidence"])
            else:
                self.tracks[track_id] = Track(
                    track_id, det["bbox"], det["label"], det["confidence"]
                )
            active_tracks.append(self.tracks[track_id])
        return active_tracks


VehicleTracker = ByteTrackTracker
