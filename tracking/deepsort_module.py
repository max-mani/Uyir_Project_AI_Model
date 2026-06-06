import numpy as np

def compute_iou(box1, box2):
    """
    Computes Intersection over Union (IoU) between two bounding boxes.
    box: [x1, y1, x2, y2]
    """
    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2])
    y2_inter = min(box1[3], box2[3])
    
    inter_area = max(0, x2_inter - x1_inter) * max(0, y1_inter - y2_inter)
    if inter_area == 0:
        # Check intersection in standard coords where y2 > y1
        x1_inter = max(box1[0], box2[0])
        y1_inter = max(box1[1], box2[1])
        x2_inter = min(box1[2], box2[2])
        y2_inter = min(box1[3], box2[3])
        inter_area = max(0, x2_inter - x1_inter) * max(0, y2_inter - y1_inter)
        
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    union_area = area1 + area2 - inter_area
    if union_area <= 0:
        return 0.0
    return float(inter_area / union_area)

class Track:
    def __init__(self, track_id, bbox, label, conf):
        self.track_id = track_id
        self.bbox = bbox  # [x1, y1, x2, y2]
        self.label = label
        self.confidence = conf
        self.history = [self.get_centroid()]  # Centroid history
        self.bbox_history = [bbox]  # Bounding box history
        self.velocities = [(0.0, 0.0)]  # Velocity vectors (vx, vy)
        self.age = 1
        self.missing_count = 0

    def get_centroid(self):
        x1, y1, x2, y2 = self.bbox
        return (float((x1 + x2) / 2.0), float((y1 + y2) / 2.0))

    def update(self, bbox, conf):
        prev_centroid = self.get_centroid()
        self.bbox = bbox
        self.confidence = conf
        curr_centroid = self.get_centroid()
        
        # Calculate velocity vector (pixels per frame)
        vx = curr_centroid[0] - prev_centroid[0]
        vy = curr_centroid[1] - prev_centroid[1]
        
        self.velocities.append((vx, vy))
        if len(self.velocities) > 30:
            self.velocities.pop(0)

        self.history.append(curr_centroid)
        if len(self.history) > 30:
            self.history.pop(0)

        self.bbox_history.append(bbox)
        if len(self.bbox_history) > 30:
            self.bbox_history.pop(0)

        self.missing_count = 0
        self.age += 1

class VehicleTracker:
    def __init__(self, iou_threshold=0.3, max_missing=10):
        self.tracks = {}
        self.next_id = 1
        self.iou_threshold = iou_threshold
        self.max_missing = max_missing

    def update(self, detections):
        """
        Updates tracks with new detections.
        detections: list of dicts: [{"bbox": [x1, y1, x2, y2], "confidence": float, "label": str}]
        Returns list of active tracks.
        """
        # Get active track IDs
        active_track_ids = list(self.tracks.keys())
        
        # If no active tracks, initialize all detections as new tracks
        if not active_track_ids:
            for det in detections:
                track = Track(self.next_id, det["bbox"], det["label"], det["confidence"])
                self.tracks[self.next_id] = track
                self.next_id += 1
            return list(self.tracks.values())

        # Greedy matching based on IoU
        matches = []
        unmatched_detections = list(range(len(detections)))
        unmatched_tracks = list(active_track_ids)

        # Build list of IoU scores
        iou_matrix = []
        for t_id in unmatched_tracks:
            track = self.tracks[t_id]
            for d_idx in unmatched_detections:
                det = detections[d_idx]
                iou = compute_iou(track.bbox, det["bbox"])
                if iou >= self.iou_threshold:
                    iou_matrix.append((iou, t_id, d_idx))

        # Sort matches by highest IoU descending
        iou_matrix.sort(key=lambda x: x[0], reverse=True)

        for iou, t_id, d_idx in iou_matrix:
            if t_id in unmatched_tracks and d_idx in unmatched_detections:
                matches.append((t_id, d_idx))
                unmatched_tracks.remove(t_id)
                unmatched_detections.remove(d_idx)

        # Update matched tracks
        for t_id, d_idx in matches:
            det = detections[d_idx]
            self.tracks[t_id].update(det["bbox"], det["confidence"])

        # Increment missing count for unmatched tracks, mark for deletion
        deleted_ids = []
        for t_id in unmatched_tracks:
            track = self.tracks[t_id]
            track.missing_count += 1
            # Add virtual velocity update (keep moving with same velocity)
            if len(track.velocities) > 0:
                vx, vy = track.velocities[-1]
                x1, y1, x2, y2 = track.bbox
                new_bbox = [x1 + vx, y1 + vy, x2 + vx, y2 + vy]
                track.bbox = new_bbox
                track.history.append(track.get_centroid())
                if len(track.history) > 30:
                    track.history.pop(0)
            
            if track.missing_count > self.max_missing:
                deleted_ids.append(t_id)

        # Delete stale tracks
        for t_id in deleted_ids:
            del self.tracks[t_id]

        # Spawn new tracks for unmatched detections
        for d_idx in unmatched_detections:
            det = detections[d_idx]
            track = Track(self.next_id, det["bbox"], det["label"], det["confidence"])
            self.tracks[self.next_id] = track
            self.next_id += 1

        # Return active tracks that have been updated recently
        return [t for t in self.tracks.values() if t.missing_count == 0]
