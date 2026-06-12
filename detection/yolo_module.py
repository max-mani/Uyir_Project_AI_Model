from ultralytics import YOLO

import config

# Shared model instance — used by both detector and ByteTrack tracker
_shared_model = None


def get_yolo_model(model_name=None):
    global _shared_model
    model_name = model_name or config.VEHICLE_MODEL_PATH
    if _shared_model is None:
        _shared_model = YOLO(model_name)
    return _shared_model


class YOLODetector:
    def __init__(self, model_name=None, conf_threshold=0.3):
        model_name = model_name or config.VEHICLE_MODEL_PATH
        self.model = get_yolo_model(model_name)
        self.conf_threshold = conf_threshold
        self.target_classes = config.TARGET_CLASSES

    def detect(self, frame):
        """
        Detects vehicles and persons in a single frame.
        Returns list of dicts with bbox, confidence, class_id, label.
        """
        results = self.model(frame, verbose=False)[0]
        detections = []

        for box in results.boxes:
            conf = float(box.conf[0])
            class_id = int(box.cls[0])

            if conf >= self.conf_threshold and class_id in self.target_classes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                detections.append({
                    "bbox": [float(x1), float(y1), float(x2), float(y2)],
                    "confidence": conf,
                    "class_id": class_id,
                    "label": self.target_classes[class_id],
                })

        return detections
