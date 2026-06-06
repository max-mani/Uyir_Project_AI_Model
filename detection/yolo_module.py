import cv2
from ultralytics import YOLO

class YOLODetector:
    def __init__(self, model_name="yolov8n.pt", conf_threshold=0.3):
        """
        Initializes the YOLOv8 detector.
        Target classes (COCO dataset):
        - 2: car
        - 3: motorcycle (bike)
        - 5: bus
        - 7: truck
        """
        self.model = YOLO(model_name)
        self.conf_threshold = conf_threshold
        # COCO class indexes for vehicles
        self.vehicle_classes = {
            2: "car",
            3: "bike",
            5: "bus",
            7: "truck"
        }

    def detect(self, frame):
        """
        Detects vehicles in a single frame.
        Returns a list of dicts: [
            {"bbox": [x1, y1, x2, y2], "confidence": float, "class_id": int, "label": str}
        ]
        """
        results = self.model(frame, verbose=False)[0]
        detections = []
        
        for box in results.boxes:
            conf = float(box.conf[0])
            class_id = int(box.cls[0])
            
            if conf >= self.conf_threshold and class_id in self.vehicle_classes:
                # Get coordinates
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                detections.append({
                    "bbox": [float(x1), float(y1), float(x2), float(y2)],
                    "confidence": conf,
                    "class_id": class_id,
                    "label": self.vehicle_classes[class_id]
                })
                
        return detections
