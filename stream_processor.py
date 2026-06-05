"""
UYIR — Stream Processor
Runs the full pipeline on a live video source.

Sources supported:
  - Laptop webcam:       python stream_processor.py --source 0
  - Video file:          python stream_processor.py --source path/to/video.mp4
  - Phone IP camera:     python stream_processor.py --source rtsp://192.168.1.5:8080/h264_ulaw.sdp
  - Real traffic camera: python stream_processor.py --source rtsp://CAMERA_IP:PORT/stream

Phone setup for RTSP testing:
  1. Install "IP Webcam" app on Android
  2. Open app → Start Server
  3. Note the IP address shown on screen (e.g., 192.168.1.5:8080)
  4. Use: rtsp://192.168.1.5:8080/h264_ulaw.sdp
"""

import cv2
import time
import argparse
import threading
import numpy as np
from paths import DEFAULT_YOLO_MODEL, SNAPSHOTS_DIR, ensure_dirs
from vehicle_tracker import VehicleTracker
from accident_detector import AccidentDetector, AccidentEvent
from firebase_uploader import FirebaseUploader


class StreamProcessor:
    """
    Main pipeline orchestrator.
    Connects video source → tracker → detector → Firebase upload.
    """

    def __init__(self,
                 source,
                 camera_id:    str   = "CAM_001",
                 location:     str   = "Coimbatore Junction",
                 model_path:   str   = None,
                 use_firebase: bool  = False,
                 show_window:  bool  = True):

        ensure_dirs()
        self.source       = source
        self.camera_id    = camera_id
        self.location     = location
        self.show_window  = show_window
        self.use_firebase = use_firebase

        if model_path is None:
            model_path = str(DEFAULT_YOLO_MODEL)
        self.tracker   = VehicleTracker(model_path=model_path, confidence=0.05)
        self.detector  = AccidentDetector(camera_id=camera_id)
        self.uploader  = FirebaseUploader(camera_id, location) if use_firebase else None

        self._running  = False
        self._fps_list = []

    def start(self):
        """Start processing the stream. Blocks until stopped."""
        cap = self._connect(self.source)
        if cap is None:
            return

        self._running = True
        frame_num     = 0

        print(f"\n[Stream] Started — source: {self.source}")
        print(f"[Stream] Camera ID: {self.camera_id}")
        print(f"[Stream] Location:  {self.location}")
        print("[Stream] Press Q to quit\n")

        while self._running:
            t_start = time.time()

            ret, frame = cap.read()
            if not ret:
                print("[Stream] Lost connection. Reconnecting in 3s...")
                cap.release()
                time.sleep(3)
                cap = self._connect(self.source)
                if cap is None:
                    break
                continue

            frame_num += 1

            # ── Step 1: Detect and track vehicles ──────────────────────────
            vehicles = self.tracker.process_frame(frame)

            # ── Step 2: Run accident logic ──────────────────────────────────
            event = self.detector.analyze(frame, vehicles, frame_num)

            # ── Step 3: Handle accident event ───────────────────────────────
            if event:
                self._on_accident(event)

            # ── Step 4: Display ─────────────────────────────────────────────
            if self.show_window:
                display = self._draw_ui(frame, vehicles, event, frame_num)
                cv2.imshow(f"UYIR — {self.camera_id}", display)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            # ── FPS calculation ──────────────────────────────────────────────
            elapsed = time.time() - t_start
            fps     = 1.0 / elapsed if elapsed > 0 else 0
            self._fps_list.append(fps)
            if len(self._fps_list) > 30:
                self._fps_list.pop(0)

        cap.release()
        cv2.destroyAllWindows()
        print("\n[Stream] Stopped.")

    def stop(self):
        self._running = False

    # ── Private methods ──────────────────────────────────────────────────────

    def _connect(self, source):
        """Connect to video source with retry logic."""
        for attempt in range(3):
            cap = cv2.VideoCapture(source)
            if cap.isOpened():
                print(f"[Stream] Connected to source: {source}")
                return cap
            print(f"[Stream] Connection attempt {attempt+1}/3 failed...")
            time.sleep(2)
        print(f"[Stream] ERROR: Could not connect to {source}")
        return None

    def _on_accident(self, event: AccidentEvent):
        """Handle a confirmed accident event."""
        print("\n" + "="*60)
        print(f"  🚨 ACCIDENT DETECTED")
        print(f"  Camera:     {event.camera_id}")
        print(f"  Time:       {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(event.timestamp))}")
        print(f"  Confidence: {event.confidence_score:.0%}")
        print(f"  Factors:    {', '.join(event.factors_triggered)}")
        print(f"  Vehicles:   {event.involved_vehicle_ids}")
        print("="*60 + "\n")

        # Save snapshot locally
        snapshot_path = str(SNAPSHOTS_DIR / f"accident_{int(event.timestamp)}.jpg")
        cv2.imwrite(snapshot_path, event.snapshot_frame)
        print(f"[Stream] Snapshot saved: {snapshot_path}")

        # Upload to Firebase in background thread
        if self.uploader:
            thread = threading.Thread(
                target=self.uploader.upload_event,
                args=(event, snapshot_path),
                daemon=True
            )
            thread.start()

    def _draw_ui(self, frame, vehicles, event, frame_num):
        """Draw detection overlay on frame."""
        display = self.tracker.draw_tracks(frame.copy(), vehicles)
        avg_fps = np.mean(self._fps_list) if self._fps_list else 0

        # Status bar
        status_color = (0, 0, 255) if event else (0, 200, 0)
        status_text  = "🚨 ACCIDENT DETECTED!" if event else "Monitoring..."
        cv2.rectangle(display, (0, 0), (display.shape[1], 40),
                      (20, 20, 20), -1)
        cv2.putText(display, f"UYIR | {self.camera_id} | Frame:{frame_num} | FPS:{avg_fps:.1f}",
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 200), 1)

        # Accident banner
        if event:
            h = display.shape[0]
            overlay = display.copy()
            cv2.rectangle(overlay, (0, h//2 - 40), (display.shape[1], h//2 + 40),
                          (0, 0, 200), -1)
            cv2.addWeighted(overlay, 0.6, display, 0.4, 0, display)
            cv2.putText(display, "ACCIDENT DETECTED",
                        (display.shape[1]//2 - 150, h//2 + 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)

        return display


# ── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="UYIR Stream Processor")
    parser.add_argument("--source",    default=0,
                        help="Video source: 0=webcam, file path, or RTSP URL")
    parser.add_argument("--camera_id", default="CAM_001",
                        help="Camera identifier")
    parser.add_argument("--location",  default="Coimbatore Junction",
                        help="Camera location name")
    parser.add_argument("--model",     default=str(DEFAULT_YOLO_MODEL),
                        help="YOLO model path (default: models/yolov8n.pt)")
    parser.add_argument("--firebase",  action="store_true",
                        help="Enable Firebase upload (needs firebase_key.json)")
    parser.add_argument("--no_window", action="store_true",
                        help="Run without display window (for Pi deployment)")
    args = parser.parse_args()

    # Handle numeric source
    source = int(args.source) if str(args.source).isdigit() else args.source

    processor = StreamProcessor(
        source       = source,
        camera_id    = args.camera_id,
        location     = args.location,
        model_path   = args.model,
        use_firebase = args.firebase,
        show_window  = not args.no_window
    )
    processor.start()
