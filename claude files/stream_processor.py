# ============================================================
# UYIR — Stream Processor
# Manikandan — Task M6
#
# Main entry point. Connects the full pipeline:
#   Camera stream → Tracker → Accident Detector → Firebase
#
# Usage:
#   python stream_processor.py                    # webcam
#   python stream_processor.py --source 0         # webcam
#   python stream_processor.py --source rtsp://192.168.1.5:8080/h264_ulaw.sdp
#   python stream_processor.py --source video.mp4
#   python stream_processor.py --no_display       # headless (Pi deployment)
#
# Phone RTSP testing (IP Webcam app on Android):
#   Install "IP Webcam" → Start Server → note the IP shown
#   Use: rtsp://YOUR_PHONE_IP:8080/h264_ulaw.sdp
# ============================================================

import cv2
import time
import argparse
import threading
import logging
import numpy as np

import config
from vehicle_tracker     import VehicleTracker
from accident_detector   import AccidentDetector, AccidentEvent
from firebase_uploader   import FirebaseUploader
from health_monitor      import HealthMonitor

# ── Logging setup ─────────────────────────────────────────────
logging.basicConfig(
    level   = logging.INFO,
    format  = "%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt = "%H:%M:%S",
)
logger = logging.getLogger("StreamProcessor")


class StreamProcessor:
    """
    Orchestrates the full UYIR pipeline for one camera.
    """

    def __init__(self,
                 source      = config.RTSP_URL,
                 show_window : bool = True):

        self.source      = source
        self.show_window = show_window
        self._running    = False

        # FPS tracking
        self._fps_list : list = []
        self._fps_lock = threading.Lock()

        # Modules
        self.tracker  = VehicleTracker(
            model_path = config.VEHICLE_MODEL_PATH,
            confidence = config.VEHICLE_CONF_THRESHOLD,
        )
        self.detector = AccidentDetector(
            camera_id = config.CAMERA_ID,
            location  = config.CAMERA_LOCATION,
        )
        self.uploader = FirebaseUploader()
        self.health   = HealthMonitor(fps_provider=self._get_fps)

    # ── Public ────────────────────────────────────────────────

    def start(self):
        """Start processing. Blocks until stopped or stream ends."""

        # Retry any offline-saved events first
        self.uploader.retry_local_events()

        # Start health monitor
        self.health.start()

        cap = self._connect(self.source)
        if cap is None:
            logger.error("Could not open stream. Exiting.")
            return

        self._running = True
        frame_count   = 0
        processed     = 0

        logger.info(f"Pipeline running | camera={config.CAMERA_ID} "
                    f"| location={config.CAMERA_LOCATION}")
        logger.info("Press Q in the window to stop.\n")

        while self._running:
            t_start = time.time()

            ret, frame = cap.read()
            if not ret:
                logger.warning("Stream lost. Reconnecting in 3s...")
                cap.release()
                time.sleep(3)
                cap = self._connect(self.source)
                if cap is None:
                    break
                continue

            frame_count += 1

            # Skip frames to reduce CPU load
            if frame_count % config.FRAME_SKIP != 0:
                continue

            processed += 1

            # Resize if needed
            frame = cv2.resize(
                frame,
                (config.FRAME_WIDTH, config.FRAME_HEIGHT)
            )

            # ── Step 1: Track vehicles ────────────────────────
            vehicles = self.tracker.process_frame(frame)

            # ── Step 2: Accident detection + 3-phase logic ────
            event = self.detector.analyze(frame, vehicles, processed)

            # ── Step 3: Handle confirmed accident ─────────────
            if event:
                self._on_accident(event)

            # ── Step 4: Display window ─────────────────────────
            if self.show_window:
                display = self._draw_ui(frame, vehicles, event, processed)
                cv2.imshow(f"UYIR | {config.CAMERA_ID}", display)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            # ── FPS calculation ────────────────────────────────
            elapsed = max(time.time() - t_start, 1e-6)
            fps     = 1.0 / elapsed
            with self._fps_lock:
                self._fps_list.append(fps)
                if len(self._fps_list) > 30:
                    self._fps_list.pop(0)

        cap.release()
        self.health.stop()
        if self.show_window:
            cv2.destroyAllWindows()
        logger.info("Pipeline stopped.")

    def stop(self):
        self._running = False

    # ── Private ───────────────────────────────────────────────

    def _connect(self, source) -> cv2.VideoCapture | None:
        """Connect to stream with retries."""
        for attempt in range(5):
            cap = cv2.VideoCapture(source)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                logger.info(f"Connected to source: {source}")
                return cap
            logger.warning(f"Connection attempt {attempt+1}/5 failed...")
            time.sleep(2)
        return None

    def _on_accident(self, event: AccidentEvent):
        """Handle a confirmed accident — log and upload."""
        print("\n" + "=" * 58)
        print("  🚨  ACCIDENT DETECTED")
        print(f"  Camera     : {event.camera_id}")
        print(f"  Location   : {event.location}")
        print(f"  Time       : {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(event.timestamp))}")
        print(f"  Confidence : {event.confidence_score:.0%}")
        print(f"  Stage 1    : {event.stage1_confidence:.0%}")
        print(f"  Phases     : {', '.join(event.phases_triggered)}")
        print(f"  Vehicles   : {event.involved_vehicle_ids}")
        print("=" * 58 + "\n")

        # Upload asynchronously — does not block detection
        self.uploader.upload_event_async(event)

    def _draw_ui(self, frame        : np.ndarray,
                 vehicles           : dict,
                 event              : AccidentEvent | None,
                 frame_num          : int) -> np.ndarray:
        """Draw detection overlay on the frame."""
        display = self.tracker.draw_tracks(frame.copy(), vehicles)
        avg_fps = self._get_fps()

        # Top status bar
        cv2.rectangle(display, (0, 0), (display.shape[1], 36),
                      (20, 20, 20), -1)
        cv2.putText(
            display,
            f"UYIR | {config.CAMERA_ID} | {config.CAMERA_LOCATION} "
            f"| Frame:{frame_num} | FPS:{avg_fps:.1f} "
            f"| Vehicles:{len(vehicles)}",
            (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.52,
            (200, 200, 200), 1
        )

        # Accident banner
        if event:
            h = display.shape[0]
            overlay = display.copy()
            cv2.rectangle(
                overlay,
                (0, h // 2 - 44),
                (display.shape[1], h // 2 + 44),
                (0, 0, 180), -1,
            )
            cv2.addWeighted(overlay, 0.65, display, 0.35, 0, display)
            cv2.putText(
                display, "ACCIDENT DETECTED",
                (display.shape[1] // 2 - 160, h // 2 + 14),
                cv2.FONT_HERSHEY_SIMPLEX, 1.3,
                (255, 255, 255), 3,
            )

        return display

    def _get_fps(self) -> float:
        with self._fps_lock:
            return round(float(np.mean(self._fps_list)), 1) \
                if self._fps_list else 0.0


# ── Entry point ───────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="UYIR Stream Processor")
    parser.add_argument(
        "--source", default=str(config.RTSP_URL),
        help="Video source: 0=webcam, RTSP URL, or video file path"
    )
    parser.add_argument(
        "--no_display", action="store_true",
        help="Run without display window (use on Pi without monitor)"
    )
    args = parser.parse_args()

    # Convert "0" string to int for webcam
    source = int(args.source) \
        if args.source.isdigit() else args.source

    processor = StreamProcessor(
        source      = source,
        show_window = not args.no_display,
    )
    processor.start()
