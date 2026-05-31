"""
UYIR — Quick Test
Run this first to verify your setup is working correctly.
It opens your webcam and runs vehicle detection for 30 seconds.

Usage:
    python test_setup.py
"""

import cv2
import time
import sys

from paths import DEFAULT_YOLO_MODEL, ensure_dirs


def test_imports():
    print("\n[Test] Checking imports...")
    errors = []
    try:
        import cv2
        print(f"  ✅ OpenCV {cv2.__version__}")
    except ImportError:
        errors.append("opencv-python")

    try:
        import numpy as np
        print(f"  ✅ NumPy {np.__version__}")
    except ImportError:
        errors.append("numpy")

    try:
        import pandas as pd
        print(f"  ✅ Pandas {pd.__version__}")
    except ImportError:
        errors.append("pandas")

    try:
        import matplotlib
        print(f"  ✅ Matplotlib {matplotlib.__version__}")
    except ImportError:
        errors.append("matplotlib")

    try:
        from ultralytics import YOLO
        print(f"  ✅ Ultralytics YOLO")
    except ImportError:
        errors.append("ultralytics")

    if errors:
        print(f"\n  ❌ Missing packages: {', '.join(errors)}")
        print(f"  Run: pip install {' '.join(errors)}")
        return False

    print("\n  All imports OK ✅")
    return True


def test_webcam():
    print("\n[Test] Testing webcam + YOLO detection for 15 seconds...")
    print("[Test] Press Q to stop early\n")

    from ultralytics import YOLO

    ensure_dirs()
    print("[Test] Loading YOLOv8m model (downloads on first run if missing)...")
    model = YOLO(str(DEFAULT_YOLO_MODEL))
    print("[Test] Model loaded ✅")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[Test] ❌ Cannot open webcam. Check your camera connection.")
        return False

    start    = time.time()
    frames   = 0

    while time.time() - start < 15:
        ret, frame = cap.read()
        if not ret:
            break

        # Run YOLO detection
        results = model(frame, classes=[0, 2, 3, 5, 7],
                        conf=0.45, verbose=False)
        annotated = results[0].plot()

        elapsed = time.time() - start
        fps     = frames / elapsed if elapsed > 0 else 0
        n_det   = len(results[0].boxes)

        cv2.putText(annotated,
                    f"UYIR Test | FPS:{fps:.1f} | Detections:{n_det}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(annotated,
                    "If you see bounding boxes on vehicles, setup is working!",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255), 1)

        cv2.imshow("UYIR Setup Test", annotated)
        frames += 1

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

    avg_fps = frames / 15
    print(f"\n[Test] Result: {frames} frames in 15s = {avg_fps:.1f} FPS ✅")
    print("[Test] If you saw vehicle bounding boxes in the window — your setup is working.")
    return True


def test_full_pipeline():
    print("\n[Test] Testing full accident detection pipeline on webcam...")
    print("[Test] Press Q to stop\n")

    from vehicle_tracker import VehicleTracker
    from accident_detector import AccidentDetector

    tracker  = VehicleTracker()
    detector = AccidentDetector(camera_id="TEST_CAM")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[Test] ❌ Cannot open webcam.")
        return

    frame_num = 0
    start     = time.time()

    while time.time() - start < 20:
        ret, frame = cap.read()
        if not ret:
            break

        frame_num += 1
        vehicles  = tracker.process_frame(frame)
        event     = detector.analyze(frame, vehicles, frame_num)

        display   = tracker.draw_tracks(frame.copy(), vehicles)

        status = "🚨 ACCIDENT" if event else f"Tracking {len(vehicles)} vehicles"
        color  = (0, 0, 255) if event else (0, 255, 0)
        cv2.putText(display, f"UYIR Pipeline | Frame:{frame_num} | {status}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
        cv2.putText(display, "Tracking active — vehicle IDs shown on bounding boxes",
                    (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        cv2.imshow("UYIR Pipeline Test", display)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("\n[Test] Pipeline test complete ✅")


if __name__ == "__main__":
    print("=" * 55)
    print("  UYIR — Setup Test")
    print("  Coimbatore Road Safety Project")
    print("=" * 55)

    if not test_imports():
        sys.exit(1)

    print("\nWhich test do you want to run?")
    print("  1 — Quick webcam + YOLO test (15 seconds)")
    print("  2 — Full pipeline test with tracking (20 seconds)")
    print("  3 — Both")

    choice = input("\nEnter 1, 2, or 3: ").strip()

    if choice in ("1", "3"):
        test_webcam()
    if choice in ("2", "3"):
        test_full_pipeline()

    print("\n✅ All tests complete. Your setup is ready.")
    print("Next step: python stream_processor.py --source 0")
