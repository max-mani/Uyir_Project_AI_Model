"""
UYIR — Data Logger
Manikandan's threshold analysis tool.

Run this on accident and non-accident video clips.
It logs all raw factor values into a CSV file.
Then load the CSV into pandas and plot distributions to find thresholds.

Usage:
    python data_logger.py --video path/to/clip.mp4 --label accident
    python data_logger.py --video path/to/normal.mp4 --label normal
"""

import cv2
import csv
import time
import argparse
import os
import numpy as np
from paths import DEFAULT_YOLO_MODEL, DEFAULT_DATA_LOG, ensure_dirs
from vehicle_tracker import VehicleTracker
from accident_detector import AccidentDetector, Thresholds


def run_logger(video_path: str, label: str, output_csv: str):
    """
    Process a video clip and log all raw measurements to CSV.

    Args:
        video_path: Path to video file or RTSP URL
        label:      "accident" or "normal" — you set this manually
        output_csv: Output CSV file path
    """

    ensure_dirs()
    tracker  = VehicleTracker(model_path=str(DEFAULT_YOLO_MODEL), confidence=0.45)
    detector = AccidentDetector(camera_id="LOG_SESSION")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video: {video_path}")
        return

    out_dir = os.path.dirname(output_csv)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    # Write CSV header
    file_exists = os.path.exists(output_csv)
    csv_file    = open(output_csv, "a", newline="")
    writer      = csv.writer(csv_file)

    if not file_exists:
        writer.writerow([
            "video_file",
            "label",               # "accident" or "normal" — manually set
            "frame_num",
            "vehicle_id",
            "class_name",
            "centroid_x",
            "centroid_y",
            "bbox_width",
            "bbox_height",
            "speed_px_per_frame",
            "avg_speed_5f",
            "speed_drop_percent",
            "direction_degrees",
            "iou_with_nearest",
            "trajectory_deviation_px",
            "optical_flow_ratio",
            "total_factor_score"
        ])

    print(f"\n[Logger] Processing: {video_path}")
    print(f"[Logger] Label: {label}")
    print(f"[Logger] Output: {output_csv}")
    print("[Logger] Press Q to stop early\n")

    frame_num  = 0
    video_name = os.path.basename(video_path)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_num += 1
        vehicles   = tracker.process_frame(frame)

        # Compute optical flow ratio for this frame
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        flow_ratio = _get_flow_ratio(gray, detector)

        # Compute IOU between all vehicle pairs
        vlist    = list(vehicles.values())
        iou_map  = _compute_all_ious(vlist)

        # Log one row per vehicle per frame
        for vid, v in vehicles.items():
            bbox   = v.bbox
            bw     = int(bbox[2] - bbox[0])
            bh     = int(bbox[3] - bbox[1])
            nearest_iou = iou_map.get(vid, 0.0)
            traj_dev    = _compute_traj_deviation(v)

            # Quick score for this vehicle
            score = 0
            if nearest_iou > Thresholds.IOU_OVERLAP:
                score += 3
            if v.get_speed_drop_percent() > Thresholds.SPEED_DROP_PERCENT:
                score += 2
            if traj_dev > Thresholds.TRAJECTORY_DEVIATION:
                score += 2
            if flow_ratio > Thresholds.OPTICAL_FLOW_SPIKE:
                score += 1

            writer.writerow([
                video_name,
                label,
                frame_num,
                vid,
                v.class_name,
                v.centroid[0],
                v.centroid[1],
                bw,
                bh,
                round(v.speed, 3),
                round(v.get_avg_speed(5), 3),
                round(v.get_speed_drop_percent(), 3),
                round(v.direction, 2),
                round(nearest_iou, 4),
                round(traj_dev, 3),
                round(flow_ratio, 4),
                score
            ])

        # Show progress
        annotated = tracker.draw_tracks(frame.copy(), vehicles)
        cv2.putText(annotated,
                    f"Frame: {frame_num} | Vehicles: {len(vehicles)} | [{label.upper()}]",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
        cv2.imshow("UYIR Data Logger", annotated)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("[Logger] Stopped by user.")
            break

    cap.release()
    csv_file.close()
    cv2.destroyAllWindows()
    print(f"\n[Logger] Done. Logged {frame_num} frames → {output_csv}")


# ── Helper functions ─────────────────────────────────────────────────────────

_prev_gray_log  = None
_flow_hist_log  = []

def _get_flow_ratio(gray, detector):
    global _prev_gray_log, _flow_hist_log
    magnitude = 0.0
    if _prev_gray_log is not None:
        try:
            flow = cv2.calcOpticalFlowFarneback(
                _prev_gray_log, gray, None,
                0.5, 3, 15, 3, 5, 1.2, 0
            )
            mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            magnitude = float(np.mean(mag))
        except Exception:
            pass
    _prev_gray_log = gray.copy()
    _flow_hist_log.append(magnitude)
    if len(_flow_hist_log) > 10:
        _flow_hist_log.pop(0)
    avg = max(0.1, float(np.mean(_flow_hist_log)))
    return magnitude / avg


def _compute_all_ious(vehicles):
    """Returns dict: vehicle_id → highest IOU with any other vehicle."""
    iou_map = {}
    for i in range(len(vehicles)):
        for j in range(i + 1, len(vehicles)):
            va, vb = vehicles[i], vehicles[j]
            iou    = _iou(va.bbox, vb.bbox)
            iou_map[va.id] = max(iou_map.get(va.id, 0.0), iou)
            iou_map[vb.id] = max(iou_map.get(vb.id, 0.0), iou)
    return iou_map


def _iou(b1, b2):
    xi1 = max(b1[0], b2[0]); yi1 = max(b1[1], b2[1])
    xi2 = min(b1[2], b2[2]); yi2 = min(b1[3], b2[3])
    inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    if inter == 0:
        return 0.0
    a1    = (b1[2]-b1[0]) * (b1[3]-b1[1])
    a2    = (b2[2]-b2[0]) * (b2[3]-b2[1])
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0.0


def _compute_traj_deviation(v):
    if len(v.centroid_history) < 3:
        return 0.0
    p2 = v.centroid_history[-3]
    p1 = v.centroid_history[-2]
    p0 = v.centroid_history[-1]
    ex = p1[0] + (p1[0] - p2[0])
    ey = p1[1] + (p1[1] - p2[1])
    return float(np.sqrt((p0[0]-ex)**2 + (p0[1]-ey)**2))


# ── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="UYIR Data Logger")
    parser.add_argument("--video",  required=True,  help="Video file path or RTSP URL")
    parser.add_argument("--label",  required=True,  choices=["accident", "normal"],
                        help="Label for this clip")
    parser.add_argument("--output", default=str(DEFAULT_DATA_LOG),
                        help="Output CSV file (default: data_logs/uyir_data_log.csv)")
    args = parser.parse_args()

    run_logger(args.video, args.label, args.output)
