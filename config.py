# ============================================================
# Accident Detection System — Configuration
# Change model paths and thresholds here only.
# ============================================================

# ── Model Paths ──────────────────────────────────────────────
VEHICLE_MODEL_PATH = "yolov8n.pt"
ACCIDENT_MODEL_PATH = "accident_model.pt"  # optional Stage-1 YOLO for stream pipeline

# ── Camera Settings ──────────────────────────────────────────
CAMERA_ID = "CAM_001"
CAMERA_LOCATION = "Gandhipuram Junction"
RTSP_URL = 0

FRAME_SKIP = 3
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720

# ── Stage 1 — Accident Model Settings ───────────────────────
ACCIDENT_CONF_THRESHOLD = 0.50
STAGE1_GATE_CONFIDENCE = 0.65

# ── Tracker Settings ─────────────────────────────────────────
VEHICLE_CONF_THRESHOLD = 0.30
TRACK_HISTORY_FRAMES = 30
TRACK_LOST_TIMEOUT = 30

# ── Phase A — Proximity & TTC ────────────────────────────────
PROXIMITY_THRESHOLD = 150
PROXIMITY_PERSON_THRESHOLD = 80
TTC_MAX_FRAMES = 8
TTC_MIN_CLOSING_SPEED = 0.5

PROXIMITY_PX_CITY = 150
PROXIMITY_PX_HIGHWAY = 220

VEHICLE_CLASSES = {"car", "bike", "bus", "truck", "motorcycle", "auto"}
PERSON_CLASS = "person"

TARGET_CLASSES = {
    0: "person",
    2: "car",
    3: "bike",
    5: "bus",
    7: "truck",
}

# ── Phase B — Trajectory Conflict ───────────────────────────
SPEED_DROP_PERCENT = 70.0
ANGLE_DIVERGENCE_DEG = 30.0
VELOCITY_SUM_STOP = 8.0

EMERGENCY_BASELINE_FRAMES = 15
EMERGENCY_RECENT_FRAMES = 3
EMERGENCY_DROP_PERCENT = 75.0
EMERGENCY_SUDDEN_RATIO = 0.65
TRAJECTORY_STOP_PREV_SPEED = 3.0
TRAJECTORY_STOP_RECENT_SPEED = 2.0
TRAJECTORY_STOP_FRAMES = 5

REL_VEL_PREV_DIFF_MIN = 8.0
REL_VEL_CURR_DIFF_MAX = 2.0

# ── Phase C — Anomaly Confirmation ──────────────────────────
OPTICAL_FLOW_SPIKE = 2.5
BBOX_DEFORM_RATIO = 0.25
FLOW_HISTORY_FRAMES = 10

# ── Gate Settings ────────────────────────────────────────────
CONSECUTIVE_FRAMES = 3
COOLDOWN_SECONDS = 20.0
FUSION_THRESHOLD = 0.55

# ── Fusion Weights (research-backed, LSTM disabled) ──────────
FUSION_WEIGHTS = {
    "trajectory_stop": 0.45,
    "ttc_critical": 0.20,
    "emergency_stop": 0.20,
    "cnn_lstm": 0.0,
    "optical_flow": 0.08,
    "flow_dispersion": 0.07,
}

# Legacy score weights for stream pipeline phase gating
SCORE_PHASE_A = 3
SCORE_PHASE_B = 2
SCORE_PHASE_C = 1
MIN_SCORE_TO_PASS = 4

# ── Firebase Settings ────────────────────────────────────────
FIREBASE_KEY_PATH = "firebase_key.json"
FIREBASE_BUCKET = "your-project-id.appspot.com"
FIRESTORE_COLLECTION = "accident_events"
HEALTH_COLLECTION = "pi_health"
HEALTH_INTERVAL_SEC = 30

# ── Local Fallback ───────────────────────────────────────────
LOCAL_EVENTS_DIR = "local_events"
SNAPSHOTS_DIR = "snapshots"

# ── Data Logger ──────────────────────────────────────────────
DATA_LOG_CSV = "uyir_data_log.csv"
