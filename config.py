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
VEHICLE_CONF_THRESHOLD = 0.15
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

# ── Phase B — Recent-motion guard ───────────────────────────
# A track that had peak speed > this in the last N frames is
# considered "recently moving" (post-crash stop ≠ parked car).
RECENTLY_MOVING_FRAMES = 15
RECENTLY_MOVING_MIN_SPEED = 3.0   # px/frame

# ── Phase C — Anomaly Confirmation ──────────────────────────
OPTICAL_FLOW_SPIKE = 2.5
BBOX_DEFORM_RATIO = 0.25
FLOW_HISTORY_FRAMES = 10

# ── Gate Settings ────────────────────────────────────────────
CONSECUTIVE_FRAMES = 3
COOLDOWN_SECONDS = 20.0
FUSION_THRESHOLD = 0.55

# ── DL Gate ──────────────────────────────────────────────────
DL_GATE_THRESHOLD = 0.55   # lstm_peak must reach this to open the gate
DL_PHASE_SIGNAL_MIN = 0.30  # a phase must reach this to count as a vote
DL_WARMUP_FRAMES = 16       # SEQUENCE_LEN // 2 — don't trust rolling peak before this

# ── Fusion Weights ────────────────────────────────────────────
# CNN-LSTM acts as a HARD GATE only — its weight is 0.
# The weight that was on cnn_lstm (0.25) is redistributed to
# trajectory_stop (+0.15) and emergency_stop (+0.05) and optical_flow (+0.05).
FUSION_WEIGHTS = {
    "trajectory_stop": 0.45,
    "emergency_stop":  0.25,
    "ttc_critical":    0.15,
    "optical_flow":    0.10,
    "flow_dispersion": 0.05,
    "cnn_lstm":        0.0,   # gate only — not a weighted contributor
}

# Legacy score weights for stream pipeline phase gating (kept for compatibility)
SCORE_PHASE_A = 3
SCORE_PHASE_B = 2
SCORE_PHASE_C = 1
MIN_SCORE_TO_PASS = 4

# ── Firebase Settings ────────────────────────────────────────
FIREBASE_KEY_PATH = "firebase_key.json"
FIREBASE_BUCKET = "kapaan-web.firebasestorage.app"
FIRESTORE_COLLECTION = "accident_events"
HEALTH_COLLECTION = "pi_health"
HEALTH_INTERVAL_SEC = 30
# Storage requires Blaze (billing). Set False to use Firestore-only on free Spark plan.
FIREBASE_USE_STORAGE = False
# When storage is off, optionally embed a JPEG thumbnail in Firestore (max ~1 MB/doc).
FIREBASE_EMBED_SNAPSHOT = True

# ── Local Fallback ───────────────────────────────────────────
LOCAL_EVENTS_DIR = "local_events"
SNAPSHOTS_DIR = "snapshots"

# ── Incident Clips ───────────────────────────────────────────
CLIP_SECONDS_BEFORE = 5
CLIP_SECONDS_AFTER = 5
INCIDENTS_DIR = "static/uploads/incidents"
INCIDENTS_INDEX = "local_events/incidents_index.json"
CLIP_BUFFER_FPS = 10

# ── Data Logger ──────────────────────────────────────────────
DATA_LOG_CSV = "uyir_data_log.csv"
