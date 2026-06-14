# ============================================================
# UYIR — Configuration File
# Change model paths and thresholds here only.
# Do not hardcode values in other files.
# ============================================================

# ── Model Paths ──────────────────────────────────────────────
# Change these to your actual model file paths
ACCIDENT_MODEL_PATH  = "accident_model.pt"   # Kishore's accident detection model
VEHICLE_MODEL_PATH   = "vehicle_model.pt"    # Vehicle detection model (Lalu/Kishore)

# ── Camera Settings ──────────────────────────────────────────
CAMERA_ID       = "CAM_001"
CAMERA_LOCATION = "Gandhipuram Junction"

# For phone testing (IP Webcam app on Android):
#   RTSP_URL = "rtsp://192.168.1.5:8080/h264_ulaw.sdp"
# For real traffic camera:
#   RTSP_URL = "rtsp://CAMERA_IP:PORT/stream"
# For webcam:
#   RTSP_URL = 0
RTSP_URL = 0

FRAME_SKIP        = 3       # Process every Nth frame (3 = ~10 FPS from 30FPS source)
FRAME_WIDTH       = 1280
FRAME_HEIGHT      = 720

# ── Stage 1 — Accident Model Settings ───────────────────────
ACCIDENT_CONF_THRESHOLD = 0.50   # Min confidence for accident model detection
STAGE1_GATE_CONFIDENCE  = 0.65   # Min confidence to pass Stage 1 to verification

# ── Stage 2 — Vehicle Tracker Settings ──────────────────────
VEHICLE_CONF_THRESHOLD  = 0.30   # Min confidence for vehicle detection
TRACK_HISTORY_FRAMES    = 30     # How many frames of history to keep per vehicle
TRACK_LOST_TIMEOUT      = 30     # Remove vehicle after N missed frames

# ── Phase A — Proximity Thresholds ──────────────────────────
# These will be tuned per camera using data_logger + threshold_analyzer
# City junction cameras (dense traffic):
PROXIMITY_PX_CITY     = 150   # pixels at 720p
# Highway cameras (fast traffic):
PROXIMITY_PX_HIGHWAY  = 220   # pixels at 720p

# Default (change per camera after analysis):
PROXIMITY_THRESHOLD   = 150

# ── Phase B — Trajectory Conflict Thresholds ────────────────
SPEED_DROP_PERCENT    = 70.0   # % speed drop = sudden stop
ANGLE_DIVERGENCE_DEG  = 30.0   # degrees direction change = conflict
VELOCITY_SUM_STOP     = 8.0    # combined px/frame = both nearly stopped

# ── Phase C — Anomaly Confirmation Thresholds ───────────────
OPTICAL_FLOW_SPIKE    = 2.5    # x times rolling average = spike
BBOX_DEFORM_RATIO     = 0.25   # 25% area change = deformation
FLOW_HISTORY_FRAMES   = 10     # rolling average window

# ── Gate Settings ────────────────────────────────────────────
CONSECUTIVE_FRAMES    = 3      # frames that must agree before alert
COOLDOWN_SECONDS      = 20.0   # seconds before same camera alerts again

# ── Score Weights ────────────────────────────────────────────
SCORE_PHASE_A         = 3
SCORE_PHASE_B         = 2
SCORE_PHASE_C         = 1
MIN_SCORE_TO_PASS     = 4

# ── Firebase Settings ────────────────────────────────────────
FIREBASE_KEY_PATH     = "firebase_key.json"
FIREBASE_BUCKET       = "your-project-id.appspot.com"   # change this
FIRESTORE_COLLECTION  = "accident_events"
HEALTH_COLLECTION     = "pi_health"
HEALTH_INTERVAL_SEC   = 30     # heartbeat every N seconds

# ── Local Fallback ───────────────────────────────────────────
LOCAL_EVENTS_DIR      = "local_events"
SNAPSHOTS_DIR         = "snapshots"

# ── Data Logger Settings (for threshold analysis) ────────────
DATA_LOG_CSV          = "uyir_data_log.csv"
