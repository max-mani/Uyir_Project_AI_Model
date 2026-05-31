"""
UYIR — Project paths (relative to repository root).
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parent

MODELS_DIR = ROOT / "models"
DATA_LOGS_DIR = ROOT / "data_logs"
SNAPSHOTS_DIR = ROOT / "snapshots"
OUTPUTS_DIR = ROOT / "outputs"
LOCAL_EVENTS_DIR = ROOT / "local_events"

DEFAULT_YOLO_MODEL = MODELS_DIR / "yolov8n.pt"
DEFAULT_DATA_LOG = DATA_LOGS_DIR / "uyir_data_log.csv"


def ensure_dirs():
    """Create standard output directories if missing."""
    for d in (MODELS_DIR, DATA_LOGS_DIR, SNAPSHOTS_DIR, OUTPUTS_DIR, LOCAL_EVENTS_DIR):
        d.mkdir(parents=True, exist_ok=True)
