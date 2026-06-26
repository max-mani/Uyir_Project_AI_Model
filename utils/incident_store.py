"""Local incident persistence and index for dashboard API."""

import json
import os
import threading
import uuid
from datetime import datetime, timezone

import config

_lock = threading.Lock()


def _base_dir():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def incidents_dir():
    path = os.path.join(_base_dir(), config.INCIDENTS_DIR)
    os.makedirs(path, exist_ok=True)
    return path


def index_path():
    path = os.path.join(_base_dir(), config.INCIDENTS_INDEX)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return path


def _load_index():
    path = index_path()
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except (json.JSONDecodeError, OSError):
        return []


def _save_index(records):
    with open(index_path(), "w") as f:
        json.dump(records, f, indent=2)


def _public_url(relative_path):
    rel = relative_path.replace("\\", "/")
    if not rel.startswith("/"):
        rel = "/" + rel
    return rel


def save_incident(record):
    """Persist an incident record and return it with id/urls set."""
    with _lock:
        incident_id = record.get("id") or str(uuid.uuid4())
        record["id"] = incident_id

        if "timestamp" not in record:
            record["timestamp"] = datetime.now(timezone.utc).isoformat()

        records = _load_index()
        records = [r for r in records if r.get("id") != incident_id]
        records.insert(0, record)
        _save_index(records)
        return record


def update_incident(incident_id, updates: dict) -> bool:
    """
    Partially update an existing incident record in the index.
    Used to set clip_url and llm_analysis after async background processing.
    Returns True if the record was found and updated.
    """
    with _lock:
        records = _load_index()
        found = False
        for record in records:
            if record.get("id") == incident_id:
                record.update(updates)
                found = True
                break
        if found:
            _save_index(records)
    return found


def clear_all_incidents() -> bool:
    """
    Delete every incident from the index and remove their media files from disk.
    Used by the dashboard 'Clear All' button.
    """
    with _lock:
        records = _load_index()
        for record in records:
            iid = record.get("id")
            if not iid:
                continue
            clip_fs, snap_fs, _, _ = build_incident_paths(iid)
            for path in (clip_fs, snap_fs):
                try:
                    if os.path.exists(path):
                        os.remove(path)
                except OSError:
                    pass
        _save_index([])
    return True


def list_incidents(limit=50):
    with _lock:
        records = _load_index()
    return records[:limit]


def get_incident(incident_id):
    with _lock:
        for record in _load_index():
            if record.get("id") == incident_id:
                return record
    return None


def delete_incident(incident_id):
    """Remove an incident from the index and delete its media files."""
    with _lock:
        records = _load_index()
        target = next((r for r in records if r.get("id") == incident_id), None)
        if target is None:
            return False

        records = [r for r in records if r.get("id") != incident_id]
        _save_index(records)

    clip_fs, snap_fs, _, _ = build_incident_paths(incident_id)
    for path in (clip_fs, snap_fs):
        try:
            if os.path.exists(path):
                os.remove(path)
        except OSError:
            pass
    return True


def build_incident_paths(incident_id):
    """Return filesystem paths and public URLs for clip/snapshot assets."""
    clip_name = f"clip_{incident_id}.mp4"
    snap_name = f"snap_{incident_id}.jpg"
    clip_fs = os.path.join(incidents_dir(), clip_name)
    snap_fs = os.path.join(incidents_dir(), snap_name)
    clip_url = _public_url(f"{config.INCIDENTS_DIR}/{clip_name}")
    snap_url = _public_url(f"{config.INCIDENTS_DIR}/{snap_name}")
    return clip_fs, snap_fs, clip_url, snap_url
