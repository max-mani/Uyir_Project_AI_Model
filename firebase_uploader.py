"""UYIR Firebase Uploader — async accident event upload with local fallback."""

import json
import logging
import os
import threading
import time
from datetime import datetime, timezone

import cv2

import config

logger = logging.getLogger("FirebaseUploader")


class FirebaseUploader:
    def __init__(self):
        self._db = None
        self._bucket = None
        self._enabled = False
        self._lock = threading.Lock()
        os.makedirs(config.LOCAL_EVENTS_DIR, exist_ok=True)
        os.makedirs(config.SNAPSHOTS_DIR, exist_ok=True)
        self._init_firebase()

    def _init_firebase(self):
        if not os.path.exists(config.FIREBASE_KEY_PATH):
            logger.warning("firebase_key.json not found — events saved locally only.")
            return
        try:
            import firebase_admin
            from firebase_admin import credentials, firestore, storage
            with self._lock:
                if not firebase_admin._apps:
                    cred = credentials.Certificate(config.FIREBASE_KEY_PATH)
                    firebase_admin.initialize_app(cred, {"storageBucket": config.FIREBASE_BUCKET})
                self._db = firestore.client()
                self._bucket = storage.bucket()
                self._enabled = True
                logger.info("Firebase connected successfully.")
        except ImportError:
            logger.warning("firebase-admin not installed. Run: pip install firebase-admin")
        except Exception as e:
            logger.error(f"Firebase init failed: {e}")

    def upload_event_async(self, event, record=None):
        threading.Thread(target=self._upload, args=(event, record), daemon=True).start()

    def upload_incident_record_async(self, record):
        """Upload a dashboard incident record (from incident_store) to Firestore + Storage."""
        threading.Thread(target=self._upload_incident_record, args=(record,), daemon=True).start()

    @property
    def enabled(self):
        return self._enabled

    def _upload_incident_record(self, record):
        from utils import incident_store

        incident_id = record.get("id", "unknown")
        clip_fs, snap_fs, _, _ = incident_store.build_incident_paths(incident_id)

        image_url = record.get("snapshot_url") or ""
        clip_url = record.get("clip_url") or ""

        if self._enabled:
            if os.path.exists(snap_fs):
                try:
                    blob = self._bucket.blob(f"incidents/{incident_id}/snapshot.jpg")
                    blob.upload_from_filename(snap_fs, content_type="image/jpeg")
                    blob.make_public()
                    image_url = blob.public_url
                except Exception as e:
                    logger.error(f"Snapshot upload failed: {e}")

            if clip_fs and os.path.exists(clip_fs):
                try:
                    blob = self._bucket.blob(f"incidents/{incident_id}/clip.mp4")
                    blob.upload_from_filename(clip_fs, content_type="video/mp4")
                    blob.make_public()
                    clip_url = blob.public_url
                except Exception as e:
                    logger.error(f"Clip upload failed: {e}")

        doc = {
            "incident_id": incident_id,
            "timestamp": record.get("timestamp"),
            "status": record.get("status", "confirmed"),
            "source": record.get("source"),
            "camera_id": record.get("camera_id"),
            "location": record.get("location"),
            "confidence_score": record.get("confidence"),
            "dl_confidence": record.get("dl_confidence"),
            "trigger_phase": record.get("trigger_phase"),
            "phases_triggered": record.get("phases_triggered"),
            "involved_vehicle_ids": record.get("involved_vehicle_ids"),
            "frame_number": record.get("frame_number"),
            "time_in_video_sec": record.get("time_in_video_sec"),
            "image_url": image_url,
            "clip_url": clip_url,
            "llm_analysis": record.get("llm_analysis"),
            "details": record.get("details") or {},
            "created_at": time.time(),
        }

        if self._enabled:
            try:
                self._db.collection(config.FIRESTORE_COLLECTION).document(incident_id).set(doc)
                logger.info(f"Incident {incident_id} ({doc['status']}) uploaded to Firestore.")
                return
            except Exception as e:
                logger.error(f"Firestore write failed: {e}")

        path = os.path.join(config.LOCAL_EVENTS_DIR, f"incident_{incident_id}.json")
        with open(path, "w") as f:
            json.dump(doc, f, indent=2)
        logger.info(f"Incident saved locally: {path}")

    def _upload(self, event, record=None):
        ts_str = datetime.fromtimestamp(event.timestamp, tz=timezone.utc).isoformat()
        ts_int = int(event.timestamp)
        snap_name = f"{event.camera_id}_{ts_int}.jpg"
        snap_path = os.path.join(config.SNAPSHOTS_DIR, snap_name)
        cv2.imwrite(snap_path, event.snapshot_frame)

        image_url = ""
        clip_url = ""
        if self._enabled:
            try:
                blob = self._bucket.blob(f"snapshots/{snap_name}")
                blob.upload_from_filename(snap_path, content_type="image/jpeg")
                blob.make_public()
                image_url = blob.public_url
            except Exception as e:
                logger.error(f"Storage upload failed: {e}")

            clip_path = getattr(event, "clip_path", None) or (record or {}).get("clip_path")
            if clip_path and os.path.exists(clip_path):
                try:
                    clip_name = f"{event.camera_id}_{ts_int}.mp4"
                    clip_blob = self._bucket.blob(f"clips/{clip_name}")
                    clip_blob.upload_from_filename(clip_path, content_type="video/mp4")
                    clip_blob.make_public()
                    clip_url = clip_blob.public_url
                except Exception as e:
                    logger.error(f"Clip upload failed: {e}")

        details = getattr(event, "fusion_details", None) or (record or {}).get("details") or {}
        doc = {
            "timestamp": ts_str,
            "camera_id": event.camera_id,
            "location": event.location,
            "confidence_score": event.confidence_score,
            "stage1_confidence": event.stage1_confidence,
            "dl_confidence": getattr(event, "cnn_lstm_confidence", event.stage1_confidence),
            "trigger_phase": getattr(event, "trigger_phase", "Weighted Fusion"),
            "phases_triggered": event.phases_triggered,
            "involved_vehicle_ids": event.involved_vehicle_ids,
            "image_url": image_url,
            "clip_url": clip_url or (record or {}).get("clip_url", ""),
            "frame_number": event.frame_num,
            "details": details,
            "created_at": time.time(),
        }

        if record:
            doc["incident_id"] = record.get("id")

        if self._enabled:
            try:
                self._db.collection(config.FIRESTORE_COLLECTION).add(doc)
                return
            except Exception as e:
                logger.error(f"Firestore write failed: {e}")

        path = os.path.join(config.LOCAL_EVENTS_DIR, f"event_{ts_int}.json")
        with open(path, "w") as f:
            json.dump(doc, f, indent=2)
        logger.info(f"Event saved locally: {path}")

    def retry_local_events(self):
        if not self._enabled:
            return
        folder = config.LOCAL_EVENTS_DIR
        for fname in os.listdir(folder):
            if not fname.endswith(".json"):
                continue
            if not (fname.startswith("event_") or fname.startswith("incident_")):
                continue
            fpath = os.path.join(folder, fname)
            try:
                with open(fpath) as f:
                    doc = json.load(f)
                incident_id = doc.get("incident_id")
                if incident_id:
                    self._db.collection(config.FIRESTORE_COLLECTION).document(incident_id).set(doc)
                else:
                    self._db.collection(config.FIRESTORE_COLLECTION).add(doc)
                os.remove(fpath)
            except Exception as e:
                logger.warning(f"Retry failed for {fname}: {e}")
