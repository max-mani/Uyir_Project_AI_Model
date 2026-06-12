# ============================================================
# UYIR — Firebase Uploader
# Manikandan — Task M7
#
# Uploads confirmed accident events to Firebase Firestore
# and snapshot images to Firebase Storage.
# Always runs in a separate thread — never blocks detection.
#
# Setup:
#   1. Go to Firebase Console → Project Settings → Service Accounts
#   2. Generate new private key → save as firebase_key.json
#   3. Update FIREBASE_BUCKET in config.py with your bucket name
#   4. pip install firebase-admin
# ============================================================

import os
import json
import time
import logging
import threading
from datetime import datetime, timezone
import cv2
import config

logger = logging.getLogger("FirebaseUploader")


class FirebaseUploader:
    """
    Handles async upload of accident events to Firebase.
    If Firebase is not configured, saves events locally as JSON.
    """

    def __init__(self):
        self._db      = None
        self._bucket  = None
        self._enabled = False
        self._lock    = threading.Lock()

        os.makedirs(config.LOCAL_EVENTS_DIR, exist_ok=True)
        os.makedirs(config.SNAPSHOTS_DIR,    exist_ok=True)

        self._init_firebase()

    # ── Firebase init ─────────────────────────────────────────

    def _init_firebase(self):
        if not os.path.exists(config.FIREBASE_KEY_PATH):
            logger.warning(
                "firebase_key.json not found. "
                "Events will be saved locally only. "
                "Add your Firebase key to enable cloud upload."
            )
            return
        try:
            import firebase_admin
            from firebase_admin import credentials, firestore, storage

            with self._lock:
                if not firebase_admin._apps:
                    cred = credentials.Certificate(config.FIREBASE_KEY_PATH)
                    firebase_admin.initialize_app(cred, {
                        "storageBucket": config.FIREBASE_BUCKET
                    })
                self._db      = firestore.client()
                self._bucket  = storage.bucket()
                self._enabled = True
                logger.info("Firebase connected successfully.")

        except ImportError:
            logger.warning(
                "firebase-admin not installed. "
                "Run: pip install firebase-admin"
            )
        except Exception as e:
            logger.error(f"Firebase init failed: {e}")

    # ── Public method — call from stream_processor ────────────

    def upload_event_async(self, event):
        """
        Fire-and-forget upload in a background thread.
        Detection pipeline never waits for this.
        """
        t = threading.Thread(
            target = self._upload,
            args   = (event,),
            daemon = True,
        )
        t.start()

    # ── Upload logic ──────────────────────────────────────────

    def _upload(self, event):
        """Full upload sequence — runs in background thread."""
        ts_str    = datetime.fromtimestamp(
            event.timestamp, tz=timezone.utc
        ).isoformat()
        ts_int    = int(event.timestamp)
        snap_name = f"{event.camera_id}_{ts_int}.jpg"
        snap_path = os.path.join(config.SNAPSHOTS_DIR, snap_name)

        # 1 ── Save snapshot to local disk
        cv2.imwrite(snap_path, event.snapshot_frame)
        logger.info(f"Snapshot saved locally: {snap_path}")

        # 2 ── Upload snapshot to Firebase Storage
        image_url = ""
        if self._enabled:
            try:
                blob_name = f"snapshots/{snap_name}"
                blob      = self._bucket.blob(blob_name)
                blob.upload_from_filename(snap_path,
                                          content_type="image/jpeg")
                blob.make_public()
                image_url = blob.public_url
                logger.info(f"Snapshot uploaded: {image_url}")
            except Exception as e:
                logger.error(f"Storage upload failed: {e}")

        # 3 ── Build Firestore document
        doc = {
            "timestamp"           : ts_str,
            "camera_id"           : event.camera_id,
            "location"            : event.location,
            "confidence_score"    : event.confidence_score,
            "stage1_confidence"   : event.stage1_confidence,
            "phases_triggered"    : event.phases_triggered,
            "involved_vehicle_ids": event.involved_vehicle_ids,
            "image_url"           : image_url,
            "frame_number"        : event.frame_num,
            "created_at"          : time.time(),
        }

        # 4 ── Write to Firestore
        if self._enabled:
            try:
                ref = self._db \
                    .collection(config.FIRESTORE_COLLECTION) \
                    .add(doc)
                logger.info(f"Event uploaded to Firestore: {ref[1].id}")
                return
            except Exception as e:
                logger.error(f"Firestore write failed: {e}")

        # 5 ── Fallback: save locally
        self._save_local(doc, ts_int)

    def _save_local(self, doc: dict, ts_int: int):
        path = os.path.join(
            config.LOCAL_EVENTS_DIR, f"event_{ts_int}.json"
        )
        with open(path, "w") as f:
            json.dump(doc, f, indent=2)
        logger.info(f"Event saved locally: {path}")

    # ── Retry pending local events ────────────────────────────

    def retry_local_events(self):
        """
        Call this on startup to retry any events that failed
        to upload during a previous offline period.
        """
        if not self._enabled:
            return

        folder = config.LOCAL_EVENTS_DIR
        files  = [f for f in os.listdir(folder) if f.endswith(".json")]

        if not files:
            return

        logger.info(f"Retrying {len(files)} locally saved events...")

        for fname in files:
            fpath = os.path.join(folder, fname)
            try:
                with open(fpath) as f:
                    doc = json.load(f)
                self._db.collection(config.FIRESTORE_COLLECTION).add(doc)
                os.remove(fpath)
                logger.info(f"Retried: {fname}")
            except Exception as e:
                logger.warning(f"Retry failed for {fname}: {e}")
