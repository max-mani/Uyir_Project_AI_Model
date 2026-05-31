"""
UYIR — Firebase Uploader
Uploads accident events to Firebase Firestore and snapshot images to Storage.
Runs in a separate thread so it never blocks the detection pipeline.

Setup:
  1. Go to Firebase Console → Project Settings → Service Accounts
  2. Generate new private key → download as "firebase_key.json"
  3. Place firebase_key.json in the same folder as this file
  4. Update FIREBASE_BUCKET below with your project's storage bucket name
"""

import json
import time
import logging
import os
from datetime import datetime

from paths import LOCAL_EVENTS_DIR, ensure_dirs

logger = logging.getLogger("FirebaseUploader")

# ── Your Firebase config ─────────────────────────────────────────────────────
FIREBASE_KEY_PATH = "firebase_key.json"
FIREBASE_BUCKET   = "your-project-id.appspot.com"   # ← change this
COLLECTION_NAME   = "accident_events"


class FirebaseUploader:
    """
    Handles async upload of accident events to Firebase.
    If Firebase is not set up yet, it logs locally instead.
    """

    def __init__(self, camera_id: str, location: str):
        self.camera_id = camera_id
        self.location  = location
        self._db       = None
        self._bucket   = None
        self._enabled  = False
        self._init_firebase()

    def _init_firebase(self):
        """Initialize Firebase connection."""
        if not os.path.exists(FIREBASE_KEY_PATH):
            logger.warning(
                "firebase_key.json not found. "
                "Events will be logged locally only. "
                "Add your Firebase key to enable cloud upload."
            )
            return
        try:
            import firebase_admin
            from firebase_admin import credentials, firestore, storage

            if not firebase_admin._apps:
                cred = credentials.Certificate(FIREBASE_KEY_PATH)
                firebase_admin.initialize_app(cred, {
                    "storageBucket": FIREBASE_BUCKET
                })

            self._db      = firestore.client()
            self._bucket  = storage.bucket()
            self._enabled = True
            logger.info("Firebase connected successfully.")

        except ImportError:
            logger.warning(
                "firebase_admin not installed. "
                "Run: pip install firebase-admin"
            )
        except Exception as e:
            logger.error(f"Firebase init failed: {e}")

    def upload_event(self, event, snapshot_path: str):
        """
        Upload accident event to Firestore + snapshot to Storage.
        Call this in a background thread from stream_processor.py
        """
        timestamp_str = datetime.fromtimestamp(event.timestamp).isoformat()
        image_url     = ""

        # Upload snapshot image to Firebase Storage
        if self._enabled and os.path.exists(snapshot_path):
            try:
                blob_name  = f"snapshots/{self.camera_id}_{int(event.timestamp)}.jpg"
                blob       = self._bucket.blob(blob_name)
                blob.upload_from_filename(snapshot_path, content_type="image/jpeg")
                blob.make_public()
                image_url  = blob.public_url
                logger.info(f"Snapshot uploaded: {image_url}")
            except Exception as e:
                logger.error(f"Storage upload failed: {e}")

        # Build Firestore document
        event_doc = {
            "timestamp":             timestamp_str,
            "camera_id":             self.camera_id,
            "location":              self.location,
            "confidence_score":      event.confidence_score,
            "factors_triggered":     event.factors_triggered,
            "involved_vehicle_ids":  event.involved_vehicle_ids,
            "image_url":             image_url,
            "frame_number":          event.frame_num,
            "created_at":            time.time()
        }

        # Upload to Firestore
        if self._enabled:
            try:
                doc_ref = self._db.collection(COLLECTION_NAME).add(event_doc)
                logger.info(f"Event uploaded to Firestore: {doc_ref[1].id}")
            except Exception as e:
                logger.error(f"Firestore upload failed: {e}")
                self._save_local(event_doc)
        else:
            self._save_local(event_doc)

    def _save_local(self, event_doc: dict):
        """Fallback — save event to local JSON file if Firebase is unavailable."""
        ensure_dirs()
        filename = LOCAL_EVENTS_DIR / f"event_{int(event_doc['created_at'])}.json"
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(event_doc, f, indent=2)
        logger.info(f"Event saved locally: {filename}")
