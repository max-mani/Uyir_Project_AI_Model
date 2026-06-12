# ============================================================
# UYIR — Pi Health Monitor
# Manikandan — Task M8
#
# Background thread. Every 30 seconds, sends Pi status to
# Firebase so the dashboard can show online/offline status,
# FPS, CPU temperature, and RAM usage per camera.
# ============================================================

import time
import threading
import logging
import config

logger = logging.getLogger("HealthMonitor")


class HealthMonitor:
    """
    Sends Pi health data to Firebase every N seconds.
    Runs in a daemon thread — stops automatically when main process exits.
    """

    def __init__(self, fps_provider=None):
        """
        fps_provider: callable that returns current FPS (float).
                      Pass a lambda from stream_processor.
        """
        self._fps_provider = fps_provider or (lambda: 0.0)
        self._running      = False
        self._thread       = None
        self._db           = None
        self._start_time   = time.time()
        self._init_firebase()

    def _init_firebase(self):
        if not __import__("os").path.exists(config.FIREBASE_KEY_PATH):
            return
        try:
            import firebase_admin
            from firebase_admin import firestore
            if firebase_admin._apps:
                self._db = firestore.client()
        except Exception:
            pass

    def start(self):
        self._running = True
        self._thread  = threading.Thread(
            target = self._loop, daemon=True
        )
        self._thread.start()
        logger.info("[Health] Monitor started.")

    def stop(self):
        self._running = False

    def _loop(self):
        while self._running:
            self._send()
            time.sleep(config.HEALTH_INTERVAL_SEC)

    def _send(self):
        data = {
            "camera_id": config.CAMERA_ID,
            "status"   : "online",
            "fps"      : round(self._fps_provider(), 2),
            "cpu_temp" : self._cpu_temp(),
            "ram_used" : self._ram_percent(),
            "uptime_s" : int(time.time() - self._start_time),
            "updated_at": time.time(),
        }

        if self._db:
            try:
                self._db \
                    .collection(config.HEALTH_COLLECTION) \
                    .document(config.CAMERA_ID) \
                    .set(data)
            except Exception as e:
                logger.warning(f"[Health] Firebase write failed: {e}")
                self._log_local(data)
        else:
            self._log_local(data)

    @staticmethod
    def _cpu_temp() -> float:
        """Read CPU temperature — works on Raspberry Pi."""
        try:
            with open("/sys/class/thermal/thermal_zone0/temp") as f:
                return round(int(f.read().strip()) / 1000.0, 1)
        except Exception:
            pass
        # Fallback for non-Pi (development machine)
        try:
            import psutil
            temps = psutil.sensors_temperatures()
            for key in ("coretemp", "cpu_thermal", "cpu-thermal"):
                if key in temps and temps[key]:
                    return round(temps[key][0].current, 1)
        except Exception:
            pass
        return 0.0

    @staticmethod
    def _ram_percent() -> float:
        try:
            import psutil
            return round(psutil.virtual_memory().percent, 1)
        except Exception:
            return 0.0

    @staticmethod
    def _log_local(data: dict):
        logger.info(
            f"[Health] status={data['status']} "
            f"fps={data['fps']} "
            f"cpu_temp={data['cpu_temp']}°C "
            f"ram={data['ram_used']}% "
            f"uptime={data['uptime_s']}s"
        )
