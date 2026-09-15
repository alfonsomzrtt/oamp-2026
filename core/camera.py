"""
core/camera.py — camera capture thread + software calibration.

CameraThread membaca frame dari cap, apply transforms,
lalu push ke frame_queue untuk dikonsumsi DetectionThread.

Cap management (open/release/switch) ada di sini juga
supaya tidak ada module lain yang perlu import cv2 untuk urusan kamera.
"""

from __future__ import annotations
import platform
import queue
import threading
import time
from typing import Optional

import cv2
import numpy as np


# ── Cap singleton ─────────────────────────────────────────────────────────────
# Diakses oleh CameraThread dan preview loop di App_Input.
# Gunakan open_camera() / release_camera() — jangan akses _cap langsung.

_cap: Optional[cv2.VideoCapture] = None
_cap_lock = threading.Lock()

# Props yang aman ditulis ke hardware (yang lain persist di driver — jangan sentuh)
_SAFE_CAP_PROPS = {cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT}


def open_camera(index: int) -> bool:
    """Buka kamera di index tertentu. Return True jika berhasil."""
    global _cap
    with _cap_lock:
        if _cap is not None and _cap.isOpened():
            _cap.release()
        if platform.system() == "Windows":
            _cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        else:
            _cap = cv2.VideoCapture(index)
        if _cap.isOpened():
            _cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
            _cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            print(f">>> [Camera] Opened index {index}")
            return True
        print(f">>> [Camera] Failed to open index {index}")
        return False


def release_camera():
    global _cap
    with _cap_lock:
        if _cap is not None:
            _cap.release()
            _cap = None


def read_frame() -> tuple[bool, Optional[np.ndarray]]:
    """Thread-safe frame read. Return (ret, frame)."""
    with _cap_lock:
        if _cap is None or not _cap.isOpened():
            return False, None
        return _cap.read()


def enumerate_cameras(max_index: int = 6) -> list[int]:
    """Coba buka setiap index, return list index yang berhasil."""
    available = []
    for i in range(max_index):
        try:
            if platform.system() == "Windows":
                c = cv2.VideoCapture(i, cv2.CAP_DSHOW)
            else:
                c = cv2.VideoCapture(i)
            if c.isOpened():
                ret, _ = c.read()
                if ret:
                    available.append(i)
            c.release()
        except Exception:
            pass
    return available


# ── Software calibration ──────────────────────────────────────────────────────

def apply_calibration(
    frame: np.ndarray,
    brightness: int = 128,
    contrast: int = 128,
    saturation: int = 128,
) -> np.ndarray:
    """
    Apply brightness/contrast/saturation sebagai software filter.
    Tidak ada hardware write — tidak persist di driver.
    Frame diasumsikan RGB.
    """
    if contrast != 128 or brightness != 128:
        alpha = contrast / 128.0
        beta  = brightness - 128
        frame = cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)

    if saturation != 128:
        hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
        hsv[:, :, 1] = np.clip(
            hsv[:, :, 1].astype(np.float32) * (saturation / 128.0),
            0, 255,
        ).astype(np.uint8)
        frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

    return frame


# ── Transform helpers ─────────────────────────────────────────────────────────

def apply_transforms(
    frame: np.ndarray,
    mirror_x: bool = False,
    mirror_y: bool = False,
    zoom: float = 1.0,
) -> np.ndarray:
    """Apply mirror dan zoom ke frame RGB."""
    if mirror_x:
        frame = cv2.flip(frame, 1)
    if mirror_y:
        frame = cv2.flip(frame, 0)
    if zoom != 1.0:
        h, w  = frame.shape[:2]
        nw, nh = int(w * zoom), int(h * zoom)
        frame  = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)
        if zoom > 1.0:
            sx = (nw - w) // 2
            sy = (nh - h) // 2
            frame = frame[sy:sy + h, sx:sx + w]
    return frame


# ── CameraThread ──────────────────────────────────────────────────────────────

class CameraThread(threading.Thread):
    """
    Producer thread: baca frame → apply transforms → push ke frame_queue.

    frame_queue berisi np.ndarray RGB, maxsize=2 (selalu frame terbaru).
    Main thread tidak perlu tahu soal cap.read() sama sekali.

    Config (mirror_x, zoom, dll.) bisa diupdate dari main thread kapan saja —
    assignment Python bersifat atomic untuk tipe primitif, aman tanpa lock.
    """

    def __init__(
        self,
        frame_queue: queue.Queue,
        *,
        mirror_x:    bool  = False,
        mirror_y:    bool  = False,
        zoom:        float = 1.0,
        calibration: bool  = False,
        brightness:  int   = 128,
        contrast:    int   = 128,
        saturation:  int   = 128,
        target_fps:  int   = 30,
    ):
        super().__init__(daemon=True, name="CameraThread")
        self.frame_queue = frame_queue

        # Mutable config — update langsung dari main thread
        self.mirror_x    = mirror_x
        self.mirror_y    = mirror_y
        self.zoom        = zoom
        self.calibration = calibration
        self.brightness  = brightness
        self.contrast    = contrast
        self.saturation  = saturation

        self._interval   = 1.0 / max(target_fps, 1)
        self._stop_event = threading.Event()
        self._fail_count = 0

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def stop(self):
        self._stop_event.set()

    @property
    def running(self) -> bool:
        return not self._stop_event.is_set()

    # ── Loop ──────────────────────────────────────────────────────────────────

    def run(self):
        print(">>> [CameraThread] Started")
        while not self._stop_event.is_set():
            t0 = time.monotonic()
            self._tick()
            elapsed = time.monotonic() - t0
            sleep   = self._interval - elapsed
            if sleep > 0:
                time.sleep(sleep)
        print(">>> [CameraThread] Stopped")

    def _tick(self):
        ret, frame = read_frame()
        if not ret or frame is None:
            self._fail_count += 1
            if self._fail_count > 60:
                print(">>> [CameraThread] Camera disconnected")
                self._stop_event.set()
            time.sleep(0.05)
            return
        self._fail_count = 0

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = apply_transforms(frame, self.mirror_x, self.mirror_y, self.zoom)
        if self.calibration:
            frame = apply_calibration(frame, self.brightness, self.contrast, self.saturation)

        self._push(frame)

    def _push(self, frame: np.ndarray):
        """Keep frame terbaru — drop yang lama jika queue penuh."""
        try:
            self.frame_queue.put_nowait(frame)
        except queue.Full:
            try:
                self.frame_queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self.frame_queue.put_nowait(frame)
            except queue.Full:
                pass