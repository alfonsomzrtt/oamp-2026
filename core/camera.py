# ── core/camera_thread.py ─────────────────────────────────────────────────────

import threading
import queue
import time
import cv2
import numpy as np


class CameraThread(threading.Thread):
    """
    Producer: baca frame dari cap, apply transforms, push ke frame_queue.
    Main thread tidak perlu tahu soal cap.read() sama sekali.
    """

    def __init__(
        self,
        frame_queue: queue.Queue,
        *,
        mirror_x: bool = False,
        mirror_y: bool = False,
        zoom: float = 1.0,
        calibration: bool = False,
        brightness: int = 128,
        contrast: int = 128,
        saturation: int = 128,
        target_fps: int = 30,
    ):
        super().__init__(daemon=True, name="CameraThread")
        self.frame_queue = frame_queue

        # config — bisa di-update dari main thread via property (thread-safe karena GIL + atomic assign)
        self.mirror_x   = mirror_x
        self.mirror_y   = mirror_y
        self.zoom       = zoom
        self.calibration = calibration
        self.brightness = brightness
        self.contrast   = contrast
        self.saturation = saturation

        self._frame_interval = 1.0 / target_fps
        self._running = threading.Event()
        self._running.set()
        self._fail_count = 0

    # ── lifecycle ────────────────────────────────────────────────────────────

    def stop(self):
        self._running.clear()

    # ── main loop ────────────────────────────────────────────────────────────

    def run(self):
        while self._running.is_set():
            t0 = time.monotonic()
            self._tick()
            elapsed = time.monotonic() - t0
            sleep = self._frame_interval - elapsed
            if sleep > 0:
                time.sleep(sleep)

    def _tick(self):
        from main import cap  # import cap global; atau inject via constructor
        if cap is None or not cap.isOpened():
            time.sleep(0.1)
            return

        ret, frame = cap.read()
        if not ret or frame is None:
            self._fail_count += 1
            if self._fail_count > 50:
                self._running.clear()  # signal: kamera mati
            return
        self._fail_count = 0

        frame = self._apply_transforms(frame)
        self._push(frame)

    def _apply_transforms(self, frame: np.ndarray) -> np.ndarray:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if self.mirror_x:
            frame = cv2.flip(frame, 1)
        if self.mirror_y:
            frame = cv2.flip(frame, 0)

        zoom = self.zoom
        if zoom != 1.0:
            h, w = frame.shape[:2]
            nw, nh = int(w * zoom), int(h * zoom)
            frame = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)
            if zoom > 1.0:
                sx, sy = (nw - w) // 2, (nh - h) // 2
                frame = frame[sy:sy + h, sx:sx + w]

        if self.calibration:
            frame = _apply_software_calibration(
                frame, self.brightness, self.contrast, self.saturation
            )
        return frame

    def _push(self, frame: np.ndarray):
        """Selalu keep frame terbaru — drop yang lama."""
        try:
            self.frame_queue.put_nowait(frame)
        except queue.Full:
            try:
                self.frame_queue.get_nowait()   # buang frame lama
            except queue.Empty:
                pass
            try:
                self.frame_queue.put_nowait(frame)
            except queue.Full:
                pass