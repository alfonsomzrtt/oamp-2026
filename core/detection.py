"""
core/detection.py — YOLO inference + block face classification thread.

Pipeline per frame:
  1. Threshold frame → binary image
  2. YOLO inference → bounding boxes
  3. classify_face() per box → face id (1-6)
  4. Geometric sort jika 4 box valid
  5. Annotate frame (rectangle + face label overlay)
  6. Push DetectionResult ke detection_queue
"""

from __future__ import annotations
import math
import queue
import threading
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np


# ── Face classification ───────────────────────────────────────────────────────

# Keep the same sampling order as the stable monolithic implementation:
# left, bottom, right, top.
_SAMPLE_POINTS = [(50, 25), (75, 50), (50, 75), (25, 50)]

# pixel pattern → face id
_FACE_PATTERNS: dict[tuple[int, ...], int] = {
    (0,   0,   0,   0  ): 1,
    (255, 255, 255, 255): 2,
    (255, 255, 0,   0  ): 3,
    (255, 0,   0,   255): 4,
    (0,   0,   255, 255): 5,
    (0,   255, 255, 0  ): 6,
}


def classify_face(
    img_thres: np.ndarray,
    x1: int, y1: int, x2: int, y2: int,
) -> int:
    """
    Classify wajah blok dari region di thresholded image.
    Pure function — thread-safe, tidak ada side effect.
    Return 0 jika bbox invalid atau pattern tidak dikenali.
    """
    h_img, w_img = img_thres.shape[:2]
    bh, bw = y2 - y1, x2 - x1
    if x1 < 0 or y1 < 0 or x2 > w_img or y2 > h_img or bh <= 0 or bw <= 0:
        return 0

    patch  = cv2.resize(img_thres[y1:y2, x1:x2], (100, 100), interpolation=cv2.INTER_AREA)
    sample = tuple(int(patch[r, c]) for r, c in _SAMPLE_POINTS)
    return _FACE_PATTERNS.get(sample, 0)


# ── Result container ──────────────────────────────────────────────────────────

@dataclass
class DetectionResult:
    """
    Immutable snapshot hasil satu frame.
    Dikirim dari DetectionThread ke main thread via detection_queue.
    """
    boxes:         list[tuple[int, int, int, int, float]]  # (x1,y1,x2,y2,conf)
    faces:         list[int]                                # face id per box
    pos_x:         list[int]                                # center_x per box
    pos_y:         list[int]                                # center_y per box
    sorted_design: list[int]                                # [f1,f2,f3,f4] atau []
    img_display:   np.ndarray                               # frame RGB terannotasi

    @property
    def block_count(self) -> int:
        return len(self.faces)

    @property
    def is_complete(self) -> bool:
        """True jika 4 blok valid terdeteksi dan sudah diurutkan."""
        return len(self.sorted_design) == 4


# ── Detection thread ──────────────────────────────────────────────────────────

class DetectionThread(threading.Thread):
    """
    Consumer frame_queue → Producer detection_queue.

    Semua heavy work (YOLO + classify + sort + annotate) selesai di sini
    sebelum hasil di-push ke detection_queue.
    Main thread hanya perlu read result dan update widget.
    """

    def __init__(
        self,
        model: Any,
        use_bantal_model: bool,
        frame_queue:      queue.Queue,
        detection_queue:  queue.Queue,
        *,
        yolo_infer_size:  int   = 640,
        conf_threshold:   float = 0.7,
        rect_tolerance:   int   = 100,   # toleransi perbedaan panjang sisi (px)
        face_assets:      tuple = (),    # ((face_img, mask), ...) index 0-based
    ):
        super().__init__(daemon=True, name="DetectionThread")
        self.model            = model
        self.use_bantal_model = use_bantal_model
        self.frame_queue      = frame_queue
        self.detection_queue  = detection_queue
        self.yolo_infer_size  = yolo_infer_size
        self.conf_threshold   = conf_threshold
        self.rect_tolerance   = rect_tolerance
        self.face_assets      = face_assets

        self._stop_event      = threading.Event()
        self._inference_count = 0

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def stop(self):
        self._stop_event.set()

    # ── Loop ──────────────────────────────────────────────────────────────────

    def run(self):
        print(">>> [DetectionThread] Started")
        while not self._stop_event.is_set():
            try:
                frame = self.frame_queue.get(timeout=0.05)
            except queue.Empty:
                continue
            try:
                result = self._process(frame)
                self._push(result)
            except Exception as e:
                print(f">>> [DetectionThread] Error: {e}")
        print(">>> [DetectionThread] Stopped")

    # ── Processing pipeline ───────────────────────────────────────────────────

    def _process(self, frame: np.ndarray) -> DetectionResult:
        # 1. Threshold
        blurred  = cv2.GaussianBlur(frame, (7, 7), 1)
        gray     = cv2.cvtColor(blurred, cv2.COLOR_RGB2GRAY)
        _, thres = cv2.threshold(gray, 175, 255, cv2.THRESH_BINARY)

        # 2. YOLO
        raw = self._infer(frame)
        self._inference_count += 1
        if self._inference_count % 60 == 0:
            print(f">>> [DetectionThread] {self._inference_count} inferences")

        # 3. Filter + classify + annotate
        annotated = frame.copy()
        boxes, faces, pos_x, pos_y = [], [], [], []

        for det in raw:
            x1, y1, x2, y2 = int(det[0]), int(det[1]), int(det[2]), int(det[3])
            conf = float(det[4])
            if conf <= self.conf_threshold:
                continue

            face_id = classify_face(thres, x1, y1, x2, y2)
            if face_id == 0:
                continue

            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            boxes.append((x1, y1, x2, y2, conf))
            faces.append(face_id)
            pos_x.append(cx)
            pos_y.append(cy)

            # Bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Face label overlay (50x50 top-left)
            self._draw_face_label(annotated, face_id, x1, y1)

        # 4. Geometric sort (hanya jika tepat 4 blok)
        sorted_design: list[int] = []
        if len(faces) == 4:
            sorted_design = self._sort_blocks(faces, pos_x, pos_y, annotated)

        return DetectionResult(
            boxes=boxes,
            faces=faces,
            pos_x=pos_x,
            pos_y=pos_y,
            sorted_design=sorted_design,
            img_display=annotated,
        )

    # ── YOLO inference ────────────────────────────────────────────────────────

    def _infer(self, frame: np.ndarray) -> list:
        """Normalize output ke list of [x1,y1,x2,y2,conf,...]."""
        if self.use_bantal_model:
            results = self.model(frame, verbose=False)
            if not results or not hasattr(results[0], "boxes"):
                return []
            out = []
            for b in results[0].boxes:
                x1, y1, x2, y2 = b.xyxy[0].cpu().numpy()
                conf = float(b.conf[0].cpu().numpy())
                out.append([x1, y1, x2, y2, conf, 0, ""])
            return out
        else:
            results = self.model(frame, size=self.yolo_infer_size)
            return results.pandas().xyxy[0].values.tolist()

    # ── Geometric sort ────────────────────────────────────────────────────────

    def _sort_blocks(
        self,
        faces: list[int],
        pos_x: list[int],
        pos_y: list[int],
        frame: np.ndarray,
    ) -> list[int]:
        """
        Sort 4 block: kiri → kanan, atas → bawah.
        Validasi rectangle shape sebelum sort.
        Draw connecting lines sebagai side effect di frame.
        Return [] jika shape tidak valid.
        """
        # Draw connections
        for i in range(4):
            for j in range(i + 1, 4):
                cv2.line(frame, (pos_x[i], pos_y[i]), (pos_x[j], pos_y[j]), (0, 0, 0), 2)

        # Validate: 4 sisi harus roughly equal
        lengths = sorted(
            math.sqrt((pos_x[i] - pos_x[j])**2 + (pos_y[i] - pos_y[j])**2)
            for i in range(4) for j in range(i + 1, 4)
        )
        sides = lengths[:4]
        if max(sides) - min(sides) > self.rect_tolerance:
            return []

        # Sort: pisah kiri/kanan via midpoint x, dalam grup sort by y
        sx  = sorted(pos_x)
        mid = (sx[1] + sx[2]) / 2
        indexed = sorted(
            enumerate(zip(pos_x, pos_y)),
            key=lambda t: (t[1][0] >= mid, t[1][1]),
        )
        return [faces[i] for i, _ in indexed]

    # ── Face label overlay ────────────────────────────────────────────────────

    def _draw_face_label(self, frame: np.ndarray, face_id: int, x1: int, y1: int):
        if not self.face_assets or face_id < 1 or face_id > len(self.face_assets):
            return
        face_img, face_mask = self.face_assets[face_id - 1]
        roi = frame[y1:y1 + 50, x1:x1 + 50]
        if roi.shape == (50, 50, 3):
            try:
                roi[np.where(face_mask)] = 0
                roi += face_img
            except Exception:
                pass

    # ── Queue push ────────────────────────────────────────────────────────────

    def _push(self, result: DetectionResult):
        """Keep result terbaru — drop yang lama jika penuh."""
        try:
            self.detection_queue.put_nowait(result)
        except queue.Full:
            try:
                self.detection_queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self.detection_queue.put_nowait(result)
            except queue.Full:
                pass