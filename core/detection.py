# ── core/detection_thread.py ─────────────────────────────────────────────────

import threading
import queue
import time
import cv2
import numpy as np
from typing import List


# Type alias: satu detection = [x1,y1,x2,y2,conf,cls_id,cls_name]
Detection = List

# Pixel sampler — 4 titik kardinal di dalam bounding box (50,25 | 75,50 | 50,75 | 25,50 di skala 100x100)
_SAMPLE_POINTS = [(50, 25), (75, 50), (50, 75), (25, 50)]

# Mapping pixel pattern → face id
_FACE_PATTERNS = {
    (0,   0,   0,   0  ): 1,
    (255, 255, 255, 255): 2,
    (255, 255, 0,   0  ): 3,
    (255, 0,   0,   255): 4,
    (0,   0,   255, 255): 5,
    (0,   255, 255, 0  ): 6,
}


def classify_face(img_thres: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> int:
    """
    Classify block face dari region di thresholded image.
    Return 0 jika tidak dikenali atau bbox invalid.
    Pure function — tidak ada side effect, bisa dipanggil dari thread manapun.
    """
    h_img, w_img = img_thres.shape[:2]
    if x1 < 0 or y1 < 0 or x2 > w_img or y2 > h_img:
        return 0
    bh, bw = y2 - y1, x2 - x1
    if bh <= 0 or bw <= 0:
        return 0

    patch = cv2.resize(img_thres[y1:y2, x1:x2], (100, 100), interpolation=cv2.INTER_AREA)
    sample = tuple(int(patch[r, c]) for r, c in _SAMPLE_POINTS)
    return _FACE_PATTERNS.get(sample, 0)


class DetectionResult:
    """Immutable result dari satu frame — dikirim ke main thread."""
    __slots__ = ("boxes", "faces", "pos_x", "pos_y", "sorted_design", "img_display")

    def __init__(self, boxes, faces, pos_x, pos_y, sorted_design, img_display):
        self.boxes          = boxes           # list of (x1,y1,x2,y2,conf)
        self.faces          = faces           # list of int (face id per box)
        self.pos_x          = pos_x           # list of int center_x
        self.pos_y          = pos_y           # list of int center_y
        self.sorted_design  = sorted_design   # [f1,f2,f3,f4] sudah diurutkan, atau []
        self.img_display    = img_display     # np.ndarray RGB — frame yang sudah diannotasi


class DetectionThread(threading.Thread):
    """
    Consumer frame_queue + Producer detection_queue.

    Melakukan:
      1. YOLO inference
      2. Threshold + face classification
      3. Geometric sort (kiri→kanan, atas→bawah)
      4. Frame annotation (rectangle + face label overlay)

    Hasil DetectionResult di-push ke detection_queue untuk main thread.
    """

    def __init__(
        self,
        model,
        use_bantal_model: bool,
        frame_queue: queue.Queue,
        detection_queue: queue.Queue,
        *,
        yolo_infer_size: int = 640,
        conf_threshold: float = 0.7,
        # Face label images untuk overlay (tuple of (img, mask) per face 1-6)
        face_assets: tuple = (),
    ):
        super().__init__(daemon=True, name="DetectionThread")
        self.model              = model
        self.use_bantal_model   = use_bantal_model
        self.frame_queue        = frame_queue
        self.detection_queue    = detection_queue
        self.yolo_infer_size    = yolo_infer_size
        self.conf_threshold     = conf_threshold
        self.face_assets        = face_assets  # [(face_img, mask), ...] index 1-based

        self._running = threading.Event()
        self._running.set()
        self._inference_count = 0

    # ── lifecycle ────────────────────────────────────────────────────────────

    def stop(self):
        self._running.clear()

    # ── main loop ────────────────────────────────────────────────────────────

    def run(self):
        while self._running.is_set():
            try:
                frame = self.frame_queue.get(timeout=0.05)
            except queue.Empty:
                continue
            except Exception:
                continue

            try:
                result = self._process(frame)
                self._push(result)
            except Exception as e:
                print(f">>> [DetectionThread] Error: {e}")

    def _process(self, frame: np.ndarray) -> DetectionResult:
        # ── 1. Threshold ─────────────────────────────────────────────────────
        blurred  = cv2.GaussianBlur(frame, (7, 7), 1)
        gray     = cv2.cvtColor(blurred, cv2.COLOR_RGB2GRAY)
        _, thres = cv2.threshold(gray, 175, 255, cv2.THRESH_BINARY)

        # ── 2. YOLO inference ─────────────────────────────────────────────────
        raw_detections = self._infer(frame)
        self._inference_count += 1

        # ── 3. Filter + classify ──────────────────────────────────────────────
        boxes, faces, pos_x, pos_y = [], [], [], []
        annotated = frame.copy()

        for det in raw_detections:
            x1, y1, x2, y2, conf = (
                int(det[0]), int(det[1]), int(det[2]), int(det[3]), float(det[4])
            )
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

            # Annotate: bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Annotate: face label overlay (top-left 50x50)
            if self.face_assets and 1 <= face_id <= len(self.face_assets):
                face_img, face_mask = self.face_assets[face_id - 1]
                roi = annotated[y1:y1 + 50, x1:x1 + 50]
                if roi.shape == (50, 50, 3):
                    try:
                        roi[np.where(face_mask)] = 0
                        roi += face_img
                    except Exception:
                        pass

        # ── 4. Geometric sort (hanya jika 4 box) ─────────────────────────────
        sorted_design = []
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

    def _infer(self, frame: np.ndarray) -> list:
        """Return list of [x1,y1,x2,y2,conf,...] — normalized ke format yang sama."""
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

    def _sort_blocks(self, faces, pos_x, pos_y, frame) -> list:
        """
        Sort 4 block: kiri → kanan, atas → bawah.
        Draw connecting lines sebagai side effect di frame.
        """
        # Draw connections
        pts = [(pos_x[i], pos_y[i]) for i in range(4)]
        for i in range(4):
            for j in range(i + 1, 4):
                cv2.line(frame, pts[i], pts[j], (0, 0, 0), 2)

        # Validate rectangle-ish shape
        import math
        lengths = []
        for i in range(4):
            for j in range(i + 1, 4):
                d = math.sqrt((pos_x[i]-pos_x[j])**2 + (pos_y[i]-pos_y[j])**2)
                lengths.append(d)
        lengths.sort()
        # 4 sisi harus roughly equal
        sides = lengths[:4]
        if max(sides) - min(sides) > 100:
            return []

        # Sort: pisah kiri/kanan via midpoint, dalam grup sort by y
        sx = sorted(pos_x)
        mid = (sx[1] + sx[2]) / 2
        indexed = sorted(
            enumerate(zip(pos_x, pos_y)),
            key=lambda t: (t[1][0] >= mid, t[1][1])
        )
        return [faces[i] for i, _ in indexed]

    def _push(self, result: DetectionResult):
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