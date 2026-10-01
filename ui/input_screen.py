"""
ui/input_screen.py — App_Input: participant registration + camera setup screen.
"""

from __future__ import annotations
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import customtkinter
from PIL import Image, ImageTk

from config import (
    CAMERA_INDEX, CAMERA_MIRROR_X, CAMERA_MIRROR_Y, CAMERA_ZOOM,
    CAMERA_CALIBRATION, CAMERA_BRIGHTNESS, CAMERA_CONTRAST, CAMERA_SATURATION,
    SLIDER_BOUNDS, remap_slider, MAX_LEVEL,
)
from state import STATE
from ui.theme import CLR, FONT, RADIUS
from ui.widgets import UIDEntryFrame
from core.camera import (
    open_camera, release_camera, read_frame,
    enumerate_cameras, apply_calibration, apply_transforms,
)
from core.game_logic import save_env_value
from api.participants import verify as verify_participant
from api.tournament import check_active_match


class InputScreen(customtkinter.CTk):
    """
    Layar pertama setelah consent.
    Menangani: UID scan/entry, kamera preview, kalibrasi, mode selection.
    Setelah destroy(), STATE sudah terisi nick_name / uid / mode.
    """

    def __init__(self, model=None, face_assets=()):
        super().__init__()
        self._model = model
        self._face_assets = face_assets
        self.user_cancelled = False
        self.title("Block Design Test")
        self.geometry("640x780")
        self.configure(fg_color=CLR.BG)
        self.minsize(480, 600)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        self._preview_running  = False
        self._current_imgtk   = None
        self._detecting_cams   = True
        self._available_cams: list[int] = []
        self._cek_in_progress  = False
        self._block_test_active = False
        self._last_detections: list = []
        self._qr_detector      = None
        self._qr_available     = False
        self._qr_last_ts       = 0.0
        self._qr_cooldown      = 3.0

        # mirror state (local, synced to STATE on save)
        self._mirror_x = STATE.camera_mirror_x
        self._mirror_y = STATE.camera_mirror_y

        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)

        self._build_title_bar()
        self._build_content()
        self._init_qr()
        self._update_mode_buttons()

    # ── Title bar ─────────────────────────────────────────────────────────────

    def _build_title_bar(self):
        bar = customtkinter.CTkFrame(self, fg_color=CLR.CARD, corner_radius=0, height=64)
        bar.grid(row=0, column=0, sticky="ew")
        bar.grid_columnconfigure(0, weight=1)
        bar.grid_propagate(False)

        customtkinter.CTkLabel(
            bar, text="BLOCK DESIGN TEST",
            font=(FONT.P, 22, "bold"), text_color=CLR.TEXT,
        ).grid(row=0, column=0, padx=20, sticky="w")

        customtkinter.CTkLabel(
            bar, text="Otak Atik Tournament",
            font=(FONT.P, 12), text_color=CLR.ACCENT,
        ).grid(row=1, column=0, padx=20, sticky="w")

        actions = customtkinter.CTkFrame(bar, fg_color="transparent")
        actions.grid(row=0, column=1, rowspan=2, padx=(0, 16), sticky="e")

        self._btn_training = customtkinter.CTkButton(
            actions, text="Latihan",
            command=lambda: self._on_mode_change("training"),
            font=(FONT.P, 11, "bold"),
            corner_radius=RADIUS.SMALL, height=32, width=72,
        )
        self._btn_training.pack(side="left", padx=(0, 2))

        self._btn_competition = customtkinter.CTkButton(
            actions, text="Kompetisi",
            command=lambda: self._on_mode_change("competition"),
            font=(FONT.P, 11, "bold"),
            corner_radius=RADIUS.SMALL, height=32, width=80,
        )
        self._btn_competition.pack(side="left", padx=(0, 10))

        theme_label = "Light" if STATE.current_mode == "dark" else "Dark"
        self._theme_btn = customtkinter.CTkButton(
            actions, text=theme_label,
            command=self._toggle_theme,
            font=(FONT.P, 11, "bold"),
            fg_color=CLR.BG, hover_color=CLR.SUBTLE_BORDER,
            text_color=CLR.TEXT,
            corner_radius=RADIUS.PILL, height=32, width=64,
        )
        self._theme_btn.pack(side="left")

    # ── Content area ──────────────────────────────────────────────────────────

    def _build_content(self):
        content = customtkinter.CTkFrame(self, fg_color=CLR.BG, corner_radius=0)
        content.grid(row=1, column=0, sticky="nsew")
        content.grid_columnconfigure(0, weight=1)
        content.grid_rowconfigure(1, weight=1)
        self._content = content

        px = 20
        self._build_uid_section(content, px)
        self._build_camera_section(content, px)
        self._build_action_buttons(content, px)

    def _build_uid_section(self, parent, px):
        card = customtkinter.CTkFrame(
            parent, fg_color=CLR.CARD, corner_radius=RADIUS.DEFAULT,
            border_width=1, border_color=CLR.SUBTLE_BORDER,
        )
        card.grid(row=0, column=0, sticky="ew", padx=px, pady=(16, 10))

        top = customtkinter.CTkFrame(card, fg_color="transparent")
        top.grid(row=0, column=0, sticky="ew", padx=14, pady=(14, 0))
        top.grid_columnconfigure(1, weight=1)

        customtkinter.CTkLabel(
            top, text="UID", font=(FONT.P, 13, "bold"),
            text_color=CLR.ACCENT, width=32,
        ).grid(row=0, column=0, padx=(0, 8))

        self._uid_entry = UIDEntryFrame(top)
        self._uid_entry.grid(row=0, column=1, sticky="ew", padx=(0, 8))
        self._uid_entry.bind_return(self._cek_uid)

        self._cek_btn = customtkinter.CTkButton(
            top, text="CEK", command=self._cek_uid,
            font=(FONT.P, 13, "bold"),
            fg_color=CLR.ACCENT, hover_color=CLR.ACCENT2,
            text_color=CLR.TEXT, corner_radius=RADIUS.DEFAULT,
            height=40, width=64,
        )
        self._cek_btn.grid(row=0, column=2)

        self._qr_indicator = customtkinter.CTkLabel(
            top, text="", font=(FONT.P, 10, "bold"),
            text_color=CLR.MUTED, width=56,
        )
        self._qr_indicator.grid(row=0, column=3, padx=(8, 0))

        self._uid_status = customtkinter.CTkLabel(
            card, text="", font=(FONT.P, 11), text_color=CLR.MUTED,
        )
        self._uid_status.grid(row=1, column=0, padx=16, pady=(4, 0), sticky="w")

        self._info_box = customtkinter.CTkFrame(card, fg_color="transparent")
        self._info_box.grid(row=2, column=0, sticky="ew", padx=14, pady=(4, 10))
        self._info_box.grid_columnconfigure(0, weight=1)
        self._info_box.grid_remove()

        self._participant_info = customtkinter.CTkLabel(
            self._info_box, text="Belum ada peserta",
            font=(FONT.P, 13), text_color=CLR.MUTED,
        )
        self._participant_info.grid(row=0, column=0, sticky="w")

        self._reset_btn = customtkinter.CTkButton(
            self._info_box, text="Ganti",
            command=self._reset_scan,
            font=(FONT.P, 11), fg_color="transparent",
            hover_color=CLR.SUBTLE_BORDER, text_color=CLR.MUTED,
            height=24, width=56, corner_radius=RADIUS.SMALL,
        )
        self._reset_btn.grid(row=0, column=1, sticky="e")

    def _build_camera_section(self, parent, px):
        cam_card = customtkinter.CTkFrame(
            parent, fg_color=CLR.CARD, corner_radius=RADIUS.DEFAULT,
            border_width=1, border_color=CLR.SUBTLE_BORDER,
        )
        cam_card.grid(row=1, column=0, sticky="nsew", padx=px, pady=(0, 10))
        cam_card.grid_columnconfigure(0, weight=1)
        cam_card.grid_rowconfigure(0, weight=1)

        # Preview label
        self._preview_lbl = customtkinter.CTkLabel(
            cam_card, text="Memulai kamera...",
            font=(FONT.P, 13), text_color=CLR.MUTED,
            fg_color="#111111", corner_radius=RADIUS.DEFAULT,
        )
        self._preview_lbl.grid(row=0, column=0, sticky="nsew", padx=12, pady=(12, 0))

        # Hint row
        hint = customtkinter.CTkFrame(cam_card, fg_color="transparent")
        hint.grid(row=1, column=0, sticky="ew", padx=12, pady=(4, 0))
        hint.grid_columnconfigure(0, weight=1)

        self._qr_hint = customtkinter.CTkLabel(
            hint, text="", font=(FONT.P, 10), text_color=CLR.MUTED,
        )
        self._qr_hint.grid(row=0, column=0, sticky="w")

        self._cam_status = customtkinter.CTkLabel(
            hint, text="Memuat...", font=(FONT.P, 11), text_color=CLR.MUTED,
        )
        self._cam_status.grid(row=0, column=1, sticky="e", padx=(12, 0))

        self._detect_result = customtkinter.CTkLabel(
            cam_card, text="", font=(FONT.P, 12),
            text_color=CLR.MUTED, wraplength=400,
        )
        self._detect_result.grid(row=2, column=0, padx=12, pady=(2, 0), sticky="w")

        self._build_camera_controls(cam_card)
        self._build_calibration(cam_card)

    def _build_camera_controls(self, parent):
        ctrl = customtkinter.CTkFrame(parent, fg_color="transparent")
        ctrl.grid(row=3, column=0, sticky="ew", padx=12, pady=(8, 12))

        self._cam_dropdown = customtkinter.CTkComboBox(
            ctrl, values=["Mendeteksi..."],
            command=self._on_camera_change,
            font=(FONT.P, 11), fg_color=CLR.BG,
            border_color=CLR.SUBTLE_BORDER, height=34, width=110,
        )
        self._cam_dropdown.pack(side="left", padx=(0, 6))

        self._mirror_x_btn = customtkinter.CTkButton(
            ctrl, text="↔", command=self._toggle_mirror_x,
            font=(FONT.P, 13, "bold"),
            fg_color=CLR.BG if self._mirror_x else CLR.CARD,
            hover_color=CLR.SUBTLE_BORDER, text_color=CLR.TEXT,
            corner_radius=RADIUS.SMALL, height=34, width=38,
        )
        self._mirror_x_btn.pack(side="left", padx=(0, 6))

        for z in [1.0, 1.5, 2.0]:
            customtkinter.CTkButton(
                ctrl, text=f"{z:.1f}x",
                command=lambda v=z: self._set_zoom(v),
                font=(FONT.P, 11, "bold"),
                fg_color=CLR.BG, hover_color=CLR.SUBTLE_BORDER,
                text_color=CLR.TEXT, corner_radius=RADIUS.SMALL,
                height=34, width=42,
            ).pack(side="left", padx=(0, 4))

        self._mirror_y_btn = customtkinter.CTkButton(
            ctrl, text="↕", command=self._toggle_mirror_y,
            font=(FONT.P, 13, "bold"),
            fg_color=CLR.BG if self._mirror_y else CLR.CARD,
            hover_color=CLR.SUBTLE_BORDER, text_color=CLR.TEXT,
            corner_radius=RADIUS.SMALL, height=34, width=38,
        )
        self._mirror_y_btn.pack(side="left", padx=(0, 2))

        customtkinter.CTkLabel(
            ctrl, text="Zoom", font=(FONT.P, 10), text_color=CLR.MUTED,
        ).pack(side="left", padx=(0, 2))

        self._zoom_slider = customtkinter.CTkSlider(
            ctrl, from_=0.5, to=3.0, number_of_steps=25,
            command=self._on_zoom_change, width=100,
        )
        self._zoom_slider.set(STATE.camera_zoom)
        self._zoom_slider.pack(side="left", padx=(0, 4))

        self._zoom_label = customtkinter.CTkLabel(
            ctrl, text=f"{STATE.camera_zoom:.1f}x",
            font=(FONT.P, 11, "bold"), text_color=CLR.ACCENT, width=36,
        )
        self._zoom_label.pack(side="left", padx=(0, 6))

        customtkinter.CTkButton(
            ctrl, text="Simpan",
            command=self._save_camera_settings,
            font=(FONT.P, 10, "bold"),
            fg_color=CLR.ACCENT, hover_color=CLR.ACCENT2,
            text_color=CLR.TEXT, corner_radius=RADIUS.SMALL,
            height=30, width=60,
        ).pack(side="left", padx=(0, 8))

        self._detect_btn = customtkinter.CTkButton(
            ctrl, text="Tes Blok",
            command=self._toggle_block_test,
            font=(FONT.P, 11, "bold"),
            fg_color=CLR.SUCCESS, hover_color=CLR.SUCCESS_HOVER,
            text_color=CLR.TEXT, corner_radius=RADIUS.SMALL, height=34,
        )
        self._detect_btn.pack(side="right")

    def _build_calibration(self, parent):
        calib = customtkinter.CTkFrame(parent, fg_color="transparent")
        calib.grid(row=4, column=0, sticky="ew", padx=12, pady=(0, 8))
        calib.grid_columnconfigure(0, weight=1)

        self._calib_var = customtkinter.BooleanVar(value=STATE.camera_calibration)
        customtkinter.CTkCheckBox(
            calib, text="Kalibrasi Kamera",
            variable=self._calib_var, command=self._on_calib_toggle,
            font=(FONT.P, 11, "bold"),
            fg_color=CLR.ACCENT, hover_color=CLR.ACCENT2,
        ).grid(row=0, column=0, sticky="w", pady=(0, 6))

        sliders = customtkinter.CTkFrame(calib, fg_color="transparent")
        sliders.grid(row=1, column=0, sticky="ew")

        for ci, (label, attr, val) in enumerate([
            ("Kecerahan", "brightness", STATE.camera_brightness),
            ("Kontras",   "contrast",   STATE.camera_contrast),
            ("Saturasi",  "saturation", STATE.camera_saturation),
        ]):
            sliders.grid_columnconfigure(ci * 3 + 1, weight=1)
            customtkinter.CTkLabel(
                sliders, text=label,
                font=(FONT.P, 10, "bold"), text_color=CLR.MUTED,
            ).grid(row=0, column=ci * 3, padx=(0 if ci == 0 else 8, 4))

            s = customtkinter.CTkSlider(
                sliders, from_=0, to=255, number_of_steps=255,
                width=80, command=lambda v, a=attr: self._on_quality_slider(a, v),
            )
            s.set(val)
            s.grid(row=0, column=ci * 3 + 1, sticky="ew")
            setattr(self, f"_slider_{attr}", s)

            lbl = customtkinter.CTkLabel(
                sliders, text=f"{round(val / 255 * 100)}%",
                font=(FONT.P, 10, "bold"), text_color=CLR.ACCENT, width=28,
            )
            lbl.grid(row=0, column=ci * 3 + 2, padx=(4, 0))
            setattr(self, f"_slider_{attr}_lbl", lbl)

        self._set_calib_sliders_state()

    def _build_action_buttons(self, parent, px):
        self._start_btn = customtkinter.CTkButton(
            parent, text="MULAI TES",
            command=self._on_start,
            font=(FONT.P, 20, "bold"),
            fg_color=CLR.ACCENT, hover_color=CLR.ACCENT2,
            text_color=CLR.TEXT, corner_radius=RADIUS.DEFAULT, height=56,
        )
        self._start_btn.grid(row=2, column=0, sticky="ew", padx=px, pady=(0, 14))

        self._mode_frame = customtkinter.CTkFrame(parent, fg_color="transparent")
        self._mode_frame.grid(row=2, column=0, sticky="ew", padx=px, pady=(0, 14))
        self._mode_frame.grid_columnconfigure((0, 1), weight=1)
        self._mode_frame.grid_remove()

        self._duel_btn = customtkinter.CTkButton(
            self._mode_frame, text="Duel 1v1",
            command=self._start_duel,
            font=(FONT.P, 14, "bold"),
            fg_color=CLR.ACCENT, hover_color=CLR.ACCENT2,
            text_color=CLR.TEXT, corner_radius=RADIUS.DEFAULT, height=56,
        )
        self._duel_btn.grid(row=0, column=0, padx=(0, 6), sticky="ew")

        self._tournament_btn = customtkinter.CTkButton(
            self._mode_frame, text="Turnamen",
            command=self._start_tournament,
            font=(FONT.P, 14, "bold"),
            fg_color=CLR.CARD, hover_color=CLR.ACCENT,
            text_color=CLR.TEXT, corner_radius=RADIUS.DEFAULT, height=56,
        )
        self._tournament_btn.grid(row=0, column=1, padx=(6, 0), sticky="ew")

    # ── QR init ───────────────────────────────────────────────────────────────

    def _init_qr(self):
        try:
            if hasattr(cv2, "QRCodeDetector"):
                self._qr_detector = cv2.QRCodeDetector()
                self._qr_available = True
                self._qr_indicator.configure(text="● QR", text_color=CLR.SUCCESS)
            else:
                self._qr_indicator.configure(text="✗ QR", text_color=CLR.DANGER)
        except Exception as e:
            print(f">>> [QR] Init failed: {e}")
            self._qr_indicator.configure(text="✗ QR", text_color=CLR.DANGER)

    # ── Camera preview ────────────────────────────────────────────────────────

    def start_camera_preview(self):
        self._preview_running = True
        threading.Thread(target=self._detect_cams_then_preview, daemon=True).start()

    def _detect_cams_then_preview(self):
        available = enumerate_cameras()
        self._available_cams = available
        if available:
            labels = [f"Kamera {i}" for i in available]
            self.after(0, lambda: self._update_cam_dropdown(labels, available))
        else:
            self.after(0, lambda: self._update_cam_dropdown(["Tidak ada kamera"], []))
        open_camera(STATE.camera_index)
        threading.Thread(target=self._preview_loop, daemon=True).start()

    def _update_cam_dropdown(self, labels, indices):
        self._detecting_cams = False
        self._cam_dropdown.configure(values=labels)
        if indices:
            idx = STATE.camera_index
            sel = indices.index(idx) if idx in indices else 0
            self._cam_dropdown.set(labels[sel])
            self._cam_status.configure(
                text=f"Kamera aktif — {len(indices)} terdeteksi",
                text_color=CLR.SUCCESS,
            )
        else:
            self._cam_dropdown.set("Tidak ada kamera")
            self._cam_status.configure(text="Tidak ada kamera", text_color=CLR.DANGER)

    def _preview_loop(self):
        qframe = 0
        while self._preview_running:
            try:
                ret, frame = read_frame()
                if not ret or frame is None:
                    time.sleep(0.05)
                    continue

                cw = max(self._preview_lbl.winfo_width(),  400)
                ch = max(self._preview_lbl.winfo_height(), 300)

                # QR detection (every 5 frames, BGR so we read before cvtColor)
                qframe += 1
                if (self._qr_available and qframe % 5 == 0
                        and time.time() - self._qr_last_ts > self._qr_cooldown):
                    try:
                        data, _, _ = self._qr_detector.detectAndDecode(frame)
                        if data and data.strip():
                            self._qr_last_ts = time.time()
                            self.after(0, lambda u=data.strip(): self._on_qr_detected(u))
                    except Exception:
                        pass

                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = apply_transforms(
                    frame, STATE.camera_mirror_x, STATE.camera_mirror_y, STATE.camera_zoom,
                )
                if STATE.camera_calibration:
                    frame = apply_calibration(
                        frame, STATE.camera_brightness,
                        STATE.camera_contrast, STATE.camera_saturation,
                    )

                if self._block_test_active:
                    frame = self._run_block_test_overlay(frame)

                # Letterbox resize
                h_f, w_f = frame.shape[:2]
                scale = min(cw / w_f, ch / h_f)
                nw, nh = int(w_f * scale), int(h_f * scale)
                img    = Image.fromarray(frame).resize((nw, nh), Image.LANCZOS)
                imgtk  = ImageTk.PhotoImage(img)
                self._current_imgtk = imgtk
                self.after(0, lambda im=imgtk: self._preview_lbl.configure(image=im))

            except Exception:
                pass
            time.sleep(0.03)

    def _run_block_test_overlay(self, frame: np.ndarray) -> np.ndarray:
        """Block detection overlay untuk mode tes — dijalankan setiap 15 frame."""
        from config import USE_BANTAL_MODEL, YOLO_INFER_SIZE

        if not hasattr(self, "_btest_frame_n"):
            self._btest_frame_n = 0
        self._btest_frame_n += 1

        if self._btest_frame_n % 15 == 0:
            try:
                if self._model is None:
                    raise RuntimeError("Model deteksi belum tersedia")
                if USE_BANTAL_MODEL:
                    res = self._model(frame, verbose=False)
                    self._last_detections = []
                    if res and hasattr(res[0], "boxes"):
                        for b in res[0].boxes:
                            x1, y1, x2, y2 = b.xyxy[0].cpu().numpy()
                            self._last_detections.append(
                                [x1, y1, x2, y2, float(b.conf[0].cpu().numpy())]
                            )
                else:
                    res = self._model(frame, size=YOLO_INFER_SIZE)
                    self._last_detections = [
                        r[:5] for r in res.pandas().xyxy[0].values.tolist()
                    ]
            except Exception:
                self._last_detections = []

        blurred = cv2.GaussianBlur(frame, (7, 7), 1)
        gray    = cv2.cvtColor(blurred, cv2.COLOR_RGB2GRAY)
        _, thres = cv2.threshold(gray, 175, 255, cv2.THRESH_BINARY)

        from core.detection import classify_face
        box_count = 0
        design    = []
        confidence_scores = []
        positions = []

        for det in self._last_detections:
            x1, y1, x2, y2, conf = int(det[0]), int(det[1]), int(det[2]), int(det[3]), float(det[4])
            if conf <= 0.7:
                continue
            confidence_scores.append(conf)
            fid = classify_face(thres, x1, y1, x2, y2)
            if fid == 0:
                continue
            design.append(fid)
            box_count += 1
            positions.append(((x1 + x2) // 2, (y1 + y2) // 2))
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                frame, f"Face {fid}", (x1, max(y1 - 8, 18)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2,
            )
            self._draw_face_label(frame, fid, x1, y1)

        confidence = (
            sum(confidence_scores) / len(confidence_scores)
            if confidence_scores else 0.0
        )
        result_text = f"Blok: {box_count} | Confidence: {confidence:.2f}"
        if len(design) == 4:
            for i in range(4):
                for j in range(i + 1, 4):
                    cv2.line(frame, positions[i], positions[j], (0, 0, 0), 2)
            order = sorted(
                range(4),
                key=lambda i: (positions[i][0] >= (
                    sorted(x for x, _ in positions)[1]
                    + sorted(x for x, _ in positions)[2]
                ) / 2, positions[i][1]),
            )
            sorted_design = [design[i] for i in order]
            result_text += f" | Face: {design} | Urutan: {sorted_design}"

        cv2.putText(frame, f"Blok: {box_count}", (12, 36),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        cv2.putText(frame, f"Confidence: {confidence:.2f}", (12, 72),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        ok = len(design) == 4
        self.after(0, lambda t=result_text, o=ok: self._detect_result.configure(
            text=t, text_color=CLR.SUCCESS if o else CLR.MUTED,
        ))
        return frame

    def _draw_face_label(self, frame: np.ndarray, face_id: int, x1: int, y1: int):
        if not self._face_assets or face_id < 1 or face_id > len(self._face_assets):
            return
        face_img, face_mask = self._face_assets[face_id - 1]
        roi = frame[y1:y1 + 50, x1:x1 + 50]
        if roi.shape == (50, 50, 3):
            try:
                roi[np.where(face_mask)] = 0
                roi += face_img
            except Exception:
                pass

    # ── Camera controls ───────────────────────────────────────────────────────

    def _on_camera_change(self, choice):
        if self._detecting_cams or not self._available_cams:
            return
        idx = self._available_cams[
            list(self._cam_dropdown.cget("values")).index(choice)
        ]
        STATE.camera_index = idx
        open_camera(idx)

    def _toggle_mirror_x(self):
        self._mirror_x = not self._mirror_x
        STATE.camera_mirror_x = self._mirror_x
        self._mirror_x_btn.configure(
            fg_color=CLR.BG if self._mirror_x else CLR.CARD
        )

    def _toggle_mirror_y(self):
        self._mirror_y = not self._mirror_y
        STATE.camera_mirror_y = self._mirror_y
        self._mirror_y_btn.configure(
            fg_color=CLR.BG if self._mirror_y else CLR.CARD
        )

    def _on_zoom_change(self, value):
        STATE.camera_zoom = round(float(value), 1)
        self._zoom_label.configure(text=f"{STATE.camera_zoom:.1f}x")

    def _set_zoom(self, value):
        self._zoom_slider.set(value)
        self._on_zoom_change(value)

    def _on_calib_toggle(self):
        STATE.camera_calibration = self._calib_var.get()
        self._set_calib_sliders_state()

    def _set_calib_sliders_state(self):
        s = "normal" if self._calib_var.get() else "disabled"
        for attr in ("brightness", "contrast", "saturation"):
            sl = getattr(self, f"_slider_{attr}", None)
            if sl:
                sl.configure(state=s)

    def _on_quality_slider(self, attr: str, value):
        val = int(round(float(value)))
        lbl = getattr(self, f"_slider_{attr}_lbl", None)
        if lbl:
            lbl.configure(text=f"{round(val / 255 * 100)}%")
        remapped = remap_slider(val, attr)
        if attr == "brightness":
            STATE.camera_brightness = remapped
        elif attr == "contrast":
            STATE.camera_contrast = remapped
        elif attr == "saturation":
            STATE.camera_saturation = remapped

    def _save_camera_settings(self):
        save_env_value("CAMERA_INDEX",      str(STATE.camera_index))
        save_env_value("CAMERA_MIRROR_X",   str(STATE.camera_mirror_x).lower())
        save_env_value("CAMERA_MIRROR_Y",   str(STATE.camera_mirror_y).lower())
        save_env_value("CAMERA_ZOOM",       str(STATE.camera_zoom))
        save_env_value("CAMERA_CALIBRATION", str(STATE.camera_calibration).lower())
        save_env_value("CAMERA_BRIGHTNESS", str(int(round(self._slider_brightness.get()))))
        save_env_value("CAMERA_CONTRAST",   str(int(round(self._slider_contrast.get()))))
        save_env_value("CAMERA_SATURATION", str(int(round(self._slider_saturation.get()))))
        import tkinter.messagebox as mb
        mb.showinfo("Tersimpan", "Pengaturan kamera disimpan ke .env")

    def _toggle_block_test(self):
        self._block_test_active = not self._block_test_active
        self._last_detections   = []
        if self._block_test_active:
            self._detect_btn.configure(text="Stop Tes", fg_color=CLR.DANGER)
            self._detect_result.configure(
                text="Deteksi aktif — blok di-highlight",
                text_color=CLR.SUCCESS,
            )
        else:
            self._detect_btn.configure(text="Tes Blok", fg_color=CLR.SUCCESS)
            self._detect_result.configure(text="")

    # ── QR ────────────────────────────────────────────────────────────────────

    def _on_qr_detected(self, uid: str):
        if self._uid_entry.get() == uid:
            return
        print(f">>> [QR] Detected: {uid}")
        self._uid_entry.set(uid)
        self._qr_indicator.configure(text="✓ QR", text_color=CLR.SUCCESS)
        self._cek_uid()

    # ── UID verification ──────────────────────────────────────────────────────

    def _cek_uid(self):
        if self._cek_in_progress:
            return
        uid = self._uid_entry.get()
        if not uid:
            self._uid_status.configure(text="Masukkan UID dulu", text_color=CLR.DANGER)
            return
        self._cek_in_progress = True
        self._cek_btn.configure(state="disabled", text="...")
        self._uid_status.configure(text="Mengecek ke server...", text_color=CLR.MUTED)

        verify_participant(uid, callback=lambda ok, data: self.after(
            0, lambda: self._on_uid_result(ok, data, uid)
        ))

    def _on_uid_result(self, exists: bool, data: dict, uid: str):
        self._cek_in_progress = False
        self._cek_btn.configure(state="normal", text="Cek")
        if not exists:
            import tkinter.messagebox as mb
            self._uid_status.configure(
                text=f"UID '{uid}' tidak ditemukan", text_color=CLR.DANGER,
            )
            mb.showerror("UID Tidak Ditemukan", f"UID '{uid}' tidak ada di server.")
            return

        participant = data.get("data", {}) if isinstance(data, dict) else {}
        STATE.current_participant_uid = uid
        STATE.nick_name               = participant.get("name", "")
        STATE.gender_code             = participant.get("gender", "")
        STATE.age_range_code          = int(participant.get("age", 0) or 0)

        self._participant_info.configure(
            text=f"{STATE.nick_name}  |  {participant.get('age','?')} th  |  {participant.get('gender','?')}",
            text_color=CLR.TEXT,
        )
        self._info_box.grid()
        print(f">>> UID verified: {uid} → {STATE.nick_name}")

        if STATE.current_mode == "competition":
            self._uid_status.configure(text="Mengecek jadwal turnamen...", text_color=CLR.MUTED)
            check_active_match(uid, callback=lambda ok, d: self.after(
                0, lambda: self._on_tournament_result(ok, d)
            ))
        else:
            self._uid_status.configure(text="Siap mulai", text_color=CLR.SUCCESS)

    def _on_tournament_result(self, has_match: bool, match_data: dict):
        if has_match:
            STATE.tournament_mode      = True
            STATE.tournament_room_code = match_data.get("room_id", "")
            STATE.tournament_opponent  = match_data.get("opponent", "?")
            STATE.tournament_is_p1     = match_data.get("is_player1", False)
            STATE.tournament_round     = match_data.get("match", {}).get("round", 0)

            slot       = "P1" if STATE.tournament_is_p1 else "P2"
            round_name = {1: "Round 1", 2: "Quarterfinal", 3: "Semifinal", 4: "Final"}.get(
                STATE.tournament_round, f"Ronde {STATE.tournament_round}"
            )
            self._uid_status.configure(
                text=f"{round_name}  |  Lawan: {STATE.tournament_opponent}  |  {slot}",
                text_color=CLR.SUCCESS,
            )
        else:
            STATE.tournament_mode = False
            self._uid_status.configure(text="Siap mulai", text_color=CLR.SUCCESS)

        if STATE.current_mode == "competition":
            self._start_btn.grid_remove()
            self._mode_frame.grid()
            self._duel_btn.configure(state="normal")
            if has_match:
                self._tournament_btn.configure(
                    state="normal", fg_color=CLR.SUCCESS, hover_color=CLR.SUCCESS_HOVER,
                )
            else:
                self._tournament_btn.configure(
                    state="disabled", fg_color=CLR.SUBTLE_BORDER,
                    hover_color=CLR.SUBTLE_BORDER,
                )

    def _reset_scan(self):
        STATE.reset_participant()
        STATE.tournament_mode = False
        self._uid_entry.set("")
        self._participant_info.configure(text="Belum ada peserta", text_color=CLR.MUTED)
        self._uid_status.configure(text="", text_color=CLR.MUTED)
        self._cek_btn.configure(state="normal", text="CEK")
        self._info_box.grid_remove()
        self._qr_last_ts = 0.0

        if STATE.current_mode == "training":
            self._uid_status.configure(
                text="Mode Latihan — UID opsional", text_color=CLR.MUTED,
            )

    # ── Mode toggle ───────────────────────────────────────────────────────────

    def _on_mode_change(self, mode: str):
        if STATE.current_mode == mode:
            return
        STATE.current_mode = mode
        self._reset_scan()
        self._update_mode_buttons()

    def _update_mode_buttons(self):
        is_training = STATE.current_mode == "training"
        self._btn_training.configure(
            fg_color=CLR.ACCENT if is_training else CLR.BG,
        )
        self._btn_competition.configure(
            fg_color=CLR.ACCENT if not is_training else CLR.BG,
        )
        self._uid_required = not is_training
        self._start_btn.configure(
            text="Mulai Latihan" if is_training else "MULAI TES",
        )

    # ── Action callbacks ──────────────────────────────────────────────────────

    def _on_start(self):
        import tkinter.messagebox as messagebox
        uid = self._uid_entry.get()
        try:
            if not uid:
                if self._uid_required:
                    raise ValueError("UID tidak boleh kosong — tekan 'Cek UID' dulu")
                STATE.nick_name = "training_user"
                STATE.current_participant_uid = ""
            elif not STATE.current_participant_uid:
                if self._uid_required:
                    raise ValueError("UID belum diverifikasi — tekan 'Cek UID' dulu")
                STATE.nick_name = "training_user"
                STATE.current_participant_uid = ""
            self.destroy()
        except ValueError as e:
            messagebox.showerror("Input Error", str(e))

    def _start_duel(self):
        STATE.tournament_mode = False
        self.destroy()

    def _start_tournament(self):
        STATE.tournament_mode = True
        self.destroy()

    def _toggle_theme(self):
        from ui.theme import set_theme
        from config import THEME
        new = "light" if THEME == "dark" else "dark"
        set_theme(new)
        import tkinter.messagebox as mb
        mb.showinfo("Tema Diubah", "Restart aplikasi untuk menerapkan tema baru.")

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def _on_close(self):
        self._preview_running = False
        self.user_cancelled   = True
        self.destroy()
        sys.exit(0)

    def destroy(self):
        self._preview_running = False
        super().destroy()

    def report_callback_exception(self, exc, val, tb):
        from core.game_logic import log_exception
        log_exception("Tk callback (InputScreen)", exc, val, tb)