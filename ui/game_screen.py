"""
ui/game_screen.py — GameScreen: main game loop.

Thread model:
  CameraThread → frame_queue → DetectionThread → detection_queue
                                                        ↓
                              Main thread ← after(33, _poll)
                              [_check_answer + widget update ONLY]
"""

from __future__ import annotations
import queue
import sys
import threading
import time
from typing import Any

import cv2
import customtkinter
import numpy as np
from PIL import Image, ImageTk

from config import DISPLAY_HALF, HIDE_CAMERA, BUTTON_MODE, MAX_LEVEL, LEVEL_ANSWERS, get_variant
from state import STATE
from ui.theme import CLR, FONT, RADIUS
from ui.widgets import SectionLabel, BorderedImageFrame, StatCard, LevelTimeRow
from core.camera import CameraThread, open_camera
from core.detection import DetectionThread, DetectionResult
from core.audio import play_sfx, sfx_for_time
from core.serial_reader import SerialReaderThread
from core.game_logic import (
    estimate_cognitive_age, compute_visuo_spatial,
    star_rating, log_exception, save_env_value,
)
from api.participants import submit_results, save_training_locally
from api.tournament import send_event as send_tournament_event
from api.duel import submit_result as submit_duel_result, poll_result as poll_duel_result
from api.client import fire_async


# ── Helpers ───────────────────────────────────────────────────────────────────

def _fire_event(event_type: str, level: int = 0, time_sec: float = 0.0):
    """Fire game event to API (async, best-effort)."""
    from config import API_SERVER_URL
    if not API_SERVER_URL or not STATE.is_multiplayer:
        return
    room = STATE.current_room_code
    fire_async(
        f"{API_SERVER_URL}/api/game/event",
        {
            "type":        event_type,
            "room_id":     room,
            "player_name": STATE.nick_name,
            "player_num":  STATE.player_num,
            "level":       level,
            "time_sec":    round(float(time_sec), 3),
        },
        fallback_filename=None,
    )
    # Also broadcast via WS
    ws = STATE.ws_client
    if ws and ws.connected:
        if event_type == "level_start":
            ws.send({"type": "level_start", "player_id": f"P{STATE.player_num}",
                     "player_name": STATE.nick_name, "player_num": STATE.player_num,
                     "level": level})
        elif event_type == "level_complete":
            ws.send({"type": "score_update", "player_id": f"P{STATE.player_num}",
                     "player_name": STATE.nick_name, "player_num": STATE.player_num,
                     "game_score": level, "blocks_hit": int(time_sec)})


class GameScreen(customtkinter.CTk):

    def __init__(self, model: Any, face_assets: tuple):
        super().__init__()
        self.title("Block Design Test")
        self.configure(fg_color=CLR.BG)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        self._model       = model
        self._face_assets = face_assets

        # ── Thread queues ─────────────────────────────────────────────────────
        self._frame_q     = queue.Queue(maxsize=2)
        self._detection_q = queue.Queue(maxsize=2)
        self._cam_thread: CameraThread | None       = None
        self._det_thread: DetectionThread | None    = None
        self._serial_thread: SerialReaderThread | None = None
        self._poll_job    = None
        self._game_running = False

        # ── Game state ────────────────────────────────────────────────────────
        self.current_question = 1
        self.current_variant  = ""
        self.start_task       = 0.0
        self.timer_running    = False
        self.start_time       = 0.0

        self.timer_task_all:     list[float] = []
        self.cognitive_age_list: list[int]   = []
        self.variant_played_list: list[str]  = []

        # task flags — True = level belum selesai
        self._task_flags = {i: True for i in range(1, 9)}

        # button mode
        self._image_visible   = False
        self._image_show_time: float | None = None
        self._img_display_dur = 5.0

        # multiplayer
        self._pending_match_start  = False
        self._pending_match_result = None
        self._heartbeat_active     = False
        self._action_bar_built     = False

        # opponent tracking
        self._opp_level    = 0
        self._opp_blocks   = 0
        self._opp_finished = False
        self._opp_name     = ""

        self._build_ui()
        self._preload_images()

        if BUTTON_MODE:
            self._serial_thread = SerialReaderThread()
            self._serial_thread.start()

        if STATE.is_multiplayer:
            self.button_0.configure(
                text="⏳ Menunggu lawan...", state="disabled", fg_color=CLR.CARD,
            )
            self._setup_ws()
            self.after(200, self._retry_send_ready)

    # ── UI build ──────────────────────────────────────────────────────────────

    def _build_ui(self):
        if DISPLAY_HALF:
            self.geometry("960x560")
            self.grid_rowconfigure(0, weight=4)
            self.grid_rowconfigure(1, weight=1)
            self.grid_rowconfigure(2, weight=0)
        else:
            self.geometry("1200x800")
            self.grid_rowconfigure(0, weight=1)
            self.grid_rowconfigure(1, weight=0)

        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)

        self._top = customtkinter.CTkFrame(self, fg_color=CLR.BG)
        self._top.grid(row=0, column=0, columnspan=2, sticky="nsew",
                       padx=(0 if DISPLAY_HALF else 10),
                       pady=(0 if DISPLAY_HALF else 10))
        self._top.grid_columnconfigure(0, weight=1)
        self._top.grid_columnconfigure(1, weight=1)
        self._top.grid_rowconfigure(0, weight=1)

        self._content = customtkinter.CTkFrame(self._top, fg_color=CLR.BG)
        self._content.grid(row=0, column=0, columnspan=2, sticky="nsew")
        self._content.grid_columnconfigure(0, weight=1)
        self._content.grid_columnconfigure(1, weight=3 if not HIDE_CAMERA else 1)
        self._content.grid_rowconfigure(0, weight=1)

        self._build_left_panel()
        if not HIDE_CAMERA:
            self._build_camera_panel()

        # Exit button (always visible)
        self._exit_btn = customtkinter.CTkButton(
            self, text="✕ Keluar",
            font=(FONT.P, 12, "bold"), width=100, height=36,
            fg_color=CLR.DANGER, hover_color=CLR.DANGER_HOVER,
            text_color=CLR.TEXT, corner_radius=RADIUS.DEFAULT,
            command=self._switch_player,
        )
        self._exit_btn.place(relx=1.0, rely=0.0, anchor="ne", x=-12, y=8)

        # Start button
        btn_font = 30 if DISPLAY_HALF else 36
        btn_pady = 10 if DISPLAY_HALF else 20
        self.button_0 = customtkinter.CTkButton(
            self._top, text="MULAI TES",
            font=(FONT.P, btn_font, "bold"),
            fg_color=CLR.ACCENT, hover_color=CLR.ACCENT2,
            text_color=CLR.TEXT, corner_radius=RADIUS.DEFAULT, height=56,
            command=self._on_start_btn,
        )
        row = 1 if DISPLAY_HALF else 2
        self.button_0.grid(row=row, column=0, columnspan=2,
                           padx=20, pady=btn_pady, sticky="ew")

        # Status bar (half mode only)
        if DISPLAY_HALF:
            self._status_bar = customtkinter.CTkFrame(
                self, fg_color=CLR.CARD, height=44,
                border_width=1, border_color=CLR.SUBTLE_BORDER,
            )
            self._status_bar.grid(row=1, column=0, columnspan=2, sticky="ew")
            self._status_bar.grid_propagate(False)
            self._status_lbl = customtkinter.CTkLabel(
                self._status_bar,
                text=f"MODE: {STATE.current_mode.upper()}  |  LEVEL 1/{MAX_LEVEL}",
                font=(FONT.P, 12, "bold"), text_color=CLR.ACCENT, anchor="w",
            )
            self._status_lbl.pack(side="left", padx=20, pady=8)

        # Results action bar (pinned bottom, hidden until test done)
        action_row = 2 if DISPLAY_HALF else 1
        self._results_bar = customtkinter.CTkFrame(
            self, fg_color=CLR.CARD, height=72,
            border_width=1, border_color=CLR.SUBTLE_BORDER,
        )
        self._results_bar.grid(row=action_row, column=0, columnspan=2, sticky="ew")
        self._results_bar.grid_propagate(False)
        self._results_bar.grid_remove()

        # Countdown overlay
        self._countdown_lbl = customtkinter.CTkLabel(
            self, text="",
            font=(FONT.P, 80, "bold"), text_color=CLR.ACCENT, fg_color="transparent",
        )

        # Celebration flash
        self._celebration = customtkinter.CTkFrame(
            self._design_frame, fg_color=CLR.SUCCESS, corner_radius=RADIUS.DEFAULT,
        )
        self._celebration_lbl = customtkinter.CTkLabel(
            self._celebration, text="",
            font=(FONT.P, 38, "bold"), text_color=CLR.BG,
        )
        self._celebration_lbl.place(relx=0.5, rely=0.5, anchor="center")

    def _build_left_panel(self):
        px = 10 if DISPLAY_HALF else 20
        py = 10 if DISPLAY_HALF else 20

        left = customtkinter.CTkFrame(self._content, fg_color=CLR.BG)
        if HIDE_CAMERA:
            left.grid(row=0, column=0, columnspan=2, sticky="", padx=px, pady=py)
        else:
            left.grid(row=0, column=0, sticky="nsew", padx=px, pady=py)
        left.grid_rowconfigure(1, weight=1)
        left.grid_columnconfigure(1, weight=1)

        # Level header
        lvl_hdr = customtkinter.CTkFrame(left, fg_color="transparent")
        lvl_hdr.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 10))
        lvl_hdr.grid_columnconfigure(0, weight=1)

        self.level_text = customtkinter.CTkLabel(
            lvl_hdr, text="LEVEL 1 / 8",
            font=(FONT.P, 14, "bold"), text_color=CLR.ACCENT,
        )
        self.level_text.grid(row=0, column=0, sticky="w")

        self._opp_badge = customtkinter.CTkLabel(
            lvl_hdr, text="",
            font=(FONT.P, 12, "bold"), text_color=CLR.MUTED, anchor="e",
        )
        if STATE.is_multiplayer:
            self._opp_badge.configure(text="Lawan: --")
            self._opp_badge.grid(row=0, column=1, sticky="e")

        # Timer frame
        tw = 120 if DISPLAY_HALF else 150
        tf = 28  if DISPLAY_HALF else 32
        self._timer_frame = customtkinter.CTkFrame(
            left, corner_radius=RADIUS.DEFAULT, fg_color=CLR.BG,
            border_width=2, border_color=CLR.ACCENT, width=tw,
        )
        self._timer_frame.grid(row=1, column=0, rowspan=2, padx=(0, 14), pady=10, sticky="ns")
        self._timer_frame.pack_propagate(False)

        customtkinter.CTkLabel(
            self._timer_frame, text="WAKTU",
            font=(FONT.P, 11, "bold"), text_color=CLR.MUTED,
        ).pack(pady=(16, 0))

        self.timer_label = customtkinter.CTkLabel(
            self._timer_frame, text="00:00",
            font=(FONT.P, tf, "bold"), text_color=CLR.ACCENT,
        )
        self.timer_label.pack(pady=(6, 4))

        self._danger_bar = customtkinter.CTkProgressBar(
            self._timer_frame, width=tw - 24, height=10, corner_radius=5,
        )
        self._danger_bar.pack(pady=(0, 18))
        self._danger_bar.set(0.0)
        self._danger_bar.configure(
            progress_color=CLR.TEAL_SAFE, fg_color=CLR.SUBTLE_BORDER,
        )

        # Level badges
        self._badge_frame = customtkinter.CTkFrame(left, fg_color="transparent")
        self._badge_frame.grid(row=3, column=0, columnspan=2, sticky="ew", pady=(8, 0))
        self._badge_frame.grid_columnconfigure(tuple(range(8)), weight=1)
        self._badges: list[customtkinter.CTkLabel] = []
        for i in range(8):
            b = customtkinter.CTkLabel(
                self._badge_frame, text=str(i + 1),
                font=(FONT.P, 14, "bold"), width=44, height=44,
                fg_color=CLR.CARD, text_color=CLR.MUTED,
                corner_radius=RADIUS.DEFAULT,
            )
            b.grid(row=0, column=i, padx=2, pady=2)
            self._badges.append(b)

        # Design section label + image
        SectionLabel(left, "Block Design").grid(
            row=0, column=1, padx=0, pady=(0, 5), sticky="nsew",
        )
        self._design_frame = BorderedImageFrame(left, border_width=3)
        self._design_frame.grid(row=1, column=1, padx=0, pady=0, sticky="nsew")
        self._design_frame.grid_propagate(False)

        from config import BASE_DIR
        _init_path = BASE_DIR / "FILES" / "TEST_1000x1000" / "0Bx.jpg"
        self._frame_w = 480 if DISPLAY_HALF else 700
        self._frame_h = 480 if DISPLAY_HALF else 700
        self._design_img_lbl = customtkinter.CTkLabel(
            self._design_frame, text="",
        )
        self._design_img_lbl.pack(expand=True, fill="both")

        try:
            _img = Image.open(_init_path).resize(
                (self._frame_w, self._frame_h), Image.Resampling.LANCZOS
            )
            self._ctk_image = customtkinter.CTkImage(
                light_image=_img, size=(self._frame_w, self._frame_h),
            )
            self._design_img_lbl.configure(image=self._ctk_image)
        except Exception:
            pass

        # Results scrollable frame (hidden until test done)
        self._results_frame = customtkinter.CTkScrollableFrame(
            left, fg_color=CLR.CARD, corner_radius=RADIUS.DEFAULT,
            border_width=1, border_color=CLR.SUBTLE_BORDER,
        )
        self._results_frame.grid(row=1, column=1, padx=0, pady=0, sticky="nsew")
        self._results_frame.grid_columnconfigure(0, weight=1)
        self._results_frame.grid_remove()

    def _build_camera_panel(self):
        px = 10 if DISPLAY_HALF else 20
        py = (40, 10) if DISPLAY_HALF else (50, 20)

        SectionLabel(self._content, "Upper Table Camera").grid(
            row=0, column=1, padx=px, pady=0, sticky="nsew",
        )
        self._cam_frame = BorderedImageFrame(self._content, border_width=3)
        self._cam_frame.grid(row=0, column=1, padx=px, pady=py, sticky="nsew")
        self._cam_frame.grid_propagate(False)
        self._cam_frame.grid_rowconfigure(0, weight=1)
        self._cam_frame.grid_columnconfigure(0, weight=1)

        self._cam_label = customtkinter.CTkLabel(self._cam_frame, text="", anchor="center")
        self._cam_label.grid(row=0, column=0, sticky="nsew")

    # ── Image preload ─────────────────────────────────────────────────────────

    def _preload_images(self):
        from config import LEVEL_PATHS
        self._cached_images: dict[str, Image.Image] = {}
        print(">>> [GameScreen] Preloading level images...")
        for variant, path in LEVEL_PATHS.items():
            try:
                img = Image.open(path).resize(
                    (self._frame_w, self._frame_h), Image.Resampling.LANCZOS,
                )
                self._cached_images[variant] = img
            except Exception as e:
                print(f">>> [GameScreen] Preload failed {variant}: {e}")
        print(f">>> [GameScreen] Cached {len(self._cached_images)} images")

    def _load_level_image(self, variant: str):
        img = self._cached_images.get(variant)
        if img is None:
            return
        ctk_img = customtkinter.CTkImage(
            light_image=img, size=(self._frame_w, self._frame_h),
        )
        self._ctk_image = ctk_img

        if BUTTON_MODE:
            self._design_img_lbl.configure(image=ctk_img)
            self._image_visible   = True
            self._image_show_time = time.time()
        else:
            self._design_img_lbl.configure(image=ctk_img)

    # ── Game flow ─────────────────────────────────────────────────────────────

    def _on_start_btn(self):
        self.button_0.grid_remove()
        self._show_level_btn(self.current_question)
        play_sfx("countdown")
        self._show_countdown(3, on_done=self._start_game)

    def _start_game(self):
        self.current_variant = get_variant(self.current_question)
        self._load_level_image(self.current_variant)
        self.start_task = time.time()
        _fire_event("level_start", level=1)

        if STATE.tournament_mode and STATE.tournament_room_code:
            send_tournament_event(STATE.tournament_room_code, "match_started")

        self._start_threads()
        self._reset_timer()
        self._start_timer()
        self._start_heartbeat()

    def _start_threads(self):
        self._cam_thread = CameraThread(
            self._frame_q,
            mirror_x=STATE.camera_mirror_x,
            mirror_y=STATE.camera_mirror_y,
            zoom=STATE.camera_zoom,
            calibration=STATE.camera_calibration,
            brightness=STATE.camera_brightness,
            contrast=STATE.camera_contrast,
            saturation=STATE.camera_saturation,
        )
        self._det_thread = DetectionThread(
            self._model, self._is_bantal(),
            self._frame_q, self._detection_q,
            face_assets=self._face_assets,
        )
        self._cam_thread.start()
        self._det_thread.start()
        self._game_running = True
        self._schedule_poll()

    def _is_bantal(self) -> bool:
        from config import USE_BANTAL_MODEL
        return USE_BANTAL_MODEL

    def _stop_threads(self):
        self._game_running = False
        if self._poll_job:
            self.after_cancel(self._poll_job)
            self._poll_job = None
        if self._cam_thread:
            self._cam_thread.stop()
        if self._det_thread:
            self._det_thread.stop()

    # ── Poll loop (main thread entry point) ───────────────────────────────────

    def _schedule_poll(self):
        if self._game_running:
            self._poll_job = self.after(33, self._poll)

    def _poll(self):
        result: DetectionResult | None = None
        while not self._detection_q.empty():
            try:
                result = self._detection_q.get_nowait()
            except queue.Empty:
                break

        if result is not None:
            self._update_cam_display(result.img_display)
            if result.is_complete:
                self._check_answer(result.sorted_design)

        self._handle_button_mode()
        self._schedule_poll()

    # ── Answer check (state machine, no I/O) ─────────────────────────────────

    def _check_answer(self, sorted_design: list):
        lvl = self.current_question
        if not self._task_flags.get(lvl, False):
            return
        expected = LEVEL_ANSWERS.get(self.current_variant, [])
        if sorted_design != expected:
            return

        # Level complete
        elapsed = round(time.time() - self.start_task, 2)
        self.timer_task_all.append(elapsed)
        self.cognitive_age_list.append(estimate_cognitive_age(elapsed))
        self.variant_played_list.append(self.current_variant)
        self._task_flags[lvl] = False

        print(f"TASK {lvl} COMPLETED in {elapsed}s")
        _fire_event("level_complete", level=lvl, time_sec=elapsed)
        self._update_badge(lvl, state="completed")

        if lvl >= self.max_level:
            def _finish_level_audio():
                play_sfx(sfx_for_time(elapsed), wait=True)
                self.after(0, self.end_test)

            threading.Thread(target=_finish_level_audio, daemon=True).start()
            return

        # Advance
        self.current_question = lvl + 1
        self.current_variant  = get_variant(self.current_question)
        self._load_level_image(self.current_variant)
        self._current_level_btn.grid_remove()
        self._show_level_btn(self.current_question)
        self.start_task = time.time()
        _fire_event("level_start", level=self.current_question)
        self._reset_timer()
        self._start_timer()
        def _play_level_transition_audio():
            play_sfx(sfx_for_time(elapsed), wait=True)
            # Transition voice is temporarily disabled; keep motivation audio first.

        threading.Thread(
            target=_play_level_transition_audio,
            daemon=True,
        ).start()

    @property
    def max_level(self) -> int:
        return MAX_LEVEL

    # ── Skip ──────────────────────────────────────────────────────────────────

    def skip_current_level(self, event=None):
        lvl = self.current_question
        if not self.timer_running or not self._task_flags.get(lvl, False):
            return
        print(f">>> Level {lvl} SURRENDERED")
        threading.Thread(target=lambda: play_sfx("skip"), daemon=True).start()

        self.timer_task_all.append(0.0)
        self._task_flags[lvl] = False
        self._update_badge(lvl, state="completed")
        _fire_event("level_complete", level=lvl, time_sec=0.0)

        self.current_question = lvl + 1
        if self.current_question <= self.max_level:
            self.current_variant = get_variant(self.current_question)
            self._load_level_image(self.current_variant)
            self._current_level_btn.grid_remove()
            self._show_level_btn(self.current_question)
            self.start_task = time.time()
            _fire_event("level_start", level=self.current_question)
            self._reset_timer()
            self._start_timer()
        else:
            self.end_test()

    # ── End test ──────────────────────────────────────────────────────────────

    def end_test(self):
        self._stop_heartbeat()
        self._stop_threads()
        self._reset_timer()
        self.timer_running = False
        if hasattr(self, "_current_level_btn"):
            self._current_level_btn.grid_remove()

        total_time = float(sum(self.timer_task_all))
        age_real   = int(STATE.age_range_code)
        is_duel    = STATE.current_mode == "competition" or STATE.tournament_mode

        if is_duel:
            age_cog      = 0
            visuo_spatial = 0
        else:
            age_cog = (
                int(sum(self.cognitive_age_list) / len(self.cognitive_age_list))
                if self.cognitive_age_list else 0
            )
            visuo_spatial = compute_visuo_spatial(age_real, age_cog)

        threading.Thread(target=lambda: play_sfx("complete"), daemon=True).start()
        self._show_celebration("TES SELESAI!", duration=2500)
        self.after(2500, lambda: self._finish_test(
            total_time, age_real, age_cog, visuo_spatial,
        ))

    def _finish_test(self, total_time, age_real, age_cog, visuo_spatial):
        uid    = STATE.current_participant_uid
        times  = self.timer_task_all
        avg    = float(sum(times) / len(times)) if times else 0.0

        payload = {
            "uid":           uid,
            "mode":          "tournament" if STATE.tournament_mode else STATE.current_mode,
            "nick_name":     STATE.nick_name,
            "gender":        STATE.gender_code,
            "age":           age_real,
            **{f"task{i+1:02d}": float(times[i]) if i < len(times) else 0.0
               for i in range(8)},
            "task_avg":      avg,
            "cognitive_age": age_cog,
            "visuo_spatial": visuo_spatial,
            "cog_age_list":  list(self.cognitive_age_list),
            "variant_list":  self.variant_played_list,
            "client_ts":     int(time.time()),
        }

        result_thread = None
        if uid:
            result_thread = submit_results(uid, payload)
        else:
            save_training_locally(payload)

        score = float(sum(times))

        if STATE.tournament_mode and STATE.tournament_room_code:
            p_num = 1 if STATE.tournament_is_p1 else 2
            send_tournament_event(
                STATE.tournament_room_code, "match_finished",
                player_num=p_num, score=score,
            )

        if STATE.is_multiplayer and not STATE.tournament_mode and STATE.current_room_code and uid:
            if result_thread:
                result_thread.join(timeout=15)
            submit_duel_result(STATE.current_room_code, uid, STATE.player_num, score)

        ws = STATE.ws_client
        if STATE.is_multiplayer and ws and ws.connected:
            ws.send({
                "type":        "GAME_OVER",
                "player_id":   f"P{STATE.player_num}",
                "player_name": STATE.nick_name,
                "player_num":  STATE.player_num,
                "game_score":  sum(1 for t in times if t > 0),
            })

        self._build_results_dashboard(age_real, total_time, age_cog, visuo_spatial)

    # ── Results dashboard ─────────────────────────────────────────────────────

    def _build_results_dashboard(self, age_real, total_time, age_cog, visuo_spatial):
        self._duel_result_lbl  = None
        self._winlose_lbl      = None

        self._design_frame.grid_remove()
        self._results_frame.grid()
        self._exit_btn.place(relx=1.0, rely=0.0, anchor="ne", x=-12, y=8)

        for w in self._results_frame.winfo_children():
            w.destroy()

        # Header
        customtkinter.CTkLabel(
            self._results_frame, text="TES SELESAI",
            font=(FONT.P, 26, "bold"), text_color=CLR.TEXT,
        ).grid(row=0, column=0, pady=(20, 6))

        customtkinter.CTkFrame(
            self._results_frame, fg_color=CLR.ACCENT, height=3, corner_radius=0,
        ).grid(row=1, column=0, sticky="ew", padx=20, pady=(0, 16))

        # Trophy
        stars, star_lbl = star_rating(total_time)
        trophy = customtkinter.CTkFrame(
            self._results_frame, fg_color=CLR.BG,
            corner_radius=RADIUS.DEFAULT, border_width=2, border_color=CLR.GOLD,
        )
        trophy.grid(row=2, column=0, sticky="ew", padx=20, pady=(0, 16))
        trophy.grid_columnconfigure(0, weight=1)

        customtkinter.CTkLabel(trophy, text=stars, font=(FONT.P, 28), text_color=CLR.GOLD).grid(row=0, column=0, pady=(16, 2))
        customtkinter.CTkLabel(trophy, text=f"{total_time:.1f}s", font=(FONT.P, 52, "bold"), text_color=CLR.GOLD).grid(row=1, column=0, pady=(0, 2))
        customtkinter.CTkLabel(trophy, text=star_lbl, font=(FONT.P, 20, "bold"), text_color=CLR.SUCCESS).grid(row=2, column=0, pady=(0, 2))
        customtkinter.CTkLabel(trophy, text="Total Waktu", font=(FONT.P, 13), text_color=CLR.MUTED).grid(row=3, column=0, pady=(0, 16))

        is_duel = STATE.current_mode == "competition" or STATE.tournament_mode

        # Win/Lose banner (competition)
        if is_duel:
            wl = customtkinter.CTkFrame(
                self._results_frame, fg_color=CLR.BG,
                corner_radius=RADIUS.DEFAULT, border_width=2, border_color=CLR.SUCCESS,
            )
            wl.grid(row=3, column=0, sticky="ew", padx=20, pady=(0, 16))
            wl.grid_columnconfigure(0, weight=1)
            wl_text  = "⏳ Menunggu hasil lawan..." if STATE.is_multiplayer and STATE.current_room_code else "Mode Latihan"
            wl_color = CLR.ACCENT if STATE.is_multiplayer else CLR.MUTED
            self._winlose_lbl = customtkinter.CTkLabel(
                wl, text=wl_text,
                font=(FONT.P, 22, "bold"), text_color=wl_color,
            )
            self._winlose_lbl.grid(row=0, column=0, pady=18)

        # Stats
        stats = customtkinter.CTkFrame(self._results_frame, fg_color="transparent")
        stats.grid(row=4, column=0, sticky="ew", padx=20, pady=(0, 12))
        if is_duel:
            stats.grid_columnconfigure((0, 1), weight=1)
            StatCard(stats, "Usia", f"{age_real} th").grid(row=0, column=0, padx=(0, 8), sticky="ew")
            StatCard(stats, "Total Waktu", f"{total_time:.1f}s").grid(row=0, column=1, padx=(8, 0), sticky="ew")
        else:
            stats.grid_columnconfigure((0, 1, 2), weight=1)
            StatCard(stats, "Usia",          f"{age_real} th").grid(row=0, column=0, padx=(0, 4), sticky="ew")
            StatCard(stats, "Usia Kognitif", f"{age_cog} th").grid(row=0, column=1, padx=4,       sticky="ew")
            StatCard(stats, "Visual-Spasial",f"{visuo_spatial}%").grid(row=0, column=2, padx=(4, 0), sticky="ew")

        # Duel result (multiplayer)
        if STATE.is_multiplayer and STATE.current_room_code and not STATE.tournament_mode:
            self._build_duel_banner()

        # Per-level times
        if self.timer_task_all:
            customtkinter.CTkLabel(
                self._results_frame, text="Waktu per Level",
                font=(FONT.P, 14, "bold"), text_color=CLR.TEXT,
            ).grid(row=5, column=0, pady=(16, 6), sticky="w", padx=20)

            for i, t in enumerate(self.timer_task_all):
                LevelTimeRow(self._results_frame, i + 1, t).grid(
                    row=6 + i, column=0, sticky="ew", padx=20, pady=2,
                )

        # Action bar
        if not self._action_bar_built:
            self._results_bar.grid_columnconfigure((0, 1, 2), weight=1)
            customtkinter.CTkButton(
                self._results_bar, text="Main Lagi",
                font=(FONT.P, 14, "bold"),
                fg_color=CLR.ACCENT, hover_color=CLR.ACCENT2,
                text_color=CLR.TEXT, corner_radius=RADIUS.DEFAULT, height=52,
                command=self._retry,
            ).grid(row=0, column=0, padx=(16, 4), pady=10, sticky="ew")
            customtkinter.CTkButton(
                self._results_bar, text="Salin Hasil",
                font=(FONT.P, 14, "bold"),
                fg_color=CLR.CARD, hover_color=CLR.SUBTLE_BORDER,
                text_color=CLR.TEXT, corner_radius=RADIUS.DEFAULT, height=52,
                command=lambda: self._copy_results(total_time, age_real),
            ).grid(row=0, column=1, padx=4, pady=10, sticky="ew")
            customtkinter.CTkButton(
                self._results_bar, text="✕ Kembali ke Beranda",
                font=(FONT.P, 14, "bold"),
                fg_color=CLR.DANGER, hover_color=CLR.DANGER_HOVER,
                text_color=CLR.TEXT, corner_radius=RADIUS.DEFAULT, height=52,
                command=self._switch_player,
            ).grid(row=0, column=2, padx=(4, 16), pady=10, sticky="ew")
            self._action_bar_built = True

        self._results_bar.grid()

        # Apply pending match result if arrived before dashboard
        if self._pending_match_result:
            w, p1s, p2s = self._pending_match_result
            self._pending_match_result = None
            self._apply_match_result(w, p1s, p2s)

    def _build_duel_banner(self):
        banner = customtkinter.CTkFrame(
            self._results_frame, fg_color=CLR.CARD,
            corner_radius=RADIUS.DEFAULT, border_width=2, border_color=CLR.ACCENT,
        )
        banner.grid(row=4, column=0, sticky="ew", padx=20, pady=(8, 8))
        banner.grid_columnconfigure(0, weight=1)

        finish_text  = "⚡ Kamu selesai duluan!" if not self._opp_finished else "⚖️ Lawan selesai lebih dulu!"
        finish_color = CLR.SUCCESS if not self._opp_finished else CLR.WARNING
        customtkinter.CTkLabel(
            banner, text=finish_text,
            font=(FONT.P, 12, "bold"), text_color=finish_color,
        ).grid(row=0, column=0, pady=(14, 4))

        self._duel_result_lbl = customtkinter.CTkLabel(
            banner, text="Menunggu hasil lawan...",
            font=(FONT.P, 18, "bold"), text_color=CLR.ACCENT,
        )
        self._duel_result_lbl.grid(row=1, column=0, pady=(4, 8))

        scores = customtkinter.CTkFrame(banner, fg_color="transparent")
        scores.grid(row=2, column=0, sticky="ew", padx=16, pady=(0, 4))
        scores.grid_columnconfigure((0, 1), weight=1)
        self._p1_score_lbl = customtkinter.CTkLabel(scores, text="P1: --", font=(FONT.P, 13, "bold"), text_color=CLR.TEXT)
        self._p1_score_lbl.grid(row=0, column=0, sticky="w")
        self._p2_score_lbl = customtkinter.CTkLabel(scores, text="P2: --", font=(FONT.P, 13, "bold"), text_color=CLR.TEXT)
        self._p2_score_lbl.grid(row=0, column=1, sticky="e")

        poll_duel_result(STATE.current_room_code, callback=self._on_duel_result)

    def _on_duel_result(self, decided: bool, data: dict):
        def _apply():
            try:
                if decided and data.get("winner"):
                    w   = data["winner"]
                    p1s = f"P1: {data.get('player1_score','--')}"
                    p2s = f"P2: {data.get('player2_score','--')}"
                    if self._p1_score_lbl: self._p1_score_lbl.configure(text=p1s)
                    if self._p2_score_lbl: self._p2_score_lbl.configure(text=p2s)
                    self._apply_match_result(w, data.get("player1_score",0), data.get("player2_score",0))
            except Exception:
                pass
        try:
            self.after(0, _apply)
        except Exception:
            pass

    def _apply_match_result(self, winner: str, p1s, p2s):
        my = str(STATE.player_num)
        if winner == "draw":
            txt, color = "🤝 SERI!", CLR.ACCENT
        elif winner == my:
            txt, color = "🏆 KAMU MENANG!", CLR.SUCCESS
        else:
            txt, color = "KAMU KALAH", CLR.DANGER
        if self._duel_result_lbl and self._duel_result_lbl.winfo_exists():
            self._duel_result_lbl.configure(text=txt, text_color=color)
        if self._winlose_lbl and self._winlose_lbl.winfo_exists():
            self._winlose_lbl.configure(text=txt, text_color=color)

    def _copy_results(self, total_time, age_real):
        lvl = sum(1 for t in self.timer_task_all if t > 0)
        text = f"BDT | Total: {total_time:.1f}s | Usia: {age_real} | Level: {lvl}/{MAX_LEVEL}"
        if self.timer_task_all:
            text += " | " + " ".join(f"L{i+1}:{t:.1f}s" for i, t in enumerate(self.timer_task_all))
        self.clipboard_clear()
        self.clipboard_append(text)

    # ── Camera display ────────────────────────────────────────────────────────

    def _update_cam_display(self, frame: np.ndarray):
        if HIDE_CAMERA or not hasattr(self, "_cam_label"):
            return
        try:
            cw = self._cam_frame.winfo_width()
            ch = self._cam_frame.winfo_height()
            if cw < 2 or ch < 2:
                return
            h, w  = frame.shape[:2]
            scale = min(cw / w, ch / h)
            if scale < 1.0:
                frame = cv2.resize(
                    frame, (int(w * scale), int(h * scale)),
                    interpolation=cv2.INTER_LINEAR,
                )
            imgtk = ImageTk.PhotoImage(image=Image.fromarray(frame))
            self._last_photo = imgtk  # prevent GC
            self._cam_label.configure(image=imgtk)
        except Exception:
            pass

    # ── Button mode ───────────────────────────────────────────────────────────

    def _handle_button_mode(self):
        if not BUTTON_MODE:
            return
        if self._image_visible and self._image_show_time:
            if time.time() - self._image_show_time >= self._img_display_dur:
                blank = customtkinter.CTkImage(
                    light_image=Image.new("RGB", (self._frame_w, self._frame_h), "gray"),
                    size=(self._frame_w, self._frame_h),
                )
                self._design_img_lbl.configure(image=blank)
                self._image_visible   = False
                self._image_show_time = None

        if self._serial_thread:
            msg = self._serial_thread.get_message()
            if msg == "disable_image" and not self._image_visible:
                self._load_level_image(self.current_variant)

    # ── Timer ─────────────────────────────────────────────────────────────────

    def _update_timer(self):
        if self.timer_running:
            elapsed = time.time() - self.start_time
            m, s    = int(elapsed // 60), int(elapsed % 60)
            self.timer_label.configure(text=f"{m:02d}:{s:02d}")
            if elapsed < 40:
                color = CLR.TEAL_SAFE
            elif elapsed < 60:
                color = CLR.ACCENT
            else:
                color = CLR.DANGER
            self._danger_bar.configure(progress_color=color)
            self._danger_bar.set(min(elapsed / 80.0, 1.0))
            self.after(200, self._update_timer)

    def _start_timer(self):
        if not self.timer_running:
            self.start_time    = time.time()
            self.timer_running = True
            self._update_timer()

    def _reset_timer(self):
        self.timer_label.configure(text="00:00")
        self.timer_running = False
        self._danger_bar.set(0.0)
        self._danger_bar.configure(progress_color=CLR.TEAL_SAFE)

    # ── Level UI ──────────────────────────────────────────────────────────────

    def _show_level_btn(self, level: int):
        self._current_level_btn = customtkinter.CTkLabel(
            self._top, text=f"Level {level}",
            font=(FONT.P, 28, "bold"),
            fg_color=CLR.ACCENT, text_color=CLR.TEXT,
            corner_radius=RADIUS.DEFAULT,
        )
        row = 1 if DISPLAY_HALF else 2
        self._current_level_btn.grid(
            row=row, column=0, columnspan=2, padx=20, pady=10, sticky="ew",
        )
        if hasattr(self, "_status_lbl"):
            self._status_lbl.configure(
                text=f"MODE: {STATE.current_mode.upper()}  |  LEVEL {level}/{MAX_LEVEL}",
            )
        self.level_text.configure(text=f"LEVEL {level} / {MAX_LEVEL}")
        self._update_badge(level, state="active")

    def _update_badge(self, level: int, state: str = "active"):
        for i, badge in enumerate(self._badges):
            n = i + 1
            if n < level:
                badge.configure(text="✓", fg_color=CLR.LEVEL_DONE, text_color=CLR.TEXT)
            elif n == level:
                if state == "active":
                    badge.configure(text=str(n), fg_color=CLR.LEVEL_ACTIVE, text_color=CLR.TEXT)
                else:
                    badge.configure(text="✓", fg_color=CLR.LEVEL_DONE, text_color=CLR.TEXT)
            else:
                badge.configure(text=str(n), fg_color=CLR.CARD, text_color=CLR.MUTED)

    # ── Countdown + celebration ───────────────────────────────────────────────

    def _show_countdown(self, count: int, on_done=None):
        if count > 0:
            self._countdown_lbl.place(relx=0.5, rely=0.5, anchor="center")
            self._countdown_lbl.configure(text=str(count))
            self.after(800, lambda: self._show_countdown(count - 1, on_done))
        else:
            self._countdown_lbl.configure(text="GO!")
            self.after(400, self._countdown_lbl.place_forget)
            if on_done:
                on_done()

    def _show_celebration(self, text: str = "HEBAT!", duration: int = 1200):
        self._celebration_lbl.configure(text=text)
        self._celebration.place(relx=0, rely=0, relwidth=1, relheight=1)
        self._celebration.lift()
        self.after(duration, self._celebration.place_forget)

    # ── Multiplayer / WS ──────────────────────────────────────────────────────

    def _setup_ws(self):
        ws = STATE.ws_client
        if ws:
            ws.on_message = self._on_ws_msg
            if ws.connected:
                return
            ws.close()
        from config import API_SERVER_URL
        if not API_SERVER_URL or not STATE.current_room_code:
            return
        from api.websocket_client import GameWebSocket
        STATE.ws_client = GameWebSocket(
            room_code=STATE.current_room_code,
            player_num=STATE.player_num,
            player_name=STATE.nick_name,
            on_message=self._on_ws_msg,
        )
        STATE.ws_client.connect()

    def _retry_send_ready(self, attempt: int = 0):
        ws = STATE.ws_client
        if ws and ws.connected:
            ws.send({"type": "player_ready", "player_id": f"P{STATE.player_num}",
                     "player_num": STATE.player_num})
            return
        if attempt >= 33:
            self.button_0.configure(text="MULAI TES", state="normal", fg_color=CLR.ACCENT)
            return
        self.after(300, lambda: self._retry_send_ready(attempt + 1))

    def _on_ws_msg(self, data: dict):
        try:
            self.after(0, lambda d=data: self._handle_ws(d))
        except Exception:
            pass

    def _handle_ws(self, data: dict):
        t = data.get("type", "")
        if t == "score_update" and data.get("player_num") != STATE.player_num:
            self._opp_level  = data.get("game_score", self._opp_level)
            self._opp_blocks = data.get("blocks_hit", self._opp_blocks)
            self._opp_name   = data.get("player_name", self._opp_name or "Lawan")
            self._refresh_opp_badge()
        elif t == "GAME_OVER" and data.get("player_num") != STATE.player_num:
            self._opp_finished = True
            self._refresh_opp_badge()
        elif t == "match_start":
            self._pending_match_start = True
            self._try_start_competition()
        elif t == "match_result":
            w, p1s, p2s = data.get("winner",""), data.get("p1_score",0), data.get("p2_score",0)
            if w:
                self._pending_match_result = (w, p1s, p2s)
                self.after(0, lambda: self._apply_match_result(w, p1s, p2s))

    def _try_start_competition(self):
        if not STATE.is_multiplayer or not self._pending_match_start:
            return
        self._pending_match_start = False
        self.button_0.grid_remove()
        self._show_level_btn(self.current_question)
        threading.Thread(target=lambda: play_sfx("countdown"), daemon=True).start()
        self._show_countdown(3, on_done=self._start_game)

    def _refresh_opp_badge(self):
        if not STATE.is_multiplayer or not hasattr(self, "_opp_badge"):
            return
        lvl  = f"L{self._opp_level}" if self._opp_level > 0 else "--"
        name = self._opp_name or "Lawan"
        if self._opp_finished:
            self._opp_badge.configure(text=f"{name}: Selesai!", text_color=CLR.SUCCESS)
        else:
            t = f" ({self._opp_blocks}s)" if self._opp_blocks else ""
            self._opp_badge.configure(text=f"{name}: {lvl}{t}", text_color=CLR.ACCENT)

    # ── Heartbeat ─────────────────────────────────────────────────────────────

    def _start_heartbeat(self):
        if not STATE.is_multiplayer:
            return
        self._heartbeat_active = True
        from config import API_SERVER_URL
        if not API_SERVER_URL:
            return
        def _loop():
            while self._heartbeat_active:
                time.sleep(15)
                from api.client import post
                post(f"{API_SERVER_URL}/api/game/event", {
                    "type": "heartbeat", "room_id": STATE.current_room_code,
                    "player_name": STATE.nick_name, "player_num": STATE.player_num,
                }, timeout=5)
        threading.Thread(target=_loop, daemon=True).start()

    def _stop_heartbeat(self):
        self._heartbeat_active = False

    # ── Navigation ────────────────────────────────────────────────────────────

    def _retry(self):
        STATE.reset_opponent()
        STATE.close_ws()
        self.destroy()

        if STATE.current_mode == "competition":
            STATE.is_multiplayer = True
            from ui.room_screen import RoomScreen
            room = RoomScreen()
            room.mainloop()
            if not room.room_ready:
                return

        _launch_game(self._model, self._face_assets)

    def _switch_player(self):
        STATE.full_reset()
        self.destroy()
        _relaunch_from_input(self._model, self._face_assets)

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def _on_close(self):
        self._stop_heartbeat()
        self._stop_threads()
        STATE.close_ws()
        self.cleanup()
        self.destroy()

    def cleanup(self):
        self._stop_threads()
        if self._serial_thread:
            self._serial_thread.stop()
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def destroy(self):
        self._stop_heartbeat()
        self.cleanup()
        super().destroy()

    def report_callback_exception(self, exc, val, tb):
        log_exception("Tk callback (GameScreen)", exc, val, tb)


# ── Module-level launch helpers (dipakai oleh _retry / _switch_player) ────────

def _launch_game(model, face_assets):
    app = GameScreen(model, face_assets)
    app.bind("<Return>", app.skip_current_level)
    try:
        if sys.platform == "linux":
            app.attributes("-zoomed", True)
        else:
            app.state("zoomed")
    except Exception:
        pass
    app.mainloop()


def _relaunch_from_input(model, face_assets):
    from ui.input_screen import InputScreen
    inp = InputScreen(model=model, face_assets=face_assets)
    inp.after(200, inp.start_camera_preview)
    try:
        if sys.platform == "linux":
            inp.attributes("-zoomed", True)
        else:
            inp.state("zoomed")
    except Exception:
        pass
    inp.mainloop()
    if not STATE.nick_name:
        return

    if STATE.current_mode == "competition":
        STATE.is_multiplayer = True
        from ui.room_screen import RoomScreen
        room = RoomScreen()
        room.mainloop()
        if not room.room_ready:
            return

    _launch_game(model, face_assets)