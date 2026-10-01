"""
state.py — OAMP mutable runtime state.

Semua state yang dulunya global variable di main.py dikumpulkan di sini
sebagai satu AppState instance (singleton: STATE).

Cara pakai:
    from state import STATE
    STATE.nick_name = "Alfonso"
    STATE.is_multiplayer = True
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from api.websocket_client import GameWebSocket


@dataclass
class AppState:
    # ── Participant ───────────────────────────────────────────────────────────
    nick_name:               str  = ""
    gender_code:             str  = ""
    age_range_code:          int  = 0
    current_participant_uid: str  = ""

    # ── Mode ──────────────────────────────────────────────────────────────────
    current_mode:     str  = "training"   # "training" | "competition"
    is_multiplayer:   bool = False
    current_room_code: str = ""
    player_num:       int  = 1

    # ── Tournament ────────────────────────────────────────────────────────────
    tournament_mode:      bool = False
    tournament_room_code: str  = ""
    tournament_opponent:  str  = ""
    tournament_is_p1:     bool = False
    tournament_round:     int  = 0

    # ── WebSocket / Opponent tracking ─────────────────────────────────────────
    ws_client:          Optional[object] = field(default=None, repr=False)  # GameWebSocket
    opponent_level:     int  = 0
    opponent_blocks:    int  = 0
    opponent_finished:  bool = False
    opponent_name:      str  = ""

    # ── Camera (runtime, dapat berubah via UI) ────────────────────────────────
    # Diinit dari config, diupdate oleh App_Input controls
    camera_index:      int   = 4
    camera_mirror_x:   bool  = False
    camera_mirror_y:   bool  = False
    camera_zoom:       float = 1.0
    camera_calibration: bool = False
    camera_brightness: int   = 128
    camera_contrast:   int   = 128
    camera_saturation: int   = 128

    def reset_participant(self):
        """Clear participant data — dipanggil saat ganti peserta."""
        self.nick_name               = ""
        self.gender_code             = ""
        self.age_range_code          = 0
        self.current_participant_uid = ""

    def reset_multiplayer(self):
        """Clear multiplayer state — dipanggil saat kembali ke App_Input."""
        self.is_multiplayer    = False
        self.current_room_code = ""
        self.player_num        = 1

    def reset_tournament(self):
        """Clear tournament state."""
        self.tournament_mode      = False
        self.tournament_room_code = ""
        self.tournament_opponent  = ""
        self.tournament_is_p1     = False
        self.tournament_round     = 0

    def reset_opponent(self):
        """Clear opponent tracking — dipanggil saat retry / switch player."""
        self.opponent_level    = 0
        self.opponent_blocks   = 0
        self.opponent_finished = False
        self.opponent_name     = ""

    def close_ws(self):
        """Tutup WebSocket jika aktif."""
        if self.ws_client is not None:
            try:
                self.ws_client.close()
            except Exception:
                pass
            self.ws_client = None

    def full_reset(self):
        """Reset semua state — dipakai saat switch player dari results screen."""
        self.close_ws()
        self.reset_participant()
        self.reset_multiplayer()
        self.reset_tournament()
        self.reset_opponent()

    def sync_from_config(self):
        """
        Inisialisasi camera fields dari config.py.
        Dipanggil sekali saat startup sebelum App_Input dibuka.
        """
        from config import (
            CAMERA_INDEX, CAMERA_MIRROR_X, CAMERA_MIRROR_Y, CAMERA_ZOOM,
            CAMERA_CALIBRATION, CAMERA_BRIGHTNESS, CAMERA_CONTRAST,
            CAMERA_SATURATION, PC_MODE, PLAYER_NUM,
        )
        self.camera_index       = CAMERA_INDEX
        self.camera_mirror_x    = CAMERA_MIRROR_X
        self.camera_mirror_y    = CAMERA_MIRROR_Y
        self.camera_zoom        = CAMERA_ZOOM
        self.camera_calibration = CAMERA_CALIBRATION
        self.camera_brightness  = CAMERA_BRIGHTNESS
        self.camera_contrast    = CAMERA_CONTRAST
        self.camera_saturation  = CAMERA_SATURATION
        self.current_mode       = PC_MODE
        self.player_num         = PLAYER_NUM


# ── Singleton ─────────────────────────────────────────────────────────────────
STATE = AppState()