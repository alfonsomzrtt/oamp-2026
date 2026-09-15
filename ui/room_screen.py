"""
ui/room_screen.py — App_Room: multiplayer lobby (create / join / ready).

Setelah destroy():
  - self.room_ready == True  → lanjut ke GameScreen
  - self.room_ready == False → user keluar, exit
STATE.current_room_code dan STATE.player_num sudah terisi.
"""

from __future__ import annotations
import threading
import time

import customtkinter

from config import API_SERVER_URL
from state import STATE
from ui.theme import CLR, FONT, RADIUS
from api.websocket_client import GameWebSocket
from core.game_logic import log_exception


class RoomScreen(customtkinter.CTk):

    def __init__(self):
        super().__init__()
        self.title("BDT — Multiplayer Lobby")
        self.geometry("620x600")
        self.configure(fg_color=CLR.BG)
        self.resizable(0, 0)
        self._center()

        self.room_ready  = False
        self._room_code: str | None = None
        self._player_num: int | None = None
        self._polling    = False
        self._i_am_ready = False

        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self._build_ui()
        self._refresh_rooms()

    # ── Init ──────────────────────────────────────────────────────────────────

    def _center(self):
        self.update_idletasks()
        w, h = self.winfo_width(), self.winfo_height()
        x = (self.winfo_screenwidth()  // 2) - (w // 2)
        y = (self.winfo_screenheight() // 2) - (h // 2)
        self.geometry(f"{w}x{h}+{x}+{y}")

    def report_callback_exception(self, exc, val, tb):
        log_exception("Tk callback (RoomScreen)", exc, val, tb)

    # ── UI build ──────────────────────────────────────────────────────────────

    def _build_ui(self):
        # Header
        header = customtkinter.CTkFrame(
            self, fg_color=CLR.CARD, corner_radius=0, height=64,
        )
        header.grid(row=0, column=0, sticky="ew")
        header.grid_propagate(False)
        header.grid_columnconfigure(0, weight=1)

        customtkinter.CTkLabel(
            header, text="Multiplayer Lobby",
            font=(FONT.P, 22, "bold"), text_color=CLR.TEXT,
        ).grid(row=0, column=0, padx=20, sticky="w")

        customtkinter.CTkButton(
            header, text="✕ Keluar",
            font=(FONT.P, 12, "bold"), width=100, height=36,
            fg_color=CLR.DANGER, hover_color=CLR.DANGER_HOVER,
            text_color=CLR.TEXT, corner_radius=RADIUS.DEFAULT,
            command=self._back_to_input,
        ).grid(row=0, column=1, padx=(0, 8))

        self._conn_pill = customtkinter.CTkLabel(
            header, text="ONLINE",
            font=(FONT.P, 10, "bold"), text_color=CLR.TEXT,
            fg_color=CLR.SUCCESS, corner_radius=RADIUS.PILL,
            width=64, height=24,
        )
        self._conn_pill.grid(row=0, column=2, padx=(0, 20))

        customtkinter.CTkFrame(
            self, fg_color=CLR.ACCENT, height=3, corner_radius=0,
        ).grid(row=1, column=0, sticky="ew")

        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(2, weight=1)

        self._build_find_panel()
        self._build_lobby_panel()

    def _build_find_panel(self):
        self._find_frame = customtkinter.CTkFrame(self, fg_color=CLR.BG)
        self._find_frame.grid(row=2, column=0, sticky="nsew", padx=24, pady=20)
        self._find_frame.grid_columnconfigure(0, weight=1)

        customtkinter.CTkButton(
            self._find_frame, text="Buat Room Baru",
            command=self._create_room,
            font=(FONT.P, 14, "bold"),
            fg_color=CLR.ACCENT, hover_color=CLR.ACCENT2,
            text_color=CLR.TEXT, corner_radius=RADIUS.DEFAULT, height=52,
        ).grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 14))

        customtkinter.CTkLabel(
            self._find_frame, text="atau masukkan kode room",
            font=(FONT.P, 12), text_color=CLR.MUTED,
        ).grid(row=1, column=0, columnspan=2, pady=(0, 8))

        code_row = customtkinter.CTkFrame(self._find_frame, fg_color="transparent")
        code_row.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(0, 12))
        code_row.grid_columnconfigure(0, weight=1)

        self._code_entry = customtkinter.CTkEntry(
            code_row, placeholder_text="Kode Room (4 huruf)",
            font=(FONT.P, 16, "bold"), height=48,
            corner_radius=RADIUS.DEFAULT, border_width=2,
            border_color=CLR.BORDER, fg_color=CLR.BG, text_color=CLR.TEXT,
        )
        self._code_entry.grid(row=0, column=0, sticky="ew", padx=(0, 10))

        customtkinter.CTkButton(
            code_row, text="Join",
            command=self._join_by_code,
            font=(FONT.P, 12, "bold"),
            fg_color=CLR.CARD, hover_color=CLR.ACCENT,
            text_color=CLR.TEXT, corner_radius=RADIUS.DEFAULT,
            height=40, width=90,
        ).grid(row=0, column=1)

        list_hdr = customtkinter.CTkFrame(self._find_frame, fg_color="transparent")
        list_hdr.grid(row=3, column=0, columnspan=2, sticky="ew", pady=(0, 6))
        list_hdr.grid_columnconfigure(0, weight=1)

        customtkinter.CTkLabel(
            list_hdr, text="Room Tersedia",
            font=(FONT.P, 13, "bold"), text_color=CLR.MUTED,
        ).grid(row=0, column=0, sticky="w")

        customtkinter.CTkButton(
            list_hdr, text="Refresh",
            command=self._refresh_rooms,
            font=(FONT.P, 11, "bold"),
            fg_color=CLR.CARD, hover_color=CLR.SUBTLE_BORDER,
            text_color=CLR.TEXT, corner_radius=RADIUS.SMALL,
            height=34, width=90,
        ).grid(row=0, column=1)

        self._room_list = customtkinter.CTkScrollableFrame(
            self._find_frame, fg_color=CLR.CARD,
            corner_radius=RADIUS.DEFAULT,
            border_width=1, border_color=CLR.SUBTLE_BORDER, height=160,
        )
        self._room_list.grid(row=4, column=0, columnspan=2, sticky="ew")
        self._room_list.grid_columnconfigure(0, weight=1)

        self._rooms_placeholder = customtkinter.CTkLabel(
            self._room_list, text="Memuat...",
            font=(FONT.P, 13, "bold"), text_color=CLR.MUTED,
        )
        self._rooms_placeholder.grid(row=0, column=0, pady=24)

    def _build_lobby_panel(self):
        self._lobby_frame = customtkinter.CTkFrame(self, fg_color=CLR.BG)
        self._lobby_frame.grid_columnconfigure(0, weight=1)

        # Room code card
        code_card = customtkinter.CTkFrame(
            self._lobby_frame, fg_color=CLR.CARD,
            corner_radius=RADIUS.DEFAULT, border_width=2, border_color=CLR.ACCENT,
        )
        code_card.grid(row=0, column=0, sticky="ew", pady=(0, 16))
        code_card.grid_columnconfigure(0, weight=1)

        customtkinter.CTkLabel(
            code_card, text="KODE ROOM",
            font=(FONT.P, 12, "bold"), text_color=CLR.MUTED,
        ).grid(row=0, column=0, pady=(16, 0))

        self._code_display = customtkinter.CTkLabel(
            code_card, text="----",
            font=(FONT.M, 44, "bold"), text_color=CLR.ACCENT,
        )
        self._code_display.grid(row=1, column=0, pady=(6, 8))

        customtkinter.CTkButton(
            code_card, text="Salin Kode",
            command=self._copy_code,
            font=(FONT.P, 11, "bold"),
            fg_color=CLR.SUBTLE_BORDER, hover_color=CLR.ACCENT,
            text_color=CLR.TEXT, corner_radius=RADIUS.SMALL,
            height=32, width=100,
        ).grid(row=2, column=0, pady=(0, 16))

        # Player cards
        players = customtkinter.CTkFrame(self._lobby_frame, fg_color="transparent")
        players.grid(row=1, column=0, sticky="ew", pady=(0, 12))
        players.grid_columnconfigure((0, 1), weight=1)

        def _player_card(col):
            c = customtkinter.CTkFrame(
                players, fg_color=CLR.CARD, corner_radius=RADIUS.DEFAULT,
                border_width=1, border_color=CLR.SUBTLE_BORDER,
            )
            c.grid(
                row=0, column=col,
                padx=(0, 8) if col == 0 else (8, 0), sticky="nsew",
            )
            c.grid_columnconfigure(0, weight=1)
            return c

        p1 = _player_card(0)
        customtkinter.CTkLabel(p1, text="PLAYER 1", font=(FONT.P, 11, "bold"), text_color=CLR.MUTED).grid(row=0, column=0, pady=(14, 4))
        self._p1_name   = customtkinter.CTkLabel(p1, text="...", font=(FONT.P, 18, "bold"), text_color=CLR.TEXT)
        self._p1_name.grid(row=1, column=0, pady=(0, 6))
        self._p1_status = customtkinter.CTkLabel(p1, text="Menunggu", font=(FONT.P, 11, "bold"), text_color=CLR.MUTED, fg_color=CLR.BG, corner_radius=RADIUS.SMALL, width=110, height=26)
        self._p1_status.grid(row=2, column=0, pady=(0, 14))

        p2 = _player_card(1)
        customtkinter.CTkLabel(p2, text="PLAYER 2", font=(FONT.P, 11, "bold"), text_color=CLR.MUTED).grid(row=0, column=0, pady=(14, 4))
        self._p2_name   = customtkinter.CTkLabel(p2, text="Menunggu...", font=(FONT.P, 18, "bold"), text_color=CLR.MUTED)
        self._p2_name.grid(row=1, column=0, pady=(0, 6))
        self._p2_status = customtkinter.CTkLabel(p2, text="", font=(FONT.P, 11, "bold"), text_color=CLR.MUTED, fg_color=CLR.BG, corner_radius=RADIUS.SMALL, width=110, height=26)
        self._p2_status.grid(row=2, column=0, pady=(0, 14))

        self._lobby_msg = customtkinter.CTkLabel(
            self._lobby_frame, text="",
            font=(FONT.P, 12), text_color=CLR.MUTED,
        )
        self._lobby_msg.grid(row=2, column=0, pady=(0, 8))

        btn_row = customtkinter.CTkFrame(self._lobby_frame, fg_color="transparent")
        btn_row.grid(row=3, column=0, sticky="ew")
        btn_row.grid_columnconfigure((0, 1), weight=1)

        self._ready_btn = customtkinter.CTkButton(
            btn_row, text="SIAP",
            command=self._mark_ready,
            font=(FONT.P, 16, "bold"),
            fg_color=CLR.CARD, hover_color=CLR.ACCENT,
            text_color=CLR.TEXT, corner_radius=RADIUS.DEFAULT, height=52,
        )
        self._ready_btn.grid(row=0, column=0, padx=(0, 8), sticky="ew")

        customtkinter.CTkButton(
            btn_row, text="Keluar Room",
            command=self._leave_room,
            font=(FONT.P, 14, "bold"),
            fg_color=CLR.DANGER, hover_color=CLR.DANGER_HOVER,
            text_color=CLR.TEXT, corner_radius=RADIUS.DEFAULT, height=52,
        ).grid(row=0, column=1, padx=(8, 0), sticky="ew")

    # ── Panel switch ──────────────────────────────────────────────────────────

    def _show_find(self):
        self._lobby_frame.grid_remove()
        self._find_frame.grid(row=2, column=0, sticky="nsew", padx=20, pady=16)

    def _show_lobby(self, room: dict):
        self._find_frame.grid_remove()
        self._lobby_frame.grid(row=2, column=0, sticky="nsew", padx=20, pady=16)
        self._i_am_ready = False
        self._ready_btn.configure(text="SIAP", fg_color=CLR.CARD, state="normal")
        self._update_lobby(room)
        self._start_polling()

    # ── Async API helper ──────────────────────────────────────────────────────

    def _api_call(self, fn, on_success, on_error=None):
        def _worker():
            try:
                result = fn()
                self.after(0, lambda r=result: on_success(r))
            except Exception as e:
                cb = on_error or self._show_err
                self.after(0, lambda m=str(e): cb(m))
        threading.Thread(target=_worker, daemon=True).start()

    def _show_err(self, msg: str):
        import tkinter.messagebox as mb
        mb.showerror("Error", msg, parent=self)

    # ── Room list ─────────────────────────────────────────────────────────────

    def _refresh_rooms(self):
        import requests
        def _fetch():
            resp = requests.get(f"{API_SERVER_URL}/api/rooms", timeout=5)
            resp.raise_for_status()
            return resp.json().get("rooms", [])
        self._api_call(_fetch, self._update_room_list)

    def _update_room_list(self, rooms: list):
        for w in self._room_list.winfo_children():
            w.destroy()
        self._room_list.grid_columnconfigure(0, weight=1)

        active = [
            r for r in rooms
            if r.get("status") in ("waiting", "ready") and not r.get("player2_name")
        ]
        if not active:
            customtkinter.CTkLabel(
                self._room_list, text="Belum ada room",
                font=(FONT.P, 13), text_color=CLR.MUTED,
            ).grid(row=0, column=0, pady=24)
            return

        for i, r in enumerate(active):
            row = customtkinter.CTkFrame(
                self._room_list, fg_color=CLR.BG,
                corner_radius=RADIUS.SMALL,
                border_width=1, border_color=CLR.SUBTLE_BORDER,
            )
            row.grid(row=i, column=0, sticky="ew", pady=4, padx=4)
            row.grid_columnconfigure(1, weight=1)

            customtkinter.CTkLabel(
                row, text=r["id"],
                font=(FONT.P, 18, "bold"), text_color=CLR.ACCENT, width=64,
            ).grid(row=0, column=0, padx=(12, 8), pady=8)

            customtkinter.CTkLabel(
                row, text=r.get("player1_name", "—"),
                font=(FONT.P, 13), text_color=CLR.TEXT,
            ).grid(row=0, column=1, sticky="w")

            p_count = 2 if r.get("player2_name") else (1 if r.get("player1_name") else 0)
            customtkinter.CTkLabel(
                row, text=f"{p_count}/2",
                font=(FONT.P, 12), text_color=CLR.MUTED, width=32,
            ).grid(row=0, column=2)

            code = r["id"]
            customtkinter.CTkButton(
                row, text="Join",
                command=lambda c=code: self._join_by_id(c),
                font=(FONT.P, 12, "bold"),
                fg_color=CLR.ACCENT, hover_color=CLR.ACCENT2,
                corner_radius=RADIUS.SMALL, height=38, width=70,
            ).grid(row=0, column=3, padx=(8, 12), pady=8)

    # ── Create / Join ─────────────────────────────────────────────────────────

    def _create_room(self):
        import requests
        def _do():
            resp = requests.post(
                f"{API_SERVER_URL}/api/rooms",
                json={"player_name": STATE.nick_name}, timeout=5,
            )
            resp.raise_for_status()
            return resp.json()
        self._api_call(_do, self._on_joined, self._show_err)

    def _join_by_code(self):
        code = self._code_entry.get().strip().upper()
        if len(code) != 4:
            self._show_err("Kode room harus 4 karakter")
            return
        self._join_by_id(code)

    def _join_by_id(self, code: str):
        import requests
        def _do():
            resp = requests.post(
                f"{API_SERVER_URL}/api/rooms/{code}/join",
                json={"player_name": STATE.nick_name}, timeout=5,
            )
            resp.raise_for_status()
            return resp.json()
        self._api_call(_do, self._on_joined, self._show_err)

    def _on_joined(self, room: dict):
        self._room_code  = room["id"]
        self._player_num = 1 if room.get("player1_name") == STATE.nick_name else 2
        self._connect_ws()
        self._show_lobby(room)

    # ── Lobby actions ─────────────────────────────────────────────────────────

    def _mark_ready(self):
        if self._i_am_ready:
            return
        import requests
        def _do():
            resp = requests.post(
                f"{API_SERVER_URL}/api/rooms/{self._room_code}/ready",
                json={"player_name": STATE.nick_name}, timeout=5,
            )
            resp.raise_for_status()
            return resp.json()
        self._api_call(_do, self._on_ready_response, self._show_err)

    def _on_ready_response(self, room: dict):
        self._i_am_ready = True
        self._ready_btn.configure(
            text="SIAP!", fg_color=CLR.SUBTLE_BORDER, state="disabled",
        )
        self._update_lobby(room)
        if room.get("status") == "playing":
            self._on_game_start()

    def _leave_room(self):
        if not self._room_code:
            self._stop_polling()
            self._show_find()
            self._refresh_rooms()
            return
        import requests
        code = self._room_code
        self._room_code = None
        self._stop_polling()
        def _do():
            try:
                requests.post(
                    f"{API_SERVER_URL}/api/rooms/{code}/leave",
                    json={"player_name": STATE.nick_name}, timeout=5,
                )
            except Exception:
                pass
        threading.Thread(target=_do, daemon=True).start()
        self._show_find()
        self._refresh_rooms()

    def _back_to_input(self):
        if self._room_code:
            import requests
            code = self._room_code
            def _do():
                try:
                    requests.post(
                        f"{API_SERVER_URL}/api/rooms/{code}/leave",
                        json={"player_name": STATE.nick_name}, timeout=5,
                    )
                except Exception:
                    pass
            threading.Thread(target=_do, daemon=True).start()
        self._stop_polling()
        STATE.close_ws()
        self.room_ready = False
        self.destroy()

    def _copy_code(self):
        if self._room_code:
            self.clipboard_clear()
            self.clipboard_append(self._room_code)

    # ── Lobby UI update ───────────────────────────────────────────────────────

    def _update_lobby(self, room: dict):
        self._code_display.configure(text=room.get("id", "----"))
        p1 = room.get("player1_name") or "—"
        p2 = room.get("player2_name")
        r1 = room.get("player1_ready", False)
        r2 = room.get("player2_ready", False)
        st = room.get("status", "waiting")

        self._p1_name.configure(text=p1)
        self._p1_status.configure(
            text="SIAP" if r1 else "Belum Siap",
            text_color=CLR.SUCCESS if r1 else CLR.MUTED,
            fg_color=CLR.SUCCESS_BG if r1 else CLR.BG,
        )
        if p2:
            self._p2_name.configure(text=p2, text_color=CLR.TEXT)
            self._p2_status.configure(
                text="SIAP" if r2 else "Belum Siap",
                text_color=CLR.SUCCESS if r2 else CLR.MUTED,
                fg_color=CLR.SUCCESS_BG if r2 else CLR.BG,
            )
        else:
            self._p2_name.configure(text="Menunggu pemain...", text_color=CLR.MUTED)
            self._p2_status.configure(text="", fg_color=CLR.BG)

        msg_map = {
            "playing": "Kedua pemain siap! Game dimulai...",
        }
        if st == "ready" and r1 and not r2:
            msg = "Menunggu Player 2 tekan SIAP..."
        elif st == "ready" and not r1 and r2:
            msg = "Menunggu Player 1 tekan SIAP..."
        elif st == "ready":
            msg = "Kedua pemain hadir. Tekan SIAP!"
        else:
            msg = msg_map.get(st, "Menunggu pemain lain bergabung...")
        self._lobby_msg.configure(text=msg)

    # ── Polling ───────────────────────────────────────────────────────────────

    def _start_polling(self):
        self._polling = True
        self.after(2000, self._poll)

    def _stop_polling(self):
        self._polling = False

    def _poll(self):
        if not self._polling or not self._room_code:
            return
        import requests
        code = self._room_code
        def _worker():
            try:
                resp = requests.get(f"{API_SERVER_URL}/api/rooms/{code}", timeout=3)
                data = None if resp.status_code == 404 else resp.json()
            except Exception:
                data = "__error__"
            try:
                self.after(0, lambda d=data: self._handle_poll(d))
            except Exception:
                pass
        threading.Thread(target=_worker, daemon=True).start()

    def _handle_poll(self, room):
        if not self._polling:
            return
        if room is None:
            self._stop_polling()
            self._room_code = None
            import tkinter.messagebox as mb
            mb.showinfo("Info", "Room dihapus oleh server", parent=self)
            self._show_find()
            self._refresh_rooms()
            return
        if room == "__error__":
            if self._polling:
                self.after(3000, self._poll)
            return
        self._update_lobby(room)
        if room.get("status") == "playing":
            self._on_game_start()
        elif self._polling:
            self.after(2000, self._poll)

    def _on_game_start(self):
        self._stop_polling()
        STATE.current_room_code = self._room_code or ""
        STATE.player_num        = self._player_num or 1
        self.room_ready         = True
        self.destroy()

    # ── WebSocket ─────────────────────────────────────────────────────────────

    def _connect_ws(self):
        if not API_SERVER_URL or not self._room_code:
            return
        STATE.close_ws()
        STATE.ws_client = GameWebSocket(
            room_code=self._room_code,
            player_num=self._player_num,
            player_name=STATE.nick_name,
            on_message=self._on_ws_message,
        )
        STATE.ws_client.connect()

    def _on_ws_message(self, data: dict):
        try:
            self.after(0, lambda d=data: self._handle_ws(d))
        except Exception:
            pass

    def _handle_ws(self, data: dict):
        msg_type = data.get("type", "")
        if msg_type == "room_update":
            room = {
                "id":            data.get("room_id", ""),
                "status":        data.get("status", ""),
                "player1_name":  data.get("player1_name", ""),
                "player2_name":  data.get("player2_name", ""),
                "player1_ready": data.get("status") in ("ready", "playing"),
                "player2_ready": data.get("status") == "playing",
            }
            self._update_lobby(room)
            if room["status"] == "playing":
                self._on_game_start()
        elif msg_type in ("match_start",):
            self._on_game_start()

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def _on_close(self):
        self._stop_polling()
        STATE.close_ws()
        if self._room_code:
            import requests
            try:
                requests.post(
                    f"{API_SERVER_URL}/api/rooms/{self._room_code}/leave",
                    json={"player_name": STATE.nick_name}, timeout=3,
                )
            except Exception:
                pass
        self.destroy()