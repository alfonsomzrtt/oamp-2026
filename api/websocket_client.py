"""
api/websocket_client.py — threaded WebSocket untuk real-time match events.
"""

from __future__ import annotations
import json
import time
import threading
from typing import Callable
from config import API_SERVER_URL

_RECONNECT_DELAYS = [1, 2, 4, 8, 16]


class GameWebSocket:
    """
    Threaded WebSocket client.

    Connect ke /ws/match/{room_code}?role=player&...
    Dispatch pesan masuk ke on_message callback (dipanggil dari WS thread —
    caller bertanggung jawab untuk after() ke main thread jika perlu update UI).
    """

    def __init__(
        self,
        room_code: str,
        player_num: int,
        player_name: str,
        on_message: Callable[[dict], None],
    ):
        self.room_code   = room_code
        self.player_num  = player_num
        self.player_name = player_name
        self.on_message  = on_message

        self._ws             = None
        self._connected      = False
        self._should_reconnect = True
        self._lock           = threading.Lock()
        self._thread: threading.Thread | None = None

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def connect(self):
        self._should_reconnect = True
        self._thread = threading.Thread(target=self._run, daemon=True, name="WSThread")
        self._thread.start()

    def close(self):
        self._should_reconnect = False
        self._connected = False
        with self._lock:
            if self._ws:
                try:
                    self._ws.close()
                except Exception:
                    pass
                self._ws = None
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2)

    @property
    def connected(self) -> bool:
        return self._connected

    # ── Send ──────────────────────────────────────────────────────────────────

    def send(self, data: dict):
        with self._lock:
            if self._ws:
                try:
                    self._ws.send(json.dumps(data))
                except Exception as e:
                    print(f">>> [WS] Send error: {e}")

    # ── Internal ─────────────────────────────────────────────────────────────

    def _build_url(self) -> str:
        from urllib.parse import quote
        base = (
            API_SERVER_URL
            .replace("http://", "ws://")
            .replace("https://", "wss://")
            .rstrip("/")
        )
        pid = f"P{self.player_num}"
        return (
            f"{base}/ws/match/{quote(self.room_code, safe='')}"
            f"?role=player&player_id={pid}"
            f"&player_name={quote(self.player_name, safe='')}"
            f"&player_num={self.player_num}"
        )

    def _run(self):
        try:
            from websocket import WebSocketApp
        except ImportError:
            print(">>> [WS] websocket-client not installed — WS disabled")
            return

        attempt = 0
        while self._should_reconnect:
            try:
                url = self._build_url()
                print(f">>> [WS] Connecting to {url}")
                ws = WebSocketApp(
                    url,
                    on_open=self._on_open,
                    on_message=self._on_msg,
                    on_error=self._on_err,
                    on_close=self._on_close,
                )
                with self._lock:
                    self._ws = ws
                ws.run_forever(ping_interval=30, ping_timeout=10)
            except Exception as e:
                print(f">>> [WS] Error: {e}")

            if not self._should_reconnect:
                break

            delay = _RECONNECT_DELAYS[min(attempt, len(_RECONNECT_DELAYS) - 1)]
            print(f">>> [WS] Reconnecting in {delay}s...")
            time.sleep(delay)
            attempt += 1

        print(">>> [WS] Disconnected permanently")

    def _on_open(self, ws):
        self._connected = True
        print(">>> [WS] Connected")

    def _on_msg(self, ws, message: str):
        try:
            self.on_message(json.loads(message))
        except Exception as e:
            print(f">>> [WS] Parse error: {e}")

    def _on_err(self, ws, error):
        print(f">>> [WS] Error: {error}")

    def _on_close(self, ws, code, msg):
        self._connected = False
        print(f">>> [WS] Closed: {code} {msg}")