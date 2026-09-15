"""
api/tournament.py — tournament cup events dan active-match check.
"""

from __future__ import annotations
import time
import threading
from typing import Callable
from config import API_SERVER_URL
from api.client import get, post, save_backup


def check_active_match(uid: str, callback: Callable[[bool, dict], None]):
    """
    GET /api/tournaments/active-match/{uid}.
    callback(has_match: bool, match_data: dict).
    """
    if not API_SERVER_URL:
        callback(False, {})
        return

    def _do():
        ok, data = get(f"{API_SERVER_URL}/api/tournaments/active-match/{uid}")
        if ok and data.get("status") == "success" and data["data"].get("has_match"):
            callback(True, data["data"])
        else:
            callback(False, {})

    threading.Thread(target=_do, daemon=True).start()


def send_event(room_code: str, event_type: str, player_num: int = 0, score: float = 0):
    """
    POST /api/tournaments/event — match_started / match_finished.
    Fire-and-forget dengan retry + local backup.
    """
    if not API_SERVER_URL:
        return

    def _do():
        payload: dict = {"room_id": room_code, "event_type": event_type}
        if player_num > 0:
            payload["player_num"] = player_num
        if score > 0:
            payload["score"] = score

        ok = False
        for attempt in range(3):
            import requests
            try:
                resp = requests.post(
                    f"{API_SERVER_URL}/api/tournaments/event",
                    json=payload, timeout=5,
                )
                if 200 <= resp.status_code < 300:
                    print(f">>> [TOURNAMENT] '{event_type}' sent ({resp.status_code})")
                    ok = True
                    break
                print(f">>> [TOURNAMENT] '{event_type}' failed ({resp.status_code}) attempt {attempt+1}/3")
            except Exception as e:
                print(f">>> [TOURNAMENT] '{event_type}' error: {e} attempt {attempt+1}/3")
            if attempt < 2:
                time.sleep(1 + attempt)

        if not ok:
            save_backup(f"tournament_{event_type}_{room_code}_{int(time.time())}.json", payload)

    threading.Thread(target=_do, daemon=True).start()