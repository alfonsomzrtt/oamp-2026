"""
api/duel.py — 1v1 duel result submission dan polling.
"""

from __future__ import annotations
import time
import threading
from typing import Callable
from config import API_SERVER_URL
from api.client import post, get, save_backup, fire_async


def submit_result(room_code: str, player_uid: str, player_num: int, score: float):
    """
    POST /api/rooms/{code}/result.
    Server menentukan pemenang saat kedua player submit.
    """
    if not API_SERVER_URL:
        return

    payload = {
        "player_uid": player_uid,
        "player_num": player_num,
        "score":      score,
    }

    fire_async(
        f"{API_SERVER_URL}/api/rooms/{room_code}/result",
        payload,
        fallback_filename=f"duel_{room_code}_{player_uid}_{int(time.time())}.json",
    )


def poll_result(
    room_code: str,
    callback: Callable[[bool, dict], None],
    interval: float = 3.0,
    max_attempts: int = 60,
):
    """
    GET /api/rooms/{code}/result secara periodik sampai winner ditentukan.
    callback(decided: bool, data: dict).
    """
    if not API_SERVER_URL:
        callback(False, {})
        return

    def _do():
        for _ in range(max_attempts):
            ok, data = get(f"{API_SERVER_URL}/api/rooms/{room_code}/result")
            if ok and data.get("winner", "") != "":
                print(f">>> [DUEL] Winner: {data['winner']}")
                callback(True, data)
                return
            time.sleep(interval)

        print(f">>> [DUEL] Poll timed out after {max_attempts} attempts")
        callback(False, {})

    threading.Thread(target=_do, daemon=True).start()