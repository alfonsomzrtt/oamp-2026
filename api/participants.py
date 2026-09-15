"""
api/participants.py — participant verification dan result submission.
"""

from __future__ import annotations
import time
import threading
from typing import Callable
from config import API_SERVER_URL
from api.client import get, post, save_backup, fire_async


def verify(uid: str, callback: Callable[[bool, dict], None]):
    """
    GET /api/v1/participants/uid/{uid} — non-blocking.
    callback(exists: bool, data: dict) dipanggil dari daemon thread.
    """
    if not API_SERVER_URL:
        callback(False, {})
        return

    def _do():
        ok, data = get(f"{API_SERVER_URL}/api/v1/participants/uid/{uid}")
        callback(ok, data)

    threading.Thread(target=_do, daemon=True).start()


def submit_results(uid: str, payload: dict) -> threading.Thread | None:
    """
    POST /api/v1/game/submit dengan retry + local backup.
    Return thread agar caller bisa join() jika perlu.
    Jika tidak ada API_SERVER_URL, simpan lokal dan return None.
    """
    if not API_SERVER_URL:
        save_backup(f"offline_{uid}_{int(time.time())}.json", payload)
        print(f">>> [API] No API_SERVER_URL — saved locally")
        return None

    def _do():
        ok = post(f"{API_SERVER_URL}/api/v1/game/submit", payload, timeout=10)
        if ok:
            print(f">>> [API] Results submitted for UID: {uid}")
        else:
            print(f">>> [API] Submit FAILED — saving locally")
            save_backup(f"failed_{uid}_{int(time.time())}.json", payload)

    t = threading.Thread(target=_do, daemon=True)
    t.start()
    return t


def save_training_locally(payload: dict):
    """Simpan hasil training tanpa UID (anonymous dataset)."""
    save_backup(f"training_{int(time.time())}.json", payload)
    print(">>> [LOCAL] Training result saved")