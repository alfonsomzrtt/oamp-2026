"""
api/client.py — low-level HTTP retry engine.

Tidak ada business logic di sini — hanya transport.
Module lain di api/ import _post() dan _save_backup() dari sini.
"""

from __future__ import annotations
import json
import time
import threading
from pathlib import Path
from config import API_SERVER_URL, RESULTS_DIR


_MAX_RETRIES   = 3
_RETRY_BACKOFF = [1, 2, 4]   # seconds


def post(url: str, payload: dict, timeout: int = 10) -> bool:
    """
    POST JSON dengan retry.
    Blocking — jalankan dari thread jika tidak mau block GUI.
    Return True jika sukses (2xx), False setelah semua retry habis.
    """
    import requests

    for attempt in range(_MAX_RETRIES):
        try:
            resp = requests.post(url, json=payload, timeout=timeout)
            if 200 <= resp.status_code < 300:
                return True
            print(f">>> [API] POST {url} → {resp.status_code} (attempt {attempt+1}/{_MAX_RETRIES})")
        except Exception as e:
            print(f">>> [API] POST {url} error: {e} (attempt {attempt+1}/{_MAX_RETRIES})")

        if attempt < _MAX_RETRIES - 1:
            time.sleep(_RETRY_BACKOFF[attempt])

    return False


def get(url: str, timeout: int = 5):
    """
    GET JSON.
    Return (True, data) jika 200, (False, {}) jika gagal.
    Blocking.
    """
    import requests

    try:
        resp = requests.get(url, timeout=timeout)
        if resp.status_code == 200:
            return True, resp.json()
        return False, {}
    except Exception as e:
        print(f">>> [API] GET {url} error: {e}")
        return False, {}


def save_backup(filename: str, payload: dict):
    """Simpan payload ke results/ sebagai JSON fallback."""
    try:
        path = RESULTS_DIR / filename
        path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        print(f">>> [LOCAL] Backup saved: {path}")
    except Exception as e:
        print(f">>> [LOCAL] Backup write failed: {e}")


def fire_async(url: str, payload: dict, fallback_filename: str | None = None):
    """
    POST di daemon thread — fire and forget.
    Jika gagal dan fallback_filename diberikan, simpan lokal.
    """
    def _do():
        ok = post(url, payload)
        if not ok and fallback_filename:
            save_backup(fallback_filename, payload)

    threading.Thread(target=_do, daemon=True).start()