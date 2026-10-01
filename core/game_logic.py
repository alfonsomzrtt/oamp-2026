"""
core/game_logic.py — pure game logic, tidak ada GUI atau I/O.

Semua fungsi di sini stateless dan thread-safe.
"""

from __future__ import annotations
import json
import sys
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np

from config import ERROR_LOG_PATH, CONSENT_FILE, CONSENT_VERSION


# ── Cognitive age estimation ──────────────────────────────────────────────────

# Time-to-age lookup table (detik → estimasi usia kognitif)
_TIME_POINTS = np.array([ 9, 10, 11, 12, 14, 16, 18, 20, 25, 30, 35, 40, 45, 50], dtype=float)
_AGE_POINTS  = np.array([20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85], dtype=float)


def estimate_cognitive_age(time_finish_sec: float) -> int:
    """
    Estimasi usia kognitif berdasarkan waktu penyelesaian satu task (detik).
    Clamp ke range [20, 90].
    """
    age = int(np.interp(time_finish_sec, _TIME_POINTS, _AGE_POINTS))
    return max(20, min(90, age))


def compute_visuo_spatial(age_real: int, age_cognitive: int) -> int:
    """
    Hitung skor visual-spasial (0-100).
    Semakin muda usia kognitif dibanding usia asli → semakin tinggi skor.
    """
    if age_cognitive > age_real:
        return max(0, 100 - (age_cognitive - age_real))
    return 100


def sfx_for_time(elapsed: float) -> str:
    """Pilih nama SFX effect berdasarkan waktu penyelesaian."""
    if   elapsed < 10: return "amazing"
    elif elapsed < 15: return "great"
    elif elapsed < 20: return "solid"
    elif elapsed < 25: return "good"
    elif elapsed < 30: return "keep_going"
    else:              return "dont_give_up"


def star_rating(total_time: float) -> tuple[str, str]:
    """
    Return (stars_str, label) berdasarkan total waktu semua level.
    """
    if   total_time < 60:  return "⭐⭐⭐⭐⭐", "LUAR BIASA!"
    elif total_time < 90:  return "⭐⭐⭐⭐",   "HEBAT!"
    elif total_time < 120: return "⭐⭐⭐",     "BAGUS!"
    elif total_time < 180: return "⭐⭐",       "LUMAYAN"
    else:                  return "⭐",         "SEMANGAT!"


# ── Exception logging ─────────────────────────────────────────────────────────

def log_exception(context: str, exc_type, exc_value, exc_tb):
    """
    Tulis traceback ke ERROR_LOG_PATH dan stderr.
    Aman dipanggil dari excepthook atau thread excepthook.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    trace     = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
    payload   = f"[{timestamp}] {context}\n{trace}\n"

    try:
        with ERROR_LOG_PATH.open("a", encoding="utf-8") as f:
            f.write(payload)
    except Exception:
        pass

    try:
        sys.__stderr__.write(payload)
        sys.__stderr__.flush()
    except Exception:
        try:
            sys.__stderr__.buffer.write(payload.encode("utf-8", "backslashreplace"))
            sys.__stderr__.buffer.flush()
        except Exception:
            pass


def install_excepthooks():
    """
    Install global + thread excepthook yang menulis ke ERROR_LOG_PATH.
    Dipanggil sekali di main.py sebelum apapun.
    """
    import threading

    def _global(exc_type, exc_value, exc_tb):
        log_exception("Unhandled exception (main thread)", exc_type, exc_value, exc_tb)

    def _thread(args):
        name = args.thread.name if args.thread else "unknown"
        log_exception(
            f"Unhandled exception (thread: {name})",
            args.exc_type, args.exc_value, args.exc_traceback,
        )

    sys.excepthook = _global
    if hasattr(threading, "excepthook"):
        threading.excepthook = _thread


# ── Informed consent ──────────────────────────────────────────────────────────

def check_consent() -> bool:
    """Return True jika consent sudah diterima dan versi cocok."""
    print(f">>> [Consent] Checking: {CONSENT_FILE}")
    if not CONSENT_FILE.exists():
        return False
    try:
        data = json.loads(CONSENT_FILE.read_text(encoding="utf-8"))
        return (
            data.get("accepted") is True
            and data.get("consent_version") == CONSENT_VERSION
        )
    except Exception:
        return False  # file rusak / format lama


def mark_consent_accepted():
    """Simpan persetujuan dengan timestamp dan versi."""
    payload = {
        "accepted":        True,
        "consent_version": CONSENT_VERSION,
        "timestamp":       datetime.now().isoformat(),
    }
    try:
        CONSENT_FILE.write_text(
            json.dumps(payload, indent=2), encoding="utf-8"
        )
        print(f">>> [Consent] Accepted — v{CONSENT_VERSION} @ {payload['timestamp']}")
    except Exception as e:
        print(f">>> [Consent] Failed to write: {e}")


# ── .env persistence helper ───────────────────────────────────────────────────

def save_env_value(key: str, value: str, env_path: Path | None = None):
    """Update atau append key=value di .env file."""
    path = env_path or (Path(".") / ".env")
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True) if path.exists() else []

    updated   = False
    new_lines = []
    for line in lines:
        stripped = line.strip()
        k = stripped.split("=")[0] if "=" in stripped else ""
        if k == key:
            new_lines.append(f"{key}={value}\n")
            updated = True
        else:
            new_lines.append(line)

    if not updated:
        new_lines.append(f"{key}={value}\n")

    path.write_text("".join(new_lines), encoding="utf-8")