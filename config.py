"""
config.py — OAMP application configuration.

Single source of truth untuk semua env vars dan compile-time constants.
Tidak ada import dari module OAMP lain di sini (no circular deps).
"""

import os
import sys
import random
from pathlib import Path
from dotenv import load_dotenv

# ── Base path resolution ──────────────────────────────────────────────────────

if getattr(sys, "frozen", False):
    BASE_DIR = Path(sys.executable).parent   # .exe
else:
    BASE_DIR = Path(__file__).parent         # .py

env_path = BASE_DIR / ".env"
load_dotenv(dotenv_path=env_path, override=True)


# ── Helper ────────────────────────────────────────────────────────────────────

def _get_bool(key: str, default: bool = False) -> bool:
    return os.getenv(key, str(default)).lower() == "true"

def _get_int(key: str, default: int = 0, min_val: int = None, max_val: int = None) -> int:
    try:
        val = int(os.getenv(key, str(default)))
        if min_val is not None and val < min_val:
            return default
        if max_val is not None and val > max_val:
            return default
        return val
    except (ValueError, TypeError):
        return default

def _get_float(key: str, default: float = 1.0, min_val: float = None, max_val: float = None) -> float:
    try:
        val = float(os.getenv(key, str(default)))
        if min_val is not None and val < min_val:
            return default
        if max_val is not None and val > max_val:
            return default
        return val
    except (ValueError, TypeError):
        return default

def sanitize_env_value(raw_value) -> str:
    """Trim dan buang inline comment (e.g. '1a,2b # notes')."""
    if raw_value is None:
        return ""
    value = str(raw_value).strip()
    if "#" in value:
        value = value.split("#", 1)[0].strip()
    return value


# ── Display / UI flags ────────────────────────────────────────────────────────

DISPLAY_HALF   = _get_bool("DISPLAY_HALF", True)
NORMAL_PATTERN = _get_bool("NORMAL_PATTERN", True)
BUTTON_MODE    = _get_bool("BUTTON_MODE", False)
HIDE_CAMERA    = _get_bool("HIDE_CAMERA", False)
THEME          = os.getenv("THEME", "dark").lower()
DEBUG_MODE     = _get_bool("DEBUG_MODE", False)


# ── Camera ────────────────────────────────────────────────────────────────────

CAMERA_INDEX      = _get_int("CAMERA_INDEX", 4)
CAMERA_MIRROR_X   = _get_bool("CAMERA_MIRROR_X", False)
CAMERA_MIRROR_Y   = _get_bool("CAMERA_MIRROR_Y", False)
CAMERA_ZOOM       = _get_float("CAMERA_ZOOM", 1.0, min_val=0.5, max_val=3.0)
CAMERA_BRIGHTNESS = _get_int("CAMERA_BRIGHTNESS", 128)
CAMERA_CONTRAST   = _get_int("CAMERA_CONTRAST", 128)
CAMERA_SATURATION = _get_int("CAMERA_SATURATION", 128)
CAMERA_CALIBRATION = _get_bool("CAMERA_CALIBRATION", False)

# Slider display bounds (UI percentage remapping)
SLIDER_BOUNDS = {
    "brightness": (64, 191),
    "contrast":   (77, 191),
    "saturation": (77, 166),
}

def remap_slider(val: int, attr: str) -> int:
    mn, mx = SLIDER_BOUNDS.get(attr, (0, 255))
    return round(mn + (val / 255) * (mx - mn))


# ── YOLO / Model ──────────────────────────────────────────────────────────────

USE_BANTAL_MODEL = _get_bool("MODEL_BANTAL", False)
YOLO_INFER_SIZE  = _get_int("YOLO_INFER_SIZE", 640)
YOLO_SKIP_FRAMES = _get_int("YOLO_SKIP_FRAMES", 2)
MEDIAPIPE_SKIP_FRAMES = _get_int("MEDIAPIPE_SKIP_FRAMES", 2)

MODEL_DIR       = BASE_DIR / "MODEL"
PATH_MODEL_BEST   = MODEL_DIR / "exp7" / "weights" / "best.pt"
PATH_MODEL_BANTAL = MODEL_DIR / "bantal" / "bantal.pt"
PATH_MODEL_YOLOV5 = MODEL_DIR / "yolov5"


# ── Game / Level ──────────────────────────────────────────────────────────────

MAX_LEVEL   = _get_int("MAX_LEVEL", 8, min_val=1, max_val=8)
PC_MODE     = os.getenv("PC_MODE", "training").lower()

LEVEL_VARIANTS: dict[int, list[str]] = {
    i: [f"{i}a", f"{i}b", f"{i}c", f"{i}d"] for i in range(1, 9)
}

# Answers — normal pattern
LEVEL_ANSWERS_NORMAL: dict[str, list[int]] = {
    "1a": [2,1,1,1], "1b": [2,1,1,2], "1c": [1,1,1,2], "1d": [1,2,2,1],
    "2a": [2,1,6,1], "2b": [3,2,1,2], "2c": [2,2,5,6], "2d": [6,2,2,6],
    "3a": [3,5,1,1], "3b": [1,5,1,6], "3c": [2,3,4,1], "3d": [6,5,3,4],
    "4a": [6,2,1,6], "4b": [3,2,2,5], "4c": [1,5,3,1], "4d": [4,2,2,6],
    "5a": [6,3,5,4], "5b": [5,4,6,3], "5c": [3,5,6,4], "5d": [4,4,6,6],
    "6a": [1,4,3,1], "6b": [6,1,5,5], "6c": [5,1,1,4], "6d": [5,4,4,5],
    "7a": [4,5,5,5], "7b": [4,5,5,6], "7c": [2,5,1,5], "7d": [1,2,3,6],
    "8a": [6,5,6,3], "8b": [6,3,4,2], "8c": [5,6,3,5], "8d": [3,6,5,4],
}

# Answers — otak atik pattern
LEVEL_ANSWERS_OTAK_ATIK: dict[str, list[int]] = {
    "1a": [1,2,1,2], "1b": [2,1,2,1], "1c": [2,1,1,2], "1d": [1,2,2,1],
    "2a": [2,4,1,1], "2b": [1,6,2,2], "2c": [1,2,2,4], "2d": [2,1,1,6],
    "3a": [4,1,5,1], "3b": [6,2,3,2], "3c": [5,1,4,1], "3d": [3,2,6,2],
    "4a": [4,1,1,6], "4b": [6,2,2,4], "4c": [1,5,3,1], "4d": [2,3,5,2],
    "5a": [4,3,5,6], "5b": [6,5,3,4], "5c": [6,6,6,6], "5d": [4,4,4,4],
    "6a": [5,3,4,6], "6b": [3,5,6,4], "6c": [6,6,4,4], "6d": [4,4,6,6],
    "7a": [3,4,6,5], "7b": [5,6,4,3], "7c": [3,6,4,5], "7d": [5,4,6,3],
    "8a": [5,4,4,5], "8b": [3,6,6,3], "8c": [4,5,3,6], "8d": [6,3,5,4],
}

LEVEL_ANSWERS = LEVEL_ANSWERS_NORMAL if NORMAL_PATTERN else LEVEL_ANSWERS_OTAK_ATIK


# ── Level image paths ─────────────────────────────────────────────────────────

def _build_level_paths() -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for variants in LEVEL_VARIANTS.values():
        for variant in variants:
            lvl_num = variant[0]
            lvl_letter = variant[1]
            if NORMAL_PATTERN:
                paths[variant] = (
                    BASE_DIR / "FILES" / "TEST_RANDOM_1500x1500"
                    / f"Lvl {lvl_num}{lvl_letter}.png"
                )
            else:
                paths[variant] = (
                    BASE_DIR / "FILES" / "TEST_RANDOM_OTAK_ATIK"
                    / f"Lvl {lvl_num}{lvl_letter}.PNG"
                )
    return paths

LEVEL_PATHS = _build_level_paths()


# ── Custom level override ─────────────────────────────────────────────────────

def parse_custom_levels(raw: str) -> dict[int, str]:
    """
    Parse '1a,2b,3c,...' → {1: '1a', 2: '2b', 3: '3c', ...}.
    Return {} jika format tidak valid.
    """
    if not raw or raw == "[]":
        return {}
    try:
        levels = [s.strip() for s in raw.split(",") if s.strip()]
        result: dict[int, str] = {}
        for lvl in levels:
            if len(lvl) != 2:
                return {}
            num, letter = lvl[0], lvl[1].lower()
            if not num.isdigit() or not (1 <= int(num) <= 8):
                return {}
            if letter not in ("a", "b", "c", "d"):
                return {}
            result[int(num)] = f"{num}{letter}"
        return result
    except Exception as e:
        print(f">>> [config] Error parsing CUSTOM_LEVEL: {e}")
        return {}

CUSTOM_LEVEL_STR = sanitize_env_value(os.getenv("CUSTOM_LEVEL", ""))
CUSTOM_LEVELS    = parse_custom_levels(CUSTOM_LEVEL_STR)

if CUSTOM_LEVELS:
    print(f">>> [config] Custom levels: {CUSTOM_LEVELS}")


def get_variant(level: int) -> str:
    """Pilih variant untuk level — custom jika ada, random jika tidak."""
    if level in CUSTOM_LEVELS:
        return CUSTOM_LEVELS[level]
    return f"{level}{random.choice(['a', 'b', 'c', 'd'])}"


# ── API / Server ──────────────────────────────────────────────────────────────

API_SERVER_URL = os.getenv("API_SERVER_URL", "").rstrip("/")
ROOM_ID        = os.getenv("ROOM_ID", "room_01")
PLAYER_NUM     = _get_int("PLAYER_NUM", 1)

CONSENT_VERSION = "1.0"
RESULTS_DIR     = BASE_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

APPDATA_DIR  = Path(os.environ.get("APPDATA", Path.home())) / "OAMP"
APPDATA_DIR.mkdir(parents=True, exist_ok=True)
CONSENT_FILE = APPDATA_DIR / "consent_accepted.json"

ERROR_LOG_PATH = BASE_DIR / "runtime_errors.log"


# ── Asset paths ───────────────────────────────────────────────────────────────

FILES_DIR = BASE_DIR / "FILES"

ASSET_PATHS = {
    # Test images
    "task_00":  FILES_DIR / "TEST_1000x1000" / "00.jpg",
    "task_0A":  FILES_DIR / "TEST_1000x1000" / "0A.jpg",
    "task_0Bx": FILES_DIR / "TEST_1000x1000" / "0Bx.jpg",
    "task_09":  FILES_DIR / "TEST_1000x1000" / "09.jpg",
    # Face labels 50x50
    **{f"face_{i:02d}": FILES_DIR / "LABEL_50x50" / f"{i:02d}.png" for i in range(1, 7)},
    # Block thumbnails 100x100
    **{f"img_{i:02d}":  FILES_DIR / "TEST_100x100" / f"{i:02d}.jpg" for i in range(10)},
}