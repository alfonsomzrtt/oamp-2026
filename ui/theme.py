"""
ui/theme.py — OAMP design tokens.

Import:
    from ui.theme import CLR, FONT, RADIUS, apply_ctk_theme
"""

from __future__ import annotations
from config import THEME
from pathlib import Path


# ── CTk global theme ─────────────────────────────────────────────────────────

def apply_ctk_theme():
    """Harus dipanggil sebelum widget CTk apapun dibuat."""
    import customtkinter
    customtkinter.set_appearance_mode("dark" if THEME == "dark" else "light")
    customtkinter.set_default_color_theme("dark-blue")


def _t(dark_val: str, light_val: str) -> str:
    return dark_val if THEME == "dark" else light_val


# ── Color palette ─────────────────────────────────────────────────────────────

class CLR:
    BG             = _t("#111117", "#F8F9FC")
    CARD           = _t("#1C1C26", "#FFFFFF")
    ACCENT         = "#E63946"
    ACCENT2        = "#C1121F"
    TEXT           = _t("#F1F5F9", "#0F172A")
    MUTED          = _t("#94A3B8", "#64748B")
    BORDER         = "#E63946"
    SUBTLE_BORDER  = _t("#2D2D3A", "#E2E8F0")
    TIMER_BG       = _t("#1C1C26", "#FFFFFF")
    DANGER         = "#E63946"
    DANGER_BG      = _t("#1A0505", "#FFF0F0")
    DANGER_HOVER   = "#9B2226"
    SUCCESS        = "#22C55E"
    SUCCESS_BG     = _t("#052E16", "#DCFCE7")
    SUCCESS_HOVER  = _t("#16A34A", "#86EFAC")
    WARNING        = "#F4A261"
    GOLD           = "#FFB800"
    GOLD_DARK      = "#D49B00"
    TEAL_SAFE      = "#34D399"
    ORANGE_WARN    = "#FB923C"
    FROST          = _t("#FFFFFF", "#F1F5F9")
    LEVEL_DONE     = "#22C55E"
    LEVEL_ACTIVE   = "#E63946"
    LEVEL_UPCOMING = _t("#2D2D3A", "#E2E8F0")


# ── Typography ────────────────────────────────────────────────────────────────

class FONT:
    PRIMARY = ("Segoe UI", "Roboto", "Helvetica Neue", "Arial", "sans-serif")
    MONO    = ("SF Mono", "Fira Code", "Consolas", "Courier New", "monospace")

    # Shorthand untuk yang paling sering dipakai
    P  = PRIMARY[0]
    M  = MONO[0]


# ── Corner radii ──────────────────────────────────────────────────────────────

class RADIUS:
    DEFAULT = 8
    SMALL   = 4
    PILL    = 12


# ── Theme persistence ─────────────────────────────────────────────────────────

def set_theme(new_theme: str) -> bool:
    """Persist theme ke .env. Perlu restart untuk apply."""
    if new_theme not in ("dark", "light"):
        return False
    env_path = Path(".") / ".env"
    lines = env_path.read_text().splitlines(keepends=True) if env_path.exists() else []
    updated = False
    new_lines = []
    for line in lines:
        if line.strip().startswith("THEME="):
            new_lines.append(f"THEME={new_theme}\n")
            updated = True
        else:
            new_lines.append(line)
    if not updated:
        new_lines.append(f"THEME={new_theme}\n")
    env_path.write_text("".join(new_lines))
    return True