"""
ui/widgets.py — reusable CustomTkinter widget components.

Semua widget di sini tidak punya state game — murni presentasi.
"""

from __future__ import annotations
import customtkinter
from ui.theme import CLR, FONT, RADIUS


# ── UID text entry ────────────────────────────────────────────────────────────

class UIDEntryFrame(customtkinter.CTkFrame):
    """
    Single-line entry untuk UID peserta dengan placeholder dan Return binding.
    Dulunya MyTextboxFrame di main.py.
    """

    def __init__(self, master, **kwargs):
        super().__init__(master, fg_color="transparent", **kwargs)
        self.grid_columnconfigure(0, weight=1)

        self._entry = customtkinter.CTkEntry(
            self,
            width=200,
            height=52,
            corner_radius=RADIUS.DEFAULT,
            font=(FONT.P, 26),
            justify="center",
            placeholder_text="Ketik atau scan UID...",
            border_width=2,
            border_color=CLR.SUBTLE_BORDER,
            fg_color=CLR.BG,
            text_color=CLR.TEXT,
        )
        self._entry.grid(row=0, column=0, sticky="ew")

    def get(self) -> str:
        return self._entry.get().strip()

    def set(self, text: str):
        self._entry.delete(0, "end")
        self._entry.insert(0, text)

    def bind_return(self, callback):
        self._entry.bind("<Return>", lambda _e: callback())

    def focus(self):
        self._entry.focus_set()


# ── Section label ─────────────────────────────────────────────────────────────

class SectionLabel(customtkinter.CTkFrame):
    """
    Label judul section dengan warna muted.
    Dulunya TextFrame di main.py.
    """

    def __init__(self, master, title: str, **kwargs):
        super().__init__(master, fg_color="transparent", **kwargs)
        self.grid_columnconfigure(0, weight=1)

        customtkinter.CTkLabel(
            self,
            text=title,
            fg_color="transparent",
            text_color=CLR.MUTED,
            font=(FONT.P, 14, "bold"),
            corner_radius=0,
        ).grid(row=0, column=0, padx=0, pady=(0, 8), sticky="w")


# ── Bordered image frame ──────────────────────────────────────────────────────

class BorderedImageFrame(customtkinter.CTkFrame):
    """
    Frame dengan border accent untuk menampilkan CTkImage / CTkLabel.
    Dulunya ImageFrame di main.py.
    """

    def __init__(
        self,
        master,
        border_width: int = 0,
        border_color: str = CLR.BORDER,
        **kwargs,
    ):
        super().__init__(
            master,
            border_width=border_width,
            border_color=border_color,
            fg_color=CLR.BG,
            corner_radius=RADIUS.DEFAULT,
            **kwargs,
        )
        self.grid_columnconfigure(0, weight=1)


# ── Stat card ─────────────────────────────────────────────────────────────────

class StatCard(customtkinter.CTkFrame):
    """
    Kecil card untuk menampilkan satu metric (title + value).
    Dipakai di results dashboard.
    """

    def __init__(self, master, title: str, value: str, **kwargs):
        super().__init__(
            master,
            fg_color=CLR.CARD,
            corner_radius=RADIUS.DEFAULT,
            border_width=1,
            border_color=CLR.SUBTLE_BORDER,
            **kwargs,
        )
        self.grid_columnconfigure(0, weight=1)

        customtkinter.CTkLabel(
            self, text=title,
            font=(FONT.P, 12), text_color=CLR.MUTED,
        ).grid(row=0, column=0, pady=(12, 0))

        customtkinter.CTkLabel(
            self, text=value,
            font=(FONT.P, 20, "bold"), text_color=CLR.TEXT,
        ).grid(row=1, column=0, pady=(0, 12))


# ── Level progress bar row ────────────────────────────────────────────────────

class LevelTimeRow(customtkinter.CTkFrame):
    """
    Satu baris di results dashboard: label level + progress bar + waktu.
    """

    def __init__(self, master, level: int, time_sec: float, max_time: float = 60.0, **kwargs):
        super().__init__(master, fg_color="transparent", **kwargs)
        self.grid_columnconfigure(1, weight=1)

        if   time_sec < 15: bar_color = CLR.LEVEL_DONE
        elif time_sec < 25: bar_color = CLR.WARNING
        else:               bar_color = CLR.DANGER

        customtkinter.CTkLabel(
            self, text=f"L{level}",
            font=(FONT.P, 12, "bold"), text_color=CLR.MUTED, width=28,
        ).grid(row=0, column=0, padx=(0, 6))

        bar_bg = customtkinter.CTkFrame(
            self, fg_color=CLR.CARD, corner_radius=3, height=20,
        )
        bar_bg.grid(row=0, column=1, sticky="ew")

        fill_pct = min(time_sec / max_time, 1.0)
        customtkinter.CTkFrame(
            bar_bg, fg_color=bar_color, corner_radius=3, height=20,
        ).place(relx=0, rely=0, relwidth=fill_pct, relheight=1.0)

        customtkinter.CTkLabel(
            self, text=f"{time_sec:.1f}s",
            font=(FONT.P, 12, "bold"), text_color=CLR.TEXT, width=50,
        ).grid(row=0, column=2, padx=(6, 0))