"""Window helpers shared by the primary application screens."""

from __future__ import annotations
import sys
import tkinter as tk


def maximize_window(window: tk.Misc) -> None:
    """Maximize a mapped Tk window, including Windows native windows."""
    def apply_maximize() -> None:
        try:
            window.update_idletasks()
            if sys.platform == "linux":
                window.attributes("-zoomed", True)
            else:
                window.state("zoomed")
        except tk.TclError:
            pass

    window.after_idle(apply_maximize)
    window.after(100, apply_maximize)
