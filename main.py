"""
main.py — OAMP entry point (thin orchestrator).

Urutan inisialisasi:
1. Exception hooks
2. config + STATE sync (via import)
3. YOLO model load
4. Face asset load
5. Consent check
6. UI flow: InputScreen → [RoomScreen] → GameScreen
"""

import sys
import traceback

import cv2

from config import BASE_DIR, USE_BANTAL_MODEL
from state import STATE
from core.game_logic import install_excepthooks, check_consent, mark_consent_accepted
from ui.theme import apply_ctk_theme


# ── Exception hooks (sebelum apapun) ─────────────────────────────────────────
install_excepthooks()


# ── Model loading ─────────────────────────────────────────────────────────────

def _load_model():
    import torch
    print("GPU CUDA:", torch.cuda.is_available())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if USE_BANTAL_MODEL:
        path = BASE_DIR / "MODEL" / "bantal" / "bantal.pt"
        try:
            from ultralytics import YOLO
            m = YOLO(str(path))
            m.to(device)
            print(">>> bantal.pt loaded (Ultralytics)")
            return m
        except Exception as e:
            print(f">>> bantal.pt failed: {e} — falling back to best.pt")

    import torch  # noqa: F811 — re-import aman, untuk kejelasan
    path = BASE_DIR / "MODEL" / "exp7" / "weights" / "best.pt"
    m = torch.hub.load(
        str(BASE_DIR / "MODEL" / "yolov5"),
        "custom", path=str(path), force_reload=True, source="local",
    )
    m.to(device)
    print(">>> best.pt loaded (YOLOv5 hub)")
    return m


# ── Face asset loading ────────────────────────────────────────────────────────

def _load_face_assets() -> tuple:
    """
    Load 6 face label images (50×50 px) untuk overlay deteksi blok.
    Return tuple of (img_bgr, mask) pairs — index 0-based.
    face_id 1 → face_assets[0], face_id 6 → face_assets[5].

    Sengaja BGR (cv2.imread default) — konsisten dengan perilaku original
    main.py yang menerapkan overlay ke RGB frame (color swap minor, bukan bug
    yang perlu difix sekarang).
    """
    assets = []
    for i in range(1, 7):
        path = BASE_DIR / "FILES" / "LABEL_50x50" / f"0{i}.png"
        img  = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Face asset tidak ditemukan: {path}")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
        assets.append((img, mask))
    print(f">>> Face assets loaded: {len(assets)} images")
    return tuple(assets)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    try:
        # CTk theme harus di-set sebelum widget apapun dibuat
        apply_ctk_theme()

        # Consent — tampilkan modal jika belum/versi lama
        if not check_consent():
            import customtkinter
            dummy = customtkinter.CTk()
            dummy.withdraw()

            def _on_accept():
                mark_consent_accepted()
                dummy.quit()
                dummy.destroy()

            from ui.consent_screen import ConsentScreen
            ConsentScreen(dummy, on_accept_callback=_on_accept)
            dummy.mainloop()

        # Load aset berat sekali di sini — tidak diulang per game session
        model       = _load_model()
        face_assets = _load_face_assets()

        # Sync camera env vars ke STATE sebelum InputScreen dibuka
        STATE.sync_from_config()

        # ── InputScreen ───────────────────────────────────────────────────────
        from ui.input_screen import InputScreen
        inp = InputScreen(model=model, face_assets=face_assets)
        inp.after(200, inp.start_camera_preview)
        inp.after(100, lambda: (
            inp.attributes("-zoomed", True) if sys.platform == "linux"
            else inp.state("zoomed")
        ))
        inp.mainloop()

        # User tutup window tanpa selesai input
        if not STATE.nick_name:
            sys.exit(0)

        # ── RoomScreen (competition + bukan tournament yang sudah punya room) ─
        if STATE.current_mode == "competition":
            STATE.is_multiplayer = True
            if not (STATE.tournament_mode and STATE.tournament_room_code):
                from ui.room_screen import RoomScreen
                room = RoomScreen()
                room.mainloop()
                if not room.room_ready:
                    sys.exit(0)

        # ── GameScreen ────────────────────────────────────────────────────────
        from ui.game_screen import _launch_game
        _launch_game(model, face_assets)

    except SystemExit:
        raise
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)

