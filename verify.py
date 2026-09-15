# verify.py — jalankan dari root project: python verify.py

import sys
import numpy as np

def assert_(val):
    assert val
    return val

errors = []

def check(label, fn):
    try:
        fn()
        print(f"  ✓  {label}")
    except Exception as e:
        print(f"  ✗  {label}: {e}")
        errors.append(label)

print("\n── config ───────────────────────────────")
check("import config",            lambda: __import__("config"))
check("config.BASE_DIR exists",   lambda: __import__("config").BASE_DIR.exists())
check("config.LEVEL_ANSWERS",     lambda: assert_(__import__("config").LEVEL_ANSWERS))
check("config.LEVEL_PATHS",       lambda: assert_(__import__("config").LEVEL_PATHS))
check("config.get_variant(1)",    lambda: __import__("config").get_variant(1))

print("\n── state ────────────────────────────────")
check("import state",             lambda: __import__("state"))
check("state.STATE instance",     lambda: assert_(__import__("state").STATE))
check("STATE.sync_from_config",   lambda: __import__("state").STATE.sync_from_config())

print("\n── ui.theme ─────────────────────────────")
check("import ui.theme",          lambda: __import__("ui.theme"))
check("CLR.ACCENT is str",        lambda: assert_(isinstance(__import__("ui.theme", fromlist=["CLR"]).CLR.ACCENT, str)))
check("FONT.P is str",            lambda: assert_(isinstance(__import__("ui.theme", fromlist=["FONT"]).FONT.P, str)))

print("\n── core.audio ───────────────────────────")
check("import core.audio",        lambda: __import__("core.audio"))
check("sfx_for_time(5)",          lambda: assert_(__import__("core.audio", fromlist=["sfx_for_time"]).sfx_for_time(5) == "amazing"))
check("sfx_for_time(99)",         lambda: assert_(__import__("core.audio", fromlist=["sfx_for_time"]).sfx_for_time(99) == "dont_give_up"))

print("\n── core.game_logic ──────────────────────")
check("import core.game_logic",   lambda: __import__("core.game_logic"))
check("estimate_cognitive_age(9)",lambda: assert_(__import__("core.game_logic", fromlist=["estimate_cognitive_age"]).estimate_cognitive_age(9) == 20))
check("compute_visuo_spatial",    lambda: assert_(__import__("core.game_logic", fromlist=["compute_visuo_spatial"]).compute_visuo_spatial(25, 30) == 95))
check("star_rating(<60s)",        lambda: assert_("LUAR BIASA" in __import__("core.game_logic", fromlist=["star_rating"]).star_rating(55)[1]))
check("check_consent callable",   lambda: callable(__import__("core.game_logic", fromlist=["check_consent"]).check_consent))

print("\n── core.serial_reader ───────────────────")
check("import core.serial_reader",lambda: __import__("core.serial_reader"))
check("SerialReaderThread class", lambda: assert_(__import__("core.serial_reader", fromlist=["SerialReaderThread"]).SerialReaderThread))

print("\n── api.client ───────────────────────────")
check("import api.client",        lambda: __import__("api.client"))
check("post callable",            lambda: callable(__import__("api.client", fromlist=["post"]).post))
check("save_backup callable",     lambda: callable(__import__("api.client", fromlist=["save_backup"]).save_backup))

print("\n── api.participants ─────────────────────")
check("import api.participants",  lambda: __import__("api.participants"))
check("verify callable",          lambda: callable(__import__("api.participants", fromlist=["verify"]).verify))

print("\n── api.tournament ───────────────────────")
check("import api.tournament",    lambda: __import__("api.tournament"))

print("\n── api.duel ─────────────────────────────")
check("import api.duel",          lambda: __import__("api.duel"))

print("\n── api.websocket_client ─────────────────")
check("import api.websocket_client", lambda: __import__("api.websocket_client"))
check("GameWebSocket class",      lambda: assert_(__import__("api.websocket_client", fromlist=["GameWebSocket"]).GameWebSocket))

print("\n── ui.widgets ───────────────────────────")
check("import ui.widgets",        lambda: __import__("ui.widgets"))

print("\n── ui.consent_screen ────────────────────")
check("import ui.consent_screen", lambda: __import__("ui.consent_screen"))


print("\n── core.camera ──────────────────────────")
check("import core.camera",        lambda: __import__("core.camera"))
check("enumerate_cameras callable", lambda: callable(__import__("core.camera", fromlist=["enumerate_cameras"]).enumerate_cameras))
check("apply_calibration callable", lambda: callable(__import__("core.camera", fromlist=["apply_calibration"]).apply_calibration))

print("\n── core.detection ───────────────────────")
check("import core.detection",      lambda: __import__("core.detection"))
check("classify_face(0,0,0,0) = 0", lambda: assert_(__import__("core.detection", fromlist=["classify_face"]).classify_face(np.zeros((100,100), dtype=np.uint8), 0, 0, 0, 0) == 0))
check("DetectionResult dataclass",  lambda: assert_(__import__("core.detection", fromlist=["DetectionResult"]).DetectionResult))

# ── Result ────────────────────────────────────────────────────────────────────


print("\n" + "─" * 42)
if errors:
    print(f"✗  {len(errors)} check gagal: {errors}")
    sys.exit(1)
else:
    print("✓  Semua check passed — Fase 3 beres")