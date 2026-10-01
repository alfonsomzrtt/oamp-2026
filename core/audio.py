"""
core/audio.py — procedural SFX dan audio playback.

Semua fungsi di sini thread-safe (tidak ada shared state).
play_sfx() dan play_audio() aman dipanggil dari daemon thread.
"""

from __future__ import annotations
from pathlib import Path
import threading
import numpy as np

from config import BASE_DIR


_AUDIO_DIR = BASE_DIR / "AUDIO"
# Set False untuk memakai tone procedural dari _SFX_BUILDERS.
USE_WAV_SFX = False
_AUDIO_LOCK = threading.Lock()
_WAV_EFFECTS = {
    "amazing": "menakjubkan.wav",
    "great": "hebat_sekali.wav",
    "solid": "mantap.wav",
    "good": "kerja_bagus.wav",
    "keep_going": "ayo_semangat.wav",
    "dont_give_up": "jangan_menyerah.wav",
    "next_level": None,
    "countdown": "hitung_mundur.wav",
    "complete": "selesai.wav",
}


# ── Playback ──────────────────────────────────────────────────────────────────

def play_audio(wav):
    """
    Putar audio.
    wav: filepath (str/Path) atau numpy array.
    """
    import sounddevice as sd
    import soundfile as sf

    if isinstance(wav, (str, Path)):
        path = Path(wav)
        if not path.exists():
            raise FileNotFoundError(f"Audio file not found: {path}")
        data, samplerate = sf.read(str(path), dtype="float32")
    else:
        data = np.asarray(wav)
        samplerate = 22050

    if data.ndim == 1:
        data = data.reshape(-1, 1)

    with _AUDIO_LOCK:
        sd.play(data, samplerate)
        sd.wait()


# ── Tone synthesis ────────────────────────────────────────────────────────────

def _tone(
    freq: float,
    duration: float,
    sr: int = 22050,
    vol: float = 0.3,
    harmonics: list[tuple[float, float]] | None = None,
) -> np.ndarray:
    """Generate sine tone dengan ADSR envelope dan optional harmonics."""
    t    = np.linspace(0, duration, int(sr * duration), endpoint=False)
    wave = vol * np.sin(2 * np.pi * freq * t)

    if harmonics:
        for h_amp, h_mult in harmonics:
            wave += vol * h_amp * np.sin(2 * np.pi * freq * h_mult * t)

    atk = min(int(0.015 * sr), len(t) // 4)
    rel = min(int(0.040 * sr), len(t) // 3)
    if atk > 1:
        wave[:atk] *= np.linspace(0, 1, atk)
    if rel > 1:
        wave[-rel:] *= np.linspace(1, 0, rel)

    return wave


def _gap(sr: int, ms: float) -> np.ndarray:
    return np.zeros(int(sr * ms / 1000))


# ── SFX library ───────────────────────────────────────────────────────────────

_SFX_BUILDERS: dict[str, callable] = {}

def _sfx(name: str):
    """Decorator untuk register SFX builder."""
    def decorator(fn):
        _SFX_BUILDERS[name] = fn
        return fn
    return decorator


SR = 22050
H  = lambda a, m: (a, m)   # harmonic shorthand


@_sfx("amazing")
def _amazing():
    return np.concatenate([
        _tone(880,  0.10, SR, 0.35, [H(0.3, 2)]),
        _gap(SR, 30),
        _tone(1100, 0.10, SR, 0.35, [H(0.3, 2)]),
        _gap(SR, 30),
        _tone(1320, 0.20, SR, 0.40, [H(0.25, 2), H(0.15, 3)]),
    ])

@_sfx("great")
def _great():
    return np.concatenate([
        _tone(660, 0.10, SR, 0.35, [H(0.3, 2)]),
        _gap(SR, 30),
        _tone(880, 0.20, SR, 0.38, [H(0.25, 2), H(0.12, 3)]),
    ])

@_sfx("solid")
def _solid():
    return np.concatenate([
        _tone(660, 0.08, SR, 0.30, [H(0.25, 2)]),
        _gap(SR, 20),
        _tone(790, 0.15, SR, 0.30, [H(0.20, 2)]),
    ])

@_sfx("good")
def _good():
    return _tone(660, 0.25, SR, 0.30, [H(0.2, 2)])

@_sfx("keep_going")
def _keep_going():
    return np.concatenate([
        _tone(440, 0.10, SR, 0.25, [H(0.15, 2)]),
        _gap(SR, 30),
        _tone(520, 0.10, SR, 0.25, [H(0.15, 2)]),
        _gap(SR, 30),
        _tone(440, 0.15, SR, 0.28, [H(0.15, 2)]),
    ])

@_sfx("dont_give_up")
def _dont_give_up():
    return np.concatenate([
        _tone(392, 0.12, SR, 0.25, [H(0.2, 2)]),
        _gap(SR, 40),
        _tone(349, 0.25, SR, 0.30, [H(0.2, 2), H(0.1, 3)]),
    ])

@_sfx("next_level")
def _next_level():
    return np.concatenate([
        _tone(523, 0.06, SR, 0.25, [H(0.2, 2)]),
        _tone(659, 0.06, SR, 0.28, [H(0.2, 2)]),
        _gap(SR, 20),
        _tone(784, 0.15, SR, 0.35, [H(0.25, 2), H(0.12, 3)]),
    ])

@_sfx("complete")
def _complete():
    return np.concatenate([
        _tone(523,  0.10, SR, 0.30, [H(0.2, 2)]),
        _tone(659,  0.10, SR, 0.32, [H(0.2, 2)]),
        _tone(784,  0.10, SR, 0.35, [H(0.2, 2), H(0.12, 3)]),
        _gap(SR, 40),
        _tone(1047, 0.35, SR, 0.42, [H(0.18, 2), H(0.1, 3), H(0.06, 4)]),
    ])

@_sfx("countdown")
def _countdown():
    return _tone(440, 0.35, SR, 0.35, [H(0.15, 2), H(0.08, 3)])

@_sfx("skip")
def _skip():
    return np.concatenate([
        _tone(523, 0.04, SR, 0.20, [H(0.15, 2)]),
        _gap(SR, 10),
        _tone(392, 0.10, SR, 0.22, [H(0.15, 2)]),
    ])


# ── Public API ────────────────────────────────────────────────────────────────

def play_sfx(
    effect: str,
    *,
    level: int | None = None,
    wait: bool = False,
):
    """
    Putar SFX pendek secara synchronous.
    Panggil dari daemon thread agar tidak block GUI:

        threading.Thread(target=lambda: play_sfx("amazing"), daemon=True).start()
    """
    import sounddevice as sd
    import soundfile as sf

    builder = _SFX_BUILDERS.get(effect)
    if builder is None:
        print(f">>> [audio] Unknown effect: {effect!r}")
        return
    try:
        with _AUDIO_LOCK:
            if USE_WAV_SFX:
                wav_name = _WAV_EFFECTS.get(effect, "")
                if effect == "next_level" and level in range(2, 9):
                    wav_name = f"lanjut_lvl{level}.wav"
                wav_path = _AUDIO_DIR / (wav_name or "")
                if wav_path.is_file():
                    data, samplerate = sf.read(str(wav_path), dtype="float32")
                    if data.ndim == 1:
                        data = data.reshape(-1, 1)
                    sd.play(data, samplerate)
                    sd.wait()
                    return

            tone = builder()
            sd.play(tone.reshape(-1, 1), SR)
            sd.wait()
    except Exception as e:
        print(f">>> [audio] play_sfx({effect!r}) failed: {e}")


def sfx_for_time(elapsed: float) -> str:
    """Pilih nama effect berdasarkan waktu penyelesaian (detik)."""
    if   elapsed < 10: return "amazing"
    elif elapsed < 15: return "great"
    elif elapsed < 20: return "solid"
    elif elapsed < 25: return "good"
    elif elapsed < 30: return "keep_going"
    else:              return "dont_give_up"