"""
core/audio.py — procedural SFX dan audio playback.

Semua fungsi di sini thread-safe (tidak ada shared state).
play_sfx() dan play_audio() aman dipanggil dari daemon thread.
"""

from __future__ import annotations
from pathlib import Path
import queue
import threading
import time
import numpy as np

from config import BASE_DIR


_AUDIO_DIR = BASE_DIR / "AUDIO"
AUDIO_DIR = _AUDIO_DIR
# Set False untuk memakai tone procedural dari _SFX_BUILDERS.
USE_WAV_SFX = True
_AUDIO_LOCK = threading.Lock()
_AUDIO_REQUEST_LOCK = threading.Lock()
_AUDIO_QUEUE: queue.Queue[tuple[str, int | None, threading.Event | None] | None] = queue.Queue(maxsize=1)
_AUDIO_WORKER: threading.Thread | None = None
_AUDIO_LAST_EFFECT: str | None = None
_AUDIO_LAST_TS = 0.0
_WAV_EFFECTS = {
    "correct": "sfx_correct.wav",
    "correct_alt": "sfx_correct_alt.wav",
    "game_finished": "sfx_game_finished.wav",
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

def _play_audio_data(data: np.ndarray, samplerate: int) -> None:
    """Main audio playback, serialized by the worker and never called directly from GUI threads."""
    import sounddevice as sd

    if data.ndim == 1:
        data = data.reshape(-1, 1)

    with _AUDIO_LOCK:
        # Start playback asynchronously so caller thread is never blocked, but wait
        # until the current effect is finished before the next queued item starts.
        sd.play(data, samplerate, blocking=False)
        sd.wait()


def play_audio(wav):
    """
    Putar audio.
    wav: filepath (str/Path) atau numpy array.
    """
    import soundfile as sf

    if isinstance(wav, (str, Path)):
        path = Path(wav)
        if not path.exists():
            raise FileNotFoundError(f"Audio file not found: {path}")
        data, samplerate = sf.read(str(path), dtype="float32")
    else:
        data = np.asarray(wav)
        samplerate = 22050

    _play_audio_data(data, samplerate)


def play_utterance_pair(first_path: str | Path, second_path: str | Path) -> None:
    """Join two compatible WAV files and play them as one continuous stream."""
    import soundfile as sf

    first_data, first_rate = sf.read(str(first_path), dtype="float32", always_2d=True)
    second_data, second_rate = sf.read(str(second_path), dtype="float32", always_2d=True)

    if first_rate != second_rate:
        raise ValueError("Utterance WAV files must use the same sample rate")
    if first_data.shape[1] != second_data.shape[1]:
        raise ValueError("Utterance WAV files must use the same number of channels")

    combined_data = np.concatenate((first_data, second_data), axis=0)
    _play_audio_data(combined_data, first_rate)


def _start_audio_worker() -> None:
    global _AUDIO_WORKER
    if _AUDIO_WORKER is not None and _AUDIO_WORKER.is_alive():
        return

    def _worker_loop() -> None:
        while True:
            item = _AUDIO_QUEUE.get()
            if item is None:
                _AUDIO_QUEUE.task_done()
                break

            effect, level, done_event = item
            try:
                _play_effect(effect, level=level)
            except Exception as exc:  # pragma: no cover - logging only
                print(f">>> [audio] worker failed for {effect!r}: {exc}")
            finally:
                if done_event is not None:
                    done_event.set()
                _AUDIO_QUEUE.task_done()

    _AUDIO_WORKER = threading.Thread(target=_worker_loop, name="AudioWorker", daemon=True)
    _AUDIO_WORKER.start()


def _play_effect(effect: str, *, level: int | None = None) -> None:
    """Play a single effect synchronously from the worker thread."""
    import sounddevice as sd
    import soundfile as sf

    builder = _SFX_BUILDERS.get(effect)
    if builder is None:
        print(f">>> [audio] Unknown effect: {effect!r}")
        return

    if USE_WAV_SFX:
        wav_name = _WAV_EFFECTS.get(effect, "")
        if effect == "next_level" and level in range(2, 9):
            wav_name = f"lanjut_lvl{level}.wav"
        wav_path = _AUDIO_DIR / (wav_name or "")
        if wav_path.is_file():
            data, samplerate = sf.read(str(wav_path), dtype="float32")
            _play_audio_data(data, samplerate)
            return

    tone = builder()
    _play_audio_data(tone.reshape(-1, 1), SR)


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

@_sfx("correct")
def _correct():
    """Soft confirmation: a precise tap followed by a warm resolving tone."""
    return np.concatenate([
        _tone(392, 0.075, SR, 0.15, [H(0.18, 2), H(0.06, 3)]),
        _gap(SR, 32),
        _tone(587, 0.09, SR, 0.18, [H(0.12, 2)]),
        _gap(SR, 26),
        _tone(784, 0.22, SR, 0.13, [H(0.18, 2), H(0.05, 3)]),
    ])

@_sfx("correct_alt")
def _correct_alt():
    """Digital-soft confirmation with a quick up-and-down contour."""
    return np.concatenate([
        _tone(330, 0.07, SR, 0.13, [H(0.12, 2)]),
        _gap(SR, 18),
        _tone(660, 0.09, SR, 0.15, [H(0.10, 2)]),
        _gap(SR, 16),
        _tone(988, 0.10, SR, 0.12, [H(0.08, 2)]),
        _gap(SR, 18),
        _tone(660, 0.15, SR, 0.10, [H(0.12, 2)]),
    ])

@_sfx("game_finished")
def _game_finished():
    """Bold fanfare for the end-of-game results screen."""
    return np.concatenate([
        _tone(262, 0.13, SR, 0.22, [H(0.16, 2), H(0.06, 3)]),
        _gap(SR, 18),
        _tone(330, 0.13, SR, 0.24, [H(0.15, 2), H(0.05, 3)]),
        _gap(SR, 18),
        _tone(392, 0.15, SR, 0.26, [H(0.14, 2), H(0.05, 3)]),
        _gap(SR, 35),
        _tone(523, 0.14, SR, 0.28, [H(0.16, 2), H(0.06, 3)]),
        _gap(SR, 16),
        _tone(659, 0.16, SR, 0.30, [H(0.15, 2), H(0.06, 3)]),
        _gap(SR, 25),
        _tone(784, 0.56, SR, 0.34, [H(0.20, 2), H(0.10, 3), H(0.04, 4)]),
    ])


# ── Public API ────────────────────────────────────────────────────────────────

def play_sfx(
    effect: str,
    *,
    level: int | None = None,
    wait: bool = False,
):
    """
    Putar SFX pendek secara asynchronous via dedicated audio worker.
    API tetap kompatibel dengan pemanggilan lama, termasuk param wait=True.
    """
    global _AUDIO_LAST_EFFECT, _AUDIO_LAST_TS

    _start_audio_worker()

    done_event = threading.Event() if wait else None
    now = time.monotonic()

    with _AUDIO_REQUEST_LOCK:
        # Drop stale queued requests when the game emits rapid consecutive effects.
        if _AUDIO_LAST_EFFECT == effect and (now - _AUDIO_LAST_TS) < 0.20:
            if done_event is not None:
                done_event.set()
            return

        try:
            _AUDIO_QUEUE.put_nowait((effect, level, done_event))
        except queue.Full:
            try:
                _AUDIO_QUEUE.get_nowait()
            except queue.Empty:
                pass
            _AUDIO_QUEUE.put_nowait((effect, level, done_event))

        _AUDIO_LAST_EFFECT = effect
        _AUDIO_LAST_TS = now

    if wait and done_event is not None:
        done_event.wait()


def sfx_for_time(elapsed: float) -> str:
    """Pilih nama effect berdasarkan waktu penyelesaian (detik)."""
    if   elapsed < 10: return "amazing"
    elif elapsed < 15: return "great"
    elif elapsed < 20: return "solid"
    elif elapsed < 25: return "good"
    elif elapsed < 30: return "keep_going"
    else:              return "dont_give_up"