"""Generate a standalone BGM preview for the OAMP app."""

from pathlib import Path

import numpy as np
import soundfile as sf


SAMPLE_RATE = 22050
DURATION = 48.0
BPM = 108
BEAT = 60.0 / BPM


def note_frequency(note: int) -> float:
    return 440.0 * (2.0 ** ((note - 69) / 12.0))


def add_note(audio: np.ndarray, start: float, length: float, note: int, volume: float) -> None:
    first = int(start * SAMPLE_RATE)
    count = min(int(length * SAMPLE_RATE), len(audio) - first)
    if count <= 0:
        return

    time = np.arange(count) / SAMPLE_RATE
    envelope = np.minimum(time / 0.025, 1.0)
    envelope *= np.minimum((length - time) / 0.12, 1.0).clip(0.0, 1.0)
    frequency = note_frequency(note)
    tone = np.sin(2 * np.pi * frequency * time)
    tone += 0.24 * np.sin(2 * np.pi * frequency * 2 * time)
    audio[first:first + count] += volume * tone * envelope


def main() -> None:
    audio = np.zeros(int(DURATION * SAMPLE_RATE), dtype=np.float32)
    chords = [
        (60, 64, 67),  # C major
        (57, 60, 64),  # A minor
        (53, 57, 60),  # F major
        (55, 59, 62),  # G major
    ]
    melody = [72, 74, 76, 79, 76, 74, 72, 67]
    total_beats = int(DURATION / BEAT)

    for beat_index in range(total_beats):
        position = beat_index * BEAT
        chord = chords[(beat_index // 8) % len(chords)]
        add_note(audio, position, BEAT * 1.8, chord[0], 0.11)
        add_note(audio, position, BEAT * 1.8, chord[1], 0.065)
        add_note(audio, position, BEAT * 1.8, chord[2], 0.055)

        if beat_index % 2 == 0:
            add_note(audio, position, BEAT * 0.72, chord[0] + 12, 0.07)

        melody_note = melody[beat_index % len(melody)]
        add_note(audio, position + BEAT * 0.25, BEAT * 0.55, melody_note, 0.12)

    # Gentle stereo width and master fade prevent clicks at the boundaries.
    left = audio * 0.92
    right = audio * 0.78
    stereo = np.column_stack((left, right))
    fade = int(0.8 * SAMPLE_RATE)
    stereo[:fade] *= np.linspace(0.0, 1.0, fade)[:, None]
    stereo[-fade:] *= np.linspace(1.0, 0.0, fade)[:, None]
    stereo /= max(1.0, float(np.max(np.abs(stereo))) * 1.15)

    output = Path(__file__).resolve().parent / "AUDIO" / "bgm_preview.wav"
    sf.write(output, stereo, SAMPLE_RATE, subtype="PCM_16")
    print(f"BGM preview dibuat: {output}")


if __name__ == "__main__":
    main()