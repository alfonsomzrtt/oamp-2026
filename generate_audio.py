"""Generate natural Indonesian game voices with ElevenLabs (Super Enthusiastic & Fast Pacing)."""

from __future__ import annotations

import os
import wave
from pathlib import Path

import requests

from config import BASE_DIR


AUDIO_DIR = BASE_DIR / "AUDIO"
API_URL = "https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"

# Teks dibuat lebih bersemangat dengan tanda seru (!) agar ekspresi AI keluar maksimal
PHRASES = {
    "sfx_correct.wav": "Yes..! Jawaban kamu benar...!",
    "menakjubkan.wav": "Wow, benar-benar menakjubkan sekali!",
    "hebat_sekali.wav": "Wow! Hebat sekali!",
    "mantap.wav": "Mantap! Keren banget!",
    "kerja_bagus.wav": "Kerja bagus! Kamu hebat!",
    "ayo_semangat.wav": "Ayo! Tetap semangat! Kamu pasti bisa!",
    "jangan_menyerah.wav": "Jangan menyerah! Ayo, kamu pasti bisa!",
    "hitung_mundur.wav": "Bersiap... tiga... dua... satu... mulai!",
    "selesai.wav": "Keren...! Test Selesai...! Kerja kamu Luar biasa!",
    "lanjut_lvl2.wav": "Hebat banget! Ayo, lanjut ke level dua!",
    "lanjut_lvl3.wav": "Keren banget! Ayo, lanjut ke level tiga!",
    "lanjut_lvl4.wav": "Luar Biasa... Lanjut ke level empat!",
    "lanjut_lvl5.wav": "Keren..! Lanjut ke level lima!",
    "lanjut_lvl6.wav": "Ayo terus... Lanjut ke level enam!",
    "lanjut_lvl7.wav": "Luar biasa....! Lanjut ke level tujuh!",
    "lanjut_lvl8.wav": "Semangat terus..! Ini level terakhir, level delapan!",
}


def _write_pcm_wav(path: Path, pcm_data: bytes, sample_rate: int = 22050) -> None:
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(sample_rate)
        output.writeframes(pcm_data)


def generate_audio(filename: str, text: str, api_key: str, voice_id: str) -> None:
    response = requests.post(
        API_URL.format(voice_id=voice_id),
        params={"output_format": "pcm_22050"},
        headers={"xi-api-key": api_key, "Content-Type": "application/json"},
        json={
            "text": text,
            "model_id": "eleven_multilingual_v2",
            "voice_settings": {
                # Parameter dioptimalkan untuk suara yang sangat antusias dan dinamis
                "stability": 0.50,         # Diturunkan agar model lebih ekspresif dan bervariasi
                "similarity_boost": 0.75,
                "style": 0.60,            # Ditingkatkan agar gaya bicara sangat hidup dan bersemangat
                "use_speaker_boost": True,
            },
        },
        timeout=120,
    )
    if not response.ok:
        raise RuntimeError(f"ElevenLabs {response.status_code}: {response.text[:300]}")
    _write_pcm_wav(AUDIO_DIR / filename, response.content)
    print(f"Saved (Enthusiastic) {filename}: {text}")


def main() -> None:
    api_key = os.getenv("ELEVENLABS_API_KEY", "").strip()
    voice_id = os.getenv("ELEVENLABS_VOICE_ID", "hpp4J3VqNfWAUOO0d1Us").strip()
    if not api_key or not voice_id:
        raise SystemExit(
            "Set ELEVENLABS_API_KEY in .env or the environment before generating audio."
        )

    AUDIO_DIR.mkdir(exist_ok=True)
    for filename, text in PHRASES.items():
        generate_audio(filename, text, api_key, voice_id)


if __name__ == "__main__":
    main()