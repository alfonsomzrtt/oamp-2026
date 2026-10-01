from pathlib import Path
import numpy as np
import sounddevice as sd
import soundfile as sf

AUDIO_DIR = Path('AUDIO')

TARGETS = [
    'sfx_correct.wav',
    'lanjut_lvl2.wav',
    'selesai.wav',
]

print('=== AUDIO FILE VALIDATION ===')
for name in TARGETS:
    path = AUDIO_DIR / name
    print(f'FILE: {name}')
    if not path.exists():
        print('  MISSING')
        continue

    info = sf.info(str(path))
    data, sr = sf.read(str(path), dtype='float32')
    print(f'  frames={info.frames}, sr={info.samplerate}, channels={info.channels}, duration={info.duration:.3f}s')
    print(f'  shape={data.shape}, min={data.min():.4f}, max={data.max():.4f}')

    print('  PLAY SINGLE...')
    sd.play(data, sr)
    sd.wait()
    print('  OK SINGLE PLAY')

print('=== PASS: single-file validation complete ===')
