# OAMP Desktop

OAMP Desktop adalah aplikasi tes Block Design Test (BDT). Peserta menyusun pola dari balok, sementara aplikasi menggunakan kamera dan model visi komputer untuk mendeteksi susunan, mengukur waktu, dan mencatat hasil.

Aplikasi dapat digunakan untuk latihan mandiri maupun sesi kompetisi yang terhubung ke backend. Input tombol ESP32 tersedia sebagai opsi tambahan.

## Menjalankan aplikasi

Persiapan awal di Windows, dari folder proyek:

```powershell
py -m venv oamp_venv
.\oamp_venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

Lalu jalankan:

```powershell
python main.py
```

`run.bat` juga dapat digunakan untuk mengaktifkan virtual environment dan membuka aplikasi.

Aplikasi memerlukan webcam serta aset proyek yang ada di folder `MODEL/`, `FILES/`, dan `AUDIO/`. Pada awal proses, model deteksi dimuat sebelum layar aplikasi dibuka, jadi startup pertama dapat memerlukan waktu.

## Pengaturan

Pengaturan lokal berada di `.env`. Gunakan `.env.example` sebagai titik awal; beberapa pengaturan yang umum:

- `API_SERVER_URL`: alamat backend. Kosongkan untuk menggunakan aplikasi tanpa koneksi backend.
- `PC_MODE`: mode awal, `training` atau `competition`.
- `CAMERA_INDEX`: memilih webcam jika perangkat memiliki lebih dari satu kamera.
- `BUTTON_MODE=true`: mengaktifkan input tombol ESP32.
- `MODEL_BANTAL=true`: memilih model bantal; nilai default menggunakan model deteksi tangan.
- `MAX_LEVEL`: jumlah level, dari 1 sampai 8.

Pengaturan kamera dan opsi lainnya dapat dilihat di `.env.example`. File `.env` berisi konfigurasi lokal; jangan masukkan ke Git.

## Audio

Audio game memakai file WAV di `AUDIO/` untuk utterance dan efek tertentu. Transisi level menggabungkan ucapan jawaban benar dan ucapan level berikutnya sebelum diputar agar keduanya terdengar berurutan. `generate_audio.py` hanya diperlukan bila ingin membuat ulang suara dengan ElevenLabs; atur `ELEVENLABS_API_KEY` di `.env` sebelum menjalankannya. Jangan menyimpan API key di source code atau commit ke Git.

`generate_bgm_preview.py` membuat file preview BGM secara terpisah dan tidak otomatis diputar oleh aplikasi.

## Struktur proyek

- `main.py`: titik masuk aplikasi dan inisialisasi model.
- `ui/`: layar persetujuan, input peserta, room, dan permainan.
- `core/`: logika permainan, kamera, deteksi, audio, dan pembacaan serial.
- `api/`: komunikasi dengan backend, termasuk room, duel, dan turnamen.
- `MODEL/`, `FILES/`, `AUDIO/`: model dan aset yang digunakan aplikasi.
- `results/`: penyimpanan hasil lokal atau cadangan.

Jangan memindahkan atau mengganti nama aset di folder `MODEL/`, `FILES/`, atau `AUDIO/` tanpa memperbarui referensi di kode.
