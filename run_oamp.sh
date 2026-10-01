#!/bin/bash
# Pindah ke direktori skrip
cd "$(dirname "$0")"

# Aktifkan venv dan jalankan aplikasi
. oamp_venv/Scripts/activate || source venv/bin/activate
python main.py