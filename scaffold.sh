#!/usr/bin/env bash
# scaffold.sh — buat folder structure OAMP

set -e

ROOT=${1:-.}   # default current dir, atau kasih path sebagai argument

mkdir -p "$ROOT"/{core,api,ui}

touch "$ROOT/main.py"
touch "$ROOT/config.py"
touch "$ROOT/state.py"

touch "$ROOT/core/camera.py"
touch "$ROOT/core/detection.py"
touch "$ROOT/core/audio.py"
touch "$ROOT/core/serial_reader.py"
touch "$ROOT/core/game_logic.py"
touch "$ROOT/core/__init__.py"

touch "$ROOT/api/client.py"
touch "$ROOT/api/participants.py"
touch "$ROOT/api/tournament.py"
touch "$ROOT/api/duel.py"
touch "$ROOT/api/websocket_client.py"
touch "$ROOT/api/__init__.py"

touch "$ROOT/ui/theme.py"
touch "$ROOT/ui/widgets.py"
touch "$ROOT/ui/consent_screen.py"
touch "$ROOT/ui/input_screen.py"
touch "$ROOT/ui/room_screen.py"
touch "$ROOT/ui/game_screen.py"
touch "$ROOT/ui/__init__.py"

echo "✓ Scaffold selesai di: $ROOT"
find "$ROOT" -not -path "*/.git/*" | sort