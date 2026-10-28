#!/usr/bin/env bash
# One-time setup on macOS for both styles (Apple Silicon or Intel). Run from the kit folder:  bash setup.sh
set -euo pipefail
cd "$(dirname "$0")"

if ! command -v brew >/dev/null 2>&1; then
  echo "Homebrew is required. Install it from https://brew.sh, then re-run this script."
  exit 1
fi

echo "==> System libraries (Cairo/Pango for Manim, ffmpeg for video)"
brew install cairo pango pkg-config ffmpeg python@3.12

PY="$(brew --prefix)/bin/python3.12"
echo "==> Python virtual environment in .venv"
"$PY" -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

echo "==> Voices (offline, free)"
python -m piper.download_voices --data-dir voices en_US-lessac-high en_US-ryan-high \
  || echo "!! Piper voice download failed — vidkit will fall back to the macOS 'say' voice."

if command -v blender >/dev/null 2>&1 || [ -x /Applications/Blender.app/Contents/MacOS/Blender ]; then
  echo "==> Blender found (needed only for the blender style)"
else
  echo "!! Blender not found. For the blender style install Blender 4.4+ from https://www.blender.org/download/"
  echo "   (or: brew install --cask blender). The manim style works without it."
fi

echo
echo "Done. Each new terminal:  source .venv/bin/activate"
echo "Start a video:  python new.py my_topic --style manim    (or --style blender)"
