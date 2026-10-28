"""Captions + narration -> video.mp4.   python finish.py   (see vidkit/finish.py for options)"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(next(p for p in HERE.parents if (p / "vidkit").is_dir())))
from vidkit.finish import main  # noqa: E402

main(HERE)
