"""Start a new video from a style's template.

    python new.py binary_search --style manim      # 2D diagrams, 3Blue1Brown style, 16:9
    python new.py how_a_pump_works --style blender # 3D objects, vertical reel with captions

Creates videos/<name>/ and prints the commands to run.
"""
import argparse
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
STEPS = {
    "manim": [
        "python narration.py                      # write the script first, then make the voice",
        "manim -pql scene.py Explainer            # 480p preview",
        "manim -qh scene.py Explainer             # 1080p final",
        "python finish.py --video media/videos/scene/1080p60/Explainer.mp4   # optional captions",
    ],
    "blender": [
        "python narration.py                      # write the script first, then make the voice",
        "blender -b -P scene.py -- --still 5      # one test frame at 5 s",
        "blender -b -P scene.py -- --draft        # fast preview frames -> frames_draft/",
        "rm -rf frames && blender -b -P scene.py  # final 1080x1920 frames",
        "python finish.py                         # captions + voice -> video.mp4",
    ],
}

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("name")
ap.add_argument("--style", choices=STEPS, required=True)
a = ap.parse_args()

# letters, digits, _ and -; "/" nests a project in a series folder (sdp_series/ep02_dns)
if not re.fullmatch(r"[\w-]+(/[\w-]+)*", a.name):
    sys.exit(f"Bad name {a.name!r}: use letters, digits, _ and -, with / only to nest (series/episode)")
dst = ROOT / "videos" / a.name
if dst.exists():
    sys.exit(f"{dst} already exists")
shutil.copytree(ROOT / "styles" / a.style / "template", dst,
                ignore=shutil.ignore_patterns("audio", "frames", "media", "__pycache__", "*.mp4"))
print(f"Created {dst.relative_to(ROOT)} ({a.style} style). Next:\n")
print(f"  cd {dst.relative_to(ROOT)}")
for s in STEPS[a.style]:
    print("  " + s)
