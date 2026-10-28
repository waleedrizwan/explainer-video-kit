"""Write the script here. One string per scene; sentences become cue points [0], [1], ...

    python narration.py          -> generate audio/ + timing.json
    python narration.py --cues   -> just list sentences with their cue numbers
"""
import sys
from pathlib import Path

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if (p / "vidkit").is_dir())))
from vidkit import narrate, print_cues  # noqa: E402

SEGMENTS = [
    # scene 0
    "Here's a list of numbers. "
    "Let's find the biggest one. "
    "Start by assuming the first number is the biggest.",
    # scene 1
    "Now walk through the rest, one at a time. "
    "If a number beats the current biggest, it takes its place. "
    "When you reach the end, whatever is left is the answer.",
]

if __name__ == "__main__":
    if "--cues" in sys.argv:
        print_cues(SEGMENTS)
    else:
        # Kokoro voices: "am_michael", "af_heart", ...; or engine="piper", voice="en_US-lessac-high"
        narrate(SEGMENTS, Path(__file__).parent, voice="am_michael", engine="kokoro", speed=1.1)
