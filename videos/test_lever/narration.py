"""Write the script here. One string per shot; sentences become cue points [0], [1], ...

    python narration.py          -> generate audio/ + timing.json
    python narration.py --cues   -> just list sentences with their cue numbers
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(next(p for p in HERE.parents if (p / "vidkit").is_dir())))
from vidkit import narrate, print_cues  # noqa: E402

SEGMENTS = [
    # shot 0
    "Here's a lever. A plank, resting on a pivot. "
    "Drop a heavy block on the short end.",
    # shot 1
    "Now drop a small ball, way out on the long end. "
    "And the heavy block goes up.",
    # shot 2
    "Distance from the pivot multiplies force. "
    "That's how a lever works.",
]

if __name__ == "__main__":
    if "--cues" in sys.argv:
        print_cues(SEGMENTS)
    else:
        # voices: "en_US-ryan-high" (male), "en_US-lessac-high" (female); engine="say" for the macOS voice
        narrate(SEGMENTS, HERE, voice="en_US-ryan-high", speed=1.1)
