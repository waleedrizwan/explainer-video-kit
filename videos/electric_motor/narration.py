"""Write the script here. One string per scene; sentences become cue points [0], [1], ...

    python narration.py          -> generate audio/ + timing.json
    python narration.py --cues   -> just list sentences with their cue numbers
"""
import sys
from pathlib import Path

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if (p / "vidkit").is_dir())))
from vidkit import narrate, print_cues  # noqa: E402

SEGMENTS = [
    # scene 0: title, draw the motor
    "How does an electric motor actually work? "
    "Let's slice a simple one open and look at it from the front.",
    # scene 1: magnets and field
    "It starts with two permanent magnets. "
    "North on the left, south on the right. "
    "Between them runs a magnetic field, pointing from north to south.",
    # scene 2: the loop
    "In the middle sits a loop of copper wire on an axle that can spin. "
    "We're looking at it end on, so we only see its two sides.",
    # scene 3: battery and current
    "Now connect a battery. "
    "Current flows out of the screen on one side, and back into it on the other.",
    # scene 4: force
    "A current inside a magnetic field feels a push. "
    "The left side is pushed up, and the right side is pushed down. "
    "So the loop turns.",
    # scene 5: dead spot and commutator
    "But once the loop stands upright, the pushes just stretch it, and it would stall. "
    "Right at that moment, the split ring flips the current.",
    # scene 6: keeps spinning
    "Now the pushes turn it the same way again. "
    "Every half turn the current flips, so the loop keeps spinning.",
    # scene 7: outro
    "No fuel. No explosions. "
    "Just magnets, and a current that switches at the right moment.",
]

if __name__ == "__main__":
    if "--cues" in sys.argv:
        print_cues(SEGMENTS)
    else:
        # voices: "en_US-lessac-high" (default), "en_US-ryan-high"; engine="say" for the macOS voice
        narrate(SEGMENTS, Path(__file__).parent)
