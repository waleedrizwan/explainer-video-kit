"""A minimal narrated 3D explainer: a lever. Everything is timed from the narration.

    python narration.py                          # voice + audio/timing.json
    blender -b -P scene.py -- --still 6          # one test frame at 6 s  -> frames/still_006.0s.png
    blender -b -P scene.py -- --draft            # fast 540-wide preview frames -> frames_draft/
    blender -b -P scene.py                       # final 1080x1920 frames (delete frames/ first)
    python finish.py                             # captions + voice -> video.mp4

Or open it in the Blender app: Scripting tab -> Open -> scene.py -> Run Script, then Space to play.
On macOS `blender` is /Applications/Blender.app/Contents/MacOS/Blender if it isn't on your PATH.
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(next(p for p in HERE.parents if (p / "vidkit").is_dir())))
from vidkit.blender import *  # noqa: E402,F403

V = setup(HERE)        # V.start(i), V.end(i), V.cue(i, k): seconds, straight from the narration
sky(); floor(); sun()

M_wood = mat("Wood", (0.55, 0.32, 0.12), rough=0.6)
M_red = mat("Red", RED, rough=0.35)
M_blue = mat("Blue", BLUE, rough=0.3)
M_steel = mat("Steel", STEEL, metal=1.0, rough=0.3)

# ---------------------------------------------------------------- objects
PIVOT_H = 1.1
R = PIVOT_H / 1.5                              # a 3-sided cylinder is a triangle: tip up, base on the floor
cyl("Pivot", R, 1.2, (0, 0, PIVOT_H - R), M_steel, verts=3, rot=(math.radians(90), 0, 0))
lever = empty("Lever", (0, 0, PIVOT_H))                        # rotates about Y
cube("Plank", (6.0, 1.0, 0.15), (1.0, 0, 0.075), M_wood, parent=lever)   # short arm 2, long arm 4
BLOCK_Z, BALL_Z = 0.15 + 0.4, 0.15 + 0.3
block = cube("Block", (0.8, 0.8, 0.8), (-1.5, 0, 5), M_red, bevel=0.05, parent=lever)
ball = sphere("Ball", 0.3, (3.6, 0, 5), M_blue, parent=lever)


def tilt(deg):
    return (0, math.radians(deg), 0)          # negative = short (left) end down


# ---------------------------------------------------------------- animation (shot 0)
key(lever, "rotation_euler", 0, tilt(0))
land = V.cue(0, 2) + 0.9                      # the block lands as "heavy block" is said
show(block, V.cue(0, 2) + 0.4)                # hidden until it falls
move(block, "location", V.cue(0, 2) + 0.4, land, (-1.5, 0, 5), (-1.5, 0, BLOCK_Z), "LINEAR")
move(lever, "rotation_euler", land, land + 0.35, tilt(0), tilt(-24))

# ---------------------------------------------------------------- shot 1
land = V.cue(1, 0) + 1.6                      # "way out on the long end"
show(ball, land - 0.5)
move(ball, "location", land - 0.5, land, (3.6, 0, 5), (3.6, 0, BALL_Z), "LINEAR")
key(lever, "rotation_euler", V.cue(1, 1) - 0.1, tilt(-24))
key(lever, "rotation_euler", V.cue(1, 1) + 0.9, tilt(14))   # "the heavy block goes up"

# ---------------------------------------------------------------- camera: one shot per segment
C = (1.0, 0, 1.2)                             # what the camera looks at
cam = Camera()
cam.shot(0, V.start(1), (-6, -9, 4), (-3, -10.5, 2.6), C)
cam.shot(V.start(1), V.start(2), (5.5, -6.5, 1.4), (3.5, -9, 2.2), C, (1.5, 0, 1.1))
cam.shot(V.start(2), V.total, (6, -8, 4.5), (-5, -9, 5), C, lens=24)

render(V)
