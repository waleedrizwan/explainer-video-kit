# Electric motor explainer — made with code + Blender

A 34-second vertical (1080×1920) video in the style of the "How does a 4-stroke engine work?" reel: simple 3D shapes on a tiled floor under a cloudy sky, narration, and yellow word-by-word captions. Every object, material, light, animation and camera cut is created by `build_scene.py`. Nothing is modelled by hand.

## What's in here

| File | What it does |
|---|---|
| `build_scene.py` | Builds and animates the whole scene in Blender, and can render it |
| `assemble.py` | Adds captions + narration to the rendered frames → MP4 |
| `voice/*.wav` | Narration, one file per shot (already generated) |
| `voice/lines.py` | The script text and when each line starts |
| `voice/make_voice.py` | Optional: regenerate the narration with a free offline voice |
| `fonts/Poppins-Bold.ttf` | Caption font (SIL Open Font License) |

## Requirements

- Blender 4.4 or newer (tested on 5.2)
- Python 3 with Pillow: `pip install pillow`
- ffmpeg on your PATH (`brew install ffmpeg` / `winget install ffmpeg` / your package manager)

## Run it

**1. Look at it in Blender (optional).** Open Blender → **Scripting** tab → **Open** → `build_scene.py` → **Run Script**. It replaces the current file's contents with the motor scene. Press `Numpad 0` for the camera view and `Space` to play.

**2. Render the frames.** Either press **Render → Render Animation** in Blender after step 1, or from a terminal in this folder:

```
blender -b -P build_scene.py -- --frames 1 816
```

Frames go to `~/electric_motor/frames` (change with `--out`). If you stop it, run the same command again and it picks up where it left off. Want a quick look first? Render one frame: `blender -b -P build_scene.py -- --still 420`.

On macOS the `blender` command is `/Applications/Blender.app/Contents/MacOS/Blender`; on Windows it's `"C:\Program Files\Blender Foundation\Blender 5.2\blender.exe"` (adjust the version).

**3. Add captions and voice.**

```
python assemble.py
```

That writes `electric_motor_explainer.mp4` in this folder.

## How long it takes

Eevee on a typical GPU: roughly 0.5–2 s per frame, so about 10–25 minutes for all 816 frames. For a faster draft use `--res 540 --samples 8`. `--engine CYCLES` looks a bit more realistic but is much slower.

## The story, shot by shot

| Time | Shot | Narration |
|---|---|---|
| 0–3.5s | Wide orbit of the motor | How does an electric motor actually work? |
| 3.5–7.5s | Front view, red N and blue S magnets | It starts with two permanent magnets. North and south. |
| 7.5–11.5s | Two bar magnets snap together | Opposite poles attract. Same poles push away. |
| 11.5–15.5s | Battery, glowing beads flow along the wires, coil glows | Now send electricity from a battery through a copper coil, |
| 15.5–19s | Coil ends light up red / blue | and the coil becomes a magnet too. |
| 19–23s | Rotor swings toward the opposite magnets | Its ends get pulled toward the opposite magnets, so it turns. |
| 23–27s | Close-up of the split-ring commutator; pole colours flip | Right as it lines up, this split ring flips the current. |
| 27–31s | Rotor and fan spin up, camera pulls back | The poles swap, so it keeps chasing, and never stops spinning. |
| 31–34s | Wide outro | No fuel. No explosions. Just magnets. |

## Changing things

- **Timing / shots:** edit `SHOTS` near the top of `build_scene.py`, and each shot's camera path in the `shot(...)` calls near the bottom.
- **Colours:** `RED`, `BLUE` and the `mat(...)` calls in the materials section.
- **Narration:** edit `voice/lines.py`, then rerun `voice/make_voice.py`, or record your own WAVs with the same file names.
- **Captions:** colour, size and position are in `draw()` in `assemble.py`.
