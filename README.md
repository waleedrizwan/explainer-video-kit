# Explainer video kit — two styles

Narrated, animated explainer videos made on your Mac with free tools. You write a script, an offline voice reads it, and the animation follows the voice sentence by sentence. Pick a style per video:

| | **manim** | **blender** |
|---|---|---|
| Looks like | Flat 2D diagrams in the 3Blue1Brown style: arrays, graphs, equations | 3D objects on a tiled floor under a cloudy sky, with camera moves |
| Default format | 16:9 landscape | 9:16 vertical (Reels / Shorts / TikTok) with yellow word-by-word captions |
| Best for | Algorithms, math, logic, data | Physical things: machines, motors, levers, engines |
| Render time | Seconds to a few minutes | Roughly 0.5–2 s per frame (10–25 min for 30 s of video) |
| Needs | Manim (installed by setup) | Blender 4.4+ (install it yourself) |

Both styles can produce either format. Manim can render vertically with captions, and Blender can render landscape with `--landscape`.

## Setup (once)

```bash
bash setup.sh                 # Homebrew libs, Python venv, Manim, Kokoro + Piper voices
source .venv/bin/activate     # do this in every new terminal
```

`requirements.txt` takes the newest compatible packages. To get the exact versions these videos were built with, run `pip install -r requirements.lock` instead. Kokoro downloads its voice model (about 330 MB) from Hugging Face the first time you narrate, and works offline after that.

Run the tests with `pip install -r requirements-dev.txt && pytest`.

For the blender style, also install [Blender](https://www.blender.org/download/) (or run `brew install --cask blender`). If `blender` isn't on your PATH, use `/Applications/Blender.app/Contents/MacOS/Blender`.

## Make a video

```bash
python new.py binary_search --style manim
python new.py how_a_lever_works --style blender
```

This copies the style's template to `videos/<name>/` and prints the commands. Every project has the same first step:

1. **Write the script** in `narration.py`, one string per scene or shot. Each sentence becomes a cue point. Run `python narration.py --cues` to see the cue numbers, then `python narration.py` to make the voice and `audio/timing.json`.
2. **Animate** in `scene.py`, timed from the cues (details below).
3. **Render and finish.**

### manim style

```python
self.seg(2)           # start scene 2's audio
self.play(...)        # animate during sentence 0
self.cue(1)           # wait for sentence 1 to begin
self.play(...)
self.finish()         # wait for the clip to end
```

```bash
manim -pql scene.py Explainer      # 480p preview
manim -qh scene.py Explainer       # 1080p final -> media/videos/scene/1080p60/Explainer.mp4
```

For a vertical reel, call `vertical()` at the top of `scene.py`, render, then add captions:

```bash
python finish.py --video media/videos/scene/1920p60/Explainer.mp4    # -> video.mp4
```

### blender style

Times are in seconds, taken from the narration. `V.start(i)` and `V.end(i)` give when shot *i*'s line plays, and `V.cue(i, k)` gives when sentence *k* starts:

```python
V = setup(HERE)
sky(); floor(); sun()
block = cube("Block", (0.8, 0.8, 0.8), (0, 0, 5), mat("Red", RED))
move(block, "location", V.cue(0, 1), V.cue(0, 1) + 0.5, (0, 0, 5), (0, 0, 0.4))   # drops on cue
cam = Camera()
cam.shot(V.start(0), V.start(1), cam_from, cam_to, look_at)                        # one shot per line
render(V)
```

Helpers in `vidkit/blender.py` include `cube`, `cyl`, `sphere`, `torus`, `text`, `wire`, `mat`, `key`, `move`, `show`, `spin` and `Camera.shot`.

```bash
blender -b -P scene.py -- --still 6     # one test frame at 6 s
blender -b -P scene.py -- --draft       # fast 540-wide preview frames -> frames_draft/
python finish.py --frames frames_draft --out draft.mp4   # watch the draft
blender -b -P scene.py                  # final 1080x1920 -> frames/ (rendering resumes if stopped)
python finish.py                        # captions + voice -> video.mp4
```

The default engine is EEVEE. For path-traced lighting, add `-- --engine CYCLES`; it renders on the Mac's GPU through Metal (`--device CPU` to turn that off). The first Cycles GPU render on a machine spends a couple of minutes compiling kernels, and later renders are much faster.

The final render resumes by skipping frames already in `frames/`, so after you change the scene or the script, run `rm -r frames` first. `finish.py` stops with an error if it finds missing frames, empty frames or frames of different sizes, instead of writing a video that's out of sync.

If you edit the script, re-run `narration.py` and re-render. Shots, cues and captions all re-time themselves.

## What's where

```
vidkit/            shared: voice (narrate), timing (Timeline), captions + MP4 (finish), Blender helpers
styles/manim/      template/ + examples/neural_networks (a real trained network, every number from results.json)
styles/blender/    template/ (a lever) + examples/electric_motor (the 34-second motor reel)
videos/            your projects, created by new.py
voices/            Piper voice models
```

The electric-motor example predates the kit. It still uses its own hand-timed `SHOTS`, `voice/` folder and `assemble.py` (see its README). For new Blender videos, use the template instead.

## Getting Claude to write them

Open Claude Code in this folder and ask for something like:

> Make a narrated explainer on how a hydraulic press works in the blender style. Use `python new.py`, write narration.py and scene.py, time every move from V.cue, render stills at a few cue times, look at them for framing problems and fix them, then render a draft and run finish.py.

> Make a narrated Manim explainer on binary search. Show the array state at every step, render a preview, look at frames for overlaps and fix them.

## Tips

- **Voices:** `narrate(..., voice="en_US-ryan-high")` gives a male voice and `speed=1.1` makes it a bit brisker. To get more voices, run `python -m piper.download_voices --data-dir voices en_GB-alan-medium` (catalogue: https://rhasspy.github.io/piper-samples/). To use macOS voices, pass `engine="say"`.
- **Check a frame from a Manim render:** `ffmpeg -ss 42 -i media/videos/scene/480p15/Explainer.mp4 -frames:v 1 frame.png`.
- **No LaTeX needed:** the Manim examples use `Text`. For `MathTex`, run `brew install --cask mactex-no-gui` (large download).
- **Captions:** the colour, size and position are in `draw()` in `vidkit/finish.py`. Pass `--no-captions` to leave them off.
