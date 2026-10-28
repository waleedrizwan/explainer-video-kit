"""
Step 3: burn yellow word-by-word captions onto the rendered frames, add the narration,
and encode a 1080x1920 MP4.

    python assemble.py                      # uses ~/electric_motor/frames
    python assemble.py --frames path/to/frames --out my_video.mp4

Needs: pip install pillow   and   ffmpeg on your PATH.
"""
import os, sys, wave, glob, argparse, subprocess
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "voice"))
from lines import LINES                       # narration text + when each line starts

ap = argparse.ArgumentParser()
ap.add_argument("--frames", default="~/electric_motor/frames")
ap.add_argument("--out", default="electric_motor_explainer.mp4")
ap.add_argument("--fps", type=int, default=24)          # must match the render's --fps
args = ap.parse_args()
SRC = os.path.abspath(os.path.expanduser(args.frames))
DST = SRC.rstrip("/\\") + "_captioned"
os.makedirs(DST, exist_ok=True)

FPS = args.fps
FONT = os.path.join(HERE, "fonts", "Poppins-Bold.ttf")
LEAD = 0.15                                   # voice starts this long after each cut

def dur(name):
    w = wave.open(os.path.join(HERE, "voice", f"{name}.wav"))
    return w.getnframes() / w.getframerate()

# caption chunks of up to 3 words, spread across each spoken line by word length
chunks = []                                   # (t0, t1, text)
for name, start, txt in LINES:
    words = txt.upper().split(); d = dur(name)
    groups, cur = [], []
    for w in words:
        cur.append(w)
        if len(cur) == 3 or w[-1] in ".,?":
            groups.append(cur); cur = []
    if cur: groups.append(cur)
    total = sum(len(" ".join(g)) for g in groups); t = start + LEAD
    for g in groups:
        span = d * len(" ".join(g)) / total
        chunks.append((t, t + span, " ".join(g))); t += span

def caption_at(sec):
    for i, (a, b, s) in enumerate(chunks):
        nxt = chunks[i + 1][0] if i + 1 < len(chunks) else b + 0.3
        if a <= sec < min(nxt, b + 0.6):
            return s
    return None

def draw(img, s):
    W, H = img.size
    s = s.replace(".", "").replace(",", "")
    size = int(W * 0.075)
    font = ImageFont.truetype(FONT, size)
    d = ImageDraw.Draw(img)
    while d.textlength(s, font=font) > W * 0.86:
        size -= 2; font = ImageFont.truetype(FONT, size)
    x = (W - d.textlength(s, font=font)) / 2; y = H * 0.66
    d.text((x, y), s, font=font, fill=(255, 214, 10),
           stroke_width=max(3, size // 9), stroke_fill=(15, 12, 0))

if __name__ == "__main__":
    files = sorted(f for f in glob.glob(os.path.join(SRC, "f_*.png")) + glob.glob(os.path.join(SRC, "f_*.jpg"))
                   if os.path.getsize(f) > 0)
    if not files:
        sys.exit(f"No frames found in {SRC}. Render them first (see README).")
    for i, path in enumerate(files):
        img = Image.open(path).convert("RGB")
        s = caption_at(i / FPS)
        if s: draw(img, s)
        img.save(os.path.join(DST, f"c_{i + 1:04d}.png"))
        if i % 100 == 0: print(f"captioned {i}/{len(files)}")

    inputs, filt = [], []
    for i, (name, start, _) in enumerate(LINES):
        inputs += ["-i", os.path.join(HERE, "voice", f"{name}.wav")]
        ms = int((start + LEAD) * 1000)
        filt.append(f"[{i + 1}:a]adelay={ms}|{ms},apad[a{i}]")
    mix = "".join(f"[a{i}]" for i in range(len(LINES)))
    filt.append(f"{mix}amix=inputs={len(LINES)}:normalize=0,"
                f"atrim=0:{len(files) / FPS:.3f},loudnorm=I=-16:TP=-1.5[aout]")
    filt.append("[0:v]scale=1080:1920:flags=lanczos,format=yuv420p[vout]")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS),
                    "-i", os.path.join(DST, "c_%04d.png"), *inputs,
                    "-filter_complex", ";".join(filt), "-map", "[vout]", "-map", "[aout]",
                    "-c:v", "libx264", "-preset", "medium", "-crf", "18",
                    "-c:a", "aac", "-b:a", "160k", "-ar", "48000", "-movflags", "+faststart",
                    args.out], check=True)
    print("wrote", os.path.abspath(args.out))
