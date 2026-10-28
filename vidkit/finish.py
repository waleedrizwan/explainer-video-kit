"""Last step for either style: burn yellow word-by-word captions, add the narration, write an MP4.

    Blender:  python finish.py                          # frames/ + audio/ -> video.mp4
    Manim:    python finish.py --video media/videos/scene/1080p60/Explainer.mp4
    Options:  --no-captions   --out name.mp4   --frames other_dir

Captions follow the real sentence timings in audio/timing.json, so they stay in sync after
you edit the script and re-run narration.py. Needs Pillow and ffmpeg.
"""
import argparse
import json
import subprocess
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from . import GAP, Timeline

FONT = Path(__file__).resolve().parent / "fonts" / "Poppins-Bold.ttf"
YELLOW, OUTLINE = (255, 214, 10), (15, 12, 0)


# ---------------------------------------------------------------- caption timing
def caption_chunks(T):
    """[(t0, t1, text)] — up to 3 words at a time, spread across each sentence by word length."""
    chunks = []
    for i, seg in enumerate(T.segs):
        cues = seg["cues"] + [seg["dur"] + GAP]
        for k, sentence in enumerate(seg["sentences"]):
            t, spoken = T.start(i) + cues[k], cues[k + 1] - cues[k] - GAP
            groups, cur = [], []
            for w in sentence.upper().split():
                cur.append(w)
                if len(cur) == 3 or w[-1] in ".,?!;:":
                    groups.append(cur)
                    cur = []
            if cur:
                groups.append(cur)
            if len(groups) > 1 and len(groups[-1]) == 1 and groups[-2][-1][-1] not in ".,?!;:":
                last = groups.pop()
                groups[-1] += last                # no lonely last word: "ONE AT A TIME"
            total = sum(len(" ".join(g)) for g in groups)
            for g in groups:
                span = spoken * len(" ".join(g)) / total
                chunks.append((t, t + span, " ".join(g)))
                t += span
    return chunks


def caption_at(chunks, sec):
    for i, (a, b, s) in enumerate(chunks):
        nxt = chunks[i + 1][0] if i + 1 < len(chunks) else b + 0.4
        if a <= sec < min(nxt, b + 0.5):
            return s
    return None


def draw(img, s, y_frac=None, size_frac=None):
    W, H = img.size
    s = s.strip(".,;:")
    size = int(min(W, H) * (size_frac or (0.075 if H > W else 0.07)))
    font = ImageFont.truetype(str(FONT), size)
    d = ImageDraw.Draw(img)
    while d.textlength(s, font=font) > W * 0.86:
        size -= 2
        font = ImageFont.truetype(str(FONT), size)
    x = (W - d.textlength(s, font=font)) / 2
    y = H * (y_frac or (0.66 if H > W else 0.80))
    d.text((x, y), s, font=font, fill=YELLOW, stroke_width=max(3, size // 9), stroke_fill=OUTLINE)


# ---------------------------------------------------------------- frame sources
def probe(video):
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                          "stream=width,height,r_frame_rate", "-of", "json", str(video)],
                         capture_output=True, text=True, check=True).stdout
    st = json.loads(out)["streams"][0]
    num, den = st["r_frame_rate"].split("/")
    return st["width"], st["height"], float(num) / float(den)


def video_frames(video, W, H):
    # -xerror: a damaged or truncated file is an error, not a quietly shorter video
    p = subprocess.Popen(["ffmpeg", "-v", "error", "-xerror", "-i", str(video), "-f", "rawvideo",
                          "-pix_fmt", "rgb24", "-"], stdout=subprocess.PIPE)
    n = W * H * 3
    try:
        while True:
            buf = p.stdout.read(n)
            if len(buf) < n:
                break
            yield Image.frombytes("RGB", (W, H), buf)
    finally:
        p.stdout.close()
        rc = p.wait()
    if rc or buf:
        raise SystemExit(f"Could not decode all of {video} (ffmpeg exit {rc}). Re-render it.")


def frame_files(src):
    """f_0001.png ... in frame order. Stops on gaps or empty files, which would put the audio out of sync."""
    files = sorted(Path(src).glob("f_*.png"), key=lambda f: int(f.stem[2:]))
    if not files:
        raise SystemExit(f"No frames in {src}. Render them first: blender -b -P scene.py")
    nums = [int(f.stem[2:]) for f in files]
    for a, b in zip(nums, nums[1:]):
        if b != a + 1:
            gap = f"Frame {a + 1} is" if b == a + 2 else f"Frames {a + 1}..{b - 1} are"
            raise SystemExit(f"{gap} missing from {src}. Finish the render first.")
    empty = [f.name for f in files if f.stat().st_size == 0]
    if empty:
        raise SystemExit(f"{empty[0]} is empty: the render is still running or was interrupted.")
    return files


def image_frames(files, W, H):
    for f in files:
        img = Image.open(f)
        if img.size != (W, H):
            raise SystemExit(f"{f.name} is {img.size[0]}x{img.size[1]}, not {W}x{H}. Frames from two renders "
                             f"are mixed: delete {f.parent.name}/ and render again.")
        yield img.convert("RGB")


# ---------------------------------------------------------------- main
def main(project_dir, argv=None):
    project_dir = Path(project_dir)
    ap = argparse.ArgumentParser(description="Captions + narration -> MP4")
    ap.add_argument("--video", help="a rendered video that already has its audio (Manim)")
    ap.add_argument("--frames", default="frames", help="folder of f_0001.png ... (Blender)")
    ap.add_argument("--fps", type=float, default=None)
    ap.add_argument("--out", default="video.mp4")
    ap.add_argument("--no-captions", action="store_true")
    ap.add_argument("--caption-y", type=float, help="top of the caption, as a fraction of the height")
    ap.add_argument("--caption-size", type=float, help="caption size, as a fraction of the shorter side")
    args = ap.parse_args(argv)

    T = Timeline(project_dir, use_recorded=bool(args.video), require_recorded=bool(args.video))
    chunks = [] if args.no_captions else caption_chunks(T)
    out = project_dir / args.out

    if args.video:
        video = Path(args.video)
        W, H, fps = probe(video)
        frames = video_frames(video, W, H)
        audio_in = ["-i", str(video)]
        audio_map = ["-map", "1:a?", "-c:a", "aac", "-b:a", "160k"]
    else:
        src = project_dir / args.frames
        files = frame_files(src)
        info = src / "info.json"
        fps = args.fps or (json.loads(info.read_text())["fps"] if info.exists() else 24)
        W, H = Image.open(files[0]).size
        frames = image_frames(files, W, H)
        audio_in, filt = [], []
        for i in range(len(T)):
            audio_in += ["-i", str(project_dir / "audio" / f"seg{i}.wav")]
            ms = int(T.start(i) * 1000)
            filt.append(f"[{i + 1}:a]adelay={ms}|{ms},apad[a{i}]")
        mix = "".join(f"[a{i}]" for i in range(len(T)))
        filt.append(f"{mix}amix=inputs={len(T)}:normalize=0,atrim=0:{len(files) / fps:.3f},"
                    f"loudnorm=I=-16:TP=-1.5[aout]")
        audio_map = ["-filter_complex", ";".join(filt), "-map", "[aout]", "-c:a", "aac", "-b:a", "160k"]

    enc = subprocess.Popen(
        ["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
         "-r", f"{fps:g}", "-i", "-", *audio_in, "-map", "0:v", *audio_map,
         "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2,format=yuv420p",
         "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-shortest", "-movflags", "+faststart",
         str(out)], stdin=subprocess.PIPE)
    try:
        for n, img in enumerate(frames):
            s = caption_at(chunks, n / fps)
            if s:
                draw(img, s, args.caption_y, args.caption_size)
            enc.stdin.write(img.tobytes())
            if n % 200 == 0:
                print(f"frame {n}")
        enc.stdin.close()
        if enc.wait():
            raise SystemExit("ffmpeg failed")
    except (SystemExit, BrokenPipeError, KeyboardInterrupt) as e:
        enc.kill()
        enc.wait()
        out.unlink(missing_ok=True)          # never leave a half-written video that looks finished
        if isinstance(e, BrokenPipeError):
            raise SystemExit("ffmpeg failed") from None
        raise
    print("wrote", out)
