"""vidkit — narrated explainer helpers, shared by the Manim and Blender styles.

    narrate(SEGMENTS, project_dir)   -> writes audio/seg0.wav ... and audio/timing.json
    Timeline(project_dir)            -> when each segment / sentence plays (used by Blender + finish)
    NarratedScene                    -> Manim Scene with seg() / cue() / until() / finish()
    vertical()                       -> make a Manim render 9:16 instead of 16:9
    vidkit.blender                   -> scene-building helpers for Blender (import inside Blender only)
    vidkit.finish                    -> burn word-by-word captions, add the voice, write the MP4

Voice: Kokoro (am_michael in the manim template) or a Piper voice from voices/ (both offline, free).
If Piper isn't available it falls back to the macOS `say` command.
"""
import json
import re
import shutil
import subprocess
import tempfile
import wave
from pathlib import Path

KIT = Path(__file__).resolve().parent.parent
VOICES = KIT / "voices"
DEFAULT_VOICE = "en_US-lessac-high"   # also downloaded by setup.sh: en_US-ryan-high
GAP = 0.3                              # silence after each sentence, seconds


def _sentences(text):
    return [s for s in re.split(r"(?<=[.?!])\s+", text.strip()) if s]


def _piper_synth(voice_name, speed=1.0):
    from piper import PiperVoice, SynthesisConfig  # noqa: import here so the fallback works without piper
    model = VOICES / f"{voice_name}.onnx"
    if not model.exists():
        raise FileNotFoundError(f"{model} missing — run setup.sh, or: python -m piper.download_voices "
                                f"--data-dir {VOICES} {voice_name}")
    voice = PiperVoice.load(str(model))
    cfg = SynthesisConfig(length_scale=1.0 / speed)

    def synth(sentence, path):
        with wave.open(str(path), "wb") as wf:
            voice.synthesize_wav(sentence, wf, syn_config=cfg)
    return synth


def _kokoro_synth(voice_name="am_michael", speed=1.0):
    """Kokoro-82M (pip install kokoro soundfile): offline, more natural than Piper.
    Voices: am_michael, af_heart, ... (American: a*, British: b*)."""
    import numpy as np
    import soundfile as sf
    from kokoro import KPipeline  # noqa: import here so other engines work without kokoro
    pipe = KPipeline(lang_code=voice_name[0], repo_id="hexgrad/Kokoro-82M")

    def synth(sentence, path):
        audio = np.concatenate([a.numpy() for _, _, a in pipe(sentence, voice=voice_name, speed=speed)])
        # Kokoro pads each clip with ~0.4 s before and ~0.65 s after; keep a short breath, GAP adds the pause
        loud = np.flatnonzero(np.abs(audio) > 0.01)
        if len(loud):
            audio = audio[max(0, loud[0] - 1200):loud[-1] + 2400]
        sf.write(str(path), audio, 24000, subtype="PCM_16")
    return synth


def _say_synth(mac_voice="Samantha", speed=1.0):
    if not shutil.which("say"):
        raise RuntimeError("No Piper and no macOS `say` command available.")

    def synth(sentence, path):
        with tempfile.TemporaryDirectory() as d:
            aiff = Path(d) / "s.aiff"
            subprocess.run(["say", "-v", mac_voice, "-r", str(int(175 * speed)), "-o", str(aiff), sentence],
                           check=True)
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(aiff), "-ac", "1", "-ar", "22050",
                            "-sample_fmt", "s16", str(path)], check=True)
    return synth


def narrate(segments, project_dir, voice=DEFAULT_VOICE, engine="auto", speed=1.0):
    """One wav per segment (scene). Each sentence is synthesized separately so scenes can
    cue animations on sentence starts. engine: 'auto' | 'piper' | 'kokoro' | 'say'.
    speed: 1.0 = the voice's normal pace, 1.15 = a bit brisker (good for short vertical videos)."""
    project_dir = Path(project_dir)
    out = project_dir / "audio"
    out.mkdir(exist_ok=True)
    if engine == "kokoro":
        synth = _kokoro_synth(voice, speed)
    elif engine in ("auto", "piper"):
        try:
            synth = _piper_synth(voice, speed)
        except Exception as e:
            if engine == "piper":
                raise
            print(f"Piper unavailable ({e}); falling back to macOS say")
            synth = _say_synth(speed=speed)
    else:
        synth = _say_synth(speed=speed)

    timing = []
    tmp = out / "_s.wav"
    for i, text in enumerate(segments):
        frames, offsets = b"", []
        sentences = _sentences(text)
        for s in sentences:
            synth(s, tmp)
            with wave.open(str(tmp), "rb") as wf:
                params, rate = wf.getparams(), wf.getframerate()
                width = wf.getsampwidth() * wf.getnchannels()
                offsets.append(len(frames) / width / rate)
                frames += wf.readframes(wf.getnframes())
            frames += b"\x00" * int(GAP * rate) * width
        with wave.open(str(out / f"seg{i}.wav"), "wb") as wf:
            wf.setparams(params)
            wf.writeframes(frames)
        dur = len(frames) / width / rate
        timing.append({"dur": dur, "cues": offsets, "sentences": sentences})
        print(f"seg{i}: {dur:5.1f}s  " + "  ".join(f"[{k}] {o:.1f}" for k, o in enumerate(offsets)))
    tmp.unlink(missing_ok=True)
    (out / "timing.json").write_text(json.dumps(timing))
    (out / "starts.json").unlink(missing_ok=True)   # stale Manim start times
    print(f"total {sum(t['dur'] for t in timing):.1f}s")
    return timing


def print_cues(segments):
    """List every sentence with its cue index — handy while writing a scene."""
    for i, text in enumerate(segments):
        print(f"== seg {i}")
        for k, s in enumerate(_sentences(text)):
            print(f"  [{k}] {s}")


# ---------------------------------------------------------------- timeline
class Timeline:
    """Where each narration segment sits on the video's clock, in seconds.

    Blender scenes lay segments end to end: LEAD of silence, then seg0, PAD, seg1, PAD, ... TAIL.
    A Manim render records its own start times (audio/starts.json), since animations can
    overrun a clip; when that file exists it wins.

        T = Timeline(project_dir)
        T.start(i), T.end(i)   segment i's audio, start / end
        T.cue(i, k)            sentence k of segment i starts
        T.total                length of the whole video
    """
    LEAD, PAD, TAIL = 0.4, 0.35, 0.8

    def __init__(self, project_dir, use_recorded=True, require_recorded=False):
        """require_recorded: the times must come from a Manim render's starts.json (finish.py --video);
        laying a Manim video out on the Blender clock would put every caption out of sync."""
        audio = Path(project_dir) / "audio"
        f = audio / "timing.json"
        if not f.exists():
            raise FileNotFoundError(f"{f} missing — run `python narration.py` first")
        self.segs = json.loads(f.read_text())
        starts, t = [], self.LEAD
        for s in self.segs:
            starts.append(t)
            t += s["dur"] + self.PAD
        rec = audio / "starts.json"
        if use_recorded and rec.exists():
            got = json.loads(rec.read_text())
            if set(got) == {str(i) for i in range(len(starts))}:
                starts = [got[str(i)] for i in range(len(starts))]
            elif require_recorded:
                raise SystemExit(f"{rec} has segments {sorted(got, key=int)}, but the narration has "
                                 f"0..{len(starts) - 1}. Re-render the whole scene, then run finish.py.")
        elif require_recorded:
            raise SystemExit(f"{rec} missing: render the Manim scene (after the last `python narration.py`), "
                             f"then run finish.py.")
        self.starts = starts
        self.total = max(st + s["dur"] for st, s in zip(starts, self.segs)) + self.TAIL

    def __len__(self):
        return len(self.segs)

    def start(self, i):
        return self.starts[i]

    def end(self, i):
        return self.starts[i] + self.segs[i]["dur"]

    def cue(self, i, k):
        return self.starts[i] + self.segs[i]["cues"][k]


def find_root(here):
    """The kit folder (the one holding vidkit/) above `here` — lets projects sit at any depth."""
    return next(p for p in Path(here).resolve().parents if (p / "vidkit").is_dir())


# ---------------------------------------------------------------- Manim side
try:
    from manim import Scene, config

    def vertical(width=8.0):
        """Call at the top of a Manim scene file for a 9:16 video (Reels / Shorts / TikTok).
        Keeps the -ql / -qh quality, swaps width and height. The frame is `width` units wide."""
        w, h = sorted((config.pixel_width, config.pixel_height))
        config.pixel_width, config.pixel_height = w, h
        config.frame_width, config.frame_height = width, width * h / w

    class NarratedScene(Scene):
        """Scene whose timeline follows the narration.

        self.seg(i)     start scene i's audio clip
        self.cue(k)     wait until sentence k of the current clip starts
        self.until(k)   seconds from now until sentence k (use as a run_time)
        self.finish()   wait until the clip ends (plus a small pad)
        """
        project_dir = None   # set to Path(__file__).parent in your subclass

        def _timing(self):
            if not hasattr(self, "_T"):
                self._T = json.loads((Path(self.project_dir) / "audio" / "timing.json").read_text())
            return self._T

        def seg(self, i):
            self.add_sound(str(Path(self.project_dir) / "audio" / f"seg{i}.wav"))
            self._i, self._t0 = i, self.renderer.time
            # remember when each clip really started, so vidkit.finish can time captions
            if not hasattr(self, "_starts"):
                self._starts = {}
            self._starts[str(i)] = self._t0
            (Path(self.project_dir) / "audio" / "starts.json").write_text(json.dumps(self._starts))

        def cue(self, k):
            left = self._t0 + self._timing()[self._i]["cues"][k] - self.renderer.time
            if left > 0.04:
                self.wait(left)

        def until(self, k):
            return max(0.3, self._t0 + self._timing()[self._i]["cues"][k] - self.renderer.time)

        def finish(self, pad=0.35):
            left = self._t0 + self._timing()[self._i]["dur"] + pad - self.renderer.time
            if left > 0.04:
                self.wait(left)
except ImportError:  # narrate() still works without manim installed (and inside Blender)
    pass
