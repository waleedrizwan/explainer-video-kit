"""vidkit without Manim renders or Blender: timing, captions, frame checks, and finish.py end to end.

    pip install -r requirements-dev.txt && pytest
"""
import json
import shutil
import subprocess
import sys
import wave
from pathlib import Path

import pytest
from PIL import Image

import vidkit
from vidkit import Timeline, _sentences, narrate
from vidkit.finish import caption_at, caption_chunks, frame_files, main, video_frames

KIT = Path(__file__).resolve().parent.parent
RATE = 22050
needs_ffmpeg = pytest.mark.skipif(not shutil.which("ffmpeg"), reason="needs ffmpeg")


def write_wav(path, seconds, rate=RATE):
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(b"\x01\x00" * int(seconds * rate))


def project(tmp_path, segs, starts=None):
    """A project folder with audio/timing.json (and starts.json if given)."""
    (tmp_path / "audio").mkdir()
    (tmp_path / "audio" / "timing.json").write_text(json.dumps(segs))
    if starts is not None:
        (tmp_path / "audio" / "starts.json").write_text(json.dumps(starts))
    return tmp_path


TWO_SENTENCES = [{"dur": 4.6, "cues": [0.0, 2.3], "sentences": ["One two.", "Three four."], "gap": 0.3}]


# ---------------------------------------------------------------- sentences
def test_sentences_split_on_end_punctuation():
    assert _sentences("Hi there. How are you? Fine!  ") == ["Hi there.", "How are you?", "Fine!"]
    assert _sentences("It costs $3.50 now.") == ["It costs $3.50 now."]


def test_sentences_known_limits():
    # documented in _sentences(): abbreviations split, a closing quote hides the end
    assert _sentences("Use e.g. a cache.") == ["Use e.g.", "a cache."]
    assert _sentences('He said "stop." Then left.') == ['He said "stop." Then left.']


# ---------------------------------------------------------------- narrate
def fake_piper(monkeypatch, seconds_per_word=0.2):
    def synth_factory(voice, speed):
        return lambda sentence, path: write_wav(path, seconds_per_word * len(sentence.split()))
    monkeypatch.setattr(vidkit, "_piper_synth", synth_factory)


def test_narrate_cues_and_gap(tmp_path, monkeypatch):
    fake_piper(monkeypatch)
    timing = narrate(["One two three. Four five.", "Six."], tmp_path, engine="piper")
    assert timing[0]["cues"] == pytest.approx([0.0, 0.6 + vidkit.GAP])
    assert timing[0]["dur"] == pytest.approx(0.6 + 0.4 + 2 * vidkit.GAP)
    assert timing[0]["gap"] == vidkit.GAP
    assert json.loads((tmp_path / "audio" / "timing.json").read_text()) == timing
    with wave.open(str(tmp_path / "audio" / "seg1.wav")) as wf:
        assert wf.getnframes() / wf.getframerate() == pytest.approx(0.2 + vidkit.GAP, abs=1e-3)


def test_narrate_removes_stale_starts(tmp_path, monkeypatch):
    fake_piper(monkeypatch)
    (tmp_path / "audio").mkdir()
    (tmp_path / "audio" / "starts.json").write_text("{}")
    narrate(["Hello."], tmp_path, engine="piper")
    assert not (tmp_path / "audio" / "starts.json").exists()


@pytest.mark.parametrize("segments", [["", "Hi."], ["Hi.", "   "]])
def test_narrate_rejects_empty_segment(tmp_path, segments):
    with pytest.raises(ValueError, match="has no text"):
        narrate(segments, tmp_path, engine="piper")


# ---------------------------------------------------------------- timeline
def test_timeline_blender_layout(tmp_path):
    p = project(tmp_path, [{"dur": 2.0, "cues": [0.0, 1.0]}, {"dur": 3.0, "cues": [0.0]}])
    T = Timeline(p, use_recorded=False)
    assert T.starts == pytest.approx([Timeline.LEAD, Timeline.LEAD + 2.0 + Timeline.PAD])
    assert T.cue(0, 1) == pytest.approx(Timeline.LEAD + 1.0)
    assert T.end(1) == pytest.approx(T.start(1) + 3.0)
    assert T.total == pytest.approx(T.end(1) + Timeline.TAIL)


def test_timeline_uses_recorded_starts(tmp_path):
    p = project(tmp_path, [{"dur": 2.0, "cues": [0.0]}] * 2, starts={"0": 0.0, "1": 5.0})
    assert Timeline(p, require_recorded=True).starts == [0.0, 5.0]


@pytest.mark.parametrize("starts, message", [
    (None, "missing"),
    ({"0": 0.0}, "Re-render"),                  # interrupted render
    ({"0": 0.0, "2": 9.0}, "Re-render"),        # wrong segment ids
])
def test_timeline_require_recorded(tmp_path, starts, message):
    p = project(tmp_path, [{"dur": 2.0, "cues": [0.0]}] * 2, starts=starts)
    with pytest.raises(SystemExit, match=message):
        Timeline(p, require_recorded=True)
    Timeline(p)                                 # without require_recorded it falls back to the Blender layout


# ---------------------------------------------------------------- captions
def test_caption_spans_match_speech(tmp_path):
    chunks = caption_chunks(Timeline(project(tmp_path, TWO_SENTENCES), use_recorded=False))
    lead = Timeline.LEAD
    assert [c[2] for c in chunks] == ["ONE TWO.", "THREE FOUR."]
    assert chunks[0][:2] == pytest.approx((lead, lead + 2.0))
    assert chunks[1][:2] == pytest.approx((lead + 2.3, lead + 4.3))   # the last sentence's gap isn't speech


def test_caption_groups_avoid_lonely_last_word(tmp_path):
    seg = [{"dur": 3.3, "cues": [0.0], "sentences": ["Requests arrive one at a time"], "gap": 0.3}]
    chunks = caption_chunks(Timeline(project(tmp_path, seg), use_recorded=False))
    assert [c[2] for c in chunks] == ["REQUESTS ARRIVE ONE", "AT A TIME"]
    seg[0]["sentences"] = ["Requests arrive one at a time today"]
    (tmp_path / "audio" / "timing.json").write_text(json.dumps(seg))
    chunks = caption_chunks(Timeline(tmp_path, use_recorded=False))
    assert [c[2] for c in chunks] == ["REQUESTS ARRIVE ONE", "AT A TIME TODAY"]


def test_caption_at(tmp_path):
    chunks = [(1.0, 2.0, "A"), (2.5, 3.0, "B")]
    assert caption_at(chunks, 0.5) is None
    assert caption_at(chunks, 1.5) == "A"
    assert caption_at(chunks, 2.2) == "A"        # held up to 0.5 s after it ends, until the next one
    assert caption_at(chunks, 2.7) == "B"
    assert caption_at(chunks, 3.6) is None


# ---------------------------------------------------------------- frames
def frames(folder, nums, size=(32, 18)):
    folder.mkdir(parents=True, exist_ok=True)
    for n in nums:
        Image.new("RGB", size, (n % 255, 0, 0)).save(folder / f"f_{n:04d}.png")
    return folder


def test_frame_files_numeric_order(tmp_path):
    files = frame_files(frames(tmp_path / "frames", [9998, 9999, 10000, 10001]))
    assert [f.name for f in files] == ["f_9998.png", "f_9999.png", "f_10000.png", "f_10001.png"]


def test_frame_files_rejects_gaps_and_empty(tmp_path):
    src = frames(tmp_path / "frames", [1, 2, 4])
    with pytest.raises(SystemExit, match="Frame 3 is missing"):
        frame_files(src)
    frames(src, [3])
    (src / "f_0005.png").write_bytes(b"")
    with pytest.raises(SystemExit, match="f_0005.png is empty"):
        frame_files(src)
    with pytest.raises(SystemExit, match="No frames"):
        frame_files(tmp_path / "nothing")


# ---------------------------------------------------------------- finish.py end to end
def blender_project(tmp_path, n_frames=24):
    p = project(tmp_path, [{"dur": 0.6, "cues": [0.0], "sentences": ["Hello there."], "gap": 0.3}])
    write_wav(p / "audio" / "seg0.wav", 0.6)
    frames(p / "frames", range(1, n_frames + 1))
    (p / "frames" / "info.json").write_text(json.dumps({"fps": 24}))
    return p


def duration(video):
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0",
                          str(video)], capture_output=True, text=True, check=True).stdout
    return float(out)


@needs_ffmpeg
def test_finish_blender_frames(tmp_path):
    p = blender_project(tmp_path)
    main(p, [])
    assert duration(p / "video.mp4") == pytest.approx(1.0, abs=0.1)


@needs_ffmpeg
def test_finish_rejects_mixed_frame_sizes(tmp_path):
    p = blender_project(tmp_path)
    frames(p / "frames", [10], size=(64, 36))    # a final-render frame among draft frames
    with pytest.raises(SystemExit, match="f_0010.png is 64x36, not 32x18"):
        main(p, [])
    assert not (p / "video.mp4").exists()        # no half-written video left behind


@needs_ffmpeg
def test_finish_rejects_truncated_video(tmp_path):
    p = blender_project(tmp_path)
    main(p, ["--out", "full.mp4"])
    data = (p / "full.mp4").read_bytes()
    (p / "cut.mp4").write_bytes(data[: len(data) // 2])
    (p / "audio" / "starts.json").write_text(json.dumps({"0": 0.0}))
    with pytest.raises(SystemExit, match="Could not decode"):
        main(p, ["--video", str(p / "cut.mp4"), "--out", "again.mp4"])
    assert not (p / "again.mp4").exists()
    assert sum(1 for _ in video_frames(p / "full.mp4", 32, 18)) == 24


@needs_ffmpeg
def test_finish_manim_video_needs_recorded_starts(tmp_path):
    p = blender_project(tmp_path)
    main(p, ["--out", "full.mp4"])
    with pytest.raises(SystemExit, match="starts.json missing"):
        main(p, ["--video", str(p / "full.mp4"), "--out", "captioned.mp4"])


@needs_ffmpeg
def test_finish_keeps_frames_past_the_end_of_the_audio(tmp_path):
    # Manim's audio track ends before its video; the captioned copy must not be cut to the audio
    p = project(tmp_path, [{"dur": 1.0, "cues": [0.0], "sentences": ["Hi."], "gap": 0.3}], starts={"0": 0.0})
    subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc=size=64x36:rate=24:duration=2",
                    "-f", "lavfi", "-i", "sine=duration=1", "-pix_fmt", "yuv420p", str(p / "render.mp4")], check=True)
    main(p, ["--video", str(p / "render.mp4")])
    assert duration(p / "video.mp4") == pytest.approx(2.0, abs=0.1)


# ---------------------------------------------------------------- new.py
@pytest.mark.parametrize("name", ["/tmp/elsewhere", "../outside", "a//b", "", "has space"])
def test_new_rejects_bad_names(name):
    r = subprocess.run([sys.executable, str(KIT / "new.py"), name, "--style", "manim"], capture_output=True, text=True)
    assert r.returncode != 0 and "Bad name" in r.stderr
