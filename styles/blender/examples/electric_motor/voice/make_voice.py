"""
Optional: regenerate the narration WAVs (they're already included) after editing lines.py.

    pip install piper-tts
    # download a voice once (any Piper voice works), e.g. en_US-ryan-high:
    #   https://huggingface.co/rhasspy/piper-voices/tree/main/en/en_US/ryan/high
    #   (save en_US-ryan-high.onnx and en_US-ryan-high.onnx.json into this folder)
    python make_voice.py

Keep each line shorter than its shot (see the start times in lines.py), or move the
shot boundaries in build_scene.py (SHOTS) to match. You can also record your own voice
or use ElevenLabs etc. -- just save one WAV per line with the same names.
"""
import os, subprocess, sys, wave
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from lines import LINES

VOICE = os.path.join(HERE, "en_US-ryan-high.onnx")
for name, start, text in LINES:
    out = os.path.join(HERE, f"{name}.wav")
    subprocess.run([sys.executable, "-m", "piper", "-m", VOICE, "--length-scale", "0.88",
                    "-f", out], input=text.encode(), check=True)
    w = wave.open(out)
    print(f"{name:8s} {w.getnframes() / w.getframerate():.2f}s  {text}")
