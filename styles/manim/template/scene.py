"""A minimal narrated explainer: find the max of a list, drawn as a step-by-step state trace.

    python narration.py
    manim -pql scene.py Explainer      # fast preview (480p)
    manim -qh scene.py Explainer       # final (1080p)
Output lands in media/videos/scene/<quality>/Explainer.mp4

The look (colours, txt(), panel(), card(), Avenir Next, dark background) comes from vidkit/look.py.

Want a vertical Reel/Short with yellow captions? Uncomment `vertical()` below, lay things out
for a tall frame, render, then:
    python finish.py --video media/videos/scene/1920p60/Explainer.mp4
"""
import sys
from pathlib import Path

from manim import *

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if (p / "vidkit").is_dir())))
from vidkit import NarratedScene, vertical  # noqa: E402,F401
from vidkit.look import *  # noqa: E402,F403

# vertical()   # 9:16 instead of 16:9

NUMS = [4, 9, 2, 7, 11, 5]


def tile(n, color=WHITE):
    return VGroup(RoundedRectangle(corner_radius=0.14, width=1.0, height=1.0, stroke_color=EDGE, stroke_width=1.5)
                  .set_fill("#262A33", 1), txt(str(n), 30, color))


class Explainer(NarratedScene):
    project_dir = Path(__file__).parent

    def construct(self):
        tiles = VGroup(*[tile(n) for n in NUMS]).arrange(RIGHT, buff=0.18).shift(DOWN * 0.6)
        row_card = card(tiles.width + 0.8, tiles.height + 1.3, tiles.get_center() + DOWN * 0.25)  # room for the pointer

        # ---- scene 0
        self.seg(0)
        self.play(FadeIn(row_card), LaggedStart(*[FadeIn(t, shift=UP * 0.2) for t in tiles], lag_ratio=0.1),
                  run_time=1.2)
        self.cue(1)
        title = txt("biggest so far", 26, DIM).move_to(UP * 2.3)
        best_val = txt(str(NUMS[0]), 40, HI).next_to(title, DOWN, buff=0.25)
        top_card = card(4.2, 1.9, VGroup(title, best_val))
        self.play(FadeIn(top_card), FadeIn(title))
        self.cue(2)
        best_box = SurroundingRectangle(tiles[0], color=HI, buff=0.06, corner_radius=0.16)
        self.play(Create(best_box), TransformFromCopy(tiles[0][1], best_val))
        self.finish()

        # ---- scene 1
        self.seg(1)
        pointer = Triangle(color=NEG, fill_opacity=1, stroke_width=0).scale(0.15).next_to(tiles[1], DOWN, buff=0.25)
        self.play(FadeIn(pointer, shift=UP * 0.2))
        best = NUMS[0]
        self.cue(1)
        for i in range(1, len(NUMS)):
            self.play(pointer.animate.next_to(tiles[i], DOWN, buff=0.25), run_time=0.4)
            if NUMS[i] > best:
                best = NUMS[i]
                new_val = txt(str(best), 40, HI).move_to(best_val)
                self.play(best_box.animate.move_to(tiles[i]), Transform(best_val, new_val), run_time=0.5)
            else:
                self.play(Indicate(tiles[i][1], color=DIM), run_time=0.3)
        self.cue(2)
        self.play(FadeOut(pointer), Circumscribe(top_card, color=GOOD, buff=0.05))
        self.finish(1.0)
