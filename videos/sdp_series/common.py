"""Shared drawing pieces for the System Design series: icons, labels, and a scene base with wipe() / trip().
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # the sdp_series folder
    from common import *
"""
import sys
from pathlib import Path

from manim import *

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if (p / "vidkit").is_dir())))
from vidkit import NarratedScene  # noqa: E402
from vidkit.look import *  # noqa: E402,F403

PURPLE_ = "#B48CFF"
SERIES = "System Design, from First Principles"


def fit(t, w):
    return t.scale_to_fit_width(w) if t.width > w else t


def node(label, color=NEG, w=2.0, h=0.8, size=20):
    r = RoundedRectangle(corner_radius=0.15, width=w, height=h, stroke_color=color, stroke_width=2.5).set_fill(PANEL, 1)
    return VGroup(r, fit(txt(label, size), w - 0.3))


def chip(label, color=DIM, size=16):
    t = txt(label, size)
    r = RoundedRectangle(corner_radius=0.22, width=t.width + 0.45, height=t.height + 0.3, stroke_color=color,
                         stroke_width=2).set_fill(PANEL, 1)
    return VGroup(r, t)


def person(color=WHITE, s=1.0):
    head = Circle(0.13, stroke_width=0).set_fill(color, 1).shift(UP * 0.2)
    pts = [np.array([0.24 * np.cos(a), 0.24 * np.sin(a) - 0.22, 0]) for a in np.linspace(0, PI, 20)]
    body = Polygon(*pts, stroke_width=0).set_fill(color, 1)
    return VGroup(head, body).scale(s)


def phone(color=WHITE, s=1.0):
    body = RoundedRectangle(corner_radius=0.1, width=0.6, height=1.05, stroke_color=color, stroke_width=2.5)
    body.set_fill(PANEL, 1)
    screen = RoundedRectangle(corner_radius=0.04, width=0.46, height=0.78, stroke_width=0).set_fill("#232836", 1)
    dot = Dot(radius=0.03, color=color).move_to(body.get_bottom() + UP * 0.08)
    return VGroup(body, screen, dot).scale(s)


def server(color=NEG, s=1.0, label=None):
    """A rack-mount server: three stacked units with status lights."""
    units = VGroup()
    for k in range(3):
        u = RoundedRectangle(corner_radius=0.06, width=1.3, height=0.32, stroke_color=color, stroke_width=2)
        u.set_fill(PANEL, 1).shift(DOWN * k * 0.38)
        led = Dot(radius=0.035, color=GOOD).move_to(u.get_right() + LEFT * 0.17)
        slot = Line(LEFT * 0.4, RIGHT * 0.2, color=EDGE, stroke_width=3).move_to(u.get_center() + LEFT * 0.2)
        units.add(VGroup(u, slot, led))
    g = VGroup(units.move_to(ORIGIN))
    if label:
        g.add(txt(label, 16, DIM).next_to(units, DOWN, buff=0.12))
    return g.scale(s)


def db(label, color=GOOD, w=1.3, h=1.0, size=18):
    ry = 0.16
    bottom = Ellipse(width=w, height=2 * ry, stroke_color=color, stroke_width=2.5).set_fill(PANEL, 1).shift(DOWN * h / 2)
    side = Rectangle(width=w, height=h, stroke_width=0).set_fill(PANEL, 1)
    left = Line(UP * h / 2, DOWN * h / 2, color=color, stroke_width=2.5).shift(LEFT * w / 2)
    right = left.copy().shift(RIGHT * w)
    top = Ellipse(width=w, height=2 * ry, stroke_color=color, stroke_width=2.5).set_fill("#20242D", 1).shift(UP * h / 2)
    lab = fit(txt(label, size), w - 0.15).shift(DOWN * 0.08)
    return VGroup(bottom, side, left, right, top, lab)


def arrow(a, b, color=EDGE, buff=0.1, sw=3):
    return Arrow(a, b, buff=buff, stroke_width=sw, color=color, max_tip_length_to_length_ratio=0.12,
                 max_stroke_width_to_length_ratio=10)


def header(s):
    return txt(s, 30, HI).to_edge(UP, buff=0.35)


def banner(s, color=HI, y=-3.3, size=22):
    t = txt(s, size, color)
    return VGroup(panel(t, buff=0.2), t).move_to(UP * y)


def term(word, meaning, color=HI):
    """A definition card: the new word, then what it means."""
    w = txt(word, 26, color, weight=BOLD)
    m = txt(meaning, 18, WHITE)
    g = VGroup(w, m).arrange(DOWN, buff=0.12)
    return VGroup(panel(g, buff=0.25), g)


def pos(m):
    return m.get_center()


class EpisodeScene(NarratedScene):
    def wipe(self, rt=0.5):
        if self.mobjects:
            self.play(*[FadeOut(m) for m in self.mobjects], run_time=rt)
        self.clear()

    def trip(self, pts, color=HI, r=0.1):
        """Animation of a dot moving through points; the dot is removed afterwards."""
        d = Dot(pts[0], radius=r, color=color).set_z_index(10)
        path = VMobject().set_points_as_corners([np.array(p, dtype=float) for p in pts])
        return Succession(FadeIn(d, run_time=0.01), MoveAlongPath(d, path), FadeOut(d, run_time=0.01))


__all__ = [n for n in dir() if not n.startswith("_")]
