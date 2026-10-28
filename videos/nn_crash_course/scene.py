"""Neural networks, from first principles — all of Google's ML Crash Course "Neural networks" module.
Running example: a thermostat deciding if a room is comfortable from temperature and humidity.
Every boundary, weight, loss and inference number for the comfort network comes from results.json
(python train.py). Fruit numbers are the course's own figures; the bowl numbers are labelled illustrative.

    python narration.py
    manim -pql scene.py Explainer      # fast preview (480p)
    manim -qh scene.py Explainer       # final (1080p) -> media/videos/scene/1080p60/Explainer.mp4
"""
import json
import sys
from pathlib import Path

import numpy as np
from manim import *

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if (p / "vidkit").is_dir())))
from vidkit import NarratedScene  # noqa: E402
from vidkit.look import *  # noqa: E402,F403

import PIL.Image as _PILImage  # noqa: E402

_fromarray = _PILImage.fromarray


def _fromarray_compat(obj, mode=None):
    """Manim's ImageMobject pastes onto Image.fromarray(np.zeros((h, w)), mode="RGBA"), which newer Pillow rejects."""
    try:
        return _fromarray(obj, mode) if mode else _fromarray(obj)
    except ValueError:
        if mode and np.ndim(obj) == 2 and not np.any(obj):
            return _PILImage.new(mode, (np.shape(obj)[1], np.shape(obj)[0]))
        raise


_PILImage.fromarray = _fromarray_compat

HERE = Path(__file__).parent
R = json.loads((HERE / "results.json").read_text())
BLUE_C, ORANGE_C = "#4C8DFF", "#F5A04A"
LIM = 3.2
W1, B1 = np.array(R["W1"]), np.array(R["b1"])
W2, B2 = np.array(R["W2"]), R["b2"]
CX, CY = np.array(R["circle"]["X"]), np.array(R["circle"]["y"])
TEMP0, TEMP_STEP, HUM0, HUM_STEP = 22, 4, 50, 10      # chart unit 1 = 4 °C across, 10 % humidity up
FRUIT = ["apple", "orange", "pear", "grape"]
FRUIT_C = ["#FF6B6B", "#F5A04A", "#9BD35B", "#B07CFF"]
OVA = [0.34, 0.18, 0.84, 0.07]                         # the course's Figure 8 (sigmoid outputs)
SCORES = [1.3, 0.9, 2.5, 0.2]                          # raw scores whose softmax is close to the course's Figure 9


def sig(z):
    return 1 / (1 + np.exp(-z))


def grid_xy(n=72):
    xs = np.linspace(-LIM, LIM, n)
    return np.meshgrid(xs, xs[::-1])


# ---------------------------------------------------------------- charts
def chart(center, side=5.6, names=("x₁", "x₂")):
    ax = Axes(x_range=[-LIM, LIM, 1], y_range=[-LIM, LIM, 1], x_length=side, y_length=side, tips=False,
              axis_config={"stroke_color": EDGE, "stroke_width": 2, "include_ticks": False}).move_to(center)
    bg = card(side + 0.3, side + 0.3, center)
    lx = txt(names[0], 18, DIM).next_to(ax.c2p(LIM, 0), DOWN + LEFT * 0.3, buff=0.12)
    ly = txt(names[1], 18, DIM).next_to(ax.c2p(0, LIM), RIGHT + DOWN * 0.3, buff=0.12)
    return ax, VGroup(bg, ax, lx, ly)


def comfort_chart(center, side=5.6):
    """Axes in rescaled units, labelled in real ones. Returns ax, frame, x labels, y labels."""
    ax = Axes(x_range=[-LIM, LIM, 1], y_range=[-LIM, LIM, 1], x_length=side, y_length=side, tips=False,
              axis_config={"stroke_color": EDGE, "stroke_width": 0}).move_to(center)
    bg = card(side + 0.3, side + 0.3, center)
    xl = VGroup(*[txt(f"{TEMP0 + TEMP_STEP * u}°C", 15, DIM).next_to(ax.c2p(u, -LIM), DOWN, buff=0.28)
                  for u in (-2, 0, 2)],
                txt("temperature →", 18, WHITE).next_to(ax.c2p(0, -LIM), DOWN, buff=0.7))
    yl = VGroup(*[txt(f"{HUM0 + HUM_STEP * v}%", 15, DIM).next_to(ax.c2p(-LIM, v), LEFT, buff=0.28)
                  for v in (-2, 0, 2)],
                txt("humidity →", 18, WHITE).rotate(PI / 2).next_to(ax.c2p(-LIM, 0), LEFT, buff=0.95))
    return ax, VGroup(bg, ax), xl, yl


def dots(ax, X, y, r=0.075):
    return VGroup(*[Dot(ax.c2p(*p), radius=r, color=BLUE_C if c else ORANGE_C).set_z_index(3)
                    for p, c in zip(X, y)])


def heat(ax, P, mix=0.38):
    """P: probability of blue on a grid (row 0 = top). Soft tint behind the dots."""
    P = np.asarray(P)[..., None]
    base = np.array(ManimColor(PANEL).to_rgb()) * 255
    blue, orange = np.array(ManimColor(BLUE_C).to_rgb()) * 255, np.array(ManimColor(ORANGE_C).to_rgb()) * 255
    col = P * blue + (1 - P) * orange
    img = ImageMobject((base * (1 - mix) + col * mix).astype(np.uint8))
    img.set_resampling_algorithm(RESAMPLING_ALGORITHMS["bilinear"])
    img.stretch_to_fit_width(ax.x_length).stretch_to_fit_height(ax.y_length).move_to(ax.c2p(0, 0))
    return img.set_z_index(-1)


def heat_mono(ax, V):
    """A neuron's output on a grid: dark where it is 0, brighter the bigger it gets."""
    V = np.asarray(V)[..., None] / max(1e-9, np.max(V))
    base = np.array(ManimColor(PANEL).to_rgb()) * 255
    hi = np.array(ManimColor(HI).to_rgb()) * 255
    img = ImageMobject((base * (1 - 0.75 * V) + hi * 0.75 * V).astype(np.uint8))
    img.set_resampling_algorithm(RESAMPLING_ALGORITHMS["bilinear"])
    img.stretch_to_fit_width(ax.x_length).stretch_to_fit_height(ax.y_length).move_to(ax.c2p(0, 0))
    return img.set_z_index(-1)


def clip_line(ax, a, b, c, color=HI, width=4):
    """The line a*x + b*y + c = 0, clipped to the chart square."""
    pts = []
    for x in (-LIM, LIM):
        if abs(b) > 1e-9 and abs(-(a * x + c) / b) <= LIM:
            pts.append((x, -(a * x + c) / b))
    for yv in (-LIM, LIM):
        if abs(a) > 1e-9 and abs(-(b * yv + c) / a) <= LIM:
            pts.append((-(b * yv + c) / a, yv))
    pts = sorted(set((round(p[0], 4), round(p[1], 4)) for p in pts))
    return Line(ax.c2p(*pts[0]), ax.c2p(*pts[-1]), color=color, stroke_width=width).set_z_index(4)


def mini_plot(f, name, center, y_range, w=3.4, h=1.9, color=HI):
    ax = Axes(x_range=[-4, 4, 1], y_range=y_range, x_length=w, y_length=h, tips=False,
              axis_config={"stroke_color": EDGE, "stroke_width": 2, "include_ticks": False}).move_to(center)
    g = ax.plot(f, x_range=[-4, 4, 0.02], color=color, stroke_width=4)
    lab = txt(name, 22, WHITE).next_to(ax, UP, buff=0.25)
    bg = card(w + 0.6, h + 1.3, VGroup(ax, lab).get_center())
    return VGroup(bg, ax, g, lab), ax


def rows(lines, size=22, color=WHITE, buff=0.28):
    return VGroup(*[txt(s, size, color) for s in lines]).arrange(DOWN, aligned_edge=LEFT, buff=buff)


def chip(s, color=WHITE, w=None, size=18, fill="#262A33"):
    t = txt(s, size, color)
    box = RoundedRectangle(corner_radius=0.15, width=w or t.width + 0.5, height=t.height + 0.4,
                           stroke_color=EDGE, stroke_width=1.5).set_fill(fill, 1)
    return VGroup(box, t.move_to(box))


# ---------------------------------------------------------------- networks
class Net(VGroup):
    """Columns of nodes at given x positions; edges between neighbouring columns."""

    def __init__(self, sizes, xs, gap=1.4, r=0.32, center_y=0.0, edge_w=2.5):
        super().__init__()
        self.cols = []
        gaps = gap if isinstance(gap, (list, tuple)) else [gap] * len(sizes)
        for n, x, g in zip(sizes, xs, gaps):
            col = VGroup(*[Circle(radius=r, stroke_color=WHITE, stroke_width=2.5).set_fill(PANEL, 1)
                           .move_to([x, center_y + (n - 1) / 2 * g - k * g, 0]) for k in range(n)])
            col.set_z_index(2)
            self.cols.append(col)
        self.edges = []
        for a, b in zip(self.cols, self.cols[1:]):
            self.edges.append(VGroup(*[VGroup(*[
                Line(u.get_center(), v.get_center(), stroke_color=DIM, stroke_width=edge_w, buff=r)
                for v in b]) for u in a]))     # edges[layer][i][j]: node i -> node j of next column
        self.add(*self.cols, *self.edges)

    def layer_edges(self, l):
        return VGroup(*[e for row in self.edges[l] for e in row])

    def into(self, l, j):
        """Edges feeding node j of column l+1."""
        return VGroup(*[row[j] for row in self.edges[l]])


def flash(edges, color=HI, rt_width=0.6):
    return ShowPassingFlash(edges.copy().set_stroke(color, 6), time_width=rt_width)


def back_flash(edges, color=BAD):
    rev = VGroup(*[Line(e.get_end(), e.get_start(), stroke_color=color, stroke_width=6)
                   for e in edges.family_members_with_points()])
    return ShowPassingFlash(rev, time_width=0.6)


def on(node, s, size=17, color=WHITE):
    return txt(s, size, color).move_to(node).set_z_index(5)


def fmt(v):
    return f"{v:.2f}".replace("-", "−")


def weighted_sum(ws, xs, b):
    parts = []
    for k, (w, x) in enumerate(zip(ws, xs)):
        if k == 0:
            parts.append(f"{fmt(w)}·{x:g}")
        else:
            parts.append(("− " if w < 0 else "+ ") + f"{abs(w):.2f}·{x:g}")
    parts.append(("− " if b < 0 else "+ ") + f"{abs(b):.2f}")
    return " ".join(parts)


def room(u, v):
    return f"{TEMP0 + TEMP_STEP * u:.0f}°C", f"{HUM0 + HUM_STEP * v:.0f}%"


class Explainer(NarratedScene):
    project_dir = HERE

    def clear_all(self, rt=0.6, keep=()):
        gone = [m for m in self.mobjects if m not in keep]
        if gone:
            self.play(*[FadeOut(m) for m in gone], run_time=rt)

    def construct(self):
        for k in range(26):
            getattr(self, f"s{k}")()

    # ============================================================ PART 1: the problem
    def s0(self):
        self.seg(0)
        title = txt("Neural networks", 54, WHITE, weight=BOLD)
        sub = txt("from first principles", 26, HI).next_to(title, DOWN, buff=0.3)
        src = txt("Google's Machine Learning Crash Course · Neural networks module", 17, DIM).next_to(sub, DOWN, buff=0.3)
        self.play(FadeIn(title, shift=UP * 0.2), FadeIn(sub), FadeIn(src), run_time=0.9)
        self.cue(1)
        self.play(FadeOut(VGroup(title, sub, src)), run_time=0.5)
        dev = RoundedRectangle(corner_radius=0.35, width=3.2, height=3.2, stroke_color=EDGE, stroke_width=2) \
            .set_fill(PANEL, 1).move_to(LEFT * 3)
        ring = Circle(radius=1.1, color=HI, stroke_width=5).move_to(dev)
        r1 = txt("26°C", 34).move_to(dev.get_center() + UP * 0.25)
        r2 = txt("55%", 22, DIM).next_to(r1, DOWN, buff=0.15)
        arrow = Arrow(LEFT * 1.1, RIGHT * 1.0, color=DIM, buff=0)
        q = txt("comfortable?", 32, HI).next_to(arrow, RIGHT, buff=0.3)
        lab1 = txt("temperature", 16, DIM).next_to(r1, UP, buff=0.12)
        self.play(FadeIn(dev), Create(ring), FadeIn(r1), FadeIn(r2), FadeIn(lab1), run_time=1.0)
        self.play(GrowArrow(arrow), FadeIn(q, shift=RIGHT * 0.2), run_time=0.8)
        thermo = VGroup(dev, ring, r1, r2, lab1, arrow, q)
        self.cue(2)
        rules = rows(["if temperature > 25 and humidity < ...", "   unless it's ... but if ...", "   ???"], 20, DIM)
        rules.move_to(DOWN * 2.6)
        self.play(thermo.animate.shift(UP * 0.9), FadeIn(rules, shift=UP * 0.2), run_time=0.9)
        self.cue(3)
        self.play(FadeOut(thermo), FadeOut(rules), run_time=0.5)
        picks = [i for i in range(len(CY)) if CY[i] == 1][:3] + [i for i in range(len(CY)) if CY[i] == 0][:3]
        picks = [picks[0], picks[3], picks[1], picks[4], picks[2], picks[5]]
        hdr = ["temperature", "humidity", "comfortable?"]
        cells = [[txt(h, 20, DIM) for h in hdr]]
        for i in picks:
            t, h = room(*CX[i])
            cells.append([txt(t, 22), txt(h, 22), txt("yes" if CY[i] else "no", 22, BLUE_C if CY[i] else ORANGE_C)])
        table = VGroup(*[m for row in cells for m in row])
        for r, row in enumerate(cells):
            for c, m in enumerate(row):
                m.move_to([-5.2 + c * 2.35, 2.2 - r * 0.62, 0])
        tcard = card(7.4, 4.9, table.get_center())
        self.play(FadeIn(tcard), LaggedStart(*[FadeIn(m) for m in table], lag_ratio=0.03), run_time=1.6)
        note = txt("one row per room you measured", 16, DIM).next_to(tcard, DOWN, buff=0.2)
        self.play(FadeIn(note), run_time=0.5)
        self.cue(4)
        arr = Arrow(RIGHT * 0.3 + UP * 0.6, RIGHT * 2.3 + UP * 0.6, color=DIM, buff=0)
        box = chip("computer finds the rule", HI, size=22).move_to(RIGHT * 4.6 + UP * 0.6)
        ml = txt("machine learning", 26, WHITE).next_to(box, UP, buff=0.35)
        self.play(GrowArrow(arr), FadeIn(box), FadeIn(ml), run_time=0.9)
        self.cue(5)
        feat_box = SurroundingRectangle(VGroup(*[cells[r][c] for r in range(7) for c in (0, 1)]),
                                        color=HI, buff=0.12, corner_radius=0.12)
        feat = txt("features", 24, HI).next_to(feat_box, DOWN, buff=0.15)
        self.play(Create(feat_box), FadeOut(note), FadeIn(feat), run_time=0.9)
        self.cue(6)
        lab_box = SurroundingRectangle(VGroup(*[cells[r][2] for r in range(7)]), color=GOOD, buff=0.12,
                                       corner_radius=0.12)
        lab = txt("label", 24, GOOD).next_to(lab_box, DOWN, buff=0.15)
        self.play(Create(lab_box), FadeIn(lab), run_time=0.9)
        self.cue(7)
        model = chip("model", HI, size=26).move_to(box)
        self.play(Transform(box, model), run_time=0.7)
        mflow = txt("features  →  model  →  label", 20, DIM).next_to(box, DOWN, buff=0.35)
        self.play(FadeIn(mflow), run_time=0.6)
        self.cue(8)
        nn = txt("neural network", 30, HI, weight=BOLD).move_to(RIGHT * 4.6 + DOWN * 1.6)
        self.play(FadeIn(nn, shift=UP * 0.2), run_time=0.8)
        self.finish()
        self.clear_all()

    def s1(self):
        self.seg(1)
        self.ax, self.chart_frame, self.xl, self.yl = comfort_chart(RIGHT * 3.4 + UP * 0.25)
        self.play(FadeIn(self.chart_frame), run_time=0.8)
        self.cue(1)
        self.play(FadeIn(self.xl, shift=UP * 0.1), run_time=0.7)
        self.cue(2)
        self.play(FadeIn(self.yl, shift=RIGHT * 0.1), run_time=0.7)
        self.cue(3)
        self.cdots = dots(self.ax, CX, CY)
        grey = self.cdots.copy().set_color(DIM)
        self.play(LaggedStart(*[GrowFromCenter(d) for d in grey], lag_ratio=0.01), run_time=1.2)
        self.cue(4)
        blue = [k for k in range(len(CY)) if CY[k]]
        orng = [k for k in range(len(CY)) if not CY[k]]
        leg_b = VGroup(Dot(color=BLUE_C, radius=0.1), txt("comfortable", 22)).arrange(RIGHT, buff=0.2)
        leg_o = VGroup(Dot(color=ORANGE_C, radius=0.1), txt("not comfortable", 22)).arrange(RIGHT, buff=0.2)
        legend = VGroup(leg_b, leg_o).arrange(DOWN, aligned_edge=LEFT, buff=0.3).move_to(LEFT * 3.8 + UP * 2.4)
        made = txt("(made-up example data)", 15, DIM).next_to(legend, DOWN, buff=0.25).align_to(legend, LEFT)
        self.play(*[grey[k].animate.set_color(BLUE_C) for k in blue], FadeIn(leg_b), run_time=0.8)
        self.cue(5)
        self.play(*[grey[k].animate.set_color(ORANGE_C) for k in orng], FadeIn(leg_o), FadeIn(made), run_time=0.8)
        self.remove(grey)
        self.add(self.cdots)
        self.cue(7)
        mid = DashedVMobject(Circle(radius=self.ax.x_length / (2 * LIM) * 1.7, color=WHITE, stroke_width=2.5),
                             num_dashes=40).move_to(self.ax.c2p(0, 0)).set_z_index(4)
        self.play(Create(mid), run_time=1.0)
        self.cue(8)
        a = self.ax
        sides = VGroup(txt("too hot", 17, ORANGE_C).move_to(a.c2p(2.55, 0.25)),
                       txt("too cold", 17, ORANGE_C).move_to(a.c2p(-2.5, 0.25)),
                       txt("too damp", 17, ORANGE_C).move_to(a.c2p(0.9, 2.95)),
                       txt("too dry", 17, ORANGE_C).move_to(a.c2p(0.9, -2.95))).set_z_index(5)
        self.play(LaggedStart(*[FadeIn(s, scale=1.2) for s in sides], lag_ratio=0.3), run_time=1.4)
        self.cue(9)
        goal = rows(["The model's job:", "draw a boundary,", "blue on one side,", "orange on the other"], 24)
        goal[0].set_color(HI)
        goal.move_to(LEFT * 3.8 + DOWN * 1.0)
        self.play(FadeOut(mid), FadeOut(sides), FadeIn(goal, shift=UP * 0.2), run_time=1.0)
        self.legend = VGroup(legend, made)
        self.left = goal
        self.finish()

    def s2(self):
        self.seg(2)
        head = txt("The simplest rule", 30, HI).move_to(LEFT * 3.8 + UP * 2.6)
        self.play(FadeOut(self.left), FadeOut(self.legend), FadeIn(head), run_time=0.7)
        self.cue(1)
        l1 = txt("temperature × ?", 26)
        l2 = txt("+ humidity × ?", 26)
        l3 = txt("+ offset", 26)
        eq = VGroup(l1, l2, l3).arrange(DOWN, aligned_edge=LEFT, buff=0.3).move_to(LEFT * 3.9 + UP * 0.75)
        hint = txt("? = how much it matters", 18, DIM).next_to(head, DOWN, buff=0.3)
        self.play(FadeIn(l1, shift=RIGHT * 0.2), FadeIn(hint), run_time=0.8)
        self.play(FadeIn(l2, shift=RIGHT * 0.2), run_time=0.7)
        self.cue(2)
        tot = Brace(VGroup(l1, l2), LEFT, color=DIM)
        self.play(GrowFromCenter(tot), run_time=0.6)
        self.cue(3)
        self.play(FadeIn(l3, shift=RIGHT * 0.2), run_time=0.7)
        self.cue(4)
        d1 = VGroup(txt("total > 0", 24), txt("→ comfortable", 24, BLUE_C)).arrange(RIGHT, buff=0.25)
        d2 = VGroup(txt("otherwise", 24), txt("→ not", 24, ORANGE_C)).arrange(RIGHT, buff=0.25)
        dec = VGroup(d1, d2).arrange(DOWN, aligned_edge=LEFT, buff=0.25).next_to(eq, DOWN, buff=0.55).align_to(eq, LEFT)
        self.play(FadeIn(d1, shift=UP * 0.2), run_time=0.7)
        self.cue(5)
        self.play(FadeIn(d2, shift=UP * 0.2), run_time=0.6)
        self.cue(6)
        w1 = txt("temperature × w₁", 26, WHITE).move_to(l1, aligned_edge=LEFT)
        w2 = txt("+ humidity × w₂", 26, WHITE).move_to(l2, aligned_edge=LEFT)
        w1[-2:].set_color(HI)
        w2[-2:].set_color(HI)
        wl = txt("w₁, w₂ = the weights", 22, HI).next_to(dec, DOWN, buff=0.45).align_to(eq, LEFT)
        self.play(Transform(l1, w1), Transform(l2, w2), FadeOut(hint), FadeIn(wl), run_time=0.9)
        self.cue(7)
        b = txt("+ b", 26).move_to(l3, aligned_edge=LEFT)
        b[-1].set_color(GOOD)
        bl = txt("b = the bias", 22, GOOD).next_to(wl, DOWN, buff=0.2).align_to(eq, LEFT)
        self.play(Transform(l3, b), FadeIn(bl), run_time=0.8)
        self.cue(8)
        learn = txt("learning = finding good w₁, w₂, b", 20, DIM).move_to(LEFT * 3.8 + DOWN * 3.45)
        self.play(FadeIn(learn, shift=UP * 0.2), run_time=0.8)
        self.left = VGroup(head, eq, tot, dec, wl, bl, learn)
        self.finish()

    def _line_tools(self):
        """Live boundary for angle/offset trackers: the line, its blue/orange tint and a wrong-side counter."""
        self.ang, self.off = ValueTracker(2.4), ValueTracker(-0.9)
        n = lambda: np.array([np.cos(self.ang.get_value()), np.sin(self.ang.get_value())])
        gx, gy = grid_xy(48)

        def tint():
            nn = n()
            return heat(self.ax, sig(4 * (nn[0] * gx + nn[1] * gy - self.off.get_value())))

        def wrong():
            nn = n()
            pred = (CX @ nn - self.off.get_value()) > 0
            k = int(np.sum(pred != (CY == 1)))
            return txt(f"rooms on the wrong side: {k}", 20, BAD if k else GOOD).next_to(self.chart_frame, UP, buff=0.15)

        line = always_redraw(lambda: clip_line(self.ax, *n(), -self.off.get_value()))
        return line, always_redraw(tint), always_redraw(wrong)

    def s3(self):
        self.seg(3)
        self.play(FadeOut(self.left), run_time=0.6)
        self.cue(1)
        flip = rows(["total = 0", "is where the answer flips"], 26)
        flip[0].set_color(HI)
        flip.move_to(LEFT * 3.8 + UP * 2.0)
        self.play(FadeIn(flip, shift=UP * 0.2), run_time=0.8)
        self.cue(2)
        self.line, self.tint, self.wrong = self._line_tools()
        self.play(Create(self.line), FadeIn(self.tint), run_time=1.0)
        straight = txt("→ always a straight line", 24, HI).next_to(flip, DOWN, buff=0.4).align_to(flip, LEFT)
        self.play(FadeIn(straight), FadeIn(self.wrong), run_time=0.7)
        self.cue(3)
        tl = txt("change the weights → it tilts", 22).next_to(straight, DOWN, buff=0.6).align_to(flip, LEFT)
        self.play(FadeIn(tl), self.ang.animate.set_value(3.0), run_time=1.0)
        self.play(self.ang.animate.set_value(1.9), run_time=1.0)
        self.cue(4)
        sl = txt("change the bias → it slides", 22).next_to(tl, DOWN, buff=0.3).align_to(flip, LEFT)
        self.play(FadeIn(sl), self.off.animate.set_value(-1.6), run_time=0.9)
        self.cue(5)
        self.play(self.off.animate.set_value(-0.6), run_time=0.9)
        self.cue(6)
        lm = txt("a linear model", 34, HI, weight=BOLD).move_to(LEFT * 3.8 + DOWN * 2.6)
        self.play(Write(lm), run_time=1.0)
        self.left = VGroup(flip, straight, tl, sl, lm)
        self.finish()

    def s4(self):
        self.seg(4)
        self.play(FadeOut(self.left), run_time=0.6)
        fence = txt("Fence in the blue rooms?", 28, HI).move_to(LEFT * 3.8 + UP * 2.2)
        self.play(FadeIn(fence), run_time=0.6)
        self.cue(1)
        self.play(self.ang.animate.set_value(0.4), self.off.animate.set_value(-1.2), run_time=0.9)
        self.cue(2)
        self.play(self.off.animate.set_value(1.2), run_time=0.9)
        self.play(self.ang.animate.set_value(4.2), self.off.animate.set_value(-0.3), run_time=1.0)
        self.cue(3)
        nope = txt("✗  never works", 28, BAD).next_to(fence, DOWN, buff=0.5)
        why = txt("blue is surrounded on every side", 20, DIM).next_to(nope, DOWN, buff=0.2)
        self.play(FadeIn(nope, shift=UP * 0.2), FadeIn(why), run_time=0.8)
        self.cue(4)
        nl = txt("a nonlinear problem", 34, HI, weight=BOLD).move_to(LEFT * 3.8 + DOWN * 0.9)
        nld = txt("no straight line can separate the classes", 18, DIM).next_to(nl, DOWN, buff=0.2)
        self.play(Write(nl), FadeIn(nld), run_time=1.1)
        self.cue(5)
        most = txt("(most real-world problems are)", 20, DIM).next_to(nld, DOWN, buff=0.5)
        self.play(FadeIn(most), run_time=0.6)
        self.left = VGroup(fence, nope, why, nl, nld, most)
        self.finish()

    def s5(self):
        self.seg(5)
        head = txt("Invent new features", 30, HI).move_to(LEFT * 3.8 + UP * 2.6)
        self.wrong.clear_updaters()
        self.play(FadeOut(self.left), FadeOut(self.line), FadeOut(self.tint), FadeOut(self.wrong), FadeIn(head),
                  run_time=0.7)
        self.cue(1)
        f1 = txt("(temperature − middle)²", 26)
        f2 = txt("(humidity − middle)²", 26)
        fs = VGroup(f1, f2).arrange(DOWN, aligned_edge=LEFT, buff=0.35).move_to(LEFT * 3.8 + UP * 1.0)
        self.play(FadeIn(f1, shift=RIGHT * 0.2), run_time=0.8)
        self.cue(2)
        self.play(FadeIn(f2, shift=RIGHT * 0.2), run_time=0.8)
        self.cue(3)
        gx, gy = grid_xy()
        r0 = 1.7
        self.ctint = heat(self.ax, sig(3 * (r0 ** 2 - gx ** 2 - gy ** 2)))
        circ = Circle(radius=self.ax.x_length / (2 * LIM) * r0, color=HI, stroke_width=4) \
            .move_to(self.ax.c2p(0, 0)).set_z_index(4)
        k = int(np.sum(((CX ** 2).sum(1) < r0 ** 2) != (CY == 1)))
        ok = txt(f"rooms on the wrong side: {k}", 20, GOOD).next_to(self.chart_frame, UP, buff=0.15)
        lin = txt("linear in the new features", 20, DIM).next_to(fs, DOWN, buff=0.45).align_to(fs, LEFT)
        lin2 = txt("= a circle on the chart  ✓", 22, GOOD).next_to(lin, DOWN, buff=0.15).align_to(fs, LEFT)
        self.play(FadeIn(self.ctint), Create(circ), FadeIn(ok), FadeIn(lin), run_time=1.4)
        self.play(FadeIn(lin2), run_time=0.6)
        self.cue(4)
        fc = txt("a feature cross", 34, HI, weight=BOLD).move_to(LEFT * 3.8 + DOWN * 1.9)
        fcd = txt("a new feature made by multiplying features", 18, DIM).next_to(fc, DOWN, buff=0.2)
        self.play(Write(fc), FadeIn(fcd), run_time=1.1)
        self.finish()
        self.clear_all()

    def s6(self):
        self.seg(6)
        guess = txt("…but someone had to guess them", 28, HI).move_to(UP * 3.2)
        self.play(FadeIn(guess), run_time=0.7)
        self.cue(1)
        ax, frame = chart(RIGHT * 3.4 + DOWN * 0.3, side=5.4)
        Xq, yq = np.array(R["quad"]["X"]), np.array(R["quad"]["y"])
        qd = dots(ax, Xq, yq)
        self.play(FadeIn(frame), run_time=0.6)
        self.cue(2)
        self.play(LaggedStart(*[GrowFromCenter(d) for d in qd], lag_ratio=0.02), run_time=1.2)
        self.cue(3)
        circ = Circle(radius=ax.x_length / (2 * LIM) * 1.7, color=BAD, stroke_width=4).move_to(ax.c2p(0, 0)).set_z_index(4)
        sq = txt("squares → circle  ✗", 24, BAD).move_to(LEFT * 3.8 + UP * 1.6)
        self.play(Create(circ), FadeIn(sq), run_time=0.9)
        self.cue(4)
        gx, gy = grid_xy()
        ht = heat(ax, sig(2.0 * gx * gy))
        mul = txt("x₁ × x₂  ✓", 24, GOOD).next_to(sq, DOWN, buff=0.35).align_to(sq, LEFT)
        self.play(FadeOut(circ), FadeIn(ht), FadeIn(mul), run_time=1.0)
        self.cue(5)
        tries = VGroup(*[txt(s, 20, DIM) for s in ["x₁²?", "x₂²?", "x₁·x₂?", "x₁²·x₂?", "x₁³?", "…"]]) \
            .arrange(RIGHT, buff=0.3).next_to(mul, DOWN, buff=0.5).align_to(sq, LEFT)
        self.play(LaggedStart(*[FadeIn(t, shift=UP * 0.15) for t in tries], lag_ratio=0.25), run_time=1.5)
        self.cue(6)
        want = txt("Want: a model that finds", 24).move_to(LEFT * 3.8 + DOWN * 1.4)
        want2 = txt("the combinations itself", 24).next_to(want, DOWN, buff=0.15)
        self.play(FadeIn(want), FadeIn(want2), run_time=0.8)
        self.cue(7)
        nn = txt("→ a neural network", 32, HI, weight=BOLD).next_to(want2, DOWN, buff=0.4)
        self.play(Write(nn), run_time=0.9)
        self.finish()
        self.clear_all()

    # ============================================================ PART 2: building a network
    def s7(self):
        self.seg(7)
        net = Net([2, 1], [-3.2, 1.8], gap=2.4, r=0.5, center_y=0.8)
        self.lin = net
        self.play(FadeIn(net.cols[1]), run_time=0.6)
        self.cue(1)
        names = VGroup(txt("temperature", 20, DIM).next_to(net.cols[0][0], LEFT, buff=0.25),
                       txt("humidity", 20, DIM).next_to(net.cols[0][1], LEFT, buff=0.25))
        self.play(LaggedStart(*[GrowFromCenter(n) for n in net.cols[0]], lag_ratio=0.3), FadeIn(names), run_time=1.0)
        self.cue(2)
        nodes = txt("nodes", 26, HI).move_to(LEFT * 0.7 + UP * 3.3)
        pts = VGroup(*[Arrow(nodes.get_bottom(), n.get_top() + UP * 0.05, color=HI, stroke_width=3, buff=0.12,
                             max_tip_length_to_length_ratio=0.08) for n in [net.cols[0][0], net.cols[1][0]]])
        self.play(FadeIn(nodes), GrowArrow(pts[0]), GrowArrow(pts[1]), run_time=0.9)
        self.cue(3)
        self.play(FadeOut(pts), FadeOut(nodes), Create(net.layer_edges(0)), run_time=0.9)
        wl = VGroup(txt("w₁", 24, HI).move_to(net.edges[0][0][0].get_center() + UP * 0.35),
                    txt("w₂", 24, HI).move_to(net.edges[0][1][0].get_center() + DOWN * 0.35))
        self.play(FadeIn(wl), run_time=0.6)
        self.cue(4)
        rule = txt("input₁·w₁ + input₂·w₂ + b", 22, HI).next_to(net.cols[1], RIGHT, buff=0.35)
        self.play(flash(net.layer_edges(0)), run_time=0.9)
        self.play(FadeIn(rule, shift=LEFT * 0.2), run_time=0.7)
        self.cue(5)
        tr = txt("Real numbers", 24, DIM).move_to(DOWN * 2.0 + LEFT * 0.7)
        self.play(FadeIn(tr), run_time=0.5)
        self.cue(6)
        iv = VGroup(on(net.cols[0][0], "2", 26), on(net.cols[0][1], "1", 26))
        self.play(FadeIn(iv, scale=1.3), run_time=0.6)
        self.cue(7)
        wv = VGroup(txt("3", 26, HI).move_to(wl[0]), txt("−1", 26, HI).move_to(wl[1]))
        self.play(Transform(wl, wv), run_time=0.6)
        self.cue(8)
        bv = txt("bias  +0.5", 22, GOOD).next_to(net.cols[1], DOWN, buff=0.3)
        self.play(FadeIn(bv), run_time=0.6)
        self.cue(9)
        calc = txt("3 × 2  +  (−1) × 1  +  0.5", 30).move_to(DOWN * 2.8 + LEFT * 0.7)
        self.play(FadeOut(tr), Write(calc), flash(net.layer_edges(0)), run_time=1.3)
        self.cue(10)
        res = txt("= 5.5", 30, HI).next_to(calc, RIGHT, buff=0.3)
        ov = on(net.cols[1][0], "5.5", 24, HI)
        self.play(FadeIn(res), FadeIn(ov, scale=1.4), run_time=0.8)
        self.finish()
        self.play(FadeOut(VGroup(wl, rule, iv, bv, calc, res, ov, net.layer_edges(0))), run_time=0.5)
        self.names = names

    def s8(self):
        self.seg(8)
        lin = self.lin
        key = txt("The key idea", 28, HI).move_to(UP * 3.6)
        self.play(FadeIn(key), run_time=0.5)
        self.cue(1)
        net = Net([2, 3, 1], [-3.2, -0.7, 1.8], gap=[2.4, 1.6, 1.6], r=0.5, center_y=0.8)
        self.play(LaggedStart(*[GrowFromCenter(n) for n in net.cols[1]], lag_ratio=0.25), run_time=1.0)
        self.play(Create(net.layer_edges(0)), Create(net.layer_edges(1)), run_time=1.0)
        self.cue(2)
        own = txt("its own weights + its own bias", 20, HI).next_to(net.cols[1][0], UP, buff=0.25)
        self.play(flash(net.into(0, 0)), Indicate(net.cols[1][0], color=HI, scale_factor=1.15), FadeOut(key), FadeIn(own),
                  run_time=1.0)
        self.cue(3)
        for j in range(3):
            self.play(flash(net.into(0, j)), Indicate(net.cols[1][j], color=HI, scale_factor=1.12), run_time=0.6)
        self.cue(4)
        self.play(flash(net.layer_edges(1)), Indicate(lin.cols[1][0], color=HI, scale_factor=1.12), FadeOut(own), run_time=1.0)
        self.cue(5)
        hbox = DashedVMobject(RoundedRectangle(corner_radius=0.3, width=1.6, height=4.4, color=HI, stroke_width=2.5),
                              num_dashes=50).move_to(net.cols[1])
        hl = txt("hidden layer", 24, HI).next_to(hbox, DOWN, buff=0.15)
        self.play(Create(hbox), FadeIn(hl), run_time=1.0)
        self.cue(6)
        seen = VGroup(txt("we see this", 18, GOOD).next_to(lin.cols[0], DOWN, buff=0.45),
                      txt("and this", 18, GOOD).next_to(lin.cols[1], DOWN, buff=0.45))
        self.play(FadeIn(seen), net.cols[1].animate.set_stroke(opacity=0.5), run_time=0.8)
        self.cue(7)
        neu = txt("its nodes: neurons", 22, HI).next_to(hl, DOWN, buff=0.15)
        self.play(FadeIn(neu), net.cols[1].animate.set_stroke(opacity=1), run_time=0.7)
        self.cue(8)
        par = rows(["parameters =", "all the weights", "+ all the biases"], 22)
        par[0].set_color(HI)
        par.move_to(RIGHT * 5.0 + UP * 0.8)
        self.play(FadeIn(par, shift=LEFT * 0.2), run_time=0.8)
        self.cue(9)
        self.clear_all(0.5)
        big = Net([3, 4, 1], [-5.6, -3.4, -1.2], gap=[1.3, 1.25, 1.0], r=0.36, center_y=0.0, edge_w=2)
        cap = txt("the course's example", 22, DIM).move_to(LEFT * 3.4 + UP * 3.2)
        self.play(FadeIn(big), FadeIn(cap), run_time=1.0)
        self.cue(10)
        p1 = txt("each hidden neuron:  3 weights + 1 bias = 4", 22).move_to(RIGHT * 3.2 + UP * 1.8)
        bias_dot = txt("+b", 16, GOOD).next_to(big.cols[1][0], UP, buff=0.08)
        self.play(flash(big.into(0, 0)), big.into(0, 0).animate.set_stroke(HI, 3), FadeIn(bias_dot),
                  FadeIn(p1, shift=LEFT * 0.2), run_time=1.0)
        self.cue(11)
        p2 = txt("4 neurons × 4 = 16", 26, HI).next_to(p1, DOWN, buff=0.35)
        self.play(*[flash(big.into(0, j)) for j in range(4)], big.layer_edges(0).animate.set_stroke(HI, 3),
                  FadeIn(p2), run_time=1.0)
        self.cue(12)
        p3 = txt("output:  4 weights + 1 bias = 5", 22).next_to(p2, DOWN, buff=0.6)
        self.play(flash(big.layer_edges(1)), big.layer_edges(1).animate.set_stroke(GOOD, 3), FadeIn(p3), run_time=1.0)
        self.cue(13)
        self.play(Indicate(p3, color=GOOD), run_time=0.6)
        self.cue(14)
        p4 = txt("16 + 5 = 21 parameters", 32, HI, weight=BOLD).next_to(p3, DOWN, buff=0.6)
        p5 = txt("linear model (3 inputs): 3 weights + 1 bias = 4", 18, DIM).next_to(p4, DOWN, buff=0.25)
        self.play(FadeIn(p4, scale=1.1), FadeIn(p5), run_time=1.0)
        self.finish()
        self.clear_all()

    def s9(self):
        self.seg(9)
        q = txt("Can this draw curves?", 32, HI).move_to(UP * 2.8)
        self.play(FadeIn(q), run_time=0.6)
        self.cue(1)
        no = txt("No.", 40, BAD, weight=BOLD).next_to(q, DOWN, buff=0.3)
        self.play(FadeIn(no, scale=1.3), run_time=0.5)
        self.cue(2)
        only = txt("every step: only × and +", 24, DIM).next_to(no, DOWN, buff=0.3)
        self.play(FadeIn(only), run_time=0.6)
        self.cue(3)
        chain = Net([1, 1, 1], [-3.6, 0, 3.6], r=0.45, center_y=-0.8)
        self.play(FadeIn(chain), run_time=0.7)
        lx = on(chain.cols[0][0], "x", 26)
        self.play(FadeIn(lx), run_time=0.3)
        self.cue(4)
        t2 = txt("× 2", 24, HI).next_to(chain.edges[0][0][0], UP, buff=0.15)
        h = on(chain.cols[1][0], "2x", 22)
        self.play(FadeIn(t2), flash(chain.layer_edges(0)), FadeIn(h), run_time=0.9)
        self.cue(5)
        t3 = txt("× 3", 24, HI).next_to(chain.edges[1][0][0], UP, buff=0.15)
        y = on(chain.cols[2][0], "6x", 22)
        self.play(FadeIn(t3), flash(chain.layer_edges(1)), FadeIn(y), run_time=0.9)
        self.cue(6)
        same = txt("= just  × 6", 30, BAD).next_to(chain, DOWN, buff=0.5)
        self.play(FadeIn(same, shift=UP * 0.2), run_time=0.7)
        self.cue(7)
        gen = txt("w₂(w₁x + b₁) + b₂  =  (w₂w₁)x + (w₂b₁ + b₂)", 26).move_to(DOWN * 3.1)
        self.play(FadeOut(same), Write(gen), run_time=1.6)
        self.cue(8)
        one = txt("= one linear model → one straight line", 24, BAD).next_to(gen, UP, buff=0.3)
        self.play(FadeIn(one, shift=UP * 0.2), run_time=0.8)
        self.finish()
        self.clear_all()

    def s10(self):
        self.seg(10)
        head = txt("Add a bend", 32, HI).move_to(UP * 3.2)
        self.play(FadeIn(head), run_time=0.6)
        self.cue(1)
        sum_n = Circle(radius=0.5, color=WHITE, stroke_width=2.5).set_fill(PANEL, 1).move_to(LEFT * 4 + UP * 1.6)
        s_lab = on(sum_n, "Σ", 26)
        bend = RoundedRectangle(corner_radius=0.2, width=1.4, height=1.0, color=HI, stroke_width=2.5) \
            .set_fill(PANEL, 1).move_to(UP * 1.6)
        glyph = VMobject(stroke_color=HI, stroke_width=4).set_points_as_corners(
            [bend.get_center() + [-0.45, -0.25, 0], bend.get_center() + [0, -0.25, 0], bend.get_center() + [0.4, 0.3, 0]])
        nxt = txt("next layer", 22, DIM).move_to(RIGHT * 4 + UP * 1.6)
        a1 = Arrow(sum_n.get_right(), bend.get_left(), color=DIM, buff=0.1)
        a2 = Arrow(bend.get_right(), nxt.get_left(), color=DIM, buff=0.15)
        ws = txt("weighted sum", 18, DIM).next_to(sum_n, DOWN, buff=0.15)
        self.play(FadeIn(sum_n), FadeIn(s_lab), FadeIn(ws), run_time=0.6)
        self.play(GrowArrow(a1), FadeIn(bend), Create(glyph), run_time=0.9)
        self.play(GrowArrow(a2), FadeIn(nxt), run_time=0.7)
        self.cue(2)
        af = txt("activation function", 26, HI).next_to(bend, DOWN, buff=0.2)
        self.play(FadeIn(af, shift=UP * 0.15), run_time=0.7)
        self.cue(4)
        specs = [(sig, "sigmoid", [0, 1, 1], "1 / (1 + e⁻ˣ)  ·  0 to 1"),
                 (np.tanh, "tanh", [-1, 1, 1], "tanh(x)  ·  −1 to 1"),
                 (lambda x: max(x, 0), "ReLU", [0, 4, 1], "max(0, x)")]
        plots = []
        for k, (f, name, yr, note) in enumerate(specs):
            g, ax = mini_plot(f, name, RIGHT * (k - 1) * 4.4 + DOWN * 1.65, yr, h=1.7)
            nt = txt(note, 16, DIM).next_to(g[0], DOWN, buff=0.12)
            plots.append((g, nt, ax))
        lr = txt("(you may know it from logistic regression)", 15, DIM).next_to(plots[0][1], DOWN, buff=0.08)
        for k, c in enumerate((4, 5, 6)):
            self.cue(c)
            g, nt, _ = plots[k]
            extra = [FadeIn(lr)] if k == 0 else []
            self.play(FadeIn(g[0]), FadeIn(g[1]), FadeIn(g[3]), run_time=0.5)
            self.play(Create(g[2]), FadeIn(nt), *extra, run_time=1.1)
        rax = plots[2][2]
        self.cue(6)
        full = txt("rectified linear unit", 18, HI).next_to(plots[2][0][0], UP, buff=0.12)
        self.play(FadeIn(full), run_time=0.5)
        self.cue(8)
        negp = Line(rax.c2p(-4, 0), rax.c2p(0, 0), color=BAD, stroke_width=7)
        self.play(Create(negp), run_time=0.7)
        self.cue(9)
        posp = Line(rax.c2p(0, 0), rax.c2p(4, 4), color=GOOD, stroke_width=7)
        self.play(Create(posp), run_time=0.7)
        self.cue(10)
        best = txt("usually works best", 18, GOOD).move_to(full)
        self.play(FadeOut(negp), FadeOut(posp), Transform(full, best), Circumscribe(plots[2][0][0], color=GOOD),
                  run_time=1.1)
        self.cue(11)
        why = txt("cheaper · helps avoid vanishing gradients", 16, GOOD).next_to(plots[2][1], DOWN, buff=0.1)
        self.play(FadeIn(why), run_time=0.7)
        self.finish()
        self.clear_all()

    def s11(self):
        self.seg(11)
        q = txt("Why does a bend help?", 30, HI).move_to(UP * 3.3)
        self.play(FadeIn(q), run_time=0.6)
        self.cue(1)
        net = Net([2, 3], [-6.2, -4.5], gap=[1.6, 2.45], r=0.32, center_y=-0.2, edge_w=2)
        nets_lab = txt("trained on the comfort data", 16, DIM).move_to(LEFT * 5.35 + DOWN * 3.6)
        self.play(FadeOut(q), FadeIn(net), FadeIn(nets_lab), run_time=0.9)
        self.cue(2)
        relu_g = VGroup(*[VMobject(stroke_color=HI, stroke_width=3).set_points_as_corners(
            [n.get_center() + [-0.13, -0.06, 0], n.get_center() + [0, -0.06, 0], n.get_center() + [0.12, 0.1, 0]])
            for n in net.cols[1]]).set_z_index(5)
        three = txt("3 ReLU neurons", 20, HI).next_to(net.cols[1], UP, buff=0.2)
        self.play(Create(relu_g), FadeIn(three), run_time=0.8)
        self.cue(3)
        maps, axs = [], []
        side = 2.0
        for j in range(3):
            c = [-2.25, net.cols[1][j].get_center()[1], 0]
            ax = Axes(x_range=[-LIM, LIM, 1], y_range=[-LIM, LIM, 1], x_length=side, y_length=side, tips=False,
                      axis_config={"stroke_width": 0}).move_to(c)
            bg = card(side + 0.12, side + 0.12, c)
            im = heat_mono(ax, R["hidden_grids"][j])
            ln = clip_line(ax, W1[0, j], W1[1, j], B1[j], color=WHITE, width=2.5)
            arrow = Arrow(net.cols[1][j].get_right(), bg.get_left(), color=DIM, buff=0.1, stroke_width=3)
            maps.append(Group(bg, im, ln, arrow))
            axs.append(ax)
        self.play(*[FadeIn(m[0]) for m in maps], *[GrowArrow(m[3]) for m in maps], run_time=0.9)
        self.cue(4)
        self.play(*[Create(m[2]) for m in maps], run_time=1.0)
        self.cue(5)
        self.play(*[FadeIn(m[1]) for m in maps], run_time=0.9)
        self.cue(6)
        grows = txt("bright = bigger output", 15, HI).next_to(maps[0][0], UP, buff=0.08)
        self.play(FadeIn(grows), run_time=0.6)
        for j, c in enumerate((7, 8, 9)):
            self.cue(c)
            lab = txt(f"neuron {j + 1}", 14, WHITE).next_to(maps[j][0], DOWN, buff=0.04)
            self.play(Indicate(net.cols[1][j], color=HI), Circumscribe(maps[j][0], color=HI, buff=0.02), FadeIn(lab),
                      run_time=0.9)
        self.cue(10)
        ax, frame = chart(RIGHT * 3.7 + DOWN * 0.1, side=5.0, names=("temp", "humidity"))
        wtxt = VGroup(*[txt(f"× {fmt(W2[j])}", 17, BAD if W2[j] < 0 else GOOD) for j in range(3)])
        arrs = VGroup()
        for j in range(3):
            a = Arrow(maps[j][0].get_right(), frame.get_left() + UP * (1.2 - 1.2 * j), color=DIM, buff=0.1, stroke_width=3)
            wtxt[j].move_to(a.get_center() + UP * 0.25)
            arrs.add(a)
        self.play(FadeIn(frame), LaggedStart(*[GrowArrow(a) for a in arrs], lag_ratio=0.2), FadeIn(wtxt), run_time=1.1)
        self.cue(11)
        img = heat(ax, R["final_grid"], mix=0.45)
        cd = dots(ax, CX, CY, r=0.06)
        lines = VGroup(*[clip_line(ax, W1[0, j], W1[1, j], B1[j], color=HI, width=2.5) for j in range(3)])
        self.play(FadeIn(img), FadeIn(cd), LaggedStart(*[Create(l) for l in lines], lag_ratio=0.3), run_time=1.6)
        self.cue(12)
        nb = txt("no hand-made features", 20, GOOD).next_to(frame, UP, buff=0.12)
        self.play(FadeIn(nb), run_time=0.6)
        self.cue(13)
        own = txt("the network found its own", 20, GOOD).next_to(frame, DOWN, buff=0.12)
        self.play(FadeIn(own), run_time=0.6)
        self.cue(14)
        more = txt("more neurons & layers → more detailed shapes", 16, HI).next_to(own, DOWN, buff=0.12)
        self.play(FadeIn(more), run_time=0.8)
        self.finish()
        self.clear_all()

    def s12(self):
        self.seg(12)
        head = txt("A neural network has three ingredients", 30, HI).move_to(UP * 3.0)
        self.play(FadeIn(head), run_time=0.7)
        cards = []
        for k, (t1, t2) in enumerate([("Nodes", "arranged in layers"), ("Weights + biases", "connecting each layer to the next"),
                                      ("Activation function", "bends each node's output")]):
            c = card(4.1, 3.6, RIGHT * (k - 1) * 4.4 + DOWN * 0.1)
            a = txt(t1, 26, WHITE).move_to(c.get_center() + DOWN * 0.55)
            b = txt(t2, 16, DIM).next_to(a, DOWN, buff=0.2)
            cards.append((c, a, b))
        mini = Net([2, 3, 1], [-5.6, -4.4, -3.2], gap=0.55, r=0.13, center_y=0.85, edge_w=1.5)
        wline = VGroup(Line(LEFT * 1.0 + UP * 0.85, RIGHT * 1.0 + UP * 0.85, color=HI, stroke_width=4),
                       txt("w", 22, HI).move_to(UP * 1.2), txt("+ b", 20, GOOD).move_to(RIGHT * 1.0 + UP * 1.2))
        rg = VMobject(stroke_color=HI, stroke_width=5).set_points_as_corners(
            [RIGHT * 3.6 + UP * 0.6, RIGHT * 4.4 + UP * 0.6, RIGHT * 5.2 + UP * 1.4])
        icons = [mini, wline, rg]
        for k in range(3):
            self.cue(1 + k)
            c, a, b = cards[k]
            self.play(FadeIn(c), FadeIn(icons[k]), FadeIn(a), FadeIn(b), run_time=0.9)
        self.cue(4)
        cav = txt("One honest caveat", 22, HI).move_to(DOWN * 2.6)
        self.play(FadeIn(cav), run_time=0.5)
        self.cue(5)
        c1 = txt("not always better than hand-made feature crosses", 20).next_to(cav, DOWN, buff=0.2)
        self.play(FadeIn(c1), run_time=0.7)
        self.cue(6)
        c2 = txt("but flexible, and works well on a huge range of problems", 20, GOOD).next_to(c1, DOWN, buff=0.15)
        self.play(FadeIn(c2), run_time=0.7)
        self.finish()
        self.clear_all()

    # ============================================================ PART 3: inference
    def s13(self):
        self.seg(13)
        x = R["infer"]["x"]
        z1 = np.array(x) @ W1 + B1
        h = np.maximum(z1, 0)
        z2 = float(h @ W2 + B2)
        out = sig(z2)
        self.inf = dict(x=x, z1=z1, h=h, z2=z2, out=out)
        net = Net([2, 3, 1], [-5.3, -3.0, -0.7], gap=1.7, r=0.45, center_y=-0.2)
        head = txt("Inference = making a prediction", 28, HI).move_to(LEFT * 3.0 + UP * 3.2)
        self.play(FadeIn(head), FadeIn(net), run_time=1.0)
        self.cue(1)
        panel_bg = card(7.0, 6.2, RIGHT * 3.55 + DOWN * 0.3)
        self.play(FadeIn(panel_bg), run_time=0.5)
        left = panel_bg.get_left()[0] + 0.35
        top = panel_bg.get_top()[1] - 0.45

        def put(m, row):
            m.move_to([0, top - 0.58 * row, 0])
            if m.width > 6.3:
                m.scale_to_fit_width(6.3)
            return m.align_to([left, 0, 0], LEFT)

        self.cue(2)
        rs = put(txt("rescale: 0 = middle of the range", 19, DIM), 0)
        self.play(FadeIn(rs), run_time=0.6)
        self.cue(3)
        t1 = put(txt(f"26°C → {x[0]:.1f}      55% → {x[1]:.1f}", 21), 1)
        xin = VGroup(*[on(n, f"{v:.1f}", 22) for v, n in zip(x, net.cols[0])])
        xl = VGroup(*[txt(s, 17, DIM).next_to(n, LEFT, buff=0.12) for s, n in zip(["temp", "hum."], net.cols[0])])
        self.play(FadeIn(t1), FadeIn(xin, scale=1.3), FadeIn(xl), run_time=0.9)
        self.cue(4)
        hv, lines = VGroup(), VGroup()
        for j in range(3):
            s = put(txt(f"h{'₁₂₃'[j]} = {weighted_sum(W1[:, j], x, B1[j])} = {fmt(z1[j])}", 19), 2.2 + j)
            v = on(net.cols[1][j], fmt(z1[j]), 16, BAD if z1[j] < 0 else WHITE)
            self.play(flash(net.into(0, j)), FadeIn(s, shift=RIGHT * 0.15), FadeIn(v), run_time=0.8)
            hv.add(v)
            lines.add(s)
        self.cue(5)
        hv2 = VGroup(*[on(net.cols[1][j], fmt(h[j]), 16, DIM if h[j] == 0 else WHITE) for j in range(3)])
        relu = put(txt(f"ReLU → {fmt(h[0])}, {fmt(h[1])}, {fmt(h[2])}", 19, HI), 5.4)
        self.play(Transform(hv, hv2), FadeIn(relu), run_time=0.9)
        self.play(Indicate(hv[0], color=BAD), run_time=0.7)
        self.cue(6)
        o1 = put(txt(f"out = {weighted_sum(W2, [round(float(v), 2) for v in h], B2)}", 19), 6.6)
        o2 = put(txt(f"= {fmt(z2)}   →   sigmoid({fmt(z2)}) = {out:.2f}", 19, HI), 7.6)
        self.play(flash(net.layer_edges(1)), FadeIn(o1, shift=RIGHT * 0.15), run_time=1.0)
        self.play(FadeIn(o2, shift=RIGHT * 0.15), run_time=0.8)
        self.cue(7)
        ov = on(net.cols[2][0], f"{out:.2f}", 18, WHITE)
        self.play(FadeIn(ov, scale=1.4), net.cols[2][0].animate.set_fill(BLUE_C, 0.85), run_time=0.8)
        self.cue(8)
        say = VGroup(txt("99%", 22, BLUE_C), txt("comfortable", 18, BLUE_C)).arrange(DOWN, buff=0.08) \
            .next_to(net.cols[2][0], DOWN, buff=0.25)
        self.play(FadeIn(say, shift=UP * 0.2), run_time=0.6)
        self.cue(9)
        arrow = Arrow(LEFT * 5.8 + DOWN * 3.3, LEFT * 0.2 + DOWN * 3.3, color=HI, stroke_width=4, buff=0)
        fwd = txt("information flows forward only", 18, HI).next_to(arrow, UP, buff=0.1)
        self.play(GrowArrow(arrow), FadeIn(fwd), run_time=1.0)
        self.finish()
        self.inf_net, self.inf_hv, self.inf_xin = net, hv, xin
        keep = [net, hv, xin, xl, ov, *net.cols, *net.edges]
        self.play(*[FadeOut(m) for m in self.mobjects if m not in keep], net.cols[2][0].animate.set_fill(PANEL, 1),
                  FadeOut(ov), run_time=0.6)

    def s14(self):
        self.seg(14)
        net, hv, xin = self.inf_net, self.inf_hv, self.inf_xin
        head = txt("Three lessons from the course's exercises", 24, HI).move_to(UP * 3.3)
        self.play(FadeIn(head), run_time=0.6)
        bx = RIGHT * 3.5
        self.cue(1)
        l1 = rows(["1.  All inputs 0  →  every weight × 0"], 20).move_to(bx + UP * 2.2)
        zeros = VGroup(*[on(n, "0", 22) for n in net.cols[0]])
        self.play(FadeIn(l1), Transform(xin, zeros), run_time=0.9)
        self.cue(2)
        rb = [max(b, 0) for b in B1]
        l1b = txt(f"hidden = ReLU(bias) = {fmt(rb[0])}, {fmt(rb[1])}, {fmt(rb[2])}", 19, DIM).next_to(l1, DOWN, buff=0.2).align_to(l1, LEFT)
        hz = VGroup(*[on(net.cols[1][j], fmt(rb[j]), 16, DIM if rb[j] == 0 else WHITE) for j in range(3)])
        self.play(FadeIn(l1b), Transform(hv, hz), run_time=0.9)
        self.cue(3)
        l2 = rows(["2.  Changing a layer only affects", "     the layers after it"], 20).next_to(l1b, DOWN, buff=0.5).align_to(l1, LEFT)
        self.play(FadeIn(l2), Indicate(net.cols[1], color=HI), run_time=0.9)
        self.play(flash(net.layer_edges(1)), Indicate(net.cols[2], color=HI), run_time=0.9)
        self.cue(4)
        x = self.inf["x"]
        back = VGroup(*[on(n, f"{v:.1f}", 22) for v, n in zip(x, net.cols[0])])
        hv_back = VGroup(*[on(net.cols[1][j], fmt(self.inf["h"][j]), 16, DIM if self.inf["h"][j] == 0 else WHITE)
                           for j in range(3)])
        l3 = rows(["3.  Neuron one"], 20).next_to(l2, DOWN, buff=0.5).align_to(l1, LEFT)
        self.play(Transform(xin, back), Transform(hv, hv_back), FadeIn(l3), Circumscribe(net.cols[1][0], color=HI), run_time=1.0)
        self.cue(5)
        a = txt(f"ReLU({fmt(self.inf['z1'][0])}) = 0", 22).next_to(l3, DOWN, buff=0.2).align_to(l1, LEFT).shift(RIGHT * 0.5)
        self.play(FadeIn(a), run_time=0.6)
        self.cue(6)
        b = txt("ReLU(−5.00) = 0", 22).next_to(a, DOWN, buff=0.15).align_to(a, LEFT)
        self.play(FadeIn(b), run_time=0.6)
        self.cue(7)
        c = txt("same value, different calculation", 20, HI).next_to(b, DOWN, buff=0.3).align_to(l1, LEFT)
        c2 = txt("→ check the calculations, not just the outputs", 18, DIM).next_to(c, DOWN, buff=0.12).align_to(l1, LEFT)
        self.play(FadeIn(c), FadeIn(c2), run_time=0.8)
        self.finish()
        self.clear_all()

    # ============================================================ PART 4: training
    def s15(self):
        self.seg(15)
        q = txt("Where did the weights come from?", 30, HI).move_to(UP * 3.2)
        self.play(FadeIn(q), run_time=0.6)
        self.cue(2)
        learned = txt("They were learned.", 26).next_to(q, DOWN, buff=0.25)
        self.play(FadeIn(learned), run_time=0.6)
        self.cue(3)
        ax, frame, xl, yl = comfort_chart(LEFT * 3.4 + DOWN * 0.5, side=5.0)
        snaps = R["snaps"]
        img = heat(ax, snaps[0]["grid"])
        cd = dots(ax, CX, CY, r=0.06)
        rnd = txt("random weights → bad guesses", 18, DIM).next_to(frame, UP, buff=0.1)
        self.play(FadeOut(learned), FadeIn(frame), FadeIn(xl), FadeIn(yl), FadeIn(img), FadeIn(cd), FadeIn(rnd), run_time=1.0)
        self.cue(4)
        how = txt("how bad?", 26, DIM).move_to(RIGHT * 3.6 + UP * 2.0)
        self.play(FadeIn(how), run_time=0.6)
        self.cue(5)
        loss = txt("the loss", 34, HI, weight=BOLD).move_to(how)
        self.play(Transform(how, loss), run_time=0.7)
        self.cue(6)
        hdr = VGroup(txt("guess", 18, DIM), txt("true label", 18, DIM), txt("loss", 18, DIM))
        for i, m in enumerate(hdr):
            m.move_to(RIGHT * (1.9 + 1.75 * i) + UP * 0.9)
        self.play(FadeIn(hdr), run_time=0.6)
        self.cue(7)
        r1 = VGroup(txt("5% blue", 22, ORANGE_C), txt("blue", 22, BLUE_C), txt(f"{-np.log(0.05):.2f}", 24, BAD))
        for i, m in enumerate(r1):
            m.move_to(RIGHT * (1.9 + 1.75 * i) + UP * 0.2)
        cw = txt("confident + wrong", 16, BAD).next_to(r1, DOWN, buff=0.1)
        self.play(FadeIn(r1), FadeIn(cw), run_time=0.8)
        self.cue(8)
        r2 = VGroup(txt("97% blue", 22, BLUE_C), txt("blue", 22, BLUE_C), txt(f"{-np.log(0.97):.2f}", 24, GOOD))
        for i, m in enumerate(r2):
            m.move_to(RIGHT * (1.9 + 1.75 * i) + DOWN * 1.0)
        cr = txt("confident + right", 16, GOOD).next_to(r2, DOWN, buff=0.1)
        self.play(FadeIn(r2), FadeIn(cr), run_time=0.8)
        f = txt("loss = −log(probability given to the true answer)", 15, DIM).move_to(RIGHT * 3.6 + DOWN * 2.15)
        self.play(FadeIn(f), run_time=0.5)
        self.cue(9)
        avg = txt(f"average over all rooms:  {snaps[0]['loss']:.2f}", 24, HI).move_to(RIGHT * 3.6 + DOWN * 2.9)
        self.play(FadeIn(avg, shift=UP * 0.2), run_time=0.8)
        self.finish()
        self.clear_all()

    def s16(self):
        self.seg(16)
        dials = VGroup()
        for k in range(6):
            c = Circle(radius=0.42, color=WHITE, stroke_width=2.5).set_fill(PANEL, 1)
            ptr = Line(c.get_center(), c.get_center() + 0.32 * np.array([np.cos(1 + k), np.sin(1 + k), 0]),
                       color=HI, stroke_width=4)
            lab = txt(f"w{'₁₂₃₄₅₆'[k]}", 16, DIM).next_to(c, DOWN, buff=0.1)
            dials.add(VGroup(c, ptr, lab))
        dials.arrange_in_grid(2, 3, buff=0.5).move_to(LEFT * 4.2 + UP * 0.6)
        dl = txt("every weight is a dial", 22, HI).next_to(dials, UP, buff=0.35)
        self.play(LaggedStart(*[FadeIn(d, scale=0.8) for d in dials], lag_ratio=0.1), FadeIn(dl), run_time=1.2)
        self.cue(1)
        ax = Axes(x_range=[-3, 3, 1], y_range=[0, 5, 1], x_length=6.0, y_length=4.4, tips=False,
                  axis_config={"stroke_color": EDGE, "stroke_width": 2, "include_ticks": False}).move_to(RIGHT * 3.3 + DOWN * 0.1)
        f = lambda w: 0.45 * (w - 0.8) ** 2 + 0.4
        curve = ax.plot(f, x_range=[-2.6, 3, 0.02], color=WHITE, stroke_width=3)
        xlab = txt("value of one weight", 16, DIM).next_to(ax, DOWN, buff=0.1)
        ylab = txt("loss", 16, DIM).next_to(ax, LEFT, buff=0.1)
        wv = ValueTracker(-2.0)
        ball = always_redraw(lambda: Dot(ax.c2p(wv.get_value(), f(wv.get_value())), radius=0.12, color=HI).set_z_index(5))
        self.play(Create(curve), FadeIn(xlab), FadeIn(ylab), FadeIn(ball), Rotate(dials[0][1], -0.6, about_point=dials[0][0].get_center()),
                  run_time=1.2)
        nudge = txt("turn it up a tiny bit… loss goes down", 16, DIM).next_to(ax, UP, buff=0.05)
        self.play(FadeIn(nudge), run_time=0.6)
        self.cue(2)
        tang = always_redraw(lambda: TangentLine(curve, alpha=(wv.get_value() + 2.6) / 5.6, length=2.0, color=BAD, stroke_width=4))
        gl = txt("gradient = the slope here", 20, BAD).move_to(RIGHT * 3.3 + UP * 2.3)
        self.play(Create(tang), FadeIn(gl), FadeOut(nudge), run_time=0.9)
        self.cue(3)
        self.play(wv.animate.set_value(-1.2), run_time=0.8)
        self.cue(4)
        br = BraceBetweenPoints(ax.c2p(-2.0, 0), ax.c2p(-1.2, 0), DOWN, color=HI)
        lrt = txt("learning rate = step size", 18, HI).next_to(br, DOWN, buff=0.05)
        self.play(GrowFromCenter(br), FadeIn(lrt), run_time=0.8)
        self.cue(5)
        self.play(FadeOut(br), FadeOut(lrt), run_time=0.3)
        for v in (-0.5, 0.05, 0.4, 0.62, 0.75):
            self.play(wv.animate.set_value(v), *[Rotate(d[1], 0.25, about_point=d[0].get_center()) for d in dials],
                      run_time=0.32)
        self.cue(6)
        gd = txt("gradient descent", 32, HI, weight=BOLD).move_to(LEFT * 4.2 + DOWN * 2.4)
        self.play(Write(gd), run_time=1.0)
        self.finish()
        self.clear_all()

    def s17(self):
        self.seg(17)
        net = Net([2, 3, 3, 1], [-5.0, -2.2, 0.6, 3.4], gap=1.5, r=0.38, center_y=-0.2)
        many = txt("millions of weights, buried in layers", 22, DIM).move_to(UP * 3.2)
        self.play(FadeIn(net), FadeIn(many), run_time=1.0)
        self.cue(1)
        bp = txt("backpropagation", 34, HI, weight=BOLD).move_to(UP * 3.2)
        self.play(Transform(many, bp), run_time=0.8)
        self.cue(2)
        err = txt("error known here", 20, BAD).next_to(net.cols[3][0], DOWN, buff=0.3)
        self.play(net.cols[3][0].animate.set_stroke(BAD, 4), FadeIn(err), run_time=0.8)
        self.cue(3)
        for l in (2, 1, 0):
            self.play(back_flash(net.layer_edges(l)), net.cols[l].animate.set_stroke(BAD, 3), run_time=0.8)
        self.cue(4)
        bl = txt("each layer's blame comes from the layer after it", 20).move_to(DOWN * 3.1)
        self.play(FadeIn(bl), run_time=0.7)
        self.cue(5)
        allg = txt("one backward pass → every gradient", 22, GOOD).move_to(DOWN * 3.1)
        self.play(*[back_flash(net.layer_edges(l), GOOD) for l in (0, 1, 2)], Transform(bl, allg), run_time=1.2)
        self.cue(6)
        keras = txt("most common training method · libraries like Keras do it automatically", 17, DIM).next_to(allg, UP, buff=0.2)
        self.play(FadeIn(keras), run_time=0.7)
        self.finish()
        self.clear_all()

    def s18(self):
        self.seg(18)
        ax, frame, xl, yl = comfort_chart(LEFT * 3.2 + DOWN * 0.1, side=5.4)
        cd = dots(ax, CX, CY, r=0.065)
        head = txt("Watching it learn", 28, HI).move_to(RIGHT * 3.7 + UP * 3.2)
        self.play(FadeIn(frame), FadeIn(xl), FadeIn(yl), FadeIn(cd), FadeIn(head), run_time=0.9)
        snaps = R["snaps"]

        def status(s):
            return rows([f"step {s['epoch']}", f"loss {s['loss']:.2f}", f"rooms right: {100 * s['acc']:.0f}%"], 24) \
                .move_to(RIGHT * 3.7 + UP * 1.5)

        curve_ax = Axes(x_range=[0, 1500, 500], y_range=[0, 1, 0.5], x_length=5.0, y_length=1.5, tips=False,
                        axis_config={"stroke_color": EDGE, "stroke_width": 2, "include_ticks": False}) \
            .move_to(RIGHT * 3.7 + DOWN * 0.6)
        cl = txt("loss", 16, DIM).next_to(curve_ax, LEFT, buff=0.15)
        losses = R["curve"]
        self.cue(1)
        img = heat(ax, snaps[0]["grid"])
        stat = status(snaps[0])
        self.play(FadeIn(img), FadeIn(stat), FadeIn(curve_ax), FadeIn(cl), run_time=0.9)
        prev = 0
        for cue, idx in ((2, [1]), (3, [2]), (4, [3, 4])):
            self.cue(cue)
            for i in idx:
                s = snaps[i]
                e = s["epoch"]
                pts = [curve_ax.c2p(10 * k, losses[k]) for k in range(prev // 10, e // 10 + 1)]
                seg = VMobject(stroke_color=HI, stroke_width=3).set_points_as_corners(pts)
                new = heat(ax, s["grid"])
                self.play(FadeIn(new), FadeOut(img), Transform(stat, status(s)), Create(seg), run_time=0.9)
                img, prev = new, e
        self.cue(5)
        Xt, yt = np.array(R["test"]["X"]), np.array(R["test"]["y"])
        rings = VGroup(*[Circle(radius=0.085, color=BLUE_C if c else ORANGE_C, stroke_width=3).move_to(ax.c2p(*p)).set_z_index(4)
                         for p, c in zip(Xt, yt)])
        tl = txt(f"{len(yt)} test rooms (rings): never trained on", 18, WHITE).move_to(RIGHT * 3.7 + DOWN * 2.0)
        self.play(cd.animate.set_opacity(0.35), LaggedStart(*[GrowFromCenter(r) for r in rings], lag_ratio=0.02),
                  FadeIn(tl), run_time=1.4)
        self.cue(6)
        tloss = txt(f"test loss {R['test']['loss']:.3f}", 26, GOOD).next_to(tl, DOWN, buff=0.25)
        self.play(FadeIn(tloss), run_time=0.7)
        self.cue(7)
        tacc = txt(f"test rooms right: {100 * R['test']['acc']:.0f}%  → learned the pattern", 18, GOOD).next_to(tloss, DOWN, buff=0.15)
        self.play(FadeIn(tacc), run_time=0.7)
        self.cue(8)
        course = txt("course's solution: 1 hidden layer, 3 ReLU neurons", 15, DIM).next_to(tacc, DOWN, buff=0.25)
        self.play(FadeIn(course), run_time=0.7)
        self.finish()
        self.clear_all()

    def s19(self):
        self.seg(19)
        head = txt("When training goes wrong", 30, HI).move_to(UP * 3.3)
        self.play(FadeIn(head), run_time=0.6)
        self.cue(1)
        mult = txt("backprop multiplies many numbers, layer after layer", 20, DIM).next_to(head, DOWN, buff=0.2)
        self.play(FadeIn(mult), run_time=0.7)

        def chain(factor, y, color):
            boxes = VGroup(*[chip(f"×{factor}", color, w=0.95, size=15) for _ in range(10)]).arrange(RIGHT, buff=0.12)
            return boxes.move_to([0.6, y, 0])

        self.cue(3)
        sp, sax = mini_plot(lambda x: sig(x) * (1 - sig(x)), "sigmoid's slope", LEFT * 5.2 + UP * 0.9, [0, 0.3, 0.1],
                            w=2.6, h=1.2)
        peak = txt("max 0.25", 16, BAD).next_to(sax.c2p(0, 0.25), UP, buff=0.05)
        self.play(FadeIn(sp), FadeIn(peak), run_time=0.9)
        self.cue(4)
        vc = chain("0.25", 1.3, BAD).shift(RIGHT * 0.9)
        val = txt("1", 22).next_to(vc, DOWN, buff=0.2)
        self.play(LaggedStart(*[FadeIn(b, shift=RIGHT * 0.1) for b in vc], lag_ratio=0.1), run_time=1.2)
        vfinal = txt(f"= {0.25 ** 10:.7f}   (under one millionth)", 22, BAD).next_to(vc, DOWN, buff=0.2)
        self.play(FadeIn(vfinal), run_time=0.7)
        self.cue(5)
        early = txt("→ early layers barely learn", 18, DIM).next_to(vfinal, DOWN, buff=0.1)
        self.play(FadeIn(early), run_time=0.5)
        self.cue(6)
        vg = txt("vanishing gradient", 26, BAD, weight=BOLD).next_to(vc, UP, buff=0.2)
        self.play(FadeIn(vg), run_time=0.6)
        self.cue(7)
        rl = txt("ReLU's slope: 1 for every positive input  ×1 ×1 ×1 … = 1", 18, GOOD).next_to(early, DOWN, buff=0.15)
        self.play(FadeIn(rl), run_time=0.7)
        self.cue(9)
        ec = chain("3", -2.0, HI).shift(RIGHT * 0.9)
        self.play(LaggedStart(*[FadeIn(b, shift=RIGHT * 0.1) for b in ec], lag_ratio=0.1), run_time=1.1)
        efinal = txt(f"= {3 ** 10:,}   → never settles", 22, HI).next_to(ec, DOWN, buff=0.2)
        self.play(FadeIn(efinal), run_time=0.6)
        self.cue(10)
        eg = txt("exploding gradient", 26, HI, weight=BOLD).next_to(ec, UP, buff=0.2)
        lw = txt("large weights", 16, DIM).next_to(ec, LEFT, buff=0.2)
        self.play(FadeIn(eg), FadeIn(lw), run_time=0.6)
        self.cue(11)
        fix = txt("fixes: batch normalization · lower learning rate", 18, GOOD).next_to(efinal, DOWN, buff=0.15)
        self.play(FadeIn(fix), run_time=0.7)
        self.finish()
        self.clear_all()

    def s20(self):
        self.seg(20)
        head = txt("ReLU's trap", 30, HI).move_to(LEFT * 3.6 + UP * 3.2)
        self.play(FadeIn(head), run_time=0.5)
        self.cue(1)
        g, ax = mini_plot(lambda x: max(x, 0), "ReLU", LEFT * 3.6 + DOWN * 0.2, [0, 4, 1], w=4.6, h=2.4)
        rng = np.random.default_rng(3)
        sums = rng.uniform(-3.8, -0.5, 9)
        pts = VGroup(*[Dot(ax.c2p(s, 0), radius=0.08, color=BAD).set_z_index(4) for s in sums])
        self.play(FadeIn(g), run_time=0.6)
        self.play(LaggedStart(*[FadeIn(p, scale=1.5) for p in pts], lag_ratio=0.08), run_time=1.0)
        allz = txt("every example's sum < 0 → output 0, slope 0", 17, BAD).next_to(g[0], DOWN, buff=0.15)
        self.play(FadeIn(allz), run_time=0.6)
        self.cue(2)
        stuck = txt("no gradient flows → stuck", 17, DIM).next_to(allz, DOWN, buff=0.1)
        self.play(FadeIn(stuck), run_time=0.5)
        self.cue(3)
        dead = txt("dead ReLU", 28, BAD, weight=BOLD).move_to(LEFT * 3.6 + UP * 2.45)
        self.play(FadeIn(dead), run_time=0.6)
        self.cue(4)
        lk = ax.plot(lambda x: x if x > 0 else 0.08 * x, x_range=[-4, 4, 0.02], color=GOOD, stroke_width=3)
        fixes = txt("fixes: lower learning rate · Leaky ReLU", 16, GOOD).next_to(stuck, DOWN, buff=0.15)
        self.play(Create(lk), FadeIn(fixes), run_time=1.0)
        self.cue(5)
        dh = txt("Dropout", 30, HI).move_to(RIGHT * 3.6 + UP * 3.2)
        net = Net([2, 4, 4, 1], [1.4, 2.9, 4.4, 5.9], gap=0.9, r=0.25, center_y=0.6, edge_w=1.5)
        self.play(FadeIn(dh), FadeIn(net), run_time=0.8)
        self.cue(6)
        rng = np.random.default_rng(5)
        hidden = list(net.cols[1]) + list(net.cols[2])
        crosses = None
        for step in range(3):
            off = rng.choice(len(hidden), 3, replace=False)
            new = VGroup(*[Cross(hidden[k], stroke_color=BAD, stroke_width=4, scale_factor=0.7) for k in off])
            steplab = txt(f"training step {step + 1}", 16, DIM).next_to(net, DOWN, buff=0.2)
            anims = [FadeIn(new), *[hidden[k].animate.set_fill(opacity=0.2) for k in range(len(hidden))]]
            if crosses is not None:
                anims += [FadeOut(crosses), FadeOut(self._steplab)]
            self.play(*anims, FadeIn(steplab), run_time=0.6)
            self.play(*[hidden[k].animate.set_fill(PANEL, 1) for k in range(len(hidden))], run_time=0.2)
            crosses, self._steplab = new, steplab
        self.cue(7)
        why = txt("no neuron can be relied on alone → better on new examples", 14, GOOD).move_to(RIGHT * 3.6 + DOWN * 1.6)
        self.play(FadeIn(why), run_time=0.7)
        self.cue(8)
        bar = Line(RIGHT * 1.6 + DOWN * 2.6, RIGHT * 5.6 + DOWN * 2.6, color=DIM, stroke_width=4)
        ends = VGroup(txt("0  none", 16, DIM).next_to(bar.get_start(), DOWN, buff=0.15),
                      txt("1  all", 16, DIM).next_to(bar.get_end(), DOWN, buff=0.15))
        rl = txt("dropout rate", 18).next_to(bar, UP, buff=0.12)
        self.play(Create(bar), FadeIn(ends), FadeIn(rl), run_time=0.8)
        self.cue(9)
        useful = Line(bar.point_from_proportion(0.1), bar.point_from_proportion(0.6), color=GOOD, stroke_width=8)
        ul = txt("useful", 16, GOOD).next_to(useful, DOWN, buff=0.15)
        self.play(Create(useful), FadeIn(ul), run_time=0.8)
        self.finish()
        self.clear_all()

    # ============================================================ PART 5: many classes
    def fruit(self, k, r=0.32):
        return VGroup(Circle(radius=r, color=FRUIT_C[k], stroke_width=0).set_fill(FRUIT_C[k], 1),
                      txt(FRUIT[k], 16).next_to(ORIGIN, DOWN, buff=r + 0.12))

    def s21(self):
        self.seg(21)
        two = VGroup(chip("comfortable?", WHITE, size=20), txt("yes / no", 20, DIM)).arrange(DOWN, buff=0.2).move_to(LEFT * 3.6 + UP * 0.6)
        self.play(FadeIn(two), run_time=0.7)
        self.cue(1)
        bl = txt("binary classification", 26, HI).next_to(two, DOWN, buff=0.5)
        self.play(FadeIn(bl), run_time=0.6)
        self.cue(2)
        self.cue(3)
        fr = VGroup(*[self.fruit(k) for k in range(4)]).arrange(RIGHT, buff=0.55).move_to(RIGHT * 3.4 + UP * 0.6)
        self.play(LaggedStart(*[FadeIn(f, scale=0.8) for f in fr], lag_ratio=0.25), run_time=1.4)
        self.cue(4)
        mc = txt("multi-class classification", 26, HI).next_to(fr, DOWN, buff=0.5)
        self.play(FadeIn(mc), run_time=0.7)
        self.finish()
        self.clear_all()

    def s22(self):
        self.seg(22)
        head = txt("One versus all", 30, HI).move_to(UP * 3.3)
        self.play(FadeIn(head), run_time=0.6)
        self.cue(1)
        sep = VGroup()
        for k in range(4):
            box = chip(f"is it {'an' if FRUIT[k][0] in 'ao' else 'a'} {FRUIT[k]}?", FRUIT_C[k], w=3.0, size=18)
            sep.add(box)
        sep.arrange(DOWN, buff=0.3).move_to(LEFT * 4.3 + DOWN * 0.2)
        cap = txt("4 separate yes/no classifiers", 18, DIM).next_to(sep, UP, buff=0.3)
        self.play(FadeIn(cap), run_time=0.5)
        for k in range(4):
            self.cue(2 + k)
            self.play(FadeIn(sep[k], shift=RIGHT * 0.2), run_time=0.5)
        self.cue(6)
        waste = txt("many classes → wasteful", 18, BAD).next_to(sep, DOWN, buff=0.3)
        self.play(FadeIn(waste), run_time=0.6)
        self.cue(7)
        net = Net([1, 3, 4, 4], [-0.6, 1.0, 2.6, 4.2], gap=[1, 1.2, 1.05, 1.05], r=0.26, center_y=-0.2, edge_w=1.5)
        pear = self.fruit(2, r=0.22)[0].next_to(net.cols[0][0], LEFT, buff=0.25)
        sigs = VGroup(*[on(n, "σ", 18, HI) for n in net.cols[3]])
        self.play(FadeIn(net), FadeIn(pear), FadeIn(sigs), sep.animate.set_opacity(0.35), cap.animate.set_opacity(0.35),
                  run_time=1.0)
        oc = txt("one network · one sigmoid output per class", 16, DIM).next_to(net, UP, buff=0.3)
        self.play(FadeIn(oc), run_time=0.5)
        labs = VGroup()
        for k in range(4):
            self.cue(8 + k)
            lab = txt(f"{FRUIT[k]}?  {OVA[k]:.2f}", 18, WHITE if OVA[k] > 0.5 else DIM).next_to(net.cols[3][k], RIGHT, buff=0.2)
            self.play(FadeIn(lab, shift=RIGHT * 0.15), flash(net.into(2, k)), run_time=0.6)
            labs.add(lab)
        self.cue(12)
        ind = txt("each answer independent", 18, DIM).move_to(RIGHT * 2.3 + DOWN * 2.8)
        self.play(FadeIn(ind), run_time=0.6)
        self.cue(13)
        tot = txt(f"total = {sum(OVA):.2f}  (not 1)", 24, BAD).next_to(ind, DOWN, buff=0.15)
        self.play(FadeIn(tot), run_time=0.7)
        self.finish()
        self.mc_net, self.mc_sigs, self.mc_labs, self.mc_pear = net, sigs, labs, pear
        keep = [net, pear, *net.cols, *net.edges]
        self.play(*[FadeOut(m) for m in self.mobjects if m not in keep],
                  Group(net, pear).animate.shift(LEFT * 4.2), run_time=0.8)

    def s23(self):
        self.seg(23)
        net = self.mc_net
        head = txt("Make the classes compete", 30, HI).move_to(UP * 3.3)
        self.play(FadeIn(head), run_time=0.6)
        self.cue(1)
        sm = txt("softmax", 30, HI, weight=BOLD).next_to(head, DOWN, buff=0.15)
        aka = txt("(the course calls it one-vs-one)", 16, DIM).next_to(sm, RIGHT, buff=0.25)
        self.play(FadeIn(sm), FadeIn(aka), run_time=0.7)
        ex = np.exp(SCORES)
        probs = ex / ex.sum()
        cols_x = [1.8, 3.25, 4.7, 6.15]
        hdr = VGroup(*[txt(s, 17, DIM).move_to([x, 1.4, 0]) for s, x in
                       zip(["", "raw score", "e^score", "÷ total"], cols_x)])
        names = VGroup(*[txt(FRUIT[k], 19, FRUIT_C[k]).move_to([cols_x[0], 0.75 - 0.6 * k, 0]) for k in range(4)])
        sc = VGroup(*[txt(f"{SCORES[k]:.1f}", 20).move_to([cols_x[1], 0.75 - 0.6 * k, 0]) for k in range(4)])
        self.play(FadeIn(hdr[:2]), FadeIn(names), FadeIn(sc), run_time=0.8)
        self.cue(2)
        e = VGroup(*[txt(f"{ex[k]:.2f}", 20).move_to([cols_x[2], 0.75 - 0.6 * k, 0]) for k in range(4)])
        self.play(FadeIn(hdr[2]), LaggedStart(*[TransformFromCopy(sc[k], e[k]) for k in range(4)], lag_ratio=0.15), run_time=1.1)
        pos = txt("always positive", 15, DIM).next_to(e, DOWN, buff=0.2)
        self.play(FadeIn(pos), run_time=0.4)
        self.cue(3)
        tl = txt(f"total {ex.sum():.2f}", 17, DIM).move_to(pos)
        p = VGroup(*[txt(f"{probs[k]:.2f}", 22, HI if k == 2 else WHITE).move_to([cols_x[3], 0.75 - 0.6 * k, 0]) for k in range(4)])
        self.play(Transform(pos, tl), FadeIn(hdr[3]), LaggedStart(*[TransformFromCopy(e[k], p[k]) for k in range(4)], lag_ratio=0.15),
                  run_time=1.2)
        self.cue(4)
        s1 = txt(f"sum = {probs.round(2).sum():.2f}", 24, GOOD).move_to([cols_x[3] - 0.45, -2.0, 0])
        self.play(FadeIn(s1, shift=UP * 0.2), run_time=0.7)
        outs = VGroup(*[txt(f"{probs[k]:.2f}", 16, HI if k == 2 else DIM).next_to(net.cols[3][k], RIGHT, buff=0.15) for k in range(4)])
        self.play(FadeIn(outs), run_time=0.6)
        self.cue(5)
        win = SurroundingRectangle(VGroup(names[2], p[2]), color=HI, buff=0.12, corner_radius=0.1)
        pc = txt(f"pear {100 * probs[2]:.0f}%", 24, HI).move_to([cols_x[1], -2.0, 0])
        self.play(Create(win), FadeIn(pc), run_time=0.8)
        self.cue(6)
        fast = txt("bonus: helps training converge faster", 16, DIM).move_to(RIGHT * 4.0 + DOWN * 2.7)
        self.play(FadeIn(fast), run_time=0.6)
        self.cue(7)
        same = txt("layer before the output: one node per class (4 = 4)", 16, HI).next_to(net, DOWN, buff=0.3)
        self.play(Indicate(net.cols[2], color=HI), Indicate(net.cols[3], color=HI), FadeIn(same), run_time=1.0)
        self.finish()
        self.clear_all()

    def s24(self):
        self.seg(24)
        head = txt("Lots of classes", 28, HI).move_to(LEFT * 3.5 + UP * 3.2)
        self.play(FadeIn(head), run_time=0.5)
        grid = VGroup(*[Square(0.32, stroke_color=EDGE, stroke_width=1).set_fill("#262A33", 1) for _ in range(60)]) \
            .arrange_in_grid(5, 12, buff=0.06).move_to(LEFT * 3.5 + UP * 1.5)
        self.play(FadeIn(grid), run_time=0.7)
        self.cue(1)
        full = txt("full softmax: compute all 60", 17).next_to(grid, DOWN, buff=0.2)
        self.play(*[s.animate.set_fill(HI, 0.8) for s in grid], FadeIn(full), run_time=0.9)
        self.cue(2)
        rng = np.random.default_rng(2)
        true = 17
        sample = set(rng.choice(60, 8, replace=False).tolist()) - {true}
        cs = txt("candidate sampling: true class + a random few", 16).next_to(full, DOWN, buff=0.15)
        self.play(*[s.animate.set_fill(GOOD if i == true else (HI if i in sample else "#262A33"), 0.9 if i == true or i in sample else 1)
                    for i, s in enumerate(grid)], FadeIn(cs), run_time=1.0)
        self.cue(3)
        one = txt("Softmax: exactly one class per example", 20, HI).move_to(RIGHT * 3.6 + UP * 3.2)
        self.play(FadeIn(one), run_time=0.7)
        self.cue(4)
        bowl = Arc(radius=1.1, start_angle=PI, angle=PI, color=WHITE, stroke_width=4).move_to(RIGHT * 3.3 + UP * 1.4)
        fr = VGroup(*[Circle(radius=0.22, stroke_width=0).set_fill(FRUIT_C[k], 1).move_to(RIGHT * (2.7 + 0.42 * i) + UP * 1.55)
                      for i, k in enumerate([0, 1, 0, 1])])
        both = txt("apples AND oranges → two yes answers", 18).next_to(bowl, DOWN, buff=0.3)
        self.play(Create(bowl), FadeIn(fr), FadeIn(both), run_time=1.0)
        self.cue(5)
        ill = [0.91, 0.87, 0.05, 0.03]
        vals = VGroup(*[txt(f"{FRUIT[k]}?  {ill[k]:.2f}", 18, GOOD if ill[k] > 0.5 else DIM) for k in range(4)]) \
            .arrange(DOWN, aligned_edge=LEFT, buff=0.15).next_to(both, DOWN, buff=0.35)
        sg = txt("separate sigmoids: one independent yes/no each", 17, HI).next_to(vals, DOWN, buff=0.25)
        il = txt("(illustrative numbers)", 14, DIM).next_to(sg, DOWN, buff=0.1)
        self.play(FadeIn(vals), FadeIn(sg), FadeIn(il), run_time=1.0)
        self.finish()
        self.clear_all()

    def s25(self):
        self.seg(25)
        head = txt("Putting it together", 34, HI).move_to(UP * 3.1)
        self.play(FadeIn(head), run_time=0.6)
        items = [("Linear model", "weights × features + bias → straight lines only"),
                 ("Hidden layers", "combine features, but alone are still linear"),
                 ("Activation functions", "add the bends that make curves possible"),
                 ("Inference", "run the network forward to predict"),
                 ("Training", "loss + backpropagation nudge every weight downhill"),
                 ("Many classes", "one-vs-all sigmoids, or softmax")]
        rws = VGroup(*[VGroup(txt(a, 24, WHITE), txt(b, 20, DIM)).arrange(RIGHT, buff=0.35) for a, b in items])
        rws.arrange(DOWN, aligned_edge=LEFT, buff=0.32).move_to(DOWN * 0.2)
        for k, r in enumerate(rws):
            self.cue(1 + k)
            self.play(FadeIn(r, shift=RIGHT * 0.25), run_time=0.7)
        self.cue(7)
        self.play(FadeOut(rws), FadeOut(head), run_time=0.6)
        end = txt("That's a neural network.", 44, WHITE, weight=BOLD)
        sub = txt("from the ground up", 24, HI).next_to(end, DOWN, buff=0.3)
        src = txt("Source: developers.google.com/machine-learning/crash-course/neural-networks", 16, DIM)
        src.next_to(sub, DOWN, buff=0.4)
        self.play(FadeIn(end, shift=UP * 0.2), FadeIn(sub), FadeIn(src), run_time=0.9)
        self.finish(1.5)
