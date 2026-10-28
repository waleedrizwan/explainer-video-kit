"""Why neural networks beat decision trees on images — handwritten digits.
Every number on screen comes from results.json (python train.py).

    python narration.py
    manim -ql scene.py Explainer      # 480p preview
    manim -qh scene.py Explainer      # 1080p final -> media/videos/scene/1080p60/Explainer.mp4
"""
import json
import sys
from pathlib import Path

import numpy as np
from manim import *

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if (p / "vidkit").is_dir())))
from vidkit import NarratedScene  # noqa: E402

HERE = Path(__file__).parent
R = json.loads((HERE / "results.json").read_text())

POS, NEG, BG = "#F2C230", "#4C8DFF", "#22262F"   # weight colours: ink here pushes up / down
HI, GOOD, BAD, DIM = "#FFD166", "#5BD38A", "#FF6B6B", "#8A93A6"
SCREEN, PANEL, EDGE = "#0E1014", "#171A21", "#2A2F3A"
FONT = "Avenir Next"
TEXT_SCALE = 1.25        # every txt() size is scaled by this
config.background_color = SCREEN

WALK = {w["label"]: w for w in R["walk"]}
ZERO, FOUR, EIGHT = WALK[0], WALK[4], WALK[8]
TWO = R["small_tree_wrong"]                      # the 2 the small tree calls an 8
W1, B1 = np.array(R["hidden_weights"]), np.array(R["hidden_bias"])
W2 = np.array(R["out_weights"])
BOTTOM = 1                                       # hidden neuron that likes ink along the bottom
FEATURED = [1, 12, 9, 3]                         # three readable neurons + one that isn't
SNAPS = R["snaps"]
SNAP_E = [s["epoch"] for s in SNAPS]
CURVE = [c["test"] for c in R["curve"]]


def txt(s, size=24, color=WHITE, **kw):
    kw.setdefault("font", FONT)
    kw.setdefault("weight", MEDIUM)
    return Text(s, font_size=round(size * TEXT_SCALE), color=color, **kw)


def panel(mob, buff=0.3, **kw):
    """Rounded dark card behind a group."""
    return RoundedRectangle(corner_radius=0.2, width=mob.width + 2 * buff, height=mob.height + 2 * buff,
                            stroke_color=EDGE, stroke_width=1.5, **kw).set_fill(PANEL, 1).move_to(mob).set_z_index(-5)


def card(w, h, center):
    return RoundedRectangle(corner_radius=0.2, width=w, height=h, stroke_color=EDGE, stroke_width=1.5) \
        .set_fill(PANEL, 1).move_to(center).set_z_index(-5)


def pct(v):
    return f"{100 * v:.1f}%"


def sig(z):
    return 1 / (1 + np.exp(-z))


# ---------------------------------------------------------------- 8x8 digit grids
def ink(v):
    return interpolate_color(ManimColor("#262A33"), WHITE, v / 16)


def digit_grid(pix, cell=0.5, values=False, size=None):
    """VGroup(squares, value labels). Pixel p is squares[p], row by row from the top left."""
    sq = VGroup(*[RoundedRectangle(corner_radius=cell * 0.14, width=cell * 0.9, height=cell * 0.9, stroke_width=0)
                  .set_fill(ink(v), 1).move_to([(p % 8) * cell, -(p // 8) * cell, 0])
                  for p, v in enumerate(pix)]).center()
    lab = VGroup(*[value_label(v, cell).move_to(sq[p]) for p, v in enumerate(pix)])
    if not values:
        lab.set_opacity(0)
    return VGroup(sq, lab)


def value_label(v, cell):
    return Text(str(v), font=FONT, weight=MEDIUM, font_size=int(cell * 40), color=BLACK if v >= 9 else GREY_A)


def grid_to(g, pix, values=True, run=True):
    """Animations that turn grid g into another digit."""
    cell = g[0][1].get_center()[0] - g[0][0].get_center()[0]
    anims = [g[0][p].animate.set_fill(ink(v), 1) for p, v in enumerate(pix)]
    new = VGroup(*[value_label(v, cell).move_to(g[0][p]) for p, v in enumerate(pix)])
    if not values:
        new.set_opacity(0)
    anims.append(Transform(g[1], new))
    return anims


def pixel_box(g, p, color=HI):
    return SurroundingRectangle(g[0][p], color=color, buff=0.03, stroke_width=5, corner_radius=0.05)


# ---------------------------------------------------------------- weight heatmaps
POS_RGB, NEG_RGB, BG_RGB = (np.array(color_to_rgb(c)) for c in (POS, NEG, BG))


def heat_colors(w):
    """Colour per pixel: yellow where ink pushes the neuron up, blue where it pushes down, scaled to the neuron's
    own strongest weight."""
    a = np.asarray(w, float)
    a = a / (np.abs(a).max() + 1e-9)
    pos, neg = np.clip(a, 0, 1)[:, None], np.clip(-a, 0, 1)[:, None]
    rgb = pos * POS_RGB + neg * NEG_RGB + (1 - pos - neg) * BG_RGB
    return [rgb_to_color(c) for c in rgb]


def set_heat(cells, w):
    for s, c in zip(cells, heat_colors(w)):
        s.set_fill(c, 1)
    return cells


def heat(w, size=2.0):
    """VGroup(64 cells, frame) — a weight vector drawn as an 8x8 map."""
    c = size / 8
    cells = VGroup(*[RoundedRectangle(corner_radius=c * 0.12, width=c * 0.9, height=c * 0.9, stroke_width=0)
                     .move_to([(p % 8) * c, -(p // 8) * c, 0]) for p in range(64)]).center()
    set_heat(cells, w)
    frame = RoundedRectangle(corner_radius=min(0.12, size * 0.06), width=size + c * 0.3, height=size + c * 0.3,
                             stroke_color=EDGE, stroke_width=1.5).move_to(cells)
    return VGroup(cells, frame)


def weights_at(k):
    k = float(np.clip(k, 0, len(SNAPS) - 1))
    i0 = int(np.floor(k)); i1 = min(i0 + 1, len(SNAPS) - 1); f = k - i0
    return (1 - f) * np.array(SNAPS[i0]["W1"]) + f * np.array(SNAPS[i1]["W1"])


def epoch_at(k):
    return float(np.interp(k, range(len(SNAP_E)), SNAP_E))


# ---------------------------------------------------------------- small decision tree
XS = [2.3, 1.15, 0.575]
YS = [2.2, 0.8, -0.6, -2.0]


def build_tree():
    nodes, edges, elabels = {}, {}, VGroup()

    def rec(n, path, x, d):
        if n["leaf"]:
            box = RoundedRectangle(corner_radius=0.1, width=0.56, height=0.56, stroke_color=GREY_B, stroke_width=2)
            lab = txt(str(n["pred"]), 22)
        else:
            lab = txt(f"px {n['pixel']} > {n['thr']:g}?", 15)
            box = RoundedRectangle(corner_radius=0.12, width=lab.width + 0.3, height=0.5,
                                   stroke_color=GREY_B, stroke_width=2)
        g = VGroup(box, lab).move_to([x, YS[d], 0])
        box.set_fill(PANEL, 1)
        nodes[path] = g
        if n["leaf"]:
            return
        for key, sgn in (("no", -1), ("yes", 1)):
            cx = x + sgn * XS[d]
            child = path + key[0]
            rec(n[key], child, cx, d + 1)
            e = Line(g.get_bottom(), nodes[child].get_top(), stroke_width=2, color=GREY_C)
            edges[child] = e
            if d < 2:
                elabels.add(txt(key, 14, DIM).move_to(e.get_center() + [sgn * 0.28, 0.08, 0]))

    rec(R["tree"], "", XS[0], 0)
    return nodes, edges, elabels


def full_tree_drawing():
    """The unlimited-depth tree: leaves evenly spaced, each split above the middle of its children."""
    L, Rt = R["full_tree"]["left"], R["full_tree"]["right"]
    pos, order = {}, []

    def leaves(i, d):
        if L[i] == -1:
            order.append(i)
            pos[i] = [None, d]
        else:
            pos[i] = [None, d]
            leaves(L[i], d + 1)
            leaves(Rt[i], d + 1)
    leaves(0, 0)
    xs = np.linspace(-6.2, 6.2, len(order))
    for x, i in zip(xs, order):
        pos[i][0] = x

    def fill(i):
        if L[i] != -1:
            pos[i][0] = (fill(L[i]) + fill(Rt[i])) / 2
        return pos[i][0]
    fill(0)
    p = lambda i: np.array([pos[i][0], 2.9 - pos[i][1] * 0.36, 0])  # noqa: E731
    lines = VGroup(*[Line(p(i), p(c), stroke_width=1.5, color=GREY_C)
                     for i in range(len(L)) if L[i] != -1 for c in (L[i], Rt[i])])
    dots = VGroup(*[Dot(p(i), radius=0.035, color=GREEN_C) for i in order])
    return VGroup(lines, dots)


class Explainer(NarratedScene):
    project_dir = Path(__file__).parent

    def clear_all(self, run_time=0.6):
        if self.mobjects:
            self.play(*[FadeOut(m) for m in self.mobjects], run_time=run_time)

    def construct(self):
        self.digit_is_numbers()
        self.decision_tree()
        self.tree_accuracy()
        self.hidden_neuron()
        self.bottom_neuron()
        self.output_layer()
        self.training()
        self.scoreboard()
        self.fair_close()

    # ============================================================ 0. a digit is 64 numbers
    def digit_is_numbers(self):
        pix = ZERO["pixels"]
        self.seg(0)
        g = digit_grid(pix, cell=0.6)
        self.play(FadeIn(g[0], scale=0.95), run_time=1.0)
        self.cue(3)
        self.play(g[1].animate.set_opacity(1), run_time=1.2)
        self.cue(4)
        scale = VGroup(*[RoundedRectangle(corner_radius=0.05, width=0.3, height=0.3, stroke_width=0).set_fill(ink(v), 1)
                         for v in range(17)]).arrange(RIGHT, buff=0.03)
        scale.next_to(g, DOWN, buff=0.35)
        ends = VGroup(txt("0 blank", 18, DIM).next_to(scale, LEFT, buff=0.2),
                      txt("16 full ink", 18, DIM).next_to(scale, RIGHT, buff=0.2))
        self.play(FadeIn(scale), FadeIn(ends), run_time=0.8)
        self.cue(5)
        idx = VGroup(*[txt(str(p), 16, HI).move_to(g[0][p]) for p in range(64)])
        vals = g[1].copy()
        self.play(FadeOut(g[1]), LaggedStart(*[FadeIn(i) for i in idx], lag_ratio=0.03), run_time=2.2)
        self.wait(0.6)
        self.play(FadeOut(idx), FadeIn(vals), run_time=0.6)
        g.submobjects[1] = vals

        self.cue(6)
        self.play(FadeOut(scale), FadeOut(ends), g.animate.scale(0.72).move_to([-4.35, 1.2, 0]), run_time=1.0)
        cols = VGroup(*[txt(str(v), 17) for v in pix])
        for p, c in enumerate(cols):
            c.move_to([-0.55 + (p % 16) * 0.46, 1.95 - (p // 16) * 0.6, 0])
        head = txt("the input: 64 numbers", 26, HI).move_to([2.9, 2.85, 0])
        self.play(FadeIn(card(4.3, 4.3, [-4.35, 1.2, 0])), FadeIn(card(7.9, 4.3, [2.9, 1.2, 0])), run_time=0.5)
        self.play(FadeIn(head), LaggedStart(*[TransformFromCopy(g[1][p], cols[p]) for p in range(64)],
                                             lag_ratio=0.02), run_time=2.5)
        self.cue(8)
        n_tr, n_te = R["n_train"], R["n_test"]
        unit = 11.0 / (n_tr + n_te)
        tr = RoundedRectangle(corner_radius=0.15, width=n_tr * unit, height=0.7, stroke_width=0).set_fill("#3A3F4B", 1)
        te = RoundedRectangle(corner_radius=0.15, width=n_te * unit, height=0.7, stroke_width=0).set_fill("#2D5BA8", 1)
        VGroup(tr, te).arrange(RIGHT, buff=0.08).move_to([0, -2.35, 0])
        trl = txt(f"{n_tr:,} digits to learn from", 22).move_to(tr)
        tel = txt(f"{n_te} to test", 22).move_to(te)
        self.play(GrowFromEdge(tr, LEFT), FadeIn(trl), run_time=1.0)
        self.cue(9)
        self.play(GrowFromEdge(te, LEFT), FadeIn(tel), run_time=0.8)
        hid = txt("never seen during learning", 18, DIM).next_to(te, DOWN, buff=0.15).align_to(te, RIGHT)
        self.play(FadeIn(hid), run_time=0.6)
        self.finish(0.2)
        self.clear_all()

    # ============================================================ 1. decision tree
    def decision_tree(self):
        self.seg(1)
        g = digit_grid(ZERO["pixels"], cell=0.4, values=True).move_to([-4.9, 0.6, 0])
        title = txt("decision tree: one pixel per question", 26).move_to([2.35, 3.35, 0])
        cards = VGroup(card(3.9, 6.3, [-4.9, -0.45, 0]), card(9.1, 6.3, [2.35, -0.45, 0]))
        self.play(FadeIn(cards), FadeIn(g), FadeIn(title), run_time=1.0)
        self.cue(2)
        rp = R["root_pixel"]
        q = txt(f"Is pixel {rp} lit at all?", 30, HI).move_to([2.3, 2.1, 0])
        self.play(FadeIn(q, shift=DOWN * 0.15), run_time=0.8)
        self.cue(3)
        box = pixel_box(g, rp)
        lbl = txt(f"pixel {rp}", 18, HI).next_to(g, UP, buff=0.15)
        self.play(Create(box), FadeIn(lbl), run_time=0.8)
        self.cue(4)
        z = txt(f"lit in only {100 * R['root_yes_by_digit']['0']:.0f}% of training zeros", 22, DIM).next_to(q, DOWN, buff=0.3)
        self.play(FadeIn(z), Indicate(g[0][rp], color=HI), run_time=1.0)

        self.cue(5)
        nodes, edges, elabels = build_tree()
        self.play(FadeOut(z), FadeOut(title), ReplacementTransform(q, nodes[""]), run_time=1.0)
        self.play(*[Create(edges[k]) for k in ("n", "y")], FadeIn(nodes["n"]), FadeIn(nodes["y"]),
                  FadeIn(elabels[:2]), run_time=0.8)
        self.cue(6)
        rest = [k for k in nodes if len(k) >= 2]
        self.play(*[Create(edges[k]) for k in rest], *[FadeIn(nodes[k]) for k in rest], FadeIn(elabels[2:]),
                  run_time=1.2)
        self.play(FadeOut(box), FadeOut(lbl), run_time=0.3)

        def walk(rec, cues, start_cue, end_cue, verdict_cue):
            path, marks, lines = "", [], VGroup()
            for k, (st, c) in enumerate(zip(rec["tree_small_path"], cues)):
                self.cue(c)
                ans = "yes" if st["yes"] else "no"
                b = pixel_box(g, st["pixel"])
                line = txt(f"pixel {st['pixel']} = {st['value']}  →  {ans}", 22,
                           HI if st["yes"] else GREY_A).move_to([-4.9, -1.55 - 0.45 * k, 0])
                path += ans[0]
                nodes_box = nodes[path[:-1]][0]
                self.play(Create(b), FadeIn(line), nodes_box.animate.set_stroke(HI, 3),
                          edges[path].animate.set_stroke(HI, 4), run_time=0.6)
                marks.append(b); lines.add(line)
                self.play(FadeOut(b), run_time=0.2)
            self.cue(verdict_cue)
            leaf = nodes[path]
            guess = rec["tree_small_guess"]
            ok = guess == rec["label"]
            v = txt(f"tree says {guess}  {'✓' if ok else '✗'}", 26, GOOD if ok else BAD).move_to([-4.9, -3.05, 0])
            self.play(leaf[0].animate.set_stroke(GOOD if ok else BAD, 4), FadeIn(v), run_time=0.6)
            return path, lines, v

        self.cue(7)
        self.play(Indicate(g, color=WHITE, scale_factor=1.05), run_time=0.6)
        path0, lines0, v0 = walk(ZERO, [8, 9, 10], 7, 12, 11)

        self.cue(13)
        reset = [nodes[path0[:i]][0].animate.set_stroke(GREY_B, 2) for i in range(len(path0) + 1)]
        reset += [edges[path0[:i]].animate.set_stroke(GREY_C, 2) for i in range(1, len(path0) + 1)]
        self.play(*reset, FadeOut(lines0), FadeOut(v0), *grid_to(g, TWO["pixels"]), run_time=0.8)
        st0 = TWO["tree_small_path"][0]
        self.cue(14)
        b = pixel_box(g, st0["pixel"])
        l1 = txt(f"pixel {st0['pixel']} = {st0['value']}  →  yes", 22, HI).move_to([-4.9, -1.55, 0])
        self.play(Create(b), FadeIn(l1), Flash(g[0][st0["pixel"]], color=HI), run_time=0.8)
        self.cue(15)
        self.play(nodes[""][0].animate.set_stroke(HI, 3), edges["y"].animate.set_stroke(HI, 4), run_time=0.6)
        self.play(FadeOut(b), run_time=0.2)
        path, lines = "y", VGroup(l1)
        for k, c in ((1, 16), (2, 17)):
            st = TWO["tree_small_path"][k]
            self.cue(c)
            ans = "yes" if st["yes"] else "no"
            b = pixel_box(g, st["pixel"])
            line = txt(f"pixel {st['pixel']} = {st['value']}  →  {ans}", 22,
                       HI if st["yes"] else GREY_A).move_to([-4.9, -1.55 - 0.45 * k, 0])
            path += ans[0]
            self.play(Create(b), FadeIn(line), nodes[path[:-1]][0].animate.set_stroke(HI, 3),
                      edges[path].animate.set_stroke(HI, 4), run_time=0.6)
            self.play(FadeOut(b), run_time=0.2)
            lines.add(line)
        self.cue(18)
        guess = TWO["tree_small_guess"]
        v = txt(f"tree says {guess}  ✗", 26, BAD).move_to([-4.9, -3.05, 0])
        self.play(nodes[path][0].animate.set_stroke(BAD, 4), FadeIn(v), run_time=0.6)
        self.cue(20)
        leaf = R["tree"]["yes"]["yes"]["no"]
        info = txt(f"{leaf['n']} training digits land here, only {100 * leaf['purity']:.0f}% are {leaf['pred']}s",
                   20, BAD).move_to([2.3, -2.85, 0])
        arrow = Arrow(info.get_top() + RIGHT * 2.0, nodes[path].get_bottom(), buff=0.08, color=BAD, stroke_width=3)
        self.play(FadeIn(info), GrowArrow(arrow), run_time=0.8)
        self.finish(0.2)
        self.tree_parts = (g, nodes, edges, elabels, lines, v, info, arrow, cards)

    # ============================================================ 2. tree accuracy + why
    def tree_accuracy(self):
        g, nodes, edges, elabels, lines, v, info, arrow, cards = self.tree_parts
        small = VGroup(*nodes.values(), *edges.values(), elabels)
        self.seg(2)
        self.play(FadeOut(VGroup(g, lines, v, info, arrow, cards)), run_time=0.5)
        self.play(small.animate.move_to([0, 0.6, 0]), run_time=0.7)
        a1 = txt(f"3 questions deep, 8 leaves:  {pct(R['tree_depth3_acc'])} of test digits right", 26).move_to([0, -2.4, 0])
        self.play(FadeIn(a1), run_time=0.6)
        self.cue(1)
        big = full_tree_drawing()
        self.play(FadeOut(small), a1.animate.scale(0.8).move_to([0, -2.95, 0]), run_time=0.6)
        self.play(Create(big[0], lag_ratio=0.01), FadeIn(big[1], lag_ratio=0.01), run_time=2.5)
        a2 = txt(f"grown fully: {R['tree_full_depth']} questions deep, {R['tree_full_leaves']} leaves", 26).move_to([0, -3.5, 0])
        self.play(FadeIn(a2), run_time=0.6)
        self.cue(2)
        a2b = txt(f"→ {pct(R['tree_full_acc'])}", 30, GOOD).next_to(a2, RIGHT, buff=0.3)
        a2b.shift(LEFT * (a2b.width + 0.3) / 2)
        self.play(a2.animate.shift(LEFT * (a2b.width + 0.3) / 2), FadeIn(a2b, shift=LEFT * 0.2), run_time=0.6)
        self.cue(3)
        self.clear_all(0.6)

        self.cue(5)
        rp = R["root_pixel"]
        n1 = digit_grid(R["nine_dark"], cell=0.42, values=True).move_to([-3.3, 0.55, 0])
        n2 = digit_grid(R["nine_lit"], cell=0.42, values=True).move_to([3.3, 0.55, 0])
        head = txt("two nines from the training data", 26).move_to([0, 3.35, 0])
        nc = VGroup(card(4.3, 4.75, [-3.3, 0.27, 0]), card(4.3, 4.75, [3.3, 0.27, 0]))
        self.play(FadeIn(head), FadeIn(nc), FadeIn(n1), FadeIn(n2), run_time=1.0)
        self.cue(6)
        b1 = pixel_box(n1, rp)
        t1 = txt(f"pixel {rp} = {R['nine_dark'][rp]}  →  no", 24, GREY_A).next_to(n1, DOWN, buff=0.3)
        self.play(Create(b1), FadeIn(t1), run_time=0.7)
        self.cue(7)
        b2 = pixel_box(n2, rp)
        t2 = txt(f"pixel {rp} = {R['nine_lit'][rp]}  →  yes", 24, HI).next_to(n2, DOWN, buff=0.3)
        self.play(Create(b2), FadeIn(t2), run_time=0.7)
        self.cue(8)
        same = txt("same digit, opposite answers to question one", 24, BAD).move_to([0, -2.6, 0])
        self.play(FadeIn(same), run_time=0.7)
        self.cue(9)
        a, b = np.array(R["nine_dark"]), np.array(R["nine_lit"])
        diff = [p for p in range(64) if (a[p] >= 8) != (b[p] >= 8)]
        boxes = VGroup(*[SurroundingRectangle(n[0][p], color=ORANGE, buff=0.02, stroke_width=3, corner_radius=0.05)
                         for p in diff for n in (n1, n2)])
        dl = txt(f"{len(diff)} squares are inked in one nine but not the other", 20, ORANGE).move_to([0, -3.13, 0])
        self.play(FadeOut(b1), FadeOut(b2), Create(boxes, lag_ratio=0.02), FadeIn(dl), run_time=1.5)
        self.cue(10)
        moral = txt("→ many special cases, one pixel at a time", 22).move_to([0, -3.62, 0])
        self.play(FadeIn(moral), run_time=0.6)
        self.finish(0.2)
        self.clear_all()

    # ============================================================ 3. one hidden neuron
    def hidden_neuron(self):
        pix = np.array(ZERO["pixels"])
        w = W1[BOTTOM]
        self.seg(3)
        g = digit_grid(pix, cell=0.325).move_to([-5.0, 0.6, 0])
        head = txt("a hidden neuron", 28).move_to([0, 3.35, 0])
        self.play(FadeIn(card(13.8, 4.6, [0, 0.35, 0])), FadeIn(g), FadeIn(head), run_time=0.8)
        self.cue(1)
        neuron = Circle(0.45, color=WHITE, stroke_width=3).move_to([-1.4, 0.6, 0])
        wires = VGroup(*[Line(s.get_center(), neuron.get_left(), stroke_width=1, color=GREY_C, stroke_opacity=0.6)
                         for s in g[0]])
        self.play(Create(wires, lag_ratio=0.01), Create(neuron), run_time=1.5)
        all64 = txt("looks at all 64 pixels, one weight each", 22, DIM).next_to(neuron, DOWN, buff=0.9)
        self.play(FadeIn(all64), run_time=0.5)
        self.cue(2)
        hw = heat(w, 2.6).move_to([-1.4, 0.6, 0])
        self.play(FadeOut(wires), FadeOut(all64), FadeOut(neuron), FadeIn(hw), run_time=1.0)
        wl = txt("its 64 weights", 20, DIM).next_to(hw, UP, buff=0.15)
        self.play(FadeIn(wl), run_time=0.4)
        self.cue(3)
        lp = VGroup(Square(0.25, stroke_width=0).set_fill(POS, 1), txt("ink here pushes it up", 20)).arrange(RIGHT, buff=0.15)
        ln = VGroup(Square(0.25, stroke_width=0).set_fill(NEG, 1), txt("ink here pushes it down", 20)).arrange(RIGHT, buff=0.15)
        lp.next_to(hw, DOWN, buff=0.3)
        ln.next_to(lp, DOWN, buff=0.15).align_to(lp, LEFT)
        self.play(FadeIn(lp), run_time=0.5)
        self.cue(4)
        self.play(FadeIn(ln), run_time=0.5)

        self.cue(5)
        prod = pix / 16 * w
        times = txt("×", 40).move_to([-3.2, 0.6, 0])
        eq = txt("=", 40).move_to([0.4, 0.6, 0])
        hp = heat(prod, 2.6).move_to([2.2, 0.6, 0])
        pl = txt("pixel × weight", 20, DIM).next_to(hp, UP, buff=0.15)
        self.play(FadeIn(times), FadeIn(eq), FadeIn(hp), FadeIn(pl), run_time=1.0)
        self.cue(6)
        z = float(prod.sum() + B1[BOTTOM])
        s1 = txt(f"sum {prod.sum():+.2f}", 24).move_to([5.3, 1.6, 0])
        s2 = txt(f"bias {B1[BOTTOM]:+.2f}", 24).next_to(s1, DOWN, buff=0.2).align_to(s1, RIGHT)
        s3 = txt(f"total {z:+.2f}", 24).next_to(s2, DOWN, buff=0.2).align_to(s1, RIGHT)
        self.play(TransformFromCopy(hp, s1), run_time=0.8)
        self.play(FadeIn(s2), run_time=0.4)
        self.play(FadeIn(s3), run_time=0.4)
        sq = txt("squash →", 24, DIM).next_to(s3, DOWN, buff=0.35).align_to(s1, LEFT)
        out = txt(f"{sig(z):.2f}", 40, POS).next_to(sq, DOWN, buff=0.2).align_to(sq, LEFT)
        self.play(FadeIn(sq), run_time=0.4)
        self.cue(7)
        self.play(FadeIn(out, scale=1.3), run_time=0.6)
        self.play(Circumscribe(out, color=POS), run_time=0.8)

        self.cue(8)
        self.clear_all(0.5)
        hs = [heat(W1[i], 1.1) for i in range(16)]
        for i, h in enumerate(hs):
            h.move_to([-4.73 + (i % 8) * 1.35, 1.2 - (i // 8) * 1.45, 0])
        h16 = txt(f"all {R['hidden']} hidden neurons, after training", 26).move_to([0, 2.9, 0])
        self.play(FadeIn(h16), LaggedStart(*[FadeIn(h) for h in hs], lag_ratio=0.05), run_time=1.5)
        self.cue(9)
        others = [hs[i] for i in range(16) if i not in FEATURED]
        self.play(*[FadeOut(h) for h in others], FadeOut(h16), *[hs[i].animate.scale(2.4 / 1.1).move_to([-4.95 + k * 3.3, 1.1, 0])
                                                                 for k, i in enumerate(FEATURED)], run_time=1.2)
        names = ["ink along\nthe bottom", "both sides,\nempty middle", "a diagonal\nstroke", "?"]
        labels = [txt(n, 22, POS if n != "?" else DIM, line_spacing=0.8).next_to(hs[i], DOWN, buff=0.25)
                  for n, i in zip(names, FEATURED)]
        for c, l in zip((10, 11, 12), labels):
            self.cue(c)
            self.play(FadeIn(l, shift=UP * 0.1), Indicate(hs[FEATURED[labels.index(l)]][1], color=POS), run_time=0.7)
        self.cue(13)
        self.play(FadeIn(labels[3]), run_time=0.5)
        self.cue(14)
        ask = txt("each one asks:  does the image contain my pattern?", 26).move_to([0, -2.45, 0])
        self.play(FadeIn(ask), run_time=0.7)
        self.cue(15)
        every = txt("all 64 pixels count a little: no single square decides", 22, DIM).move_to([0, -3.15, 0])
        self.play(FadeIn(every), run_time=0.7)
        self.finish(0.2)
        self.clear_all()

    # ============================================================ 4. bottom-stroke neuron on three digits
    def bottom_neuron(self):
        self.seg(4)
        hw = heat(W1[BOTTOM], 2.0).move_to([0, 2.45, 0])
        hl = txt("ink along the bottom", 22, POS).next_to(hw, RIGHT, buff=0.3)
        self.play(FadeIn(hw), FadeIn(hl), run_time=0.8)
        recs = [ZERO, EIGHT, FOUR]
        for k, (rec, c) in enumerate(zip(recs, (1, 2, 3))):
            x = -4.2 + 4.2 * k
            g = digit_grid(rec["pixels"], cell=0.3).move_to([x, -0.5, 0])
            a = rec["hidden"][BOTTOM]
            bg = RoundedRectangle(corner_radius=0.11, width=2.4, height=0.22, stroke_width=0).set_fill(BG, 1).move_to([x, -2.25, 0])
            bar = RoundedRectangle(corner_radius=0.11, width=max(2.4 * a, 0.22), height=0.22, stroke_width=0).set_fill(POS, 1)
            bar.align_to(bg, LEFT).set_y(-2.25)
            val = txt(f"{a:.2f}", 34, POS if a > 0.5 else GREY_A).move_to([x, -2.95, 0])
            self.cue(c)
            self.play(FadeIn(card(3.3, 4.6, [x, -1.45, 0])), FadeIn(g), run_time=0.5)
            self.play(FadeIn(bg), GrowFromEdge(bar, LEFT), FadeIn(val), run_time=0.8)
        self.finish(0.3)
        self.clear_all()

    # ============================================================ 5. output layer + a digit flowing through
    def hidden_bars(self, acts):
        bars, vals = VGroup(), VGroup()
        for i, a in enumerate(acts):
            y = self.hy[i]
            bars.add(Rectangle(width=max(1.5 * a, 0.01), height=0.26, stroke_width=0).set_fill(POS, 1)
                     .move_to([-1.95, y, 0], aligned_edge=LEFT))
            vals.add(txt(f"{a:.2f}", 14, GREY_A).move_to([-0.15, y, 0]))
        return VGroup(bars, vals)

    def out_bars(self, probs, hi=()):
        bars, vals = VGroup(), VGroup()
        for j, p in enumerate(probs):
            y = self.oy[j]
            c = GOOD if j in hi else GREY_B
            bars.add(Rectangle(width=max(2.6 * p, 0.01), height=0.4, stroke_width=0).set_fill(c, 1)
                     .move_to([3.15, y, 0], aligned_edge=LEFT))
            vals.add(txt(f"{100 * p:.0f}%", 18, c if p >= 0.05 else GREY_C)
                     .next_to(bars[-1], RIGHT, buff=0.1))
        return VGroup(bars, vals)

    def output_layer(self):
        self.hy = [3.1 - i * 0.43 for i in range(16)]
        self.oy = [2.8 - j * 0.64 for j in range(10)]
        self.seg(5)
        icons = VGroup(*[heat(W1[i], 0.36).move_to([-2.3, self.hy[i], 0]) for i in range(16)])
        hh = txt("16 hidden", 20, DIM).move_to([-1.5, 3.62, 0])
        self.play(FadeIn(icons), FadeIn(hh), run_time=0.8)
        self.cue(1)
        olabels = VGroup(*[txt(str(j), 26).move_to([2.8, self.oy[j], 0]) for j in range(10)])
        oh = txt("10 outputs", 20, DIM).move_to([4.0, 3.62, 0])
        self.play(LaggedStart(*[FadeIn(l) for l in olabels], lag_ratio=0.08), FadeIn(oh), run_time=1.0)
        self.cue(2)
        mx = np.abs(W2).max()
        wires = [[Line([0.25, self.hy[i], 0], [2.55, self.oy[j], 0], stroke_width=1.5,
                       color=POS if W2[i, j] >= 0 else NEG,
                       stroke_opacity=0.1 + 0.4 * abs(W2[i, j]) / mx) for j in range(10)] for i in range(16)]
        allw = VGroup(*[w for row in wires for w in row])
        self.play(Create(allw, lag_ratio=0.003), run_time=1.6)
        self.cue(3)
        hl = SurroundingRectangle(icons[BOTTOM], color=POS, buff=0.04)
        others = VGroup(*[wires[i][j] for i in range(16) for j in range(10) if i != BOTTOM])
        mine = VGroup(*wires[BOTTOM])
        self.play(Create(hl), others.animate.set_stroke(opacity=0.04), mine.animate.set_stroke(width=3, opacity=0.9),
                  run_time=0.8)
        self.cue(4)

        def wlabel(j):
            v = W2[BOTTOM, j]
            t = txt(f"weight {v:+.1f}", 22, POS if v >= 0 else NEG).next_to(olabels[j], RIGHT, buff=0.3)
            return VGroup(BackgroundRectangle(t, fill_opacity=0.85, buff=0.06), t)
        l2 = wlabel(2)
        self.play(FadeIn(l2), Indicate(olabels[2], color=POS), wires[BOTTOM][2].animate.set_stroke(width=6), run_time=0.8)
        self.cue(5)
        l4, l7 = wlabel(4), wlabel(7)
        self.play(FadeIn(l4), FadeIn(l7), Indicate(olabels[4], color=NEG), Indicate(olabels[7], color=NEG),
                  wires[BOTTOM][4].animate.set_stroke(width=6), wires[BOTTOM][7].animate.set_stroke(width=6), run_time=0.9)
        self.cue(7)
        self.play(FadeOut(VGroup(l2, l4, l7)), FadeOut(hl), others.animate.set_stroke(opacity=0.12),
                  mine.animate.set_stroke(width=1.5, opacity=0.4), run_time=0.7)
        soft = txt("scores → chances that add up to 100%", 16, DIM).move_to([4.1, -3.65, 0])
        self.play(FadeIn(soft), run_time=0.5)

        self.cue(8)
        g = digit_grid(FOUR["pixels"], cell=0.3).move_to([-5.6, 0, 0])
        ih = txt("64 pixels", 20, DIM).move_to([-5.6, 3.62, 0])
        inw = VGroup(*[Line([-4.4, 0, 0], [-2.55, self.hy[i], 0], stroke_width=1, color=GREY_C, stroke_opacity=0.5)
                       for i in range(16)])
        self.play(FadeIn(g), FadeIn(ih), Create(inw), run_time=1.0)
        self.cue(9)
        self.play(Indicate(g, color=WHITE, scale_factor=1.04),
                  ShowPassingFlash(inw.copy().set_stroke(HI, 3, 1), time_width=0.5), run_time=1.0)
        self.cue(10)
        hb = self.hidden_bars(np.zeros(16))
        self.add(hb)
        hb_new = self.hidden_bars(FOUR["hidden"])
        self.play(Transform(hb, hb_new), run_time=1.2)
        self.cue(11)
        ob = self.out_bars(np.zeros(10))
        self.add(ob)
        self.play(ShowPassingFlash(allw.copy().set_stroke(HI, 2, 1), time_width=0.4),
                  Transform(ob, self.out_bars(FOUR["net_probs"], hi=(4,))), run_time=1.3)
        self.play(Circumscribe(VGroup(olabels[4], ob[1][4]), color=GOOD), run_time=0.8)

        self.cue(12)
        tl = txt(f"the {TWO['label']} the small\ntree called an {TWO['tree_small_guess']}", 16, DIM, line_spacing=0.8).next_to(g, DOWN, buff=0.3)
        self.play(*grid_to(g, TWO["pixels"], values=False), FadeIn(tl), run_time=0.9)
        self.cue(13)
        self.play(Transform(hb, self.hidden_bars(TWO["hidden"])), run_time=0.9)
        self.play(Transform(ob, self.out_bars(TWO["net_probs"], hi=(2,))), run_time=1.0)
        self.play(Indicate(ob[1][2], color=GOOD), Indicate(ob[1][3], color=WHITE), run_time=0.8)
        self.finish(0.3)
        self.clear_all()

    # ============================================================ 6. training
    def training(self):
        self.seg(6)
        k = ValueTracker(0)
        hs = [heat(SNAPS[0]["W1"][i], 1.25) for i in range(16)]
        for i, h in enumerate(hs):
            h.move_to([-5.3 + (i % 4) * 1.47, 1.85 - (i // 4) * 1.47, 0])
            h[0].add_updater(lambda m, i=i: set_heat(m, weights_at(k.get_value())[i]))
        head = txt("16 hidden neurons", 22, DIM).move_to([-3.6, 3.3, 0])
        self.cue(2)
        self.play(FadeIn(card(6.2, 6.4, [-3.1, -0.35, 0])), run_time=0.3)
        self.play(FadeIn(head), LaggedStart(*[FadeIn(h) for h in hs], lag_ratio=0.04), run_time=1.2)
        self.cue(3)
        noise = txt("random start", 22, NEG).next_to(head, RIGHT, buff=0.3)
        self.play(FadeIn(noise), run_time=0.4)

        self.cue(4)
        ax = Axes(x_range=[0, 60, 10], y_range=[0, 1, 0.25], x_length=5.0, y_length=4.0,
                  axis_config={"color": DIM, "include_tip": False, "stroke_width": 2}).move_to([3.95, -0.55, 0])
        ylab = VGroup(*[txt(f"{int(v * 100)}%", 16, DIM).next_to(ax.c2p(0, v), LEFT, buff=0.12) for v in (0, 0.5, 1)])
        xlab = VGroup(*[txt(str(e), 16, DIM).next_to(ax.c2p(e, 0), DOWN, buff=0.12) for e in (0, 20, 40, 60)])
        xt = txt("epochs", 18, DIM).next_to(ax, DOWN, buff=0.45)
        yt = txt("test accuracy", 18, DIM).next_to(ax.c2p(0, 1), UP, buff=0.2).shift(RIGHT * 0.6)
        guide = DashedLine(ax.c2p(0, R["tree_full_acc"]), ax.c2p(60, R["tree_full_acc"]), color=GOOD, stroke_width=2, stroke_opacity=0.7)

        def line():
            e = epoch_at(k.get_value())
            n = int(np.floor(e))
            pts = [ax.c2p(i, CURVE[i]) for i in range(n + 1)]
            if e > n and n + 1 < len(CURVE):
                f = e - n
                pts.append(ax.c2p(e, (1 - f) * CURVE[n] + f * CURVE[n + 1]))
            m = VMobject(color=POS, stroke_width=4)
            if len(pts) > 1:
                m.set_points_as_corners(pts)
            else:
                m.set_points_as_corners([pts[0], pts[0] + RIGHT * 1e-3])
            return m
        curve = always_redraw(line)
        dot = always_redraw(lambda: Dot(curve.get_end(), color=POS, radius=0.07))
        read = always_redraw(lambda: txt(
            f"epoch {int(round(epoch_at(k.get_value())))}   ·   {pct(CURVE[int(round(epoch_at(k.get_value())))])}",
            26, WHITE).move_to([3.6, 2.4, 0]))
        self.play(FadeIn(card(6.5, 6.4, [3.6, -0.35, 0])), Create(ax), FadeIn(ylab), FadeIn(xlab), FadeIn(xt), FadeIn(yt), run_time=1.0)
        self.add(curve, dot, read)
        self.play(Flash(dot, color=POS), run_time=0.6)
        self.cue(5)
        self.play(FadeOut(noise), run_time=0.4)

        steps = [(7, 1), (8, 4), (9, 6), (10, 8), (11, 9), (12, 10)]   # (cue, snapshot index)
        for c, target in steps:
            self.cue(c)
            self.play(k.animate.set_value(target), run_time=min(2.2, self.until(c + 1) - 0.1) if c < 12 else 2.2,
                      rate_func=linear)
        tl = txt(f"full tree {pct(R['tree_full_acc'])}", 16, GOOD).next_to(ax.c2p(38, R["tree_full_acc"]), DOWN, buff=0.1)
        fin = SurroundingRectangle(read, color=POS, buff=0.12)
        self.play(Create(fin), Create(guide), FadeIn(tl), run_time=0.8)
        for h in hs:
            h[0].clear_updaters()
        self.finish(0.3)
        self.remove(curve, dot, read)
        self.clear_all()

    # ============================================================ 7. scoreboard
    def scoreboard(self):
        self.seg(7)
        head = txt(f"on {R['n_test']} test digits neither model has seen", 26).move_to([0, 3.35, 0])
        self.play(FadeIn(head), FadeIn(card(13.8, 2.1, [0, 1.95, 0])), run_time=0.6)
        rows = []
        for k, (name, acc, c, cue) in enumerate((("decision tree", R["tree_full_acc"], GREY_B, 1),
                                                ("neural network", R["net_acc"], POS, 2))):
            y = 2.3 - k * 0.9
            lab = txt(name, 23).move_to([-6.45, y, 0], aligned_edge=LEFT)
            bar = RoundedRectangle(corner_radius=0.12, width=6.2 * acc, height=0.5, stroke_width=0).set_fill(c, 1) \
                .move_to([-3.35, y, 0], aligned_edge=LEFT)
            val = txt(pct(acc), 28, c).next_to(bar, RIGHT, buff=0.15)
            self.cue(cue)
            self.play(FadeIn(lab), GrowFromEdge(bar, LEFT), FadeIn(val), run_time=0.9)
            rows.append((bar, val))
        for k, (key, cue) in enumerate((("tree", 3), ("net", 4))):
            wrong = round(R["n_test"] * (1 - (R["tree_full_acc"] if key == "tree" else R["net_acc"])))
            t = txt(f"{wrong} wrong", 24, BAD).move_to([5.55, 2.3 - k * 0.9, 0])
            self.cue(cue)
            self.play(FadeIn(t), run_time=0.5)

        self.cue(5)
        ex = R["tree_misses_net_gets"]
        picks = [ex[0], ex[1], ex[3]]
        cards = VGroup()
        for k, (rec, c) in enumerate(zip(picks, (6, 7, 8))):
            x = -4.2 + 4.2 * k
            g = digit_grid(rec["pixels"], cell=0.27).move_to([x, -1.0, 0])
            t1 = txt(f"tree: {rec['tree_full_guess']}  ✗", 20, BAD).move_to([x, -2.4, 0])
            t2 = txt(f"network: {rec['net_guess']}  ✓  ({100 * max(rec['net_probs']):.0f}%)", 20, GOOD).move_to([x, -2.9, 0])
            cd = card(3.9, 3.85, [x, -1.55, 0])
            if k == 0:
                sub = txt("tree wrong, network right", 20, DIM).move_to([0, 0.62, 0])
                self.play(FadeIn(sub), run_time=0.5)
                cards.add(sub)
            self.cue(c)
            self.play(FadeIn(cd), FadeIn(g), FadeIn(t1), FadeIn(t2), run_time=0.7)
            cards.add(cd, g, t1, t2)

        self.cue(9)
        self.play(FadeOut(cards), run_time=0.5)
        self.cue(10)
        tally = VGroup()
        for k, (label, n, c) in enumerate((("fooled only the network", R["n_net_misses_tree_gets"], POS),
                                           ("fooled only the tree", R["n_tree_misses_net_gets"], GREY_B))):
            y = -0.2 - k * 1.5
            lab = txt(f"{label}: {n}", 24, c).move_to([-6.4, y + 0.4, 0], aligned_edge=LEFT)
            sqs = VGroup(*[RoundedRectangle(corner_radius=0.04, width=0.2, height=0.2, stroke_width=0).set_fill(c, 1)
                           for _ in range(n)]).arrange_in_grid(
                cols=24, buff=0.07).move_to([-6.4, y - 0.25, 0], aligned_edge=UL)
            tally.add(VGroup(lab, sqs))
        self.play(FadeIn(tally[0][0]), LaggedStart(*[FadeIn(s) for s in tally[0][1]], lag_ratio=0.05), run_time=1.0)
        self.cue(11)
        self.play(FadeIn(tally[1][0]), LaggedStart(*[FadeIn(s) for s in tally[1][1]], lag_ratio=0.03), run_time=1.5)
        self.finish(0.3)
        self.clear_all()

    # ============================================================ 8. a fair close
    def fair_close(self):
        self.seg(8)
        q = txt("so are networks just better?", 30).move_to([0, 0.3, 0])
        self.play(FadeIn(q), run_time=0.6)
        self.cue(1)
        na = txt("not always", 30, HI).next_to(q, DOWN, buff=0.3)
        self.play(FadeIn(na), run_time=0.5)
        self.cue(2)
        self.play(FadeOut(q), FadeOut(na), run_time=0.4)
        lh = txt("spreadsheet data", 26).move_to([-3.6, 3.35, 0])
        self.play(FadeIn(card(6.7, 5.8, [-3.6, -0.1, 0])), run_time=0.3)
        cols = ["hours studied", "age", "price"]
        table = VGroup()
        for c, name in enumerate(cols):
            x = -5.6 + c * 2.0
            cell = VGroup(RoundedRectangle(corner_radius=0.08, width=1.9, height=0.5, stroke_width=0).set_fill("#2D3240", 1),
                          txt(name, 16)).move_to([x, 2.3, 0])
            table.add(VGroup(cell, *[RoundedRectangle(corner_radius=0.06, width=1.9, height=0.36, stroke_width=0)
                                     .set_fill(BG, 1).move_to([x, 1.85 - r * 0.42, 0]) for r in range(5)]))
        self.play(FadeIn(lh), FadeIn(table), run_time=0.8)
        mean = txt("each column means something", 20, DIM).move_to([-3.6, -0.6, 0])
        self.play(FadeIn(mean), run_time=0.5)
        self.cue(3)
        self.play(Indicate(table[0][0], color=HI), Indicate(table[1][0], color=HI), Indicate(table[2][0], color=HI),
                  run_time=1.2)
        self.cue(4)
        hl = SurroundingRectangle(table[0], color=HI, buff=0.04)
        qq = txt("hours studied > cutoff?", 22, HI).move_to([-3.6, -1.3, 0])
        self.play(Create(hl), FadeIn(qq), run_time=0.8)
        self.cue(5)
        tv = txt("trees: often as good, or better", 20, GOOD).move_to([-3.6, -2.1, 0])
        self.play(FadeIn(tv), run_time=0.7)

        self.cue(6)
        rh = txt("pixels", 26).move_to([3.6, 3.35, 0])
        self.play(FadeIn(card(6.7, 5.8, [3.6, -0.1, 0])), run_time=0.3)
        g = digit_grid(ZERO["pixels"], cell=0.28).move_to([2.2, 1.3, 0])
        b = pixel_box(g, R["root_pixel"])
        one = txt(f"pixel {R['root_pixel']} alone: ?", 20, DIM).next_to(g, DOWN, buff=0.2)
        self.play(FadeIn(rh), FadeIn(g), Create(b), FadeIn(one), run_time=0.8)
        self.cue(7)
        hs = VGroup(*[heat(W1[i], 0.75) for i in FEATURED[:3]]).arrange(DOWN, buff=0.18).move_to([5.3, 1.3, 0])
        self.play(LaggedStart(*[FadeIn(h, shift=LEFT * 0.2) for h in hs], lag_ratio=0.3), run_time=1.0)
        self.cue(8)
        comb = txt("meaning lives in combinations", 20, POS).move_to([3.6, -0.75, 0])
        self.play(FadeIn(comb), run_time=0.6)
        self.cue(10)
        mini = digit_grid(FOUR["pixels"], cell=0.1).move_to([1.5, -1.6, 0])
        wave = FunctionGraph(lambda x: 0.3 * np.sin(9 * x) * np.exp(-((x) ** 2) * 2.0), x_range=[-0.8, 0.8],
                             color=POS, stroke_width=3).move_to([3.3, -1.6, 0])
        word = txt("the cat sat", 16, font="Menlo", weight=NORMAL).move_to([5.4, -1.6, 0])
        labs = VGroup(*[txt(s, 18, DIM).move_to([x, -3.1, 0]) for s, x in (("pixels", 1.5), ("sound", 3.3), ("text", 5.4))]).set_y(-2.15)
        win = txt("networks shine on raw inputs", 22, GOOD).move_to([3.6, -2.65, 0])
        self.play(LaggedStart(FadeIn(mini), Create(wave), FadeIn(word), lag_ratio=0.3), FadeIn(labs), run_time=1.4)
        self.play(FadeIn(win), run_time=0.6)
        self.finish(1.2)
        self.clear_all(1.0)
