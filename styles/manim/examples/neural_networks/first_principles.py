"""Neural networks from first principles — narrated Manim explainer.
Run train.py's save() and narration.py first, then:  manim -qm first_principles.py FirstPrinciples"""
import json
from pathlib import Path
import numpy as np
from manim import *

HERE = Path(__file__).parent
T = json.loads((HERE / "audio" / "timing.json").read_text())
RES = json.loads((HERE / "results.json").read_text())

NAMES = RES["names"]
DATA = np.array(RES["data"])
XS, YS = DATA[:, :2], DATA[:, 2]
SNAPS = RES["snaps"]
LOSSES = RES["losses"]
EVERY = RES["snap_every"]

PASS_C, FAIL_C = GREEN_C, RED_C
Q1_C, Q2_C = "#58C4DD", PURPLE_B          # study question, sleep question
POS_W, NEG_W = GREY_A, ORANGE              # wire colours by weight sign
HI, DIM = YELLOW, GREY_B

# trained hidden index -> diagram slot (top = study-leaning, bottom = sleep-leaning)
SLOT = [1, 0]


def sig(z):
    return 1 / (1 + np.exp(-np.clip(z, -60, 60)))


def txt(s, size=24, color=WHITE, **kw):
    return Text(s, font_size=size, color=color, **kw)


# ---------- the hand-built network
def hand_hidden(x, y):
    return sig(3 * x - 13.5), sig(3 * y - 13.5)


def hand_out(x, y):
    a, b = hand_hidden(x, y)
    return sig(10 * a + 10 * b - 15)


# ---------- trained network at fractional snapshot index
def params_at(k):
    k = float(np.clip(k, 0, len(SNAPS) - 1))
    i0 = int(np.floor(k)); i1 = min(i0 + 1, len(SNAPS) - 1); f = k - i0
    return {key: (1 - f) * np.array(SNAPS[i0][key]) + f * np.array(SNAPS[i1][key]) for key in SNAPS[0]}


def net_forward(P, x, y):
    X = np.stack([np.asarray(x, float), np.asarray(y, float)], -1)
    h = sig(X @ P["W1"] + P["b1"])
    return h, sig(h @ P["W2"] + P["b2"])


def net_error(P):
    _, out = net_forward(P, XS[:, 0], XS[:, 1])
    return float(((out - YS) ** 2).sum())


def epoch_to_k(e):
    return e / EVERY


class FirstPrinciples(Scene):
    # ---------- narration sync
    def seg(self, i):
        self.add_sound(str(HERE / "audio" / f"seg{i}.wav"))
        self._i, self._t0 = i, self.renderer.time

    def cue(self, k):
        left = self._t0 + T[self._i]["cues"][k] - self.renderer.time
        if left > 0.04:
            self.wait(left)

    def until(self, k):
        """seconds from now until cue k"""
        return max(0.3, self._t0 + T[self._i]["cues"][k] - self.renderer.time)

    def finish(self, pad=0.35):
        left = self._t0 + T[self._i]["dur"] + pad - self.renderer.time
        if left > 0.04:
            self.wait(left)

    # ---------- chart helpers
    def make_chart(self):
        ax = Axes(x_range=[0, 10, 1], y_range=[0, 10, 1], x_length=5.6, y_length=5.6,
                  axis_config={"color": DIM, "include_tip": False, "stroke_width": 2}).move_to([-3.75, -0.35, 0])
        nums = VGroup(*[txt(str(v), 16, DIM).next_to(ax.c2p(v, 0), DOWN, buff=0.12) for v in (0, 5, 10)],
                      *[txt(str(v), 16, DIM).next_to(ax.c2p(0, v), LEFT, buff=0.12) for v in (5, 10)])
        xl = txt("hours studied →", 22, DIM).next_to(ax, DOWN, buff=0.42)
        yl = txt("↑ hours slept", 22, DIM).next_to(ax.c2p(0, 10), UP, buff=0.18).shift(RIGHT * 0.55)
        self.ax = ax
        return ax, nums, xl, yl

    def region(self, fn, mode="passfail", color=None, alpha=0.32, res=140):
        """Shade the chart: fn(x, y) -> value in [0,1]."""
        g = (np.arange(res) + 0.5) / res * 10
        gx, gy = np.meshgrid(g, g[::-1])
        p = np.clip(fn(gx, gy), 0, 1)
        img = np.zeros((res, res, 4))
        if mode == "passfail":
            c0, c1 = np.array(color_to_rgb(FAIL_C)), np.array(color_to_rgb(PASS_C))
            img[..., :3] = (1 - p)[..., None] * c0 + p[..., None] * c1
            img[..., 3] = alpha
        else:
            img[..., :3] = np.array(color_to_rgb(color))
            img[..., 3] = alpha * p
        im = ImageMobject((img * 255).astype(np.uint8))
        im.set_resampling_algorithm(RESAMPLING_ALGORITHMS["linear"])
        ax = self.ax
        w = ax.c2p(10, 0)[0] - ax.c2p(0, 0)[0]
        h = ax.c2p(0, 10)[1] - ax.c2p(0, 0)[1]
        im.stretch_to_fit_width(w).stretch_to_fit_height(h).move_to(ax.c2p(5, 5))
        im.set_z_index(-2)
        return im

    def bound_line(self, a, b, c, color, width=4):
        """Line a*x + b*y + c = 0 clipped to the 0..10 box."""
        pts = []
        if abs(b) > 1e-9:
            for x in (0, 10):
                y = -(a * x + c) / b
                if 0 <= y <= 10:
                    pts.append((x, y))
        if abs(a) > 1e-9:
            for y in (0, 10):
                x = -(b * y + c) / a
                if 0 <= x <= 10:
                    pts.append((x, y))
        uniq = []
        for p in pts:
            if all(np.hypot(p[0] - q[0], p[1] - q[1]) > 1e-6 for q in uniq):
                uniq.append(p)
        if len(uniq) < 2:
            return Line(self.ax.c2p(0, 0), self.ax.c2p(0.001, 0), stroke_opacity=0)
        return Line(self.ax.c2p(*uniq[0]), self.ax.c2p(*uniq[1]), color=color, stroke_width=width).set_z_index(1)

    # ---------- network diagram (right panel)
    def make_net(self, xs=(1.45, 3.85, 6.2), ys=1.35, r=0.42):
        inp = VGroup(*[Circle(r, color=WHITE, stroke_width=3).set_fill(BLACK, 1).move_to([xs[0], y, 0]) for y in (ys, -ys)])
        hid = VGroup(*[Circle(r, color=WHITE, stroke_width=3).set_fill(BLACK, 1).move_to([xs[1], y, 0]) for y in (ys, -ys)])
        out = Circle(r, color=WHITE, stroke_width=3).set_fill(BLACK, 1).move_to([xs[2], 0, 0])

        def wire(a, b):
            d = normalize(b.get_center() - a.get_center())
            return Line(a.get_center() + d * r, b.get_center() - d * r, stroke_width=3, color=POS_W)
        w1 = [[wire(inp[i], hid[j]) for j in range(2)] for i in range(2)]  # w1[input][hidden]
        w2 = [wire(hid[j], out) for j in range(2)]
        return inp, hid, out, w1, w2

    def style_wire(self, ln, w):
        return ln.set_stroke(POS_W if w >= 0 else NEG_W, width=float(np.clip(1 + 1.3 * abs(w), 1, 9)),
                             opacity=float(np.clip(0.25 + 0.4 * abs(w), 0.25, 1)))

    # =====================================================================
    def construct(self):
        ax, nums, xl, yl = self.make_chart()

        # ================= 0. the problem
        self.seg(0)
        colx = [1.3, 2.9, 4.4, 5.9]
        heads = VGroup(*[txt(s, 22, DIM).move_to([x, 2.95, 0]) for s, x in zip(["student", "studied", "slept", "passed"], colx)])
        rule = Line([0.6, 2.68, 0], [6.6, 2.68, 0], color=DIM, stroke_width=2)
        rows = VGroup()
        for i, (n, (st, sl, ps)) in enumerate(zip(NAMES, DATA)):
            y = 2.35 - i * 0.5
            rows.add(VGroup(txt(n, 22).move_to([colx[0], y, 0]), txt(f"{int(st)} h", 22).move_to([colx[1], y, 0]),
                            txt(f"{int(sl)} h", 22).move_to([colx[2], y, 0]),
                            txt("pass" if ps else "fail", 22, PASS_C if ps else FAIL_C).move_to([colx[3], y, 0])))
        self.play(FadeIn(heads[0]), Create(rule), run_time=0.8)
        self.cue(2)
        self.play(FadeIn(heads[1], shift=DOWN * 0.15), run_time=0.6)
        self.play(FadeIn(heads[2], shift=DOWN * 0.15), run_time=0.6)
        self.cue(3)
        self.play(FadeIn(heads[3]), LaggedStart(*[FadeIn(r, shift=RIGHT * 0.15) for r in rows], lag_ratio=0.12), run_time=2.2)
        self.cue(4)
        self.play(Create(ax), FadeIn(nums), run_time=1.2)
        self.cue(5)
        self.play(FadeIn(xl, shift=RIGHT * 0.2), run_time=0.6)
        self.cue(6)
        self.play(FadeIn(yl, shift=UP * 0.2), run_time=0.6)
        dots, names = VGroup(), VGroup()
        for i, (x, y) in enumerate(XS):
            d = Dot(ax.c2p(x, y), radius=0.1, color=PASS_C if YS[i] else FAIL_C).set_z_index(3)
            dots.add(d)
            names.add(txt(NAMES[i], 16, GREY_A).next_to(d, UP, buff=0.08).set_z_index(3))
        self.cue(7)
        ps = [i for i in range(10) if YS[i] == 1]
        fs = [i for i in range(10) if YS[i] == 0]
        self.play(*[TransformFromCopy(rows[i][3], dots[i]) for i in ps], *[FadeIn(names[i]) for i in ps], run_time=1.0)
        self.cue(8)
        self.play(*[TransformFromCopy(rows[i][3], dots[i]) for i in fs], *[FadeIn(names[i]) for i in fs], run_time=1.0)
        self.finish(0.2)

        # ================= 1. a weighted score
        self.seg(1)
        self.play(FadeOut(VGroup(heads, rule, rows)), run_time=0.6)
        parts = [txt("score =", 22), txt("w₁", 22, Q1_C), txt("× studied", 22), txt("+", 22), txt("w₂", 22, Q2_C),
                 txt("× slept", 22), txt("−", 22), txt("bias", 22, HI)]
        f = VGroup(*parts).arrange(RIGHT, buff=0.14).move_to([3.75, 2.6, 0])
        self.play(FadeIn(f[0]), run_time=0.5)
        self.cue(1)
        self.play(FadeIn(f[1:3], shift=UP * 0.15), run_time=0.7)
        self.cue(2)
        self.play(FadeIn(f[3:6], shift=UP * 0.15), run_time=0.7)
        self.cue(3)
        self.play(FadeIn(f[6:8], shift=UP * 0.15), run_time=0.7)
        self.cue(4)
        wnote = txt("weights: how much each one matters", 20, GREY_A).move_to([3.75, 1.85, 0])
        self.play(Indicate(f[1], color=Q1_C), Indicate(f[4], color=Q2_C), FadeIn(wnote), run_time=1.0)
        self.cue(5)
        bnote = txt("bias: the bar the score must clear", 20, GREY_A).move_to([3.75, 1.4, 0])
        self.play(Indicate(f[7], color=HI), FadeIn(bnote), run_time=1.0)
        self.cue(6)
        rule_t = txt("score above 0  →  pass", 26).move_to([3.75, 0.5, 0])
        self.play(FadeIn(rule_t, shift=UP * 0.15), run_time=0.7)
        self.cue(7)
        vals = [txt("1", 22, Q1_C).move_to(f[1]), txt("1", 22, Q2_C).move_to(f[4]), txt("10.5", 22, HI).move_to(f[7], aligned_edge=LEFT)]
        self.play(Transform(f[1], vals[0]), Transform(f[4], vals[1]), Transform(f[7], vals[2]), run_time=1.0)
        ex = txt("Ana: 8 + 7 − 10.5 = 4.5  →  pass", 22, PASS_C).move_to([3.75, -0.4, 0])
        ex2 = txt("Eli: 3 + 3 − 10.5 = −4.5  →  fail", 22, FAIL_C).move_to([3.75, -0.9, 0])
        self.play(FadeIn(ex), Indicate(dots[0], scale_factor=1.6), run_time=1.0)
        self.play(FadeIn(ex2), Indicate(dots[4], scale_factor=1.6), run_time=1.0)
        self.cue(8)
        line1 = self.bound_line(1, 1, -10.5, WHITE)
        self.play(Create(line1), run_time=1.2)
        self.cue(9)
        hard = self.region(lambda x, y: sig(30 * (x + y - 10.5)))
        self.play(FadeIn(hard), run_time=1.2)
        self.finish(0.2)

        # ================= 2. squash -> neuron
        self.seg(2)
        self.play(FadeOut(VGroup(wnote, bnote, rule_t, ex, ex2)), f.animate.scale(0.85).move_to([3.75, 2.9, 0]), run_time=0.6)
        self.cue(1)
        allnone = txt("pass / fail is all or nothing", 22, GREY_A).move_to([3.75, 2.2, 0])
        self.play(FadeIn(allnone), run_time=0.6)
        self.cue(2)
        sax = Axes(x_range=[-8, 8, 4], y_range=[0, 1, 0.5], x_length=5.2, y_length=2.4,
                   axis_config={"color": DIM, "include_tip": False}).move_to([3.75, -0.1, 0])
        scurve = sax.plot(lambda z: sig(z), x_range=[-8, 8], color=HI, stroke_width=4)
        s_lbls = VGroup(txt("score", 18, DIM).next_to(sax, DOWN, buff=0.15),
                        txt("1", 18, DIM).next_to(sax.c2p(-8, 1), LEFT, buff=0.12),
                        txt("0", 18, DIM).next_to(sax.c2p(-8, 0), LEFT, buff=0.12),
                        txt("squash", 22, HI).next_to(sax.c2p(-8, 1), RIGHT, buff=0.2).shift(UP * 0.2))
        self.play(FadeOut(allnone), Create(sax), Create(scurve), FadeIn(s_lbls), run_time=1.4)
        probe = ValueTracker(0.0)
        pdot = always_redraw(lambda: Dot(sax.c2p(probe.get_value(), sig(probe.get_value())), color=WHITE, radius=0.09))
        ptxt = always_redraw(lambda: txt(f"{probe.get_value():+.0f} → {sig(probe.get_value()):.2f}", 22)
                             .next_to(sax.c2p(probe.get_value(), sig(probe.get_value())), UP, buff=0.2))
        self.cue(3)
        self.add(pdot, ptxt)
        self.play(probe.animate.set_value(6), run_time=1.2)
        self.cue(4)
        self.play(probe.animate.set_value(-6), run_time=1.4)
        self.cue(5)
        self.play(probe.animate.set_value(0), run_time=1.0)
        self.cue(6)
        soft = self.region(lambda x, y: sig(x + y - 10.5))
        chance = txt("shading = chance of passing", 20, GREY_A).next_to(ax, UP, buff=0.12).shift(RIGHT * 1.2)
        self.play(FadeOut(hard), FadeIn(soft), FadeIn(chance), run_time=1.2)
        self.cue(7)
        self.remove(pdot, ptxt)
        self.play(FadeOut(VGroup(sax, scurve, s_lbls)), FadeOut(f), run_time=0.5)
        n_in = VGroup(*[Circle(0.42, color=WHITE, stroke_width=3).set_fill(BLACK, 1).move_to([1.5, y, 0]) for y in (1.3, -1.3)])
        n_lbl = VGroup(txt("studied", 20, DIM).next_to(n_in[0], UP, buff=0.12), txt("slept", 20, DIM).next_to(n_in[1], DOWN, buff=0.12))
        neuron = Circle(0.75, color=WHITE, stroke_width=4).set_fill(BLACK, 1).move_to([4.4, 0, 0])
        n_w = VGroup(*[Line(c.get_center() + normalize(neuron.get_center() - c.get_center()) * 0.42,
                            neuron.get_center() - normalize(neuron.get_center() - c.get_center()) * 0.75,
                            stroke_width=4, color=POS_W) for c in n_in])
        n_wl = VGroup(txt("× 1", 20, Q1_C).move_to(n_w[0].get_center() + UP * 0.32 + RIGHT * 0.1),
                      txt("× 1", 20, Q2_C).move_to(n_w[1].get_center() + DOWN * 0.32 + RIGHT * 0.1))
        n_out = Arrow(neuron.get_right(), [6.6, 0, 0], buff=0.05, color=WHITE, stroke_width=4)
        n_out_l = txt("chance", 20, DIM).next_to(n_out, UP, buff=0.1)
        nlab = txt("a neuron", 28, HI).move_to([4.4, 2.6, 0])
        self.play(FadeIn(nlab), GrowFromCenter(neuron), FadeIn(n_in), FadeIn(n_lbl), Create(n_w), run_time=1.0)
        self.cue(8)
        ins = VGroup(txt("8", 22).move_to(n_in[0]), txt("7", 22).move_to(n_in[1]))
        self.play(FadeIn(ins), *[ShowPassingFlash(w.copy().set_stroke(HI, 8), time_width=0.6) for w in n_w], run_time=1.0)
        self.cue(9)
        self.play(FadeIn(n_wl), run_time=0.6)
        self.cue(10)
        sigma = txt("8 + 7 − 10.5", 15).move_to(neuron).shift(UP * 0.18)
        bias_l = txt("bias −10.5", 20, HI).next_to(neuron, DOWN, buff=0.2)
        self.play(FadeIn(sigma), FadeIn(bias_l), run_time=0.7)
        self.cue(11)
        icon_ax = Axes(x_range=[-4, 4], y_range=[0, 1], x_length=0.8, y_length=0.32, axis_config={"stroke_width": 0})
        icon = icon_ax.plot(lambda z: sig(2 * z), color=HI, stroke_width=3).move_to(neuron).shift(DOWN * 0.22)
        self.play(Create(icon), run_time=0.5)
        res_t = txt("0.99", 22, PASS_C).next_to(n_out, DOWN, buff=0.1)
        self.play(GrowArrow(n_out), FadeIn(n_out_l), FadeIn(res_t), run_time=0.8)
        self.finish(0.2)

        # ================= 3. one line is not enough
        self.seg(3)
        neuron_grp = VGroup(n_in, n_lbl, neuron, n_w, n_wl, n_out, n_out_l, nlab, ins, sigma, bias_l, icon, res_t)
        self.play(FadeOut(neuron_grp), FadeOut(chance), run_time=0.5)
        ring_d = Circle(0.22, color=HI, stroke_width=4).move_to(dots[3]).set_z_index(4)
        self.play(Create(ring_d), run_time=0.6)
        self.cue(1)
        dev_t = txt("Dev: studied 9 h, slept 3 h → failed", 24, FAIL_C).move_to([3.75, 2.4, 0])
        self.play(FadeIn(dev_t), run_time=0.7)
        self.cue(2)
        says = txt("line says: pass ✗", 24, HI).move_to([3.75, 1.85, 0])
        self.play(FadeIn(says), Indicate(dots[3], color=HI, scale_factor=1.8), run_time=1.0)
        self.cue(3)
        ring_j = Circle(0.22, color=HI, stroke_width=4).move_to(dots[9]).set_z_index(4)
        jo_t = txt("Jo: studied 3 h, slept 9 h → failed", 24, FAIL_C).move_to([3.75, 1.1, 0])
        self.play(Create(ring_j), FadeIn(jo_t), run_time=0.8)
        # live line: a*x + b*y + c = 0 with a=cos(t), b=sin(t), passing through point (px, py)
        th = ValueTracker(PI / 4)
        px = ValueTracker(5.25)

        def abc():
            a, b = np.cos(th.get_value()), np.sin(th.get_value())
            c = -(a * px.get_value() + b * 5.25)
            return a, b, c

        def wrong():
            a, b, c = abc()
            pred = (a * XS[:, 0] + b * XS[:, 1] + c) > 0
            return int((pred != (YS == 1)).sum())
        live_line = always_redraw(lambda: self.bound_line(*abc(), WHITE))
        live_reg = always_redraw(lambda: self.region(lambda x, y: sig(30 * (abc()[0] * x + abc()[1] * y + abc()[2])), res=90))
        counter = always_redraw(lambda: txt(f"students on the wrong side: {wrong()}", 24, HI).move_to([3.75, 0.1, 0]))
        self.cue(4)
        self.remove(line1, soft)
        self.add(live_reg, live_line)
        self.play(FadeIn(counter), run_time=0.5)
        self.cue(5)
        self.play(th.animate.set_value(PI / 4 + 0.55), run_time=1.0)
        self.play(th.animate.set_value(PI / 4 - 0.5), run_time=1.2)
        self.cue(6)
        self.play(px.animate.set_value(6.4), run_time=0.9)
        self.play(px.animate.set_value(4.3), th.animate.set_value(PI / 4 + 0.35), run_time=1.2)
        self.play(px.animate.set_value(5.25), th.animate.set_value(PI / 4), run_time=0.9)
        self.cue(8)
        one = txt("one neuron = one straight line", 26).move_to([3.75, -1.2, 0])
        self.play(FadeIn(one, shift=UP * 0.15), run_time=0.7)
        self.cue(9)
        oneq = txt("= one question", 26).move_to([3.75, -1.75, 0])
        self.play(FadeIn(oneq, shift=UP * 0.15), run_time=0.6)
        self.cue(10)
        needs = txt("this pattern needs two", 26, HI).move_to([3.75, -2.5, 0])
        self.play(FadeIn(needs, shift=UP * 0.15), run_time=0.6)
        self.finish(0.2)

        # ================= 4. two smaller questions
        self.seg(4)
        self.remove(live_reg, live_line, counter)
        self.play(FadeOut(VGroup(dev_t, says, jo_t, one, oneq, needs, ring_d, ring_j)), run_time=0.6)
        inp, hid, out, w1, w2 = self.make_net()
        in_lbl = VGroup(txt("studied", 20, DIM).next_to(inp[0], UP, buff=0.12), txt("slept", 20, DIM).next_to(inp[1], DOWN, buff=0.12))
        self.play(FadeIn(inp), FadeIn(in_lbl), run_time=0.8)
        self.cue(1)
        q1 = txt("studied enough?", 22, Q1_C).next_to(hid[0], UP, buff=0.15)
        self.play(GrowFromCenter(hid[0]), run_time=0.6)
        self.cue(2)
        self.play(FadeIn(q1, shift=DOWN * 0.1), run_time=0.6)
        self.cue(3)
        w1[0][0].set_stroke(Q1_C, 6)
        w1[1][0].set_stroke(DIM, 1.5, opacity=0.35)
        wl1 = VGroup(txt("× 3", 20, Q1_C).next_to(w1[0][0], UP, buff=0.08),
                     txt("× 0", 18, DIM).move_to(w1[1][0].point_from_proportion(0.78) + LEFT * 0.38))
        self.play(Create(w1[0][0]), Create(w1[1][0]), FadeIn(wl1), run_time=1.0)
        self.cue(4)
        l_study = self.bound_line(1, 0, -4.5, Q1_C)
        l_study_t = txt("4.5 h", 18, Q1_C).next_to(ax.c2p(4.5, 10), UP, buff=0.08)
        self.play(Create(l_study), FadeIn(l_study_t), run_time=1.0)
        self.cue(5)
        reg_s = self.region(lambda x, y: hand_hidden(x, y)[0], mode="tint", color=Q1_C, alpha=0.22)
        self.play(FadeIn(reg_s), run_time=0.8)
        self.cue(6)
        self.play(GrowFromCenter(hid[1]), run_time=0.6)
        self.cue(7)
        q2 = txt("slept enough?", 22, Q2_C).next_to(hid[1], DOWN, buff=0.15)
        self.play(FadeIn(q2, shift=UP * 0.1), run_time=0.6)
        self.cue(8)
        w1[1][1].set_stroke(Q2_C, 6)
        w1[0][1].set_stroke(DIM, 1.5, opacity=0.35)
        wl2 = VGroup(txt("× 3", 20, Q2_C).next_to(w1[1][1], DOWN, buff=0.08),
                     txt("× 0", 18, DIM).move_to(w1[0][1].point_from_proportion(0.78) + LEFT * 0.38))
        self.play(Create(w1[1][1]), Create(w1[0][1]), FadeIn(wl2), run_time=1.0)
        self.cue(9)
        l_sleep = self.bound_line(0, 1, -4.5, Q2_C)
        l_sleep_t = txt("4.5 h", 18, Q2_C).next_to(ax.c2p(10, 4.5), RIGHT, buff=0.08)
        self.play(Create(l_sleep), FadeIn(l_sleep_t), run_time=1.0)
        self.cue(10)
        reg_b = self.region(lambda x, y: hand_hidden(x, y)[1], mode="tint", color=Q2_C, alpha=0.22)
        self.play(FadeIn(reg_b), run_time=0.8)
        self.cue(11)
        tag_d = VGroup(txt("study ✓", 18, Q1_C), txt("sleep ✗", 18, Q2_C)).arrange(DOWN, buff=0.05, aligned_edge=LEFT) \
            .next_to(dots[3], RIGHT, buff=0.12).set_z_index(5)
        hv = VGroup(txt("yes", 20, Q1_C).move_to(hid[0]), txt("no", 20, Q2_C).move_to(hid[1]))
        self.play(FadeIn(tag_d), Indicate(dots[3], color=HI, scale_factor=1.8), FadeIn(hv), run_time=1.0)
        self.cue(12)
        tag_j = VGroup(txt("study ✗", 18, Q1_C), txt("sleep ✓", 18, Q2_C)).arrange(DOWN, buff=0.05, aligned_edge=LEFT) \
            .next_to(dots[9], RIGHT, buff=0.12).set_z_index(5)
        hv2 = VGroup(txt("no", 20, Q1_C).move_to(hid[0]), txt("yes", 20, Q2_C).move_to(hid[1]))
        self.play(FadeIn(tag_j), Indicate(dots[9], color=HI, scale_factor=1.8), Transform(hv, hv2), run_time=1.0)
        self.finish(0.2)

        # ================= 5. combine
        self.seg(5)
        self.play(FadeOut(hv), FadeOut(tag_d), FadeOut(tag_j), run_time=0.4)
        for w in w2:
            w.set_stroke(PASS_C, 6)
        self.play(GrowFromCenter(out), Create(w2[0]), Create(w2[1]), run_time=1.0)
        self.cue(1)
        q3 = txt("both yes?", 22, PASS_C).next_to(out, UP, buff=0.2).shift(RIGHT * 0.15)
        self.play(FadeIn(q3, shift=DOWN * 0.1), run_time=0.6)
        self.cue(2)
        wl3 = VGroup(txt("× 10", 20, PASS_C).move_to(w2[0].point_from_proportion(0.3) + UP * 0.3),
                     txt("× 10", 20, PASS_C).move_to(w2[1].point_from_proportion(0.3) + DOWN * 0.3))
        b3 = txt("bias −15", 20, HI).next_to(out, DOWN, buff=0.15)
        self.play(FadeIn(wl3), FadeIn(b3), run_time=0.8)
        self.cue(3)
        m1 = txt("yes + yes:  10 + 10 − 15 = +5  →  pass", 20, PASS_C).move_to([3.85, -2.85, 0])
        self.play(FadeIn(m1, shift=UP * 0.1), run_time=0.6)
        self.cue(4)
        m2 = txt("yes + no:   10 + 0 − 15 = −5  →  fail", 20, FAIL_C).move_to([3.85, -3.3, 0])
        self.play(FadeIn(m2, shift=UP * 0.1), run_time=0.6)
        self.cue(5)
        reg_o = self.region(lambda x, y: hand_out(x, y))
        self.play(FadeOut(reg_s), FadeOut(reg_b), FadeIn(reg_o),
                  l_study.animate.set_stroke(opacity=0.55), l_sleep.animate.set_stroke(opacity=0.55), run_time=1.2)
        self.cue(6)
        ok = txt("10 / 10 correct", 24, PASS_C).next_to(ax, UP, buff=0.12).shift(RIGHT * 1.6)
        self.play(FadeIn(ok), LaggedStart(*[Indicate(d, scale_factor=1.6, color=d.get_color()) for d in dots], lag_ratio=0.08), run_time=1.6)
        self.finish(0.2)

        # ================= 6. name the layers
        self.seg(6)
        self.play(FadeOut(VGroup(m1, m2, ok)), run_time=0.5)
        self.cue(1)
        cols = [SurroundingRectangle(VGroup(inp, in_lbl), color=HI, buff=0.15, corner_radius=0.1),
                SurroundingRectangle(VGroup(hid, q1, q2), color=HI, buff=0.12, corner_radius=0.1),
                SurroundingRectangle(VGroup(out, q3, b3), color=HI, buff=0.12, corner_radius=0.1)]
        self.play(*[Create(c) for c in cols], run_time=1.0)
        self.play(*[FadeOut(c) for c in cols], run_time=0.6)
        self.cue(2)
        lay_y = -3.2
        L_in = txt("input layer", 22, HI).move_to([1.45, lay_y, 0])
        self.play(FadeIn(L_in, shift=UP * 0.1), Indicate(inp, color=HI), run_time=0.9)
        self.cue(3)
        ivals = VGroup(txt("8", 22).move_to(inp[0]), txt("7", 22).move_to(inp[1]))
        self.play(FadeIn(ivals), run_time=0.6)
        self.cue(4)
        nocalc = txt("no math here", 18, DIM).next_to(L_in, UP, buff=0.12)
        self.play(FadeIn(nocalc), run_time=0.6)
        self.cue(6)
        L_out = txt("output layer", 22, HI).move_to([6.2, lay_y, 0])
        self.play(FadeOut(nocalc), FadeIn(L_out, shift=UP * 0.1), Indicate(out, color=HI), run_time=0.9)
        self.cue(7)
        oval = txt("0.99", 20, PASS_C).move_to(out)
        self.play(FadeIn(oval), run_time=0.5)
        self.cue(8)
        L_hid = txt("hidden layer", 22, HI).move_to([3.85, lay_y, 0])
        self.play(FadeIn(L_hid, shift=UP * 0.1), Indicate(hid, color=HI), run_time=0.9)
        self.cue(9)
        self.play(Indicate(q1, color=Q1_C), Indicate(q2, color=Q2_C), run_time=1.0)
        self.cue(10)
        # the data table has no such column
        mini_cols = VGroup(*[txt(s, 18, GREY_A) for s in ["studied", "slept", "passed"]]).arrange(RIGHT, buff=0.35)
        ghost = txt("studied enough?", 18, Q1_C)
        mini = VGroup(mini_cols, ghost).arrange(RIGHT, buff=0.45).move_to([-3.2, 3.4, 0])
        cross = Cross(ghost, stroke_color=FAIL_C, stroke_width=4)
        data_l = txt("the data:", 18, DIM).next_to(mini, LEFT, buff=0.2)
        self.play(FadeIn(mini_cols), FadeIn(data_l), FadeOut(yl), run_time=0.7)
        self.cue(11)
        self.play(FadeIn(ghost), run_time=0.4)
        self.play(Create(cross), run_time=0.5)
        self.cue(12)
        made = txt("made up in the middle", 20, HI).next_to(L_hid, UP, buff=0.12)
        self.play(FadeIn(made), Indicate(hid, color=HI), run_time=0.9)
        self.cue(13)
        small_net = VGroup(inp, hid, out, *[w for row in w1 for w in row], *w2, in_lbl, q1, q2, q3, wl1, wl2, wl3, b3,
                           ivals, oval, L_in, L_hid, L_out, made)
        chart_stuff = Group(ax, nums, xl, dots, names, reg_o, l_study, l_sleep, l_study_t, l_sleep_t, mini, cross, data_l)
        self.play(FadeOut(small_net), FadeOut(chart_stuff), run_time=0.8)
        # a bigger network
        layers_n = [9, 7, 7, 7, 3]
        bxs = np.linspace(-5.2, 5.2, len(layers_n))
        big_nodes = []
        for n, x in zip(layers_n, bxs):
            gap = 0.62 if n > 3 else 0.9
            big_nodes.append(VGroup(*[Circle(0.17, color=WHITE, stroke_width=2).set_fill(BLACK, 1)
                                      .move_to([x, (n - 1) / 2 * gap - i * gap, 0]) for i in range(n)]))
        big_edges = VGroup(*[Line(a.get_center(), b.get_center(), stroke_width=0.8, color=DIM, stroke_opacity=0.5)
                             for l in range(len(layers_n) - 1) for a in big_nodes[l] for b in big_nodes[l + 1]]).set_z_index(-1)
        self.play(Create(big_edges, lag_ratio=0.002), *[FadeIn(g) for g in big_nodes], run_time=1.4)
        self.cue(14)
        pix = VGroup(*[Square(0.22, stroke_width=1, color=DIM).set_fill(interpolate_color(BLACK, WHITE, v), 1)
                       for v in np.random.default_rng(2).random(36)]).arrange_in_grid(6, 6, buff=0.02).move_to([-6.45, 0, 0])
        bl = VGroup(txt("thousands of inputs", 20, HI).next_to(big_nodes[0], DOWN, buff=0.3),
                    txt("(pixels)", 18, DIM))
        bl[1].next_to(bl[0], DOWN, buff=0.08)
        self.play(FadeIn(pix), FadeIn(bl), run_time=0.8)
        self.cue(15)
        hb = SurroundingRectangle(VGroup(*big_nodes[1:4]), color=HI, buff=0.2, corner_radius=0.1)
        hbl = txt("many hidden layers", 20, HI).next_to(hb, DOWN, buff=0.15)
        self.play(Create(hb), FadeIn(hbl), run_time=0.8)
        for l in range(len(layers_n) - 1):
            ls = [Line(a.get_center(), b.get_center(), stroke_width=4, color=HI) for a in big_nodes[l] for b in big_nodes[l + 1]][::3]
            self.play(*[ShowPassingFlash(x, time_width=0.7) for x in ls], run_time=0.45)
        self.finish(0.2)

        # ================= 7. random start + error
        self.seg(7)
        self.play(FadeOut(Group(big_edges, *big_nodes, pix, bl, hb, hbl)), run_time=0.6)
        # rebuild the small network, hand weights still on
        self.play(FadeIn(Group(ax, nums, xl, yl, dots, names, reg_o, l_study, l_sleep)),
                  FadeIn(VGroup(inp, hid, out, *[w for row in w1 for w in row], *w2, in_lbl, q1, q2, q3, wl1, wl2, wl3, b3)),
                  run_time=0.8)
        self.cue(1)
        self.play(Indicate(VGroup(wl1, wl2, wl3, b3), color=HI), run_time=1.2)
        self.cue(2)
        self.cue(3)
        # shrink network to the top right, swap in random weights
        net_core = VGroup(inp, hid, out, *[w for row in w1 for w in row], *w2)
        self.play(FadeOut(VGroup(q1, q2, q3, wl1, wl2, wl3, b3, in_lbl)), run_time=0.4)
        self.play(net_core.animate.scale(0.72).move_to([3.85, 1.75, 0]), run_time=0.8)
        in_lbl2 = VGroup(txt("studied", 16, DIM).next_to(inp[0], LEFT, buff=0.1), txt("slept", 16, DIM).next_to(inp[1], LEFT, buff=0.1))
        rnd = txt("random weights", 20, HI).move_to([3.85, 3.45, 0])
        self.play(FadeIn(in_lbl2), FadeIn(rnd), run_time=0.4)

        kt = ValueTracker(0.0)

        def P():
            return params_at(kt.get_value())

        def restyle_wires(_m=None):
            p = P()
            for i in range(2):
                for slot in range(2):
                    self.style_wire(w1[i][slot], p["W1"][i][SLOT[slot]])
            for slot in range(2):
                self.style_wire(w2[slot], p["W2"][SLOT[slot]])
        wire_holder = VMobject()
        wire_holder.add_updater(restyle_wires)
        self.add(wire_holder)

        def hline(slot, col):
            p = P(); j = SLOT[slot]
            return self.bound_line(p["W1"][0][j], p["W1"][1][j], p["b1"][j], col)
        t_line1 = always_redraw(lambda: hline(0, Q1_C))
        t_line2 = always_redraw(lambda: hline(1, Q2_C))
        t_reg = always_redraw(lambda: self.region(lambda x, y: net_forward(P(), x, y)[1], res=90))
        self.play(FadeOut(reg_o), FadeOut(l_study), FadeOut(l_sleep), FadeIn(t_reg), FadeIn(t_line1), FadeIn(t_line2), run_time=1.0)
        self.cue(4)
        guesses = always_redraw(lambda: VGroup(*[
            txt(f"{g:.2f}", 16, PASS_C if (g > 0.5) == bool(YS[i]) else FAIL_C).next_to(dots[i], UP, buff=0.08).set_z_index(4)
            for i, g in enumerate(net_forward(P(), XS[:, 0], XS[:, 1])[1])]))
        self.play(FadeOut(names), FadeIn(guesses), run_time=0.8)
        gl = txt("numbers = the network's guesses", 18, GREY_A).next_to(ax, UP, buff=0.1).shift(RIGHT * 1.45)
        self.play(FadeIn(gl), run_time=0.5)
        self.cue(5)
        ef = txt("error = (guess − answer)²", 26).move_to([3.85, -0.35, 0])
        self.play(Write(ef), run_time=1.4)
        exg = np.ravel(net_forward(params_at(0), XS[0, 0], XS[0, 1])[1])[0]
        eg = txt(f"Ana: ({float(exg):.2f} − 1)² = {(float(exg) - 1) ** 2:.2f}", 20, GREY_A).move_to([3.85, -0.95, 0])
        self.play(FadeIn(eg), Indicate(dots[0], scale_factor=1.6), run_time=0.8)
        self.cue(6)
        sumt = txt("added up over all 10 students", 20, GREY_A).move_to([3.85, -1.45, 0])
        self.play(FadeIn(sumt), run_time=0.6)
        self.cue(7)
        err_live = always_redraw(lambda: txt(f"error  {net_error(P()):.2f}", 34, FAIL_C).move_to([3.85, -2.35, 0]))
        self.play(FadeIn(err_live, scale=1.2), run_time=0.7)
        self.cue(9)
        lower = txt("lower is better", 20, DIM).next_to([3.85, -2.35, 0], DOWN, buff=0.4)
        self.play(FadeIn(lower), run_time=0.5)
        self.finish(0.2)

        # ================= 8. slopes + backprop
        self.seg(8)
        self.play(FadeOut(VGroup(eg, sumt, lower, gl)), ef.animate.scale(0.8).move_to([3.85, -0.4, 0]), run_time=0.5)
        self.cue(1)
        # nudge one real weight and show the real change in error
        p0 = params_at(0)
        p1 = {k: v.copy() for k, v in p0.items()}
        j = SLOT[0]
        p1["W2"][j] += 0.3
        e0, e1 = net_error(p0), net_error(p1)
        hl = w2[0].copy().set_stroke(HI, 9)
        nud = txt(f"nudge this weight up:  error {e0:.2f} → {e1:.2f}", 20, HI).move_to([3.85, -1.1, 0])
        self.play(Create(hl), run_time=0.5)
        self.play(FadeIn(nud), run_time=0.6)
        self.cue(2)
        direction = "down" if e1 < e0 else "up"
        slope_t = txt(f"slope: error goes {direction}  →  keep going {'up' if e1 < e0 else 'down'}", 20, GREY_A).move_to([3.85, -1.55, 0])
        self.play(FadeIn(slope_t), run_time=0.6)
        self.cue(3)
        all_w = [w for row in w1 for w in row] + w2
        self.play(FadeOut(hl), *[ShowPassingFlash(w.copy().set_stroke(HI, 8), time_width=0.8) for w in all_w], run_time=1.0)
        self.cue(4)
        bp = txt("backpropagation: output → back to inputs", 20, FAIL_C).move_to([3.85, 3.45, 0])
        self.play(FadeOut(rnd), FadeIn(bp), run_time=0.5)
        for grp in (w2, [w for row in w1 for w in row]):
            self.play(*[ShowPassingFlash(w.copy().set_stroke(FAIL_C, 8).reverse_points(), time_width=0.8) for w in grp], run_time=0.9)
        self.cue(5)
        steps = always_redraw(lambda: txt(f"step {int(round(kt.get_value() * EVERY)):,}", 24).move_to([3.85, -2.95, 0]))
        self.play(FadeIn(steps), run_time=0.4)
        self.cue(6)
        goal = txt("× 20,000", 26, HI).next_to([3.85, -2.95, 0], RIGHT, buff=1.0)
        self.play(FadeIn(goal), run_time=0.5)
        self.finish(0.2)

        # ================= 9. training
        self.seg(9)
        self.play(FadeOut(VGroup(nud, slope_t, goal, bp, ef)), run_time=0.4)
        self.play(kt.animate.set_value(epoch_to_k(200)), run_time=self.until(1) + 0.2, rate_func=linear)
        self.play(kt.animate.set_value(epoch_to_k(1000)), run_time=self.until(2) + 0.4, rate_func=linear)
        self.cue(2)
        self.play(kt.animate.set_value(epoch_to_k(4000)), run_time=max(0.5, self.until(3) - 0.2), rate_func=linear)
        self.play(kt.animate.set_value(epoch_to_k(20000)), run_time=1.6, rate_func=rush_from)
        self.cue(4)
        pf = params_at(len(SNAPS) - 1)
        wire_holder.clear_updaters()
        s, sl = SLOT[0], SLOT[1]
        top_t = txt(f"{pf['W1'][0][s]:.1f}×studied {pf['W1'][1][s]:+.1f}×slept", 18, Q1_C).next_to(hid[0], UP, buff=0.12)
        bot_t = txt(f"{pf['W1'][0][sl]:.1f}×studied {pf['W1'][1][sl]:+.1f}×slept", 18, Q2_C).next_to(hid[1], DOWN, buff=0.12)
        self.play(Indicate(hid, color=HI), run_time=0.9)
        self.cue(5)
        self.play(FadeIn(bot_t), Indicate(t_line2, color=Q2_C), run_time=0.9)
        lq2 = txt("mostly sleep", 18, Q2_C).next_to(bot_t, DOWN, buff=0.08)
        self.play(FadeIn(lq2), run_time=0.4)
        self.cue(6)
        self.play(FadeIn(top_t), Indicate(t_line1, color=Q1_C), run_time=0.9)
        lq1 = txt("mostly study", 18, Q1_C).next_to(top_t, UP, buff=0.08)
        self.play(FadeIn(lq1), run_time=0.4)
        self.finish(0.2)

        # ================= 10. new student + recap
        self.seg(10)
        new = RES["new"]
        star = Star(5, outer_radius=0.16, color=HI, fill_opacity=1).move_to(ax.c2p(*new)).set_z_index(5)
        star_l = txt("new: 7 h, 4 h", 18, HI).next_to(star, DOWN, buff=0.1).set_z_index(5)
        self.remove(err_live)
        self.play(FadeOut(steps), GrowFromCenter(star), FadeIn(star_l), run_time=0.8)
        self.cue(1)
        nin = VGroup(txt("7", 18).move_to(inp[0]), txt("4", 18).move_to(inp[1]))
        self.play(FadeIn(nin), run_time=0.5)
        self.cue(2)
        hn = RES["new_hidden"]
        hv_top = txt(f"{hn[SLOT[0]]:.2f}", 18, Q1_C).move_to(hid[0])
        self.play(*[ShowPassingFlash(w.copy().set_stroke(HI, 7), time_width=0.7) for w in (w1[0][0], w1[1][0])], run_time=0.6)
        self.play(FadeIn(hv_top), run_time=0.4)
        y1 = txt("study: yes", 20, Q1_C).move_to([3.85, -0.95, 0])
        self.play(FadeIn(y1), run_time=0.4)
        self.cue(3)
        hv_bot = txt(f"{hn[SLOT[1]]:.2f}", 18, Q2_C).move_to(hid[1])
        self.play(*[ShowPassingFlash(w.copy().set_stroke(HI, 7), time_width=0.7) for w in (w1[0][1], w1[1][1])], run_time=0.6)
        self.play(FadeIn(hv_bot), run_time=0.4)
        y2 = txt("sleep: probably not", 20, Q2_C).move_to([3.85, -1.4, 0])
        self.play(FadeIn(y2), run_time=0.4)
        self.cue(4)
        self.play(*[ShowPassingFlash(w.copy().set_stroke(HI, 7), time_width=0.7) for w in w2], run_time=0.6)
        ov = txt(f"{RES['new_out']:.2f}", 18, FAIL_C).move_to(out)
        self.play(FadeIn(ov), run_time=0.4)
        ans = txt(f"chance of passing: {RES['new_out']:.2f}", 26, FAIL_C).move_to([3.85, -2.2, 0])
        self.play(FadeIn(ans, shift=UP * 0.1), run_time=0.5)
        self.cue(5)
        pf_t = txt("probably fails", 24, FAIL_C).move_to([3.85, -2.75, 0])
        self.play(FadeIn(pf_t), run_time=0.5)
        self.cue(6)
        self.remove(t_reg, t_line1, t_line2, guesses)
        self.play(FadeOut(Group(ax, nums, xl, yl, dots, star, star_l, y1, y2, ans, pf_t, nin, hv_top, hv_bot, ov,
                                top_t, bot_t, lq1, lq2, in_lbl2)), run_time=0.7)
        # final picture: the network with its layer names, plus the three ideas
        net_core.generate_target()
        net_core.target.scale(1 / 0.72).move_to([3.85, 0.15, 0])
        self.play(MoveToTarget(net_core), run_time=0.8)
        for i in range(2):
            for slot in range(2):
                self.style_wire(w1[i][slot], pf["W1"][i][SLOT[slot]])
        names_l = VGroup(txt("input", 22, HI).move_to([1.45, -2.4, 0]), txt("hidden", 22, HI).move_to([3.85, -2.4, 0]),
                         txt("output", 22, HI).move_to([6.2, -2.4, 0]))
        self.play(FadeIn(names_l), run_time=0.6)
        self.cue(7)
        r1 = txt("neuron = weighted score, squashed", 26).move_to([-3.6, 1.3, 0])
        self.play(FadeIn(r1, shift=UP * 0.15), Indicate(out, color=HI), run_time=0.9)
        self.cue(8)
        r2 = txt("hidden layer = smaller questions", 26).move_to([-3.6, 0.0, 0])
        self.play(FadeIn(r2, shift=UP * 0.15), Indicate(hid, color=HI), run_time=0.9)
        self.cue(9)
        r3 = txt("training = nudge weights downhill", 26).move_to([-3.6, -1.3, 0])
        self.play(FadeIn(r3, shift=UP * 0.15), *[ShowPassingFlash(w.copy().set_stroke(FAIL_C, 8).reverse_points(), time_width=0.8)
                                                 for w in all_w], run_time=1.0)
        self.finish(1.5)
