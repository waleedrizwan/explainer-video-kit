"""Large language models from first principles, covering all three pages of Google's Machine Learning Crash Course
LLM module (CC BY 4.0). The rain and dog-trick probabilities are the course's own; the worked examples in toy.py
(N-gram counts, 2-D attention vectors) are hand-sized numbers; anything else illustrative is labelled on screen.

    python narration.py
    manim -pql scene.py Explainer      # 480p preview
    manim -qh scene.py Explainer       # 1080p final -> media/videos/scene/1080p60/Explainer.mp4
"""
import sys
from pathlib import Path

from manim import *

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(next(p for p in HERE.parents if (p / "vidkit").is_dir())))
sys.path.insert(0, str(HERE))
from vidkit import NarratedScene  # noqa: E402
from vidkit.look import *  # noqa: E402,F403
from toy import BLEND, EXPS, ORANGE_NEXT, ORANGE_TOTAL, OTHERS, SCORES, SHRINK, VEC, WEIGHTS  # noqa: E402

CHIP = "#262A33"
ORANGE = "#FF9F43"

RAIN = [("cook soup", 9.4), ("warm up a kettle", 5.2), ("cower", 3.6), ("nap", 2.5), ("relax", 2.2)]
SENT = "The animal didn't cross the street because it was too tired".split()
IT = SENT.index("it")
W_TIRED = [0.25, 0.95, 0.15, 0.1, 0.1, 0.3, 0.12, 0, 0.15, 0.1, 0.35]
W_WIDE = [0.2, 0.3, 0.15, 0.1, 0.12, 0.95, 0.12, 0, 0.15, 0.1, 0.35]
CHAPTERS = ["The guessing game", "Counting: N-grams", "Neural networks", "Transformers & attention",
            "Training & generating", "Making it useful"]


# ---------------------------------------------------------------- helpers
def chip(s, size=22, color=WHITE, fill=CHIP, pad=0.18, stroke=EDGE):
    t = txt(s, size, color)
    box = RoundedRectangle(corner_radius=0.12, width=t.width + 2 * pad, height=max(t.height, 0.32) + 2 * pad * 0.8,
                           stroke_color=stroke, stroke_width=1.5).set_fill(fill, 1)
    return VGroup(box, t.move_to(box))


def header(s):
    return txt(s, 34).to_edge(UP, buff=0.6)


def boxed(title, sub=None, color=HI, w=None, h=1.6, tsize=26, ssize=18):
    parts = [txt(title, tsize, color)] + ([txt(sub, ssize, DIM)] if sub else [])
    g = VGroup(*parts).arrange(DOWN, buff=0.2)
    return VGroup(card(max(w or 0, g.width + 0.7, 3.0), h, g.get_center()), g)


def grid(cols, rows, cell=0.3, color=NEG):
    return VGroup(*[RoundedRectangle(corner_radius=cell * 0.15, width=cell * 0.8, height=cell * 0.8, stroke_width=0)
                    .set_fill(color, 0.55).move_to([c * cell, -r * cell, 0])
                    for r in range(rows) for c in range(cols)]).center()


def arc(src, dst, w=1.0, color=NEG, angle=0.6 * PI):
    a, b = src.get_top() + UP * 0.05, dst.get_top() + UP * 0.05
    return ArcBetweenPoints(a, b, angle=angle if b[0] < a[0] else -angle,
                            stroke_color=color, stroke_width=1 + 7 * w, stroke_opacity=max(w, 0.08))


def bars(rows, x0=-0.8, scale=4.6, top=0.5, step=0.62, color=NEG, size=24, fmt="{}%"):
    """Horizontal bar chart: rows of (label, value). Each row is VGroup(label, bar, value)."""
    vmax = max(v for _, v in rows)
    out = VGroup()
    for i, (s, v) in enumerate(rows):
        y = top - step * i
        lab = txt(s, size).move_to([x0 - 0.4, y, 0], aligned_edge=RIGHT)
        bar = Rectangle(width=max(v / vmax * scale, 0.06), height=0.38, stroke_width=0).set_fill(color, 0.85)
        bar.move_to([x0, y, 0], aligned_edge=LEFT)
        val = txt(fmt.format(v), size - 2, DIM).next_to(bar, RIGHT, buff=0.15)
        out.add(VGroup(lab, bar, val))
    return out


def grow_row(r):
    return AnimationGroup(FadeIn(r[0]), GrowFromEdge(r[1], LEFT), FadeIn(r[2]))


def network(layers=(3, 4, 4, 1), w=5.0, h=2.8, center=ORIGIN):
    xs = np.linspace(-w / 2, w / 2, len(layers))
    cols = VGroup()
    for x, n in zip(xs, layers):
        ys = np.linspace(h / 2, -h / 2, n) if n > 1 else [0]
        cols.add(VGroup(*[Circle(radius=0.17, stroke_color=WHITE, stroke_width=2).set_fill(PANEL, 1).move_to([x, y, 0])
                          for y in ys]))
    rng = np.random.default_rng(5)
    edges = VGroup(*[Line(a.get_center(), b.get_center(), stroke_width=1 + 2.5 * rng.random(),
                          stroke_color=POS if rng.random() > 0.4 else NEG, stroke_opacity=0.7).set_z_index(-1)
                     for c0, c1 in zip(cols[:-1], cols[1:]) for a in c0 for b in c1])
    return VGroup(edges, cols).move_to(center)


class Explainer(NarratedScene):
    project_dir = HERE

    def clear_all(self, keep=()):
        keep = list(keep) + ([self.tag] if getattr(self, "tag", None) else [])
        out = [m for m in self.mobjects if m not in keep]
        if out:
            self.play(*[FadeOut(m) for m in out], run_time=0.5)

    def chapter(self, n):
        """Full-screen chapter card, then a small tag in the corner for the rest of the chapter."""
        self.clear_all(keep=[])
        if getattr(self, "tag", None):
            self.play(FadeOut(self.tag), run_time=0.3)
        num = txt(f"Chapter {n}", 24, DIM)
        name = txt(CHAPTERS[n - 1], 46, HI)
        g = VGroup(num, name).arrange(DOWN, buff=0.25)
        self.play(FadeIn(g, shift=UP * 0.2), run_time=0.6)
        self.wait(0.8)
        self.tag = txt(f"{n} · {CHAPTERS[n - 1]}", 16, DIM).to_corner(UL, buff=0.3)
        self.play(ReplacementTransform(g, self.tag), run_time=0.6)

    def construct(self):
        self.tag = None
        self.s0_intro()
        self.chapter(1)
        for f in [self.s1_keyboard, self.s2_tokens, self.s3_rain, self.s4_use]:
            f()
        self.chapter(2)
        for f in [self.s5_ngrams, self.s6_counting, self.s7_context, self.s8_sparsity]:
            f()
        self.chapter(3)
        for f in [self.s9_nn, self.s10_rnn, self.s11_vanish]:
            f()
        self.chapter(4)
        for f in [self.s12_llm, self.s13_encdec, self.s14_selfq, self.s15_animal, self.s16_math, self.s17_dirs,
                  self.s18_heads, self.s19_size]:
            f()
        self.chapter(5)
        for f in [self.s20_train, self.s21_max, self.s22_chat, self.s23_benefits, self.s24_train_probs,
                  self.s25_use_probs]:
            f()
        self.chapter(6)
        for f in [self.s26_foundation, self.s27_finetune, self.s28_pet, self.s29_distill, self.s30_prompt,
                  self.s31_offline, self.s32_bias]:
            f()
        self.s33_recap()

    # ================================================================ 0
    def s0_intro(self):
        self.seg(0)
        q = chip("Why is the sky blue?", 26, fill="#1E3A5F", stroke=NEG).move_to([2.6, 2.0, 0])
        self.play(FadeIn(q, shift=LEFT * 0.3), run_time=0.6)
        words = "Sunlight is made of many colours. Air scatters blue light the most, so blue reaches your eyes " \
                "from every part of the sky.".split()
        ans = VGroup(*[txt(w, 24) for w in words])
        lines, line = VGroup(), VGroup()
        for w in ans:
            line.add(w)
            if len(line) == 6:
                lines.add(line.copy())
                line = VGroup()
        if len(line):
            lines.add(line)
        for ln in lines:
            ln.arrange(RIGHT, buff=0.14)
        lines.arrange(DOWN, aligned_edge=LEFT, buff=0.2).move_to([-1.2, -0.3, 0])
        bubble = card(lines.width + 0.8, lines.height + 0.7, lines.get_center())
        self.play(FadeIn(bubble), run_time=0.4)
        flat = [w for ln in lines for w in ln]
        self.play(LaggedStart(*[FadeIn(w) for w in flat], lag_ratio=0.25), run_time=self.until(2) - 0.6)
        self.cue(2)
        big = txt("Guess the next word.", 48, HI)
        self.play(FadeOut(VGroup(q, bubble, lines)), run_time=0.4)
        self.play(Write(big), run_time=0.8)
        self.cue(3)
        title = txt("Large Language Models", 50).shift(UP * 2.4)
        sub = txt("from first principles · after Google's Machine Learning Crash Course", 20, DIM)
        sub.next_to(title, DOWN, buff=0.25)
        self.play(FadeOut(big), FadeIn(title, shift=UP * 0.2), FadeIn(sub))
        items = VGroup()
        for i, c in enumerate(CHAPTERS):
            n = txt(str(i + 1), 26, HI).move_to([-3.4, 0.7 - 0.62 * i, 0])
            items.add(VGroup(n, txt(c, 26).move_to([-2.95, n.get_y(), 0], aligned_edge=LEFT)))
        for k, idx in zip(range(4, 8), [[0], [1, 2], [3], [4, 5]]):
            self.cue(k)
            self.play(*[FadeIn(items[i], shift=RIGHT * 0.2) for i in idx], run_time=0.6)
        self.finish()

    # ================================================================ chapter 1
    def s1_keyboard(self):
        self.seg(1)
        h = header("What is a language model?")
        self.play(FadeIn(h))
        field = chip("I'll be home  |", 28, pad=0.3).move_to([-3.4, 1.2, 0])
        field[0].stretch_to_fit_width(5.2)
        field[1].align_to(field[0], LEFT).shift(RIGHT * 0.3)
        phone = RoundedRectangle(corner_radius=0.35, width=6.0, height=4.4, stroke_color=EDGE, stroke_width=2)
        phone.set_fill(PANEL, 1).move_to([-3.4, 0.0, 0]).set_z_index(-5)
        self.play(FadeIn(phone), FadeIn(field))
        self.cue(1)
        sugg = VGroup(*[chip(s, 26) for s in ["soon", "late", "by"]]).arrange(RIGHT, buff=0.25).move_to([-3.4, -0.1, 0])
        self.play(LaggedStart(*[FadeIn(s, shift=UP * 0.15) for s in sugg], lag_ratio=0.2))
        self.cue(2)
        probs = [0.55, 0.3, 0.15]
        pb = VGroup(*[Rectangle(width=0.5, height=1.2 * p / 0.55, stroke_width=0).set_fill(NEG, 0.85)
                      .next_to(s, DOWN, buff=0.15) for s, p in zip(sugg, probs)])
        for b in pb:
            b.align_to(pb[0], UP)
        pl = txt("how likely? (illustrative)", 16, DIM).next_to(pb, DOWN, buff=0.15)
        self.play(*[GrowFromEdge(b, UP) for b in pb], FadeIn(pl))
        self.cue(3)
        lm = txt("= a language model", 30, HI).move_to([3.3, 1.6, 0])
        self.play(FadeIn(lm, shift=LEFT * 0.2))
        self.cue(4)
        d = VGroup(txt("estimates the probability", 24), txt("of a piece of text", 24),
                   txt("within a longer sequence", 24)).arrange(DOWN, buff=0.12).next_to(lm, DOWN, buff=0.4)
        self.play(FadeIn(d))
        self.cue(5)
        line = NumberLine(x_range=[0, 100, 25], length=9, color=DIM, include_numbers=False).move_to([0, -3.0, 0])
        labs = VGroup(*[txt(f"{v}%", 16, DIM).next_to(line.n2p(v), DOWN, buff=0.15) for v in [0, 25, 50, 75, 100]])
        e0 = txt("never", 16, DIM).next_to(line.n2p(0), UP, buff=0.15)
        e1 = txt("always", 16, DIM).next_to(line.n2p(100), UP, buff=0.15)
        dot = Dot(line.n2p(55), color=HI)
        self.play(Create(line), FadeIn(labs), FadeIn(e0), FadeIn(e1), FadeIn(dot, scale=2))
        self.finish()
        self.clear_all()

    def s2_tokens(self):
        self.seg(2)
        h = header("Tokens")
        sent = txt("When I hear rain on my roof", 32).shift(UP * 1.6)
        self.play(FadeIn(h), FadeIn(sent))
        pieces = VGroup(*[chip(w, 26) for w in "When I hear rain on my roof".split()]).arrange(RIGHT, buff=0.1)
        pieces.move_to(sent)
        self.play(ReplacementTransform(sent, pieces), run_time=0.8)
        cols = VGroup()
        for name, ps in [("word", ["kitchen"]), ("subword", ["un", "believ", "able"]), ("character", list("kitchen"))]:
            row = VGroup(*[chip(p, 24, pad=0.12) for p in ps]).arrange(RIGHT, buff=0.06)
            cols.add(VGroup(row, txt(name, 20, DIM).next_to(row, DOWN, buff=0.25)))
        cols.arrange(RIGHT, buff=0.9).shift(DOWN * 0.3)
        for k in range(3):
            self.cue(1 + k)
            self.play(FadeIn(cols[k], shift=UP * 0.2), run_time=0.6)
        self.cue(4)
        self.play(Circumscribe(cols[1], color=HI, buff=0.15))
        self.cue(5)
        rare = VGroup(*[chip(p, 22, stroke=GOOD, pad=0.12) for p in ["un", "believ", "ably"]]).arrange(RIGHT, buff=0.06)
        rl = txt("a word it never saw whole, spelled from known pieces", 20, GOOD)
        g = VGroup(rare, rl).arrange(RIGHT, buff=0.35).move_to([0, -2.2, 0])
        self.play(FadeIn(g, shift=UP * 0.2))
        self.cue(6)
        note = txt("in this video: one word ≈ one token", 22, HI).to_edge(DOWN, buff=0.4)
        self.play(FadeIn(note))
        self.finish()
        self.clear_all()

    def s3_rain(self):
        self.seg(3)
        h = header("The course's example")
        self.play(FadeIn(h))
        self.cue(1)
        sentence = txt("When I hear rain on my roof, I  _______  in my kitchen.", 28).shift(UP * 2.0)
        self.play(FadeIn(sentence))
        self.cue(2)
        lab = txt("every possible filler gets a probability", 20, DIM).next_to(sentence, DOWN, buff=0.25)
        self.play(FadeIn(lab))
        rows = bars(RAIN, top=0.55)
        for k, r in enumerate(rows):
            self.cue(3 + k)
            self.play(grow_row(r), run_time=0.5)
        self.cue(8)
        toks = VGroup(*[chip(w, 18, pad=0.1, stroke=HI) for w in "warm up a kettle".split()]).arrange(RIGHT, buff=0.05)
        toks.next_to(rows[1], RIGHT, buff=1.4)
        tl = txt("4 tokens", 18, HI).next_to(toks, DOWN, buff=0.12)
        self.play(TransformFromCopy(rows[1][0], toks), FadeIn(tl))
        self.cue(9)
        self.play(Indicate(toks, color=HI))
        self.cue(10)
        total = sum(v for _, v in RAIN)
        s = txt(f"9.4 + 5.2 + 3.6 + 2.5 + 2.2 = {total:.1f}%", 24, WHITE).move_to([0, -2.6, 0])
        rest = txt(f"the other {100 - total:.1f}% is spread over thousands of other fillers", 20, DIM)
        rest.next_to(s, DOWN, buff=0.2)
        self.play(FadeIn(s))
        self.play(FadeIn(rest))
        self.finish()
        self.clear_all()

    def s4_use(self):
        self.seg(4)
        h = header("From probabilities to text")
        rows = bars(RAIN, top=1.6)
        self.play(FadeIn(h), FadeIn(rows))
        self.cue(2)
        pick = SurroundingRectangle(rows[0], color=HI, buff=0.12, corner_radius=0.12)
        note = txt("take the most likely", 24, HI).move_to([0, -1.6, 0])
        self.play(Create(pick), FadeIn(note))
        self.cue(3)
        cut = DashedLine([-4.5, 0, 0], [4.8, 0, 0], color=GOOD, stroke_width=2)
        cut.move_to([0.15, (rows[2].get_bottom()[1] + rows[3].get_top()[1]) / 2, 0])
        note2 = txt("or pick at random above a threshold → more varied text", 24, GOOD).move_to(note)
        self.play(Create(cut), Transform(note, note2), pick.animate.set_color(GOOD))
        for k in [1, 2, 0, 2, 1]:
            self.play(pick.animate.move_to(rows[k]).stretch_to_fit_width(rows[k].width + 0.24), run_time=0.25)
        self.cue(4)
        self.clear_all(keep=[h])
        # generation loop
        self.cue(5)
        base = txt("When I hear rain on my roof, I", 26).move_to([-1.8, 1.8, 0])
        self.play(FadeIn(base))
        cursor = base.get_right() + RIGHT * 0.15
        for w in ["cook", "soup", "for", "dinner."]:
            t = chip(w, 24, stroke=GOOD).move_to(cursor, aligned_edge=LEFT)
            self.play(FadeIn(t, shift=DOWN * 0.2), run_time=0.35)
            cursor = t.get_right() + RIGHT * 0.1
        loop = txt("predict  →  add  →  predict again  =  generating text", 22, GOOD).move_to([0, 0.9, 0])
        self.play(FadeIn(loop))
        self.cue(6)
        tr = VGroup(txt("The cat is black.", 22), txt("→", 22, DIM), chip("Le chat est noir.", 22, stroke=GOOD))
        tr.arrange(RIGHT, buff=0.25)
        trl = txt("translating", 20, HI)
        trg = VGroup(trl, tr).arrange(DOWN, buff=0.2).move_to([0, -0.5, 0])
        self.play(FadeIn(trg))
        self.cue(7)
        sm = VGroup(txt("[ a 10-page report ]", 22, DIM), txt("→", 22, DIM), chip("Sales rose 4%.", 22, stroke=GOOD))
        sm.arrange(RIGHT, buff=0.25)
        smg = VGroup(txt("summarizing", 20, HI), sm).arrange(DOWN, buff=0.2).move_to([0, -1.9, 0])
        self.play(FadeIn(smg))
        self.cue(8)
        pat = txt("statistical patterns of tokens  →  an internal picture of language", 22, POS).to_edge(DOWN, buff=0.4)
        self.play(FadeIn(pat))
        self.finish()
        self.clear_all()

    # ================================================================ chapter 2
    def s5_ngrams(self):
        self.seg(5)
        h = header("Where do the probabilities come from?")
        self.play(FadeIn(h))
        self.cue(1)
        cnt = txt("Count.", 44, HI)
        self.play(FadeIn(cnt, scale=1.2))
        self.cue(2)
        self.play(Transform(h, header("N-grams")), FadeOut(cnt))
        defn = txt("N-gram = a run of N words in a row from the training text", 24, DIM).shift(UP * 2.2)
        self.play(FadeIn(defn))
        self.cue(3)
        phrase = "you are very nice".split()
        words = VGroup(*[chip(w, 30) for w in phrase]).arrange(RIGHT, buff=0.15).shift(UP * 1.1)
        self.play(LaggedStart(*[FadeIn(w, shift=UP * 0.2) for w in words], lag_ratio=0.15))

        def run(n, y, color, label, first_cue):
            lab = txt(label, 22, color).move_to([-5.0, y, 0])
            win = SurroundingRectangle(words[0:n], color=color, buff=0.1, corner_radius=0.15)
            outs = VGroup(*[chip(" ".join(phrase[i:i + n]), 24, stroke=color) for i in range(5 - n)])
            outs.arrange(RIGHT, buff=0.3).move_to([0.8, y, 0])
            self.play(Create(win), FadeIn(lab), run_time=0.5)
            for i in range(5 - n):
                self.cue(first_cue + i)
                if i:
                    self.play(win.animate.move_to(words[i:i + n]), run_time=0.3)
                self.play(TransformFromCopy(words[i:i + n], outs[i]), run_time=0.45)
            self.play(FadeOut(win), run_time=0.3)

        self.cue(4)
        run(2, -0.4, NEG, "2-grams\n(bigrams)", 5)
        self.cue(8)
        run(3, -1.9, POS, "3-grams\n(trigrams)", 9)
        self.finish()
        self.clear_all()

    def s6_counting(self):
        self.seg(6)
        h = header("A trigram model, by counting")
        prompt = VGroup(chip("?", 28, DIM), chip("?", 28, DIM), chip("____", 28, DIM)).arrange(RIGHT, buff=0.15)
        prompt.shift(UP * 2.0)
        win = SurroundingRectangle(prompt[:2], color=POS, buff=0.1, corner_radius=0.15)
        wl = txt("the two words before the blank", 18, POS).next_to(win, LEFT, buff=0.3)
        self.play(FadeIn(h), FadeIn(prompt), Create(win), FadeIn(wl))
        self.cue(1)
        p2 = VGroup(chip("orange", 28), chip("is", 28), chip("____", 28, DIM)).arrange(RIGHT, buff=0.15).move_to(prompt)
        self.play(Transform(prompt, p2), win.animate.become(SurroundingRectangle(p2[:2], color=POS, buff=0.1,
                                                                                  corner_radius=0.15)),
                  wl.animate.next_to(p2[:2], LEFT, buff=0.4))
        self.cue(2)
        look = txt("search the training text for  “orange is ___”", 20, DIM).next_to(prompt, DOWN, buff=0.3)
        self.play(FadeIn(look))
        self.cue(3)
        rows = bars([(w, c) for w, c in ORANGE_NEXT], x0=-1.4, scale=3.6, top=0.3, fmt="{}×", color=ORANGE)
        foot = txt("toy training text (made up)", 16, DIM).to_edge(DOWN, buff=0.3)
        self.play(LaggedStart(*[grow_row(r) for r in rows], lag_ratio=0.3), FadeIn(foot), run_time=1.8)
        self.cue(4)
        tot = txt(f"total  =  {ORANGE_TOTAL}", 24, WHITE).next_to(rows, DOWN, buff=0.3).align_to(rows[0][1], LEFT)
        self.play(FadeIn(tot))
        self.cue(5)
        pcts = VGroup(*[txt(f"÷ {ORANGE_TOTAL} = {100 * c / ORANGE_TOTAL:.0f}%", 22, HI).next_to(r[2], RIGHT, buff=0.4)
                        for r, (_, c) in zip(rows, ORANGE_NEXT)])
        self.play(LaggedStart(*[FadeIn(p, shift=LEFT * 0.2) for p in pcts], lag_ratio=0.2))
        self.cue(6)
        self.play(Indicate(pcts[0], color=HI), Indicate(pcts[1], color=HI))
        self.cue(7)
        done = txt("a working language model, built by counting", 24, GOOD).move_to(look)
        self.play(Transform(look, done))
        self.finish()
        self.clear_all()

    def s7_context(self):
        self.seg(7)
        h = header("Context")
        self.play(FadeIn(h))
        self.cue(1)
        a = VGroup(txt("orange is ripe", 28), txt("→ the fruit", 24, ORANGE)).arrange(RIGHT, buff=0.3)
        b = VGroup(txt("orange is cheerful", 28), txt("→ the colour", 24, ORANGE)).arrange(RIGHT, buff=0.3)
        a.move_to([0, 1.8, 0])
        self.play(FadeIn(a))
        self.cue(2)
        b.next_to(a, DOWN, buff=0.3)
        self.play(FadeIn(b))
        self.cue(3)
        qq = txt("which one? depends on what came earlier", 22, DIM).next_to(b, DOWN, buff=0.3)
        self.play(FadeIn(qq))
        self.cue(4)
        self.play(Transform(qq, txt("that earlier information = context", 24, HI).move_to(qq)))
        self.cue(5)
        self.play(FadeOut(VGroup(a, b, qq)))
        self.cue(6)
        acts = VGroup(*[boxed(f"Act {i}", w=2.4, h=1.0, tsize=24) for i in (1, 2, 3)]).arrange(RIGHT, buff=0.4)
        acts.move_to([-2.6, 1.5, 0])
        back = CurvedArrow(acts[2].get_top() + UP * 0.05, acts[0].get_top() + UP * 0.05, angle=0.6, color=HI)
        rl = txt("you remember the characters", 18, HI).next_to(acts, DOWN, buff=0.2)
        self.play(FadeIn(acts))
        self.play(Create(back), FadeIn(rl))
        self.cue(7)
        joke = VGroup(boxed("setup", w=2.2, h=1.0, tsize=24), boxed("punchline", w=2.6, h=1.0, tsize=24))
        joke.arrange(RIGHT, buff=0.6).move_to([4.2, 1.5, 0])
        jb = CurvedArrow(joke[1].get_top() + UP * 0.05, joke[0].get_top() + UP * 0.05, angle=0.8, color=HI)
        self.play(FadeIn(joke), Create(jb))
        self.cue(8)
        tgt = chip("____", 26, DIM)
        bef, aft = txt("before", 22, NEG), txt("after", 22, NEG)
        row = VGroup(bef, tgt, aft).arrange(RIGHT, buff=1.4).move_to([0, -0.6, 0])
        ar = VGroup(Arrow(bef.get_right(), tgt.get_left(), color=NEG, buff=0.15),
                    Arrow(aft.get_left(), tgt.get_right(), color=NEG, buff=0.15))
        cl = txt("context can come from either side of the token being predicted", 18, DIM).next_to(row, DOWN, buff=0.2)
        self.play(FadeIn(row), GrowArrow(ar[0]), GrowArrow(ar[1]), FadeIn(cl))
        self.cue(9)
        self.play(FadeOut(VGroup(acts, back, rl, joke, jb, row, ar, cl)))
        ws = "I peeled the orange . The orange is ____".split()
        line = VGroup(*[chip(w, 24) for w in ws]).arrange(RIGHT, buff=0.1).move_to([0, 0.6, 0])
        line[1:4].set_color(GOOD)
        self.play(FadeIn(line))
        self.cue(10)
        win = SurroundingRectangle(line[6:8], color=POS, buff=0.1, corner_radius=0.15)
        self.play(Create(win), line[:6].animate.set_opacity(0.25))
        tl = txt("a trigram only sees 2 words of context  →  lots of mistakes", 22, BAD).move_to([0, -0.8, 0])
        self.play(FadeIn(tl))
        self.finish()
        self.clear_all()

    def s8_sparsity(self):
        self.seg(8)
        h = header("Why not longer N-grams?")
        self.play(FadeIn(h))
        self.cue(1)
        sub = txt("the longer the sequence, the rarer it is", 24, DIM).shift(UP * 2.3)
        self.play(FadeIn(sub))
        phrases = ["orange is", "the orange is", "she said the orange is", "then she said the orange is",
                   "and then she said the orange is"]
        counts = [1200, 140, 6, 2, 1]
        rows = VGroup()
        for i, (p, c) in enumerate(zip(phrases, counts)):
            y = 1.3 - 0.6 * i
            lab = txt(p, 22).move_to([-0.6, y, 0], aligned_edge=RIGHT)
            w = 0.06 + 4.2 * np.log10(c + 1) / np.log10(counts[0] + 1)
            bar = Rectangle(width=w, height=0.36, stroke_width=0).set_fill(NEG if c > 1 else BAD, 0.85)
            bar.move_to([-0.2, y, 0], aligned_edge=LEFT)
            rows.add(VGroup(lab, bar, txt(f"{c:,}", 20, DIM).next_to(bar, RIGHT, buff=0.15)))
        cap = txt("times seen in training (illustrative, log scale)", 16, DIM).next_to(rows, DOWN, buff=0.3)
        for k in range(3):
            self.cue(2 + k)
            self.play(grow_row(rows[k]), *([FadeIn(cap)] if k == 0 else []), run_time=0.5)
        self.cue(5)
        self.play(grow_row(rows[3]), run_time=0.5)
        self.cue(6)
        self.play(grow_row(rows[4]), run_time=0.5)
        self.cue(7)
        self.play(Circumscribe(rows[4], color=BAD, buff=0.1))
        self.cue(8)
        wall = txt("counting hits a wall", 28, BAD).to_edge(DOWN, buff=0.5)
        self.play(FadeIn(wall))
        self.cue(9)
        self.play(Transform(wall, txt("we need to learn context, not just count it", 28, HI).move_to(wall)))
        self.finish()
        self.clear_all()

    # ================================================================ chapter 3
    def s9_nn(self):
        self.seg(9)
        h = header("A neural network, in one minute")
        self.play(FadeIn(h))
        self.cue(2)
        net = network(w=3.8, center=[-3.3, 0.2, 0])
        self.play(FadeIn(net[1]), Create(net[0]), run_time=1.0)
        pl = txt("each line = an adjustable number", 20, DIM).next_to(net, DOWN, buff=0.3)
        pl2 = txt("(a parameter, or weight)", 18, DIM).next_to(pl, DOWN, buff=0.1)
        self.play(FadeIn(pl), FadeIn(pl2))
        self.cue(3)
        inl = txt("numbers\nin", 18, DIM).next_to(net, LEFT, buff=0.3)
        outl = txt("prediction", 18, DIM).next_to(net, RIGHT, buff=0.3)
        self.play(FadeIn(inl), FadeIn(outl))
        self.play(LaggedStart(*[Indicate(c, color=HI, scale_factor=1.15) for c in net[1]], lag_ratio=0.3), run_time=1.4)
        self.cue(4)
        steps = VGroup(txt("1  show an example", 22), txt("2  measure how wrong", 22), txt("3  nudge every weight", 22))
        steps.arrange(DOWN, aligned_edge=LEFT, buff=0.2).move_to([3.8, 1.0, 0])
        self.play(LaggedStart(*[FadeIn(s, shift=LEFT * 0.2) for s in steps], lag_ratio=0.4), run_time=1.5)
        rng = np.random.default_rng(9)
        self.play(*[e.animate.set_stroke(width=1 + 2.5 * rng.random()) for e in net[0]], run_time=0.8)
        self.cue(5)
        bp = Arrow(net.get_right() + DOWN * 1.9, net.get_left() + DOWN * 1.9, color=BAD, buff=0)
        bpl = txt("backpropagation: works out the nudges, output → input", 16, BAD).next_to(bp, DOWN, buff=0.1)
        self.play(FadeOut(VGroup(pl, pl2)), GrowArrow(bp), FadeIn(bpl))
        self.cue(6)
        self.play(FadeOut(VGroup(steps)))
        tokc = chip("cat", 26).move_to([1.4, 0.6, 0])
        vec = VGroup(*[chip(f"{v:+.1f}", 18, pad=0.1) for v in [0.2, -1.3, 0.7, 0.4]]).arrange(DOWN, buff=0.05)
        vec.next_to(tokc, RIGHT, buff=0.8)
        ea = Arrow(tokc.get_right(), vec.get_left(), color=DIM, buff=0.1)
        el = txt("embedding", 20, HI).next_to(vec, UP, buff=0.15)
        self.play(FadeIn(tokc), GrowArrow(ea), FadeIn(vec), FadeIn(el))
        self.cue(7)
        pts = {"cat": (4.4, 1.2), "dog": (5.0, 0.8), "car": (4.6, -0.9), "truck": (5.3, -1.2)}
        dots = VGroup(*[VGroup(Dot([x, y, 0], color=GOOD if n in ("cat", "dog") else POS),
                               txt(n, 16).next_to([x, y, 0], RIGHT, buff=0.08)) for n, (x, y) in pts.items()])
        sl = txt("similar meaning → similar numbers", 16, DIM).move_to([4.5, -1.8, 0])
        self.play(FadeIn(dots), FadeIn(sl))
        self.finish()
        self.clear_all()

    def _rnn_chain(self, toks, y=0.3):
        xs = np.linspace(-5.4, 5.4, len(toks))
        words = VGroup(*[chip(t, 22).move_to([x, y - 1.9, 0]) for t, x in zip(toks, xs)])
        cells = VGroup(*[RoundedRectangle(corner_radius=0.15, width=1.0, height=1.0, stroke_color=EDGE, stroke_width=2)
                         .set_fill(PANEL, 1).move_to([x, y, 0]) for x in xs])
        ups = VGroup(*[Arrow(w.get_top(), c.get_bottom(), buff=0.1, color=DIM, stroke_width=3,
                             max_tip_length_to_length_ratio=0.25) for w, c in zip(words, cells)])
        links = VGroup(*[Arrow(cells[i].get_right(), cells[i + 1].get_left(), buff=0.05, color=HI, stroke_width=3,
                               max_tip_length_to_length_ratio=0.3) for i in range(len(toks) - 1)])
        return words, cells, ups, links

    def s10_rnn(self):
        self.seg(10)
        h = header("Recurrent neural networks")
        self.play(FadeIn(h))
        toks = "I peeled the orange . The orange is".split()
        words, cells, ups, links = self._rnn_chain(toks)
        self.cue(1)
        self.play(FadeIn(words), FadeIn(cells), run_time=0.8)
        self.cue(2)
        mlab = txt("memory", 18, HI).next_to(cells[0], UP, buff=0.15)
        self.play(FadeIn(mlab), cells[0].animate.set_stroke(HI))
        self.cue(3)
        self.mem = VGroup()
        for i, c in enumerate(cells):
            fill = interpolate_color(ManimColor(PANEL), ManimColor(ORANGE), 0.9 if i < 4 else 0.5)
            self.play(GrowArrow(ups[i]), c.animate.set_fill(fill, 1).set_stroke(HI if i == len(cells) - 1 else EDGE),
                      *([GrowArrow(links[i - 1])] if i else []), run_time=0.3)
        self.cue(4)
        keep = txt("keep: “orange”, “peeled”", 20, GOOD).move_to([-2.5, 2.0, 0])
        drop = txt("ignore: “the”, “.”", 20, DIM).move_to([2.5, 2.0, 0])
        self.play(FadeIn(keep), FadeIn(drop), words[1][0].animate.set_stroke(GOOD, 2.5),
                  words[3][0].animate.set_stroke(GOOD, 2.5), words[2].animate.set_opacity(0.4),
                  words[4].animate.set_opacity(0.4))
        self.cue(5)
        lis = txt("like listening to someone speak", 22, HI).to_edge(DOWN, buff=0.5)
        self.play(FadeIn(lis))
        self.cue(6)
        self.play(Transform(lis, txt("context across several sentences  —  far more than an N-gram", 22, GOOD)
                            .move_to(lis)))
        self.finish()
        self.clear_all()

    def s11_vanish(self):
        self.seg(11)
        h = header("Where recurrent networks fall short")
        self.play(FadeIn(h))
        toks = ["I", "peeled", "the", "orange", ".", "…", "it", "is"]
        words, cells, ups, links = self._rnn_chain(toks, y=0.9)
        self.cue(1)
        self.play(FadeIn(words), FadeIn(cells), FadeIn(links), FadeIn(ups))
        squeeze = txt("everything passes through one memory", 20, HI).next_to(cells, UP, buff=0.3)
        self.play(FadeIn(squeeze), LaggedStart(*[Indicate(l, color=HI) for l in links], lag_ratio=0.15))
        self.cue(2)
        mem = VGroup()
        for i, c in enumerate(cells):
            o = 0.95 * 0.6 ** i
            b = Rectangle(width=0.7, height=0.2, stroke_width=0).set_fill(ORANGE, o).move_to(c).shift(UP * 0.22)
            mem.add(VGroup(b, txt("orange", 12).move_to(b).set_opacity(max(o, 0.06))))
        self.play(LaggedStart(*[FadeIn(m) for m in mem], lag_ratio=0.15), run_time=1.2)
        self.cue(3)
        vg = txt("the vanishing gradient problem", 28, BAD).move_to(squeeze)
        self.play(ReplacementTransform(squeeze, vg))
        self.cue(4)
        self.play(FadeOut(VGroup(words, ups)))
        sig = VGroup(*[txt("1", 22, BAD).next_to(cells[-1], DOWN, buff=0.35)])
        self.play(FadeIn(sig[0]))
        self.cue(5)
        mult = txt("× ½ at every step back", 22, BAD).to_edge(DOWN, buff=1.3)
        self.play(FadeIn(mult))
        self.cue(6)
        for i in range(len(cells) - 2, -1, -1):
            k = len(cells) - 1 - i
            v = SHRINK[k]
            t = txt(f"{v:.3g}" if v >= 0.01 else f"{v:.4f}", 20, BAD).next_to(cells[i], DOWN, buff=0.35)
            t.set_opacity(max(0.25, v ** 0.35))
            sig.add(t)
            self.play(FadeIn(t, shift=LEFT * 0.3), run_time=0.22)
        ten = txt(f"10 steps back:  ½ × ½ × … × ½  =  {SHRINK[10]:.4f}  ≈  one thousandth", 24, BAD).move_to(mult)
        self.play(Transform(mult, ten))
        self.cue(7)
        self.play(Indicate(sig[-1], color=BAD, scale_factor=1.4))
        self.cue(8)
        want = txt("what we want: look at the whole context at once", 26, HI).to_edge(DOWN, buff=0.5)
        self.play(FadeIn(want))
        self.finish()
        self.clear_all()

    # ================================================================ chapter 4
    def s12_llm(self):
        self.seg(12)
        h = header("Large language models")
        self.play(FadeIn(h))
        self.cue(1)
        d = txt("LLM: predicts a token or a sequence of tokens — even many paragraphs", 22, DIM).shift(UP * 2.2)
        self.play(FadeIn(d))
        self.cue(2)
        cmp = VGroup(boxed("N-grams", "count", DIM, w=3.0, h=1.2), boxed("RNNs", "token by token", DIM, w=3.0, h=1.2),
                     boxed("LLMs", "whole context at once", GOOD, w=3.6, h=1.2)).arrange(RIGHT, buff=0.4).shift(UP * 0.6)
        self.play(LaggedStart(*[FadeIn(c) for c in cmp], lag_ratio=0.3))
        self.cue(3)
        p = boxed("far more parameters", w=5.0, h=1.0, tsize=26).move_to([-3.0, -1.3, 0])
        self.play(FadeIn(p, shift=UP * 0.2))
        self.cue(4)
        c = boxed("far more context", w=5.0, h=1.0, tsize=26).move_to([3.0, -1.3, 0])
        toks = VGroup(*[Dot(radius=0.07, color=WHITE) for _ in range(9)]).arrange(RIGHT, buff=0.3).move_to([3.0, -2.6, 0])
        hub = Dot([3.0, -2.0, 0], radius=0.01)
        lines = VGroup(*[Line(t.get_center(), c.get_bottom(), stroke_width=1.5, color=GOOD) for t in toks])
        self.play(FadeIn(c, shift=UP * 0.2), FadeIn(toks))
        self.play(*[Create(l) for l in lines], run_time=0.6)
        self.remove(hub)
        self.cue(5)
        tf = txt("built on the Transformer", 30, HI).to_edge(DOWN, buff=0.25)
        self.play(FadeIn(tf), Indicate(cmp[2], color=GOOD))
        self.finish()
        self.clear_all()

    def s13_encdec(self):
        self.seg(13)
        h = header("The Transformer")
        self.play(FadeIn(h))
        src = VGroup(*[chip(w, 22) for w in "The cat is black".split()]).arrange(DOWN, buff=0.15).move_to([-5.6, -0.2, 0])
        enc = VGroup(RoundedRectangle(corner_radius=0.2, width=2.4, height=2.6, stroke_color=NEG, stroke_width=2)
                     .set_fill(NEG, 0.15), txt("Encoder", 26, NEG)).move_to([-2.6, -0.2, 0])
        rep = grid(2, 6, 0.32, HI).move_to([0, -0.2, 0])
        dec = VGroup(RoundedRectangle(corner_radius=0.2, width=2.4, height=2.6, stroke_color=POS, stroke_width=2)
                     .set_fill(POS, 0.12), txt("Decoder", 26, POS)).move_to([2.6, -0.2, 0])
        out = VGroup(*[chip(w, 22) for w in "Le chat est noir".split()]).arrange(DOWN, buff=0.15).move_to([5.6, -0.2, 0])
        ar = [Arrow(src.get_right(), enc.get_left(), buff=0.12, color=DIM),
              Arrow(enc.get_right(), rep.get_left(), buff=0.12, color=DIM),
              Arrow(rep.get_right(), dec.get_left(), buff=0.12, color=DIM),
              Arrow(dec.get_right(), out.get_left(), buff=0.12, color=DIM)]
        self.play(FadeIn(src))
        self.cue(1)
        self.play(FadeIn(enc), FadeIn(dec))
        self.cue(2)
        self.play(GrowArrow(ar[0]), Indicate(enc, color=NEG, scale_factor=1.05))
        self.play(GrowArrow(ar[1]), LaggedStart(*[FadeIn(s, scale=0.5) for s in rep], lag_ratio=0.05))
        lab = txt("intermediate representation (numbers that capture meaning)", 18, HI).move_to([0, -2.0, 0])
        self.play(FadeIn(lab))
        self.cue(3)
        self.play(GrowArrow(ar[2]), Indicate(dec, color=POS, scale_factor=1.05))
        self.play(GrowArrow(ar[3]), LaggedStart(*[FadeIn(o, shift=LEFT * 0.2) for o in out], lag_ratio=0.2))
        self.cue(4)
        big = txt("each half: an enormous neural network", 22, DIM).move_to([0, 2.2, 0])
        self.play(FadeIn(big), Indicate(enc[0], color=NEG), Indicate(dec[0], color=POS))
        self.cue(5)
        self.play(VGroup(src, enc, rep, ar[0], ar[1], lab).animate.set_opacity(0.2),
                  Transform(big, txt("many popular LLMs: decoder only — they just keep writing", 22, POS).move_to(big)))
        self.finish()
        self.clear_all()

    def s14_selfq(self):
        self.seg(14)
        h = header("Self-attention")
        self.play(FadeIn(h))
        self.cue(1)
        row = VGroup(*[chip(w, 24) for w in "the cat sat on the mat".split()]).arrange(RIGHT, buff=0.12).move_to([0, -0.4, 0])
        self.play(FadeIn(row))
        self.cue(2)
        q = txt("“How much does each other token affect\nthe way I should interpret this one?”", 26, HI).shift(UP * 1.9)
        tgt = row[1]
        self.play(FadeIn(q), tgt[0].animate.set_stroke(HI, 3))
        arcs = VGroup(*[arc(tgt, row[j], 0.6) for j in range(len(row)) if j != 1])
        self.play(LaggedStart(*[Create(a) for a in arcs], lag_ratio=0.15))
        self.cue(3)
        sl = txt("self = the tokens of its own input", 22, NEG).next_to(row, DOWN, buff=0.4)
        self.play(FadeIn(sl))
        self.cue(4)
        self.play(FadeOut(VGroup(q, arcs)), VGroup(row, sl).animate.shift(UP * 1.2))
        fr = VGroup(*[chip(w, 24, stroke=POS) for w in "le chat".split()]).arrange(RIGHT, buff=0.12).move_to([0, -2.0, 0])
        cross = VGroup(*[Line(f.get_top(), r.get_bottom(), color=POS, stroke_width=2, stroke_opacity=0.6)
                         for f in fr for r in row[:3]])
        cl = txt("other attention: compares to a different sequence (e.g. the translation)", 18, POS)
        cl.next_to(fr, DOWN, buff=0.25)
        self.play(FadeIn(fr), Create(cross), FadeIn(cl))
        self.cue(5)
        self.play(FadeOut(VGroup(fr, cross, cl)), Indicate(sl, color=NEG))
        self.finish()
        self.clear_all()

    def _sent_row(self, words, y=-2.0):
        r = VGroup(*[chip(w, 21, pad=0.12) for w in words]).arrange(RIGHT, buff=0.1)
        if r.width > 13.2:
            r.scale_to_fit_width(13.2)
        return r.move_to([0, y, 0])

    def s15_animal(self):
        self.seg(15)
        h = header("Which word does “it” mean?")
        self.play(FadeIn(h))
        self.cue(1)
        r1 = self._sent_row(SENT)
        self.play(LaggedStart(*[FadeIn(c, shift=UP * 0.15) for c in r1], lag_ratio=0.06), run_time=1.3)
        self.cue(2)
        alls = VGroup(*[arc(r1[i], r1[j], 0.15, angle=0.5 * PI) for i in range(11) for j in range(i + 1, 11)])
        cnt = txt("11 words × 10 others = 110 attention scores", 22, DIM).shift(UP * 2.3)
        self.play(Create(alls), FadeIn(cnt), run_time=1.5)
        self.cue(3)
        self.play(FadeOut(alls), r1[IT][0].animate.set_stroke(HI, 3).set_fill(HI, 0.25))
        self.cue(5)
        ask = txt("it  =  the animal?  or the street?", 28, HI).move_to(cnt)
        self.play(Transform(cnt, ask))
        self.cue(6)
        a1 = VGroup(*[arc(r1[IT], r1[j], w) for j, w in enumerate(W_TIRED) if j != IT])
        foot = txt("line thickness = attention weight (illustrative)", 16, DIM).to_edge(DOWN, buff=0.3)
        self.play(LaggedStart(*[Create(a) for a in a1], lag_ratio=0.08), FadeIn(foot), run_time=1.3)
        self.play(r1[1][0].animate.set_stroke(NEG, 3).set_fill(NEG, 0.3),
                  Transform(cnt, txt("it  →  the animal", 30, NEG).move_to(cnt)))
        self.cue(8)
        r2 = self._sent_row(SENT[:-1] + ["wide"]).align_to(r1, LEFT)
        r2[IT][0].set_stroke(HI, 3).set_fill(HI, 0.25)
        r2[5][0].set_stroke(NEG, 3).set_fill(NEG, 0.3)
        a2 = VGroup(*[arc(r2[IT], r2[j], w) for j, w in enumerate(W_WIDE) if j != IT])
        self.play(Transform(r1, r2), Transform(a1, a2), run_time=1.2)
        self.cue(9)
        self.play(Transform(cnt, txt("it  →  the street", 30, NEG).move_to(cnt)))
        self.cue(10)
        self.play(Indicate(r1[-1], color=HI, scale_factor=1.2))
        self.finish()
        self.clear_all()

    def s16_math(self):
        self.seg(16)
        h = header("How attention scores are computed")
        self.play(FadeIn(h))
        self.cue(1)
        cols = {"it": HI, "animal": NEG, "street": POS, "tired": GOOD}
        lists = VGroup(*[txt(f"{w}  =  ({VEC[w][0]:.1f}, {VEC[w][1]:.1f})", 22, cols[w]) for w in ["it"] + OTHERS])
        lists.arrange(DOWN, aligned_edge=LEFT, buff=0.18).move_to([3.4, 0.9, 0])
        self.play(FadeIn(lists))
        self.cue(2)
        ax = Axes(x_range=[-0.4, 1.8, 0.5], y_range=[-0.8, 1.4, 0.5], x_length=4.6, y_length=4.4,
                  axis_config={"color": EDGE, "include_ticks": False, "include_tip": False}).move_to([-3.6, -0.6, 0])
        vecs = {}
        for w in ["it"] + OTHERS:
            a = Arrow(ax.c2p(0, 0), ax.c2p(*VEC[w]), buff=0, color=cols[w], stroke_width=5,
                      max_tip_length_to_length_ratio=0.15)
            lab = txt(w, 18, cols[w]).next_to(a.get_end(), UR if w != "street" else DR, buff=0.05)
            vecs[w] = VGroup(a, lab)
        self.play(Create(ax), *[GrowArrow(vecs[w][0]) for w in vecs], *[FadeIn(vecs[w][1]) for w in vecs])
        self.cue(3)
        self.play(FadeOut(lists))
        dot1 = txt("it · animal  =  1.0 × 1.4  +  1.0 × 0.6  =  2.0", 22).move_to([2.9, 0.9, 0])
        self.play(FadeIn(dot1))
        self.cue(4)
        dl = txt("a dot product", 20, HI).next_to(dot1, DOWN, buff=0.15)
        self.play(FadeIn(dl))
        self.cue(5)
        same = txt("same direction → high score", 18, DIM).next_to(dl, DOWN, buff=0.1)
        self.play(FadeIn(same), Indicate(vecs["animal"], color=NEG))
        # score table (fades in as the dot-product notes fade out)
        hdr = VGroup(*[txt(s, 18, DIM) for s in ["word", "score", "e^score", "weight"]])
        xs = [1.2, 2.7, 4.3, 5.9]
        for t, x in zip(hdr, xs):
            t.move_to([x, 1.1, 0])
        rows = {}
        order = [0, 2, 1]                                    # animal, tired, street as the narration reads them
        for rank, i in enumerate(order):
            y = 0.5 - 0.55 * rank
            w = OTHERS[i]
            rows[i] = [txt(w, 22, cols[w]).move_to([xs[0], y, 0]), txt(f"{SCORES[i]:.1f}", 22).move_to([xs[1], y, 0]),
                       txt(f"{EXPS[i]:.2f}", 22, DIM).move_to([xs[2], y, 0]),
                       txt(f"{WEIGHTS[i]:.2f}", 24, HI).move_to([xs[3], y, 0])]
        self.play(FadeOut(VGroup(dot1, dl, same)), FadeIn(hdr[:2]), run_time=0.7)
        for k, i in enumerate(order):
            self.cue(6 + k)
            self.play(FadeIn(rows[i][0]), FadeIn(rows[i][1]), Indicate(vecs[OTHERS[i]][0], color=cols[OTHERS[i]]),
                      run_time=0.5)
        self.cue(9)
        sm = txt(f"softmax: weight = e^score ÷ sum  ({sum(EXPS):.2f})", 20, HI).move_to([3.5, -1.6, 0])
        self.play(FadeIn(hdr[2:]), FadeIn(sm), *[FadeIn(rows[i][2]) for i in order])
        self.cue(10)
        self.play(*[FadeIn(rows[i][3], shift=LEFT * 0.2) for i in order])
        sum1 = txt("weights add up to 1", 18, DIM).next_to(sm, DOWN, buff=0.15)
        self.play(FadeIn(sum1))
        self.cue(11)
        bl = VGroup(txt("new it = 0.63·animal + 0.23·tired + 0.14·street", 18),
                    txt(f"= ({BLEND[0]:.2f}, {BLEND[1]:.2f})", 18)).arrange(DOWN, buff=0.08).move_to([3.5, -2.75, 0])
        new = Arrow(ax.c2p(0, 0), ax.c2p(*BLEND), buff=0, color=WHITE, stroke_width=8,
                    max_tip_length_to_length_ratio=0.2).set_z_index(5)
        self.play(FadeIn(bl))
        self.play(GrowArrow(new), *[Indicate(vecs[w][0], color=WHITE) for w in OTHERS], run_time=1.0)
        self.cue(12)
        nl = txt("new “it”", 18).next_to(ax.c2p(*BLEND), UP + LEFT * 0.3, buff=0.15).set_z_index(5)
        self.play(FadeIn(nl), vecs["it"].animate.set_opacity(0.3))
        self.cue(14)
        qk = txt("real models: vectors pass through learned weights first (queries & keys)", 18, DIM)
        qk.to_edge(DOWN, buff=0.12)
        self.play(FadeIn(qk))
        self.finish()
        self.clear_all()

    def s17_dirs(self):
        self.seg(17)
        h = header("One direction, or both")
        self.play(FadeIn(h))

        def demo(x, both, title, color):
            dots = VGroup(*[Dot(radius=0.13, color=WHITE) for _ in range(7)]).arrange(RIGHT, buff=0.55).move_to([x, -0.4, 0])
            dots[3].set_color(HI).scale(1.3)
            if not both:
                for d in dots[4:]:
                    d.set_opacity(0.2)
            arcs = VGroup(*[ArcBetweenPoints(dots[j].get_top() + UP * 0.05, dots[3].get_top() + UP * 0.05,
                                             angle=-0.6 * PI if j < 3 else 0.6 * PI, stroke_color=color, stroke_width=3)
                            for j in range(7) if j != 3 and (both or j < 3)])
            return VGroup(dots, arcs, txt(title, 24, color).move_to([x, 1.6, 0]))

        bi = demo(-3.4, True, "bidirectional", NEG)
        uni = demo(3.4, False, "unidirectional", POS)
        self.cue(1)
        self.play(FadeIn(bi[0]), FadeIn(bi[2]))
        self.play(LaggedStart(*[Create(a) for a in bi[1]], lag_ratio=0.1))
        self.cue(2)
        self.play(FadeIn(uni[0]), FadeIn(uni[2]))
        self.play(LaggedStart(*[Create(a) for a in uni[1]], lag_ratio=0.15))
        self.cue(4)
        qs = VGroup(*[txt("?", 24, DIM).next_to(d, DOWN, buff=0.2) for d in uni[0][4:]])
        nl = txt("not written yet", 18, DIM).next_to(qs, DOWN, buff=0.15)
        self.play(FadeIn(qs), FadeIn(nl))
        self.cue(5)
        el = txt("encoders", 26, NEG).next_to(bi[0], DOWN, buff=1.3)
        el2 = txt("understand a whole sequence", 18, DIM).next_to(el, DOWN, buff=0.1)
        self.play(FadeIn(el), FadeIn(el2))
        self.cue(6)
        dl = txt("decoders", 26, POS).next_to(uni[0], DOWN, buff=1.3)
        dl2 = txt("generate token by token", 18, DIM).next_to(dl, DOWN, buff=0.1)
        self.play(FadeIn(dl), FadeIn(dl2))
        self.finish()
        self.clear_all()

    def s18_heads(self):
        self.seg(18)
        h = header("Many heads, many layers")
        self.play(FadeIn(h))
        rng = np.random.default_rng(3)
        heads = VGroup()
        jobs = ["pronoun → noun", "grammar", "?", "?"]
        for k, col in enumerate([NEG, POS, GOOD, BAD]):
            d = VGroup(*[Dot(radius=0.06, color=WHITE) for _ in range(6)]).arrange(RIGHT, buff=0.22)
            arcs = VGroup()
            for _ in range(4):
                i, j = rng.choice(6, 2, replace=False)
                arcs.add(ArcBetweenPoints(d[i].get_top(), d[j].get_top(), angle=0.6 * PI * np.sign(i - j),
                                          stroke_color=col, stroke_width=2.5))
            g = VGroup(d, arcs)
            box = RoundedRectangle(corner_radius=0.12, width=1.9, height=1.3, stroke_color=col, stroke_width=1.5)
            box.set_fill(PANEL, 1).move_to(g)
            heads.add(VGroup(box, g, txt(f"head {k + 1}", 14, DIM).next_to(box, DOWN, buff=0.08)))
        heads.arrange_in_grid(2, 2, buff=(0.3, 0.45)).move_to([-3.4, -0.2, 0])
        layer = SurroundingRectangle(heads, color=EDGE, buff=0.3, corner_radius=0.2)
        lt = txt("one self-attention layer", 20, DIM).next_to(layer, UP, buff=0.15)
        self.cue(2)
        self.play(FadeIn(layer), FadeIn(lt), LaggedStart(*[FadeIn(x, scale=0.8) for x in heads], lag_ratio=0.2))
        self.cue(3)
        rnd = txt("each starts random → learns its own job", 18, HI).next_to(layer, DOWN, buff=0.2)
        self.play(FadeIn(rnd))
        jl = VGroup()
        for k in (0, 1):
            self.cue(4 + k)
            t = txt(jobs[k], 14, HI).move_to(heads[k][2])
            jl.add(t)
            self.play(Transform(heads[k][2], t), Indicate(heads[k][0], color=HI))
        self.cue(6)
        comb = Arrow(layer.get_right(), [2.3, layer.get_y(), 0], color=DIM, buff=0.1)
        cl = txt("combine heads\n(e.g. weighted average)", 16, DIM).next_to(comb, DOWN, buff=0.1)
        self.play(GrowArrow(comb), FadeIn(cl))
        self.cue(7)
        stack = VGroup(*[VGroup(RoundedRectangle(corner_radius=0.12, width=3.0, height=0.55, stroke_color=EDGE,
                                                 stroke_width=1.5)
                                .set_fill(interpolate_color(ManimColor(NEG), ManimColor(POS), k / 3), 0.35),
                                txt(f"layer {k + 1}", 18)) for k in range(4)]).arrange(UP, buff=0.3)
        stack.move_to([3.9, -0.2, 0])
        ups = VGroup(*[Arrow(stack[k].get_top(), stack[k + 1].get_bottom(), buff=0.02, color=DIM, stroke_width=2,
                             max_tip_length_to_length_ratio=0.4) for k in range(3)])
        self.play(LaggedStart(*[FadeIn(s, shift=UP * 0.2) for s in stack], lag_ratio=0.25))
        self.cue(8)
        self.play(LaggedStart(*[GrowArrow(u) for u in ups], lag_ratio=0.3))
        self.cue(9)
        lo = txt("basic syntax", 18, NEG).next_to(stack, DOWN, buff=0.2)
        self.play(FadeIn(lo))
        self.cue(10)
        hi = txt("sentiment · context · themes", 18, POS).next_to(stack, UP, buff=0.2)
        self.play(FadeIn(hi))
        self.finish()
        self.clear_all()

    def s19_size(self):
        self.seg(19)
        h = header("Why are Transformers so large?")
        self.play(FadeIn(h))
        self.cue(1)
        big = txt("100,000,000,000", 54, HI).shift(UP * 1.5)
        unit = txt("parameters  (hundreds of billions)", 22, DIM).next_to(big, DOWN, buff=0.2)
        self.play(FadeIn(big), FadeIn(unit))
        self.play(Transform(big, txt("1,000,000,000,000", 54, HI).move_to(big)),
                  Transform(unit, txt("parameters  (trillions)", 22, DIM).move_to(unit)), run_time=1.0)
        self.cue(2)
        rule = boxed("usual advice: smaller models", "fewer resources per prediction", DIM, w=5.4, h=1.4)
        rule.move_to([-3.4, -1.5, 0])
        self.play(FadeIn(rule))
        self.cue(3)
        ax = Axes(x_range=[0, 10, 1], y_range=[0, 10, 1], x_length=4.4, y_length=2.2,
                  axis_config={"color": DIM, "include_ticks": False, "tip_length": 0.18}).move_to([3.3, -1.5, 0])
        xl = txt("parameters", 16, DIM).next_to(ax.x_axis, DOWN, buff=0.12)
        yl = txt("quality", 16, DIM).next_to(ax.y_axis, LEFT, buff=0.12)
        self.play(Create(ax), FadeIn(xl), FadeIn(yl))
        self.cue(4)
        curve = ax.plot(lambda x: 0.9 * x + 0.5, x_range=[0, 9.5], color=GOOD, stroke_width=4)
        ql = txt("Transformers: bigger keeps winning (qualitative)", 16, GOOD).next_to(ax, UP, buff=0.15)
        self.play(Create(curve), FadeIn(ql), rule.animate.set_opacity(0.4))
        self.finish()
        self.clear_all()

    # ================================================================ chapter 5
    def s20_train(self):
        self.seg(20)
        h = header("How an LLM learns")
        self.play(FadeIn(h))
        self.cue(1)
        game = txt("the guessing game, billions of times", 24, DIM).shift(UP * 2.2)
        self.play(FadeIn(game))
        self.cue(2)
        ws = "The cat sat on the mat".split()
        row = VGroup(*[chip(w, 28) for w in ws]).arrange(RIGHT, buff=0.12).shift(UP * 1.0)
        self.play(FadeIn(row))
        hid = chip("?", 28, HI, stroke=HI).move_to(row[-1])
        real = row[-1].copy()
        self.play(Transform(row[-1], hid))
        self.cue(3)
        guess = VGroup(txt("model guesses:", 22, DIM), chip("sofa", 26, stroke=BAD)).arrange(RIGHT, buff=0.3)
        guess.move_to([-2.6, -0.4, 0])
        ans = VGroup(txt("real word:", 22, DIM), real.copy().set_stroke(GOOD)).arrange(RIGHT, buff=0.3)
        ans.move_to([2.6, -0.4, 0])
        self.play(FadeIn(guess), FadeIn(ans))
        net = network(layers=(3, 4, 3), w=3.0, h=1.4, center=[0, -2.2, 0])
        nudge = txt("nudge parameters (backpropagation)", 18, HI).next_to(net, RIGHT, buff=0.4)
        rng = np.random.default_rng(2)
        self.play(FadeIn(net), FadeIn(nudge))
        self.play(*[e.animate.set_stroke(width=1 + 2.5 * rng.random()) for e in net[0]], run_time=0.7)
        self.cue(4)
        self.play(FadeOut(VGroup(net, nudge)))
        nl = txt("no human labels needed", 26, GOOD).move_to([0, -1.9, 0])
        self.play(FadeIn(nl))
        self.cue(5)
        ul = txt("the text supplies its own answers  →  unsupervised learning", 22, HI).next_to(nl, DOWN, buff=0.3)
        self.play(FadeIn(ul))
        self.finish()
        self.clear_all()

    def s21_max(self):
        self.seg(21)
        h = header("Generation is autocomplete")
        self.play(FadeIn(h))
        self.cue(1)
        ac = txt("autocomplete?", 36, DIM)
        self.play(FadeIn(ac))
        self.cue(3)
        self.play(Transform(ac, txt("autocomplete for thousands of tokens", 30, HI)))
        self.cue(4)
        self.play(ac.animate.scale(0.6).move_to([0, 2.3, 0]).set_opacity(0.6))
        given = txt("My dog, Max, knows how to perform many traditional dog tricks.", 24).move_to([0, 1.3, 0])
        mask = chip("____________ (masked sentence)", 22, DIM).next_to(given, DOWN, buff=0.3)
        self.play(FadeIn(given), FadeIn(mask))
        self.cue(5)
        rows = [("For example, he can sit, stay, and roll over.", 3.1),
                ("For example, he knows how to sit, stay, and roll over.", 2.9)]
        out = VGroup()
        for s, p in rows:
            r = VGroup(txt(f"{p}%", 26, HI), txt(s, 22)).arrange(RIGHT, buff=0.4)
            out.add(r)
        out.arrange(DOWN, aligned_edge=LEFT, buff=0.35).move_to([0, -1.4, 0])
        self.cue(6)
        self.play(FadeIn(out[0], shift=UP * 0.2))
        self.cue(7)
        self.play(FadeIn(out[1], shift=UP * 0.2))
        self.finish()
        self.clear_all()

    def s22_chat(self):
        self.seg(22)
        h = header("A chat is the same game")
        self.play(FadeIn(h))
        self.cue(1)
        qtext = txt("What is the easiest trick to teach a dog?", 28)
        qcard = VGroup(card(qtext.width + 0.8, 1.0, ORIGIN), qtext).shift(UP * 1.8)
        lab = txt("given text", 16, DIM).next_to(qcard, UP, buff=0.1)
        blank = chip("________", 26, DIM).move_to([-2.6, 0.5, 0], aligned_edge=LEFT)
        bl = txt("imaginary blank", 16, DIM).next_to(blank, DOWN, buff=0.1)
        self.play(FadeIn(lab), FadeIn(blank), FadeIn(bl))
        self.cue(2)
        self.play(FadeIn(qcard, shift=UP * 0.2))
        self.cue(3)
        self.play(FadeOut(VGroup(blank, bl)))
        steps = [["Sit", "Shake", "Stay"], ["is", "usually", "."], ["usually", "the", "often"],
                 ["first.", "easiest.", "best."]]
        cursor = [-2.6, 0.5, 0]
        for opts in steps:
            col = VGroup(*[chip(o, 20) for o in opts]).arrange(DOWN, buff=0.1).move_to([0, -1.4, 0])
            self.play(FadeIn(col), run_time=0.25)
            self.play(col[0][0].animate.set_stroke(GOOD, 3).set_fill(GOOD, 0.25), run_time=0.2)
            tok = chip(opts[0], 26, fill=PANEL, stroke=GOOD).move_to(cursor, aligned_edge=LEFT)
            self.play(TransformFromCopy(col[0], tok), FadeOut(col), run_time=0.4)
            cursor = tok.get_right() + RIGHT * 0.12
        self.cue(4)
        self.clear_all(keep=[h])
        wp = txt("A train leaves at 3 pm going 60 km/h. When has it gone 150 km?", 22).shift(UP * 1.0)
        wpa = txt("At 5:30 pm.", 26, GOOD).next_to(wp, DOWN, buff=0.4)
        self.play(FadeIn(wp))
        self.play(FadeIn(wpa))
        self.cue(5)
        lk = txt("looks like reasoning  ·  underneath: autocompleting the problem", 22, HI).to_edge(DOWN, buff=0.8)
        self.play(FadeIn(lk))
        self.finish()
        self.clear_all()

    def s23_benefits(self):
        self.seg(23)
        h = header("What LLMs are good at")
        self.play(FadeIn(h))
        cards = VGroup(boxed("Clear text", "for almost any audience", GOOD, w=3.9, h=1.8),
                       boxed("Trained tasks", "good predictions", GOOD, w=3.9, h=1.8),
                       boxed("Untrained tasks?", "some claim it, others dispute it", HI, w=4.6, h=1.8))
        cards.arrange(RIGHT, buff=0.3)
        for k in range(3):
            self.cue(1 + k)
            self.play(FadeIn(cards[k], shift=UP * 0.2))
        self.finish()
        self.clear_all()

    def s24_train_probs(self):
        self.seg(24)
        h = header("Problems: training")
        self.play(FadeIn(h))
        a = boxed("Enormous dataset", "huge amounts of text", HI, h=1.4)
        b = boxed("Months of compute", "and lots of electricity", HI, h=1.4)
        c = boxed("Parallelism", "thousands of chips in step", HI, h=1.4)
        VGroup(a, b, c).arrange(RIGHT, buff=0.25).move_to([0, 1.4, 0])
        self.cue(1)
        self.play(FadeIn(a, shift=UP * 0.2))
        self.cue(2)
        self.play(FadeIn(b, shift=UP * 0.2))
        self.cue(3)
        self.play(FadeIn(c, shift=UP * 0.2))
        self.cue(4)
        chips = VGroup(*[RoundedRectangle(corner_radius=0.06, width=0.42, height=0.42, stroke_color=NEG, stroke_width=1.5)
                         .set_fill(NEG, 0.3) for _ in range(40)]).arrange_in_grid(4, 10, buff=0.18).move_to([0, -1.5, 0])
        links = VGroup(*[Line(chips[i].get_center(), chips[i + 1].get_center(), stroke_width=1, color=DIM)
                         .set_z_index(-1) for i in range(39) if (i + 1) % 10])
        tl = txt("e.g. Tensor Processing Units (TPUs) working in parallel", 18, DIM).next_to(chips, DOWN, buff=0.25)
        self.play(LaggedStart(*[FadeIn(x, scale=0.5) for x in chips], lag_ratio=0.02), Create(links), FadeIn(tl))
        self.play(LaggedStart(*[x.animate.set_fill(HI, 0.6) for x in chips], lag_ratio=0.02), run_time=1.0)
        self.finish()
        self.clear_all()

    def s25_use_probs(self):
        self.seg(25)
        h = header("Problems: using the model (inference)")
        self.play(FadeIn(h))
        self.cue(1)
        hal = boxed("Hallucinations", None, BAD, w=4.0, h=1.0).move_to([-3.6, 1.6, 0])
        self.play(FadeIn(hal))
        self.cue(2)
        ex = VGroup(txt("“The Eiffel Tower was finished in 1899.”", 20), txt("fluent · wrong (1889)", 18, BAD))
        ex.arrange(DOWN, buff=0.12).next_to(hal, DOWN, buff=0.25)
        why = txt("likely text ≠ checked text", 18, DIM).next_to(ex, DOWN, buff=0.15)
        self.play(FadeIn(ex), FadeIn(why))
        self.cue(3)
        cost = boxed("Compute & electricity", "every prediction", HI, w=4.4, h=1.3).move_to([3.4, 1.6, 0])
        self.play(FadeIn(cost))
        self.cue(4)
        ax = Axes(x_range=[0, 10, 1], y_range=[0, 10, 1], x_length=4.0, y_length=1.9,
                  axis_config={"color": DIM, "include_ticks": False, "tip_length": 0.15}).move_to([3.4, -1.3, 0])
        xl = txt("training data", 16, DIM).next_to(ax.x_axis, DOWN, buff=0.1)
        self.play(Create(ax), FadeIn(xl))
        self.cue(5)
        up = ax.plot(lambda x: 0.8 * x + 1, x_range=[0, 9.5], color=BAD, stroke_width=4)
        dn = ax.plot(lambda x: 9 - 0.7 * x, x_range=[0, 9.5], color=GOOD, stroke_width=4)
        ul = txt("training cost", 14, BAD).next_to(ax.c2p(9.5, 8.6), UP, buff=0.05)
        dl = txt("cost to run", 14, GOOD).next_to(ax.c2p(9.5, 2.4), DOWN, buff=0.05)
        tq = txt("(qualitative)", 14, DIM).next_to(ax, UP, buff=0.05)
        self.play(Create(up), Create(dn), FadeIn(ul), FadeIn(dl), FadeIn(tq))
        self.cue(6)
        bias = boxed("Bias", "like any ML model", BAD, w=4.0, h=1.3).move_to([-3.6, -2.3, 0])
        self.play(FadeIn(bias))
        self.finish()
        self.clear_all()

    # ================================================================ chapter 6
    def s26_foundation(self):
        self.seg(26)
        h = header("Foundation models")
        self.play(FadeIn(h))
        base = grid(12, 7, 0.3).move_to([-3.4, 0.2, 0])
        self.play(LaggedStart(*[FadeIn(s) for s in base], lag_ratio=0.004), run_time=0.8)
        self.cue(1)
        names = VGroup(*[txt(n, 22, HI) for n in ["foundation model", "base model", "pre-trained model"]])
        names.arrange(DOWN, buff=0.12).next_to(base, DOWN, buff=0.25)
        self.play(LaggedStart(*[FadeIn(n) for n in names], lag_ratio=0.3))
        self.cue(2)
        knows = txt("knows grammar · words · idioms", 22).move_to([3.4, 2.0, 0])
        self.play(FadeIn(knows))
        self.cue(3)
        poem = txt("can write: explanations, even poetry", 22).next_to(knows, DOWN, buff=0.3)
        self.play(FadeIn(poem))
        self.cue(4)
        tasks = VGroup(boxed("Regression", "predict a number", NEG, w=3.0, h=1.2),
                       boxed("Classification", "pick a category", NEG, w=3.0, h=1.2)).arrange(RIGHT, buff=0.3)
        tasks.move_to([3.4, -0.6, 0])
        self.cue(5)
        self.play(FadeIn(tasks[0]))
        self.cue(6)
        self.play(FadeIn(tasks[1]))
        self.cue(7)
        pf = txt("a platform, not a finished solution", 24, HI).next_to(tasks, DOWN, buff=0.4)
        self.play(FadeIn(pf))
        self.finish()
        self.clear_all()

    def s27_finetune(self):
        self.seg(27)
        h = header("Fine-tuning")
        self.play(FadeIn(h))
        base = grid(12, 7, 0.3).move_to([-3.4, 0.3, 0])
        bl = txt("foundation model", 22).next_to(base, DOWN, buff=0.2)
        self.play(FadeIn(base), FadeIn(bl))
        self.cue(1)
        pat = txt("strong pattern-spotting → little extra training", 18, DIM).move_to([3.0, 2.2, 0])
        self.play(FadeIn(pat))
        self.cue(2)
        exs = VGroup(*[chip(s, 18, stroke=c) for s, c in [("“Server down!” → urgent", BAD), ("“Lunch Friday?” → not urgent", GOOD),
                                                         ("“Invoice overdue” → urgent", BAD)]])
        exs.arrange(DOWN, aligned_edge=LEFT, buff=0.12).move_to([3.3, 0.6, 0])
        arr = Arrow(exs.get_left(), base.get_right(), color=GOOD, buff=0.25)
        self.play(LaggedStart(*[FadeIn(e, shift=LEFT * 0.2) for e in exs], lag_ratio=0.2), GrowArrow(arr))
        self.cue(3)
        few = txt("a few hundred to a few thousand examples", 20, GOOD).next_to(exs, DOWN, buff=0.25)
        self.play(FadeIn(few))
        self.cue(4)
        self.play(*[Indicate(e, color=HI, scale_factor=1.04) for e in exs])
        self.cue(5)
        better = txt("better predictions on this task", 22, HI).move_to([3.3, -2.3, 0])
        self.play(FadeIn(better), base.animate.set_fill(GOOD, 0.55))
        self.finish()
        self.clear_all()

    def s28_pet(self):
        self.seg(28)
        h = header("The cost of fine-tuning")
        self.play(FadeIn(h))
        a = grid(12, 7, 0.28).move_to([-3.4, 0.4, 0])
        b = grid(12, 7, 0.28).move_to([3.4, 0.4, 0])
        al = txt("standard fine-tuning", 22, HI).next_to(a, UP, buff=0.2)
        bl = txt("parameter-efficient tuning", 22, GOOD).next_to(b, UP, buff=0.2)
        self.play(FadeIn(a), FadeIn(al))
        self.cue(1)
        for _ in range(2):
            self.play(a.animate.set_fill(HI, 0.9), run_time=0.35)
            self.play(a.animate.set_fill(NEG, 0.55), run_time=0.35)
        an = txt("every parameter, every backprop step", 18, DIM).next_to(a, DOWN, buff=0.2)
        self.play(FadeIn(an))
        self.cue(2)
        self.play(FadeIn(b), FadeIn(bl))
        few = VGroup(*[b[i] for i in np.random.default_rng(1).choice(len(b), 7, replace=False)])
        for _ in range(2):
            self.play(few.animate.set_fill(GOOD, 1), run_time=0.35)
            self.play(few.animate.set_fill(NEG, 0.55), run_time=0.35)
        bn = txt("only a small subset", 18, DIM).next_to(b, DOWN, buff=0.2)
        self.play(FadeIn(bn), few.animate.set_fill(GOOD, 1))
        self.cue(3)
        sz = txt("size?", 30, HI).move_to([0, -2.4, 0])
        self.play(FadeIn(sz))
        self.cue(5)
        same = txt("10 billion parameters  →  still 10 billion parameters", 26, HI).move_to(sz)
        self.play(Transform(sz, same))
        self.finish()
        self.clear_all()

    def s29_distill(self):
        self.seg(29)
        h = header("Distillation")
        self.play(FadeIn(h))
        big = grid(12, 7, 0.3).move_to([-3.6, 0.3, 0])
        bl = txt("fine-tuned model", 22).next_to(big, DOWN, buff=0.2)
        self.play(FadeIn(big), FadeIn(bl))
        self.cue(1)
        rng = np.random.default_rng(4)
        idle = VGroup(*[big[i] for i in rng.choice(len(big), 55, replace=False)])
        il = txt("much of it irrelevant to this task", 18, DIM).next_to(bl, DOWN, buff=0.1)
        self.play(idle.animate.set_fill(DIM, 0.2), FadeIn(il))
        self.cue(2)
        self.play(Transform(bl, txt("teacher", 22).move_to(bl)), big.animate.set_fill(NEG, 0.55))
        self.cue(3)
        small = grid(4, 3, 0.3, POS).move_to([3.4, 0.6, 0])
        sl = txt("student", 22).next_to(small, DOWN, buff=0.2)
        flow = Arrow(big.get_right(), small.get_left(), color=POS, buff=0.3)
        cp = txt("learns to copy its predictions", 16, DIM).next_to(flow, DOWN, buff=0.15)
        self.play(GrowArrow(flow), FadeIn(cp))
        self.play(FadeIn(small, scale=0.6), FadeIn(sl))
        self.cue(4)
        pros = VGroup(txt("much faster", 22, GOOD), txt("far fewer resources", 22, GOOD)).arrange(DOWN, buff=0.12)
        pros.next_to(sl, DOWN, buff=0.3)
        self.play(FadeIn(pros, shift=UP * 0.2))
        self.cue(5)
        con = txt("predictions not quite as good", 22, BAD).next_to(pros, DOWN, buff=0.15)
        self.play(FadeIn(con, shift=UP * 0.2))
        self.finish()
        self.clear_all()

    def s30_prompt(self):
        self.seg(30)
        h = header("Prompt engineering")
        self.play(FadeIn(h))
        who = VGroup(boxed("engineers", "fine-tune · distill", DIM, w=3.6, h=1.3),
                     boxed("users", "write the prompt", HI, w=3.6, h=1.3)).arrange(RIGHT, buff=0.6).shift(UP * 0.3)
        self.play(FadeIn(who[0]))
        self.cue(1)
        self.play(FadeIn(who[1]))
        self.cue(3)
        learn = txt("humans learn from examples — so do LLMs", 22, HI).next_to(who, DOWN, buff=0.4)
        self.play(FadeIn(learn))
        self.cue(4)
        self.play(FadeOut(VGroup(who, learn)))
        task = txt("task: name a fruit's botanical class", 20, DIM).next_to(h, DOWN, buff=0.2)
        self.play(FadeIn(task))

        def column(x, title, lines, ans, ans_color):
            t = txt(title, 26, HI)
            body = VGroup(*[txt(l, 22) for l in lines]).arrange(DOWN, aligned_edge=LEFT, buff=0.2)
            blank = txt("____", 22, DIM).next_to(body[-1], RIGHT, buff=0.15)
            box = card(4.3, 3.6, [x, -0.8, 0])
            t.move_to(box).align_to(box, UP).shift(DOWN * 0.35)
            VGroup(body, blank).next_to(t, DOWN, buff=0.45).align_to(box, LEFT).shift(RIGHT * 0.35)
            out = txt(ans, 22, ans_color).move_to(blank, aligned_edge=LEFT)
            return box, t, body, blank, out

        z = column(-4.5, "zero-shot", ["apple:"], "Apple Inc.?", BAD)
        o = column(0, "one-shot", ["peach: drupe", "apple:"], "pome", GOOD)
        f = column(4.5, "few-shot", ["plum: drupe", "pear: pome", "lemon:"], "hesperidium", GOOD)
        self.cue(5)
        self.play(FadeIn(z[0]), FadeIn(z[1]), FadeIn(z[2]), FadeIn(z[3]))
        self.cue(6)
        self.play(Transform(z[3], z[4]))
        self.cue(7)
        self.play(FadeIn(o[0]), FadeIn(o[1]), FadeIn(o[2]), FadeIn(o[3]))
        self.cue(8)
        self.play(Transform(o[3], o[4]))
        self.cue(9)
        self.play(FadeIn(f[0]), FadeIn(f[1]))
        for k in range(3):
            self.cue(10 + k)
            self.play(FadeIn(f[2][k]), *([FadeIn(f[3])] if k == 2 else []), run_time=0.4)
        self.play(Transform(f[3], f[4]))
        self.cue(13)
        ts = txt("examples ≈ a tiny training set", 18, DIM).next_to(f[0], DOWN, buff=0.15)
        self.play(FadeIn(ts), Indicate(f[2][:2], color=HI))
        self.cue(14)
        lock = txt("parameters unchanged — prompting only uses what the model already knows", 22, HI)
        lock.to_edge(DOWN, buff=0.25)
        self.play(FadeIn(lock))
        self.finish()
        self.clear_all()

    def s31_offline(self):
        self.seg(31)
        h = header("Online vs offline inference")
        self.play(FadeIn(h))
        self.cue(1)
        req = chip("request", 22).move_to([-4.8, 1.5, 0])
        mdl = boxed("LLM", None, NEG, w=1.8, h=1.0, tsize=24).move_to([-1.6, 1.5, 0])
        ans = chip("answer", 22).move_to([1.6, 1.5, 0])
        a1 = Arrow(req.get_right(), mdl.get_left(), buff=0.1, color=DIM)
        a2 = Arrow(mdl.get_right(), ans.get_left(), buff=0.1, color=DIM)
        self.play(FadeIn(req), GrowArrow(a1), FadeIn(mdl), GrowArrow(a2), FadeIn(ans))
        self.cue(2)
        on = txt("online: answer each request the moment it arrives", 20, NEG).move_to([-1.6, 2.3, 0])
        self.play(FadeIn(on))
        self.cue(3)
        slow = txt("huge model → too slow", 20, BAD).next_to(mdl, DOWN, buff=0.2)
        self.play(FadeIn(slow), Wiggle(mdl))
        self.cue(4)
        off = txt("offline (bulk / static) inference", 22, POS).move_to([0, -0.2, 0])
        self.play(FadeIn(off))
        self.cue(5)
        job = boxed("LLM batch job", "once a week", POS, w=3.2, h=1.2).move_to([-4.0, -1.6, 0])
        cache = VGroup(*[chip(s, 16, pad=0.1) for s in ["query → label", "query → label", "query → label"]])
        cache.arrange(DOWN, buff=0.06).move_to([0.2, -1.6, 0])
        cl = txt("cached predictions", 16, DIM).next_to(cache, DOWN, buff=0.1)
        live = chip("live lookup: instant", 18, stroke=GOOD).move_to([4.4, -1.6, 0])
        b1 = Arrow(job.get_right(), cache.get_left(), buff=0.15, color=DIM)
        b2 = Arrow(cache.get_right(), live.get_left(), buff=0.15, color=DIM)
        self.play(FadeIn(job), GrowArrow(b1), FadeIn(cache), FadeIn(cl))
        self.play(GrowArrow(b2), FadeIn(live))
        self.cue(6)
        self.play(Indicate(job, color=POS))
        self.cue(7)
        self.clear_all(keep=[h])
        ex = VGroup(txt("Google Search", 30, HI), txt("800+ synonyms for “Covid vaccine”", 28),
                    txt("in 50+ languages, computed offline", 24, DIM)).arrange(DOWN, buff=0.3).shift(UP * 0.5)
        self.play(LaggedStart(*[FadeIn(e, shift=UP * 0.2) for e in ex], lag_ratio=0.3))
        self.cue(8)
        m = txt("live queries matched against the cached list", 22, GOOD).next_to(ex, DOWN, buff=0.5)
        self.play(FadeIn(m))
        self.finish()
        self.clear_all()

    def s32_bias(self):
        self.seg(32)
        h = header("Use LLMs responsibly")
        self.play(FadeIn(h))
        d1 = boxed("training data", None, DIM, w=3.4, h=1.0, tsize=22).move_to([-4.2, 0.9, 0])
        d2 = boxed("distillation data", None, DIM, w=3.4, h=1.0, tsize=22).move_to([-4.2, -0.9, 0])
        m = boxed("LLM", None, NEG, w=2.0, h=1.4).move_to([0, 0, 0])
        b = boxed("its biases", None, BAD, w=3.0, h=1.0, tsize=24).move_to([4.2, 0, 0])
        self.cue(1)
        self.play(FadeIn(d1), FadeIn(d2), FadeIn(m))
        self.play(GrowArrow(Arrow(d1.get_right(), m.get_left(), color=DIM, buff=0.1)),
                  GrowArrow(Arrow(d2.get_right(), m.get_left(), color=DIM, buff=0.1)))
        self.play(GrowArrow(Arrow(m.get_right(), b.get_left(), color=BAD, buff=0.1)), FadeIn(b))
        self.cue(2)
        f = txt("follow the course's data and fairness guidance", 22, HI).to_edge(DOWN, buff=0.8)
        self.play(FadeIn(f))
        self.finish()
        self.clear_all(keep=[])
        self.play(FadeOut(self.tag), run_time=0.3)
        self.tag = None

    def s33_recap(self):
        self.seg(33)
        h = header("Recap")
        self.play(FadeIn(h))
        lines = [("Language model", "probabilities for the next token"),
                 ("N-grams", "count — but too little context"),
                 ("RNNs", "carry context forward — but it fades"),
                 ("Transformers", "self-attention over all tokens at once"),
                 ("Training", "fill in blanks → autocomplete at scale"),
                 ("Adapting", "fine-tune · distill · prompt"),
                 ("Watch out", "hallucination · cost · bias")]
        items = VGroup()
        for i, (a, b) in enumerate(lines):
            y = 2.1 - 0.62 * i
            items.add(VGroup(txt(a, 24, HI).move_to([-2.6, y, 0], aligned_edge=RIGHT),
                             txt(b, 22).move_to([-2.2, y, 0], aligned_edge=LEFT)))
        for k in range(7):
            self.cue(1 + k)
            self.play(FadeIn(items[k], shift=RIGHT * 0.2), run_time=0.5)
        self.cue(8)
        src = txt("Google Machine Learning Crash Course · “Introduction to Large Language Models” · CC BY 4.0", 16, DIM)
        src.to_edge(DOWN, buff=0.3)
        url = txt("developers.google.com/machine-learning/crash-course/llm", 20, NEG).next_to(src, UP, buff=0.12)
        self.play(FadeIn(url), FadeIn(src))
        self.finish(1.5)
