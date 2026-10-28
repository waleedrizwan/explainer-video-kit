"""The System Design Primer (github.com/donnemartin/system-design-primer), explained in ~4 minutes.

    python narration.py
    manim -pql scene.py Explainer      # fast preview (480p)
    manim -qh scene.py Explainer       # final (1080p)

Every number on screen comes from the primer's README (availability tables, latency numbers).
"""
import sys
from pathlib import Path

from manim import *

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if (p / "vidkit").is_dir())))
from vidkit import NarratedScene  # noqa: E402
from vidkit.look import *  # noqa: E402,F403

PURPLE_ = "#B48CFF"


# ---------------------------------------------------------------- building pieces
def fit(t, w):
    return t.scale_to_fit_width(w) if t.width > w else t


def node(label, color=NEG, w=2.0, h=0.8, size=20):
    r = RoundedRectangle(corner_radius=0.15, width=w, height=h, stroke_color=color, stroke_width=2.5).set_fill(PANEL, 1)
    return VGroup(r, fit(txt(label, size), w - 0.3))


def chip(label, color=DIM, size=16):
    t = txt(label, size)
    r = RoundedRectangle(corner_radius=0.22, width=t.width + 0.45, height=0.5, stroke_color=color,
                         stroke_width=2).set_fill(PANEL, 1)
    return VGroup(r, t)


def person(color=WHITE, s=1.0):
    head = Circle(0.13, stroke_width=0).set_fill(color, 1).shift(UP * 0.2)
    pts = [np.array([0.24 * np.cos(a), 0.24 * np.sin(a) - 0.22, 0]) for a in np.linspace(0, PI, 20)]
    body = Polygon(*pts, stroke_width=0).set_fill(color, 1)
    return VGroup(head, body).scale(s)


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


def banner(s, color=HI, y=-3.3):
    t = txt(s, 22, color)
    return VGroup(panel(t, buff=0.2), t).move_to(UP * y)


def pos(m):
    return m.get_center()


class Explainer(NarratedScene):
    project_dir = Path(__file__).parent

    # ---- helpers
    def wipe(self, rt=0.5):
        if self.mobjects:
            self.play(*[FadeOut(m) for m in self.mobjects], run_time=rt)
        self.clear()

    def trip(self, pts, color=HI, r=0.1):
        """Animation of a dot moving through points; the dot is removed afterwards."""
        d = Dot(pts[0], radius=r, color=color).set_z_index(10)
        path = VMobject().set_points_as_corners([np.array(p, dtype=float) for p in pts])
        return Succession(FadeIn(d, run_time=0.01), MoveAlongPath(d, path), FadeOut(d, run_time=0.01))

    def construct(self):
        for i in range(11):
            getattr(self, f"s{i}")()

    # ---------------------------------------------------------------- 0 intro
    def s0(self):
        self.seg(0)
        title = txt("The System Design Primer", 52, WHITE, weight=BOLD).shift(UP * 1.3)
        sub = txt("github.com/donnemartin/system-design-primer", 22, DIM).next_to(title, DOWN, buff=0.3)
        self.play(Write(title), run_time=1.6)
        self.play(FadeIn(sub, shift=UP * 0.1))
        self.cue(1)
        star = Star(5, outer_radius=0.32, inner_radius=0.14, stroke_width=0).set_fill(HI, 1)
        count = ValueTracker(0)
        num = always_redraw(lambda: txt(f"{int(count.get_value()):,}", 40, HI).next_to(star, RIGHT, buff=0.25))
        stars = VGroup(star, txt("stars on GitHub", 24, DIM)).shift(DOWN * 0.3)
        lbl = stars[1]
        self.add(num)
        row = VGroup(star, num, lbl)
        star.move_to(LEFT * 2.9 + DOWN * 0.6)
        final_w = txt("375,840", 40).width
        lbl.next_to(star, RIGHT, buff=0.25 + final_w + 0.3)
        self.play(FadeIn(star, scale=0.5), FadeIn(lbl), count.animate.set_value(375840), run_time=2.5,
                  rate_func=rate_functions.ease_out_cubic)
        self.cue(2)
        tags = VGroup(chip("free", GOOD), chip("open source", GOOD)).arrange(RIGHT, buff=0.3).shift(DOWN * 1.3)
        self.play(LaggedStart(*[FadeIn(t, shift=UP * 0.15) for t in tags], lag_ratio=0.3))
        self.cue(3)
        num.clear_updaters()
        self.play(VGroup(title, sub).animate.shift(UP * 1.1), FadeOut(row), FadeOut(tags), run_time=0.6)
        g1 = VGroup(txt("Goal 1", 18, DIM), txt("Design systems", 30), txt("that serve millions", 30))
        g1.arrange(DOWN, buff=0.15)
        g2 = VGroup(txt("Goal 2", 18, DIM), txt("Walk into the", 30), txt("interview prepared", 30))
        g2.arrange(DOWN, buff=0.15)
        c1 = VGroup(card(5.4, 2.6, ORIGIN), g1).move_to(LEFT * 3.1 + DOWN * 0.7)
        c2 = VGroup(card(5.4, 2.6, ORIGIN), g2).move_to(RIGHT * 3.1 + DOWN * 0.7)
        c1[0].set_stroke(NEG, 2.5)
        c2[0].set_stroke(POS, 2.5)
        self.play(FadeIn(c1, shift=UP * 0.2))
        self.cue(4)
        self.play(FadeIn(c2, shift=UP * 0.2))
        self.finish()

    # ---------------------------------------------------------------- 1 the map
    def s1(self):
        self.wipe()
        self.seg(1)
        cols = [
            ("Trade-offs", HI, ["Performance vs scalability", "Latency vs throughput", "Availability vs consistency",
                                "Consistency patterns", "Availability patterns"]),
            ("Building blocks", NEG, ["DNS", "CDN", "Load balancer", "Reverse proxy", "Application layer",
                                      "Database", "Cache", "Asynchronism"]),
            ("Communication", GOOD, ["HTTP", "TCP", "UDP", "RPC", "REST", "Security"]),
        ]
        rng = np.random.default_rng(7)
        heads, groups, everything = [], [], []
        for c, (name, color, items) in enumerate(cols):
            x = (c - 1) * 4.6
            heads.append(txt(name, 26, color).move_to([x, 2.55, 0]))
            g = VGroup(*[chip(s) for s in items])
            for k, ch in enumerate(g):
                ch.move_to([x, 1.75 - k * 0.6, 0])
                ch.generate_target()
                ch.target[0].set_stroke(color, 2)
                ch.move_to([rng.uniform(-5.6, 5.6), rng.uniform(-3.2, 3.2), 0]).rotate(rng.uniform(-0.35, 0.35))
                ch.set_opacity(0.55)
                everything.append(ch)
            groups.append(g)
        self.play(LaggedStart(*[FadeIn(ch, scale=0.7) for ch in everything], lag_ratio=0.05), run_time=2.2)
        self.cue(1)
        self.play(LaggedStart(*[MoveToTarget(ch) for ch in everything], lag_ratio=0.02), run_time=2.0)
        for k in range(3):
            self.cue(k + 2)
            self.play(FadeIn(heads[k], shift=DOWN * 0.15),
                      LaggedStart(*[Indicate(ch, color=cols[k][1], scale_factor=1.06) for ch in groups[k]],
                                  lag_ratio=0.12), run_time=1.6)
        self.finish()

    # ---------------------------------------------------------------- 2 request journey
    def s2(self):
        self.wipe()
        self.seg(2)
        h = header("How the blocks fit together")
        user = VGroup(person(), txt("User", 18, DIM)).arrange(DOWN, buff=0.12).move_to([-6.0, 0, 0])
        self.play(FadeIn(h), FadeIn(user, shift=RIGHT * 0.2))

        self.cue(1)
        dns = node("DNS", PURPLE_, 1.6).move_to([-3.6, 2.2, 0])
        a_dns = arrow(user, dns)
        ip = txt("example.com  →  93.184.216.34", 16, DIM).next_to(dns, RIGHT, buff=0.3)
        self.play(FadeIn(dns), GrowArrow(a_dns))
        self.play(self.trip([pos(user), pos(dns), pos(user)], PURPLE_), FadeIn(ip, run_time=0.6), run_time=1.6)

        self.cue(2)
        cdn = node("CDN", POS, 1.6).move_to([-3.6, -2.2, 0])
        a_cdn = arrow(user, cdn)
        files = txt("images · CSS · video", 16, DIM).next_to(cdn, RIGHT, buff=0.3)
        self.play(FadeIn(cdn), GrowArrow(a_cdn))
        self.play(self.trip([pos(user), pos(cdn), pos(user)], POS), FadeIn(files, run_time=0.6), run_time=1.6)

        self.cue(3)
        lb = node("Load balancer", NEG, 2.3).move_to([-2.9, 0, 0])
        web = VGroup(*[node("Web server", NEG, 1.9, 0.62, 17).move_to([0.0, y, 0]) for y in (1.0, 0, -1.0)])
        a_lb = arrow(user, lb)
        a_web = VGroup(*[arrow(lb, w, buff=0.08) for w in web])
        self.play(FadeIn(lb), GrowArrow(a_lb), run_time=0.7)
        self.play(LaggedStart(*[FadeIn(w) for w in web], lag_ratio=0.2), *[GrowArrow(a) for a in a_web])
        self.play(*[self.trip([pos(user), pos(lb), pos(w)]) for w in web], run_time=1.4)

        self.cue(4)
        cache = node("Cache", BAD, 1.6).move_to([2.9, 1.7, 0])
        dbase = db("Database").move_to([5.6, 0.0, 0])
        a_cache = arrow(web, cache)
        a_db = arrow(web, dbase)
        self.play(FadeIn(cache), FadeIn(dbase), GrowArrow(a_cache), GrowArrow(a_db))
        self.play(self.trip([pos(web[1]), pos(cache)], HI), run_time=0.7)
        miss = txt("miss", 18, BAD).next_to(cache, UP, buff=0.12)
        self.play(FadeIn(miss, scale=1.3), run_time=0.4)
        self.play(self.trip([pos(web[1]), pos(dbase), pos(web[1])], HI), run_time=1.4)
        self.play(FadeOut(miss), run_time=0.3)

        self.cue(5)
        queue = VGroup(node("", DIM, 1.9, 0.62), txt("Queue", 16, DIM)).move_to([2.9, -1.9, 0])
        queue[1].next_to(queue[0], DOWN, buff=0.1)
        workers = node("Workers", GOOD, 1.7).move_to([5.6, -1.9, 0])
        a_q = arrow(web, queue[0])
        a_w = arrow(queue[0], workers)
        self.play(FadeIn(queue), FadeIn(workers), GrowArrow(a_q), GrowArrow(a_w))
        jobs = VGroup(*[Square(0.2, stroke_width=0).set_fill(HI, 1) for _ in range(4)])
        anims = []
        for k, j in enumerate(jobs):
            slot = queue[0].get_center() + RIGHT * (0.6 - k * 0.4)
            anims.append(Succession(Wait(k * 0.35), FadeIn(j.move_to(web[2]), run_time=0.1),
                                    j.animate(run_time=0.7).move_to(slot)))
        self.play(*anims, run_time=2.2)
        self.play(LaggedStart(*[j.animate.move_to(workers).set_opacity(0) for j in jobs], lag_ratio=0.25),
                  run_time=1.8)
        self.finish()

    # ---------------------------------------------------------------- 3 definitions
    def s3(self):
        self.wipe()
        self.seg(3)
        lc = card(6.5, 4.9, [-3.45, -0.2, 0])
        rc = card(6.5, 4.9, [3.45, -0.2, 0])
        lt = txt("Performance vs scalability", 22, HI).move_to([-3.45, 1.9, 0])
        rt = txt("Latency vs throughput", 22, HI).move_to([3.45, 1.9, 0])
        self.play(FadeIn(lc), FadeIn(rc), FadeIn(lt), FadeIn(rt))

        self.cue(1)
        srv1 = node("Server", NEG, 1.5, 0.7).move_to([-1.6, 0.85, 0])
        p1 = person(s=0.9).move_to([-5.3, 0.85, 0])
        a1 = arrow(p1, srv1)
        slow = txt("1 user: slow", 18, BAD).move_to([-3.45, 0.2, 0])
        perf = txt("→ performance problem", 18).next_to(slow, DOWN, buff=0.12)
        self.play(FadeIn(p1), FadeIn(srv1), GrowArrow(a1))
        self.play(self.trip([pos(p1), pos(srv1)], BAD), run_time=2.0)
        self.play(FadeIn(slow), FadeIn(perf))

        self.cue(2)
        crowd = VGroup(*[person(s=0.55) for _ in range(9)]).arrange_in_grid(3, 3, buff=0.12).move_to([-5.3, -1.15, 0])
        srv2 = node("Server", NEG, 1.5, 0.7).move_to([-1.6, -1.15, 0])
        a2 = arrow(crowd, srv2)
        sc = txt("fast for 1, slow under load", 18, BAD).move_to([-3.45, -1.95, 0])
        scal = txt("→ scalability problem", 18).next_to(sc, DOWN, buff=0.12)
        self.play(LaggedStart(*[FadeIn(p) for p in crowd], lag_ratio=0.05), FadeIn(srv2), GrowArrow(a2))
        self.play(*[self.trip([pos(p), pos(srv2)], BAD, 0.06) for p in crowd], run_time=1.6)
        self.play(FadeIn(sc), FadeIn(scal))

        self.cue(3)
        pipe1 = Line([1.0, 0.85, 0], [5.9, 0.85, 0], color=EDGE, stroke_width=14)
        self.play(Create(pipe1), run_time=0.5)
        pk = Square(0.26, stroke_width=0).set_fill(HI, 1).move_to(pipe1.get_start())
        self.add(pk)
        self.play(pk.animate.move_to(pipe1.get_end()), run_time=1.3, rate_func=linear)
        br = BraceBetweenPoints(pipe1.get_start(), pipe1.get_end(), DOWN, color=DIM).shift(DOWN * 0.05)
        lat = txt("latency = time for one action", 18).next_to(br, DOWN, buff=0.1)
        self.play(FadeOut(pk), GrowFromCenter(br), FadeIn(lat))

        self.cue(4)
        pipe2 = Line([1.0, -1.15, 0], [5.9, -1.15, 0], color=EDGE, stroke_width=14)
        self.play(Create(pipe2), run_time=0.4)
        pks = [Square(0.2, stroke_width=0).set_fill(GOOD, 1).move_to(pipe2.get_start()) for _ in range(7)]
        self.play(LaggedStart(*[Succession(FadeIn(p, run_time=0.01), p.animate(rate_func=linear).move_to(pipe2.get_end()),
                                           FadeOut(p, run_time=0.01)) for p in pks], lag_ratio=0.18), run_time=2.2)
        thr = txt("throughput = actions per second", 18).move_to([3.45, -1.8, 0])
        self.play(FadeIn(thr))

        self.cue(5)
        b = banner("Aim for maximal throughput with acceptable latency")
        self.play(FadeIn(b, shift=UP * 0.2))
        self.finish()

    # ---------------------------------------------------------------- 4 CAP
    def s4(self):
        self.wipe()
        self.seg(4)
        h = header("The CAP theorem")
        self.play(FadeIn(h))

        self.cue(1)
        V = {"C": np.array([-3.7, 1.7, 0]), "A": np.array([-5.7, -1.5, 0]), "P": np.array([-1.7, -1.5, 0])}
        edges = {k: Line(V[k[0]], V[k[1]], color=EDGE, stroke_width=6) for k in ("CA", "CP", "AP")}
        dots = {k: VGroup(Circle(0.38, stroke_color=WHITE, stroke_width=2.5).set_fill(PANEL, 1),
                          txt(k, 26, WHITE, weight=BOLD)).move_to(v) for k, v in V.items()}
        names = {"C": txt("Consistency", 18).next_to(dots["C"], UP, buff=0.15),
                 "A": txt("Availability", 18).next_to(dots["A"], DOWN, buff=0.15),
                 "P": txt("Partition tolerance", 18).next_to(dots["P"], DOWN, buff=0.15)}
        pick = txt("pick two", 20, DIM).move_to([-3.7, -0.4, 0])
        self.play(*[Create(e) for e in edges.values()], run_time=1.0)
        self.play(LaggedStart(*[AnimationGroup(FadeIn(dots[k], scale=0.6), FadeIn(names[k])) for k in "CAP"],
                              lag_ratio=0.5), FadeIn(pick), run_time=2.4)

        self.cue(2)
        must = txt("always needed", 16, GOOD).next_to(names["P"], DOWN, buff=0.08)
        self.play(dots["P"][0].animate.set_stroke(GOOD, 4), names["P"].animate.set_color(GOOD), FadeIn(must))
        self.play(Indicate(dots["P"], color=GOOD))

        self.cue(3)
        na = node("Node A", NEG, 1.8, 0.8).move_to([1.6, 1.3, 0])
        nb = node("Node B", NEG, 1.8, 0.8).move_to([5.4, 1.3, 0])
        va = txt("v1", 18, DIM).next_to(na, DOWN, buff=0.12)
        vb = txt("v1", 18, DIM).next_to(nb, DOWN, buff=0.12)
        link_ = Line(na.get_right(), nb.get_left(), color=DIM, stroke_width=4)
        self.play(FadeIn(na), FadeIn(nb), Create(link_), FadeIn(va), FadeIn(vb), run_time=0.8)
        cut = Cross(stroke_color=BAD, stroke_width=6).scale(0.25).move_to(link_)
        self.play(link_.animate.set_color(BAD), Create(cut), run_time=0.6)
        self.play(Transform(va, txt("v2", 18, HI).move_to(va)), Indicate(na, color=HI), run_time=0.8)
        client = VGroup(person(s=0.8), txt("reader", 16, DIM)).arrange(DOWN, buff=0.1).move_to([5.4, -1.0, 0])
        self.play(FadeIn(client))

        self.cue(4)
        self.play(self.trip([pos(client[0]), pos(nb)], WHITE), run_time=0.7)
        cp = VGroup(txt("CP:", 22, BAD, weight=BOLD), txt("error or timeout, never stale", 20)).arrange(RIGHT, buff=0.2)
        cp.move_to([3.5, -2.55, 0])
        self.play(FadeIn(cp), edges["CP"].animate.set_color(BAD), dots["C"][0].animate.set_stroke(BAD, 4))

        self.cue(5)
        ap = VGroup(txt("AP:", 22, HI, weight=BOLD), txt("answers with v1, a bit stale", 20)).arrange(RIGHT, buff=0.2)
        ap.move_to([3.5, -2.55, 0])
        self.play(FadeOut(cp), edges["CP"].animate.set_color(EDGE), dots["C"][0].animate.set_stroke(WHITE, 2.5),
                  run_time=0.4)
        self.play(self.trip([pos(client[0]), pos(nb), pos(client[0])], WHITE), run_time=1.0)
        self.play(FadeIn(ap), edges["AP"].animate.set_color(HI), dots["A"][0].animate.set_stroke(HI, 4))
        self.finish()

    # ---------------------------------------------------------------- 5 nines
    def s5(self):
        self.wipe()
        self.seg(5)
        h = header("Availability in nines")
        sub = txt("downtime allowed per year", 18, DIM).move_to([0, 2.6, 0])
        self.play(FadeIn(h), FadeIn(sub))

        x0, unit = -2.6, 7.6 / 8.766   # bar width per hour of downtime
        def row(y, pct, nines, hours, label, color):
            name = VGroup(txt(pct, 28, WHITE, weight=BOLD), txt(nines, 16, DIM)).arrange(DOWN, buff=0.05)
            name.move_to([-4.6, y, 0])
            bar = Rectangle(width=max(hours * unit, 0.05), height=0.5, stroke_width=0).set_fill(color, 1)
            bar.move_to([x0, y, 0], aligned_edge=LEFT)
            val = txt(label, 20, color).next_to(bar, RIGHT, buff=0.2)
            return name, bar, val

        self.cue(1)
        n3, b3, v3 = row(1.65, "99.9%", "three nines", 8.766, "8h 46m", BAD)
        self.play(FadeIn(n3))
        self.play(GrowFromEdge(b3, LEFT), run_time=1.2)
        v3.move_to(b3).set_color(SCREEN)
        self.play(FadeIn(v3))

        self.cue(2)
        n4, b4, v4 = row(0.45, "99.99%", "four nines", 0.8766, "52m 36s", HI)
        self.play(FadeIn(n4))
        self.play(GrowFromEdge(b4, LEFT), run_time=0.8)
        self.play(FadeIn(v4))

        self.cue(3)
        div = Line([-6.5, -0.45, 0], [6.5, -0.45, 0], color=EDGE)
        wire = txt("wiring matters", 18, DIM).move_to([0, -0.8, 0])
        self.play(Create(div), FadeIn(wire))

        self.cue(4)
        f1 = node("Foo  99.9%", NEG, 2.0, 0.6, 16).move_to([-5.0, -2.1, 0])
        f2 = node("Bar  99.9%", NEG, 2.0, 0.6, 16).move_to([-2.4, -2.1, 0])
        s_lab = txt("in sequence", 18, DIM).move_to([-3.7, -1.35, 0])
        s_arrow = arrow(f1, f2, DIM)
        s_res = txt("= 99.8%", 26, BAD, weight=BOLD).move_to([-3.7, -3.05, 0])
        self.play(FadeIn(s_lab), FadeIn(f1), FadeIn(f2), GrowArrow(s_arrow))
        self.play(FadeIn(s_res, shift=UP * 0.15))

        self.cue(5)
        p1 = node("Foo  99.9%", NEG, 2.0, 0.6, 16).move_to([3.6, -1.75, 0])
        p2 = node("Bar  99.9%", NEG, 2.0, 0.6, 16).move_to([3.6, -2.55, 0])
        ins = Dot([1.6, -2.15, 0], color=DIM)
        outs = Dot([5.6, -2.15, 0], color=DIM)
        wires = VGroup(*[VGroup(Line(ins.get_center(), p.get_left(), color=DIM), Line(p.get_right(), outs.get_center(),
                                                                                   color=DIM)) for p in (p1, p2)])
        p_lab = txt("in parallel", 18, DIM).move_to([3.6, -1.0, 0])
        p_res = txt("= 99.9999%", 26, GOOD, weight=BOLD).next_to(outs, RIGHT, buff=0.15)
        self.play(FadeIn(p_lab), FadeIn(p1), FadeIn(p2), FadeIn(ins), FadeIn(outs), Create(wires))
        p_res.move_to([3.6, -3.25, 0])
        self.play(FadeIn(p_res, shift=UP * 0.15))
        self.finish()

    # ---------------------------------------------------------------- 6 cache-aside
    def s6(self):
        self.wipe()
        self.seg(6)
        h = header("Caching: the cache-aside pattern")
        self.play(FadeIn(h))
        self.cue(1)
        client = person().move_to([-6.0, 0.2, 0])
        app = node("App", NEG, 1.7, 0.9).move_to([-3.2, 0.2, 0])
        cache = node("Cache", BAD, 2.2, 1.0).move_to([0.6, 1.6, 0])
        dbase = db("Database", GOOD, 1.6, 1.0).move_to([0.6, -1.4, 0])
        arrs = VGroup(arrow(client, app), arrow(app, cache), arrow(app, dbase))
        steps = VGroup(*[txt(s, 18, DIM) for s in
                         ("1  check the cache", "2  miss: load from DB", "3  store a copy", "4  return it")])
        steps.arrange(DOWN, aligned_edge=LEFT, buff=0.3).move_to([4.8, 0.2, 0])
        self.play(FadeIn(client), FadeIn(app), FadeIn(cache), FadeIn(dbase), Create(arrs), FadeIn(steps), run_time=1.2)

        def hl(k, color=WHITE):
            return steps[k].animate.set_color(color)

        self.cue(2)
        self.play(hl(0), self.trip([pos(client), pos(app), pos(cache)], HI), run_time=1.2)
        miss = txt("miss", 18, BAD).next_to(cache, UP, buff=0.1)
        self.play(FadeIn(miss, scale=1.3), run_time=0.4)

        self.cue(3)
        self.play(hl(1), FadeOut(miss), self.trip([pos(app), pos(dbase), pos(app)], HI), run_time=1.6)

        self.cue(4)
        entry = txt("user:42", 15, HI).next_to(cache[1], DOWN, buff=0.06)
        self.play(hl(2), cache[1].animate.shift(UP * 0.17), self.trip([pos(app), pos(cache)], HI), run_time=0.9)
        self.play(FadeIn(entry, scale=1.4), run_time=0.4)
        self.play(hl(3), self.trip([pos(app), pos(client)], HI), run_time=0.8)

        self.cue(5)
        self.play(steps.animate.set_color(DIM), run_time=0.3)
        self.play(self.trip([pos(client), pos(app), pos(cache)], GOOD), run_time=0.6)
        hit = txt("hit", 18, GOOD).next_to(cache, UP, buff=0.1)
        self.play(FadeIn(hit, scale=1.3), run_time=0.3)
        self.play(self.trip([pos(cache), pos(app), pos(client)], GOOD), run_time=0.6)

        self.cue(6)
        diagram = VGroup(client, app, cache, dbase, arrs, steps, entry, hit)
        self.play(FadeOut(diagram), run_time=0.5)
        strat = VGroup(chip("Cache-aside", BAD, 22), chip("Write-through", NEG, 22), chip("Write-behind", NEG, 22),
                       chip("Refresh-ahead", NEG, 22)).arrange(RIGHT, buff=0.35).shift(UP * 0.4)
        note = txt("each with its trade-offs", 20, DIM).next_to(strat, DOWN, buff=0.5)
        self.play(LaggedStart(*[FadeIn(s, shift=UP * 0.2) for s in strat], lag_ratio=0.25), run_time=1.6)
        self.play(FadeIn(note))
        self.finish()

    # ---------------------------------------------------------------- 7 database scaling
    def s7(self):
        self.wipe()
        self.seg(7)
        h = header("Scaling the database")
        self.play(FadeIn(h))
        X = [-4.6, 0, 4.6]
        dividers = VGroup(*[Line([x, 2.4, 0], [x, -2.4, 0], color=EDGE) for x in (-2.3, 2.3)])

        self.cue(1)
        t1 = txt("Replication", 24, GOOD).move_to([X[0], 2.3, 0])
        prim = db("Primary", GOOD).move_to([X[0], 0.9, 0])
        reps = VGroup(*[db("Replica", GOOD, 1.2, 0.8, 16).move_to([X[0] + dx, -1.4, 0]) for dx in (-1.0, 1.0)])
        ra = VGroup(*[arrow(prim, r, GOOD) for r in reps])
        self.play(FadeIn(t1), FadeIn(prim), Create(dividers))
        self.play(*[TransformFromCopy(prim, r) for r in reps], *[GrowArrow(a) for a in ra], run_time=1.2)
        reads = txt("reads spread out", 16, DIM).move_to([X[0], -2.5, 0])
        self.play(FadeIn(reads))

        self.cue(2)
        t2 = txt("Federation", 24, NEG).move_to([X[1], 2.3, 0])
        big = db("One DB", NEG, 1.6).move_to([X[1], 0.9, 0])
        parts = VGroup(*[db(n, NEG, 1.15, 0.8, 15).move_to([X[1] + dx, -1.4, 0])
                         for n, dx in (("Users", -1.35), ("Products", 0), ("Forums", 1.35))])
        self.play(FadeIn(t2), FadeIn(big))
        self.play(*[TransformFromCopy(big, p) for p in parts], big.animate.set_opacity(0.3), run_time=1.4)
        by = txt("split by function", 16, DIM).move_to([X[1], -2.5, 0])
        self.play(FadeIn(by))

        self.cue(3)
        t3 = txt("Sharding", 24, POS).move_to([X[2], 2.3, 0])
        cols = [BAD, HI, NEG]
        rows = VGroup(*[Rectangle(width=2.4, height=0.14, stroke_width=0).set_fill(cols[k // 3], 1)
                        for k in range(9)]).arrange(DOWN, buff=0.05).move_to([X[2], 1.1, 0])
        tbl = SurroundingRectangle(rows, color=DIM, buff=0.1, corner_radius=0.08)
        tl = txt("users table", 15, DIM).next_to(tbl, DOWN, buff=0.08)
        shards = VGroup(*[db(n, POS, 1.15, 0.8, 15).move_to([X[2] + dx, -1.6, 0])
                          for n, dx in (("A–H", -1.35), ("I–P", 0), ("Q–Z", 1.35))])
        self.play(FadeIn(t3), FadeIn(rows), Create(tbl), FadeIn(tl))
        self.play(FadeIn(shards))
        self.play(*[rows[k].animate.scale(0.4).move_to(shards[k // 3][4].get_center() + UP * (0.25 + 0.09 * (k % 3)))
                    for k in range(9)], FadeOut(tbl), FadeOut(tl), run_time=1.3)
        sl = txt("each holds a slice", 16, DIM).move_to([X[2], -2.5, 0])
        self.play(FadeIn(sl))

        self.cue(4)
        q = txt("SQL or NoSQL?", 22, HI)
        kinds = VGroup(*[chip(s, PURPLE_, 15) for s in ("key-value", "document", "wide column", "graph")])
        line = VGroup(q, *kinds).arrange(RIGHT, buff=0.25).move_to([0, -3.3, 0])
        self.play(FadeIn(q), LaggedStart(*[FadeIn(k, shift=UP * 0.15) for k in kinds], lag_ratio=0.2), run_time=1.6)
        self.finish()

    # ---------------------------------------------------------------- 8 latency numbers
    def s8(self):
        self.wipe()
        self.seg(8)
        h = header("Latency numbers every programmer should know")
        heads = VGroup(txt("operation", 16, DIM).move_to([-5.0, 2.55, 0]),
                       txt("time (log scale)", 16, DIM).move_to([-1.2, 2.55, 0]),
                       txt("if 1 ns were 1 second", 16, DIM).move_to([5.4, 2.55, 0]))
        data = [("L1 cache reference", 0.5, "0.5 ns", "half a second", DIM),
                ("Main memory", 100, "100 ns", "1.7 minutes", GOOD),
                ("Datacenter round trip", 5e5, "0.5 ms", "5.8 days", HI),
                ("Disk seek", 1e7, "10 ms", "3.8 months", POS),
                ("CA → Netherlands → CA", 1.5e8, "150 ms", "4.8 years", BAD)]
        x0 = -3.0
        rows = []
        for k, (name, ns, val, human, color) in enumerate(data):
            y = 1.85 - k * 0.98
            lab = fit(txt(name, 18), 3.4).move_to([-5.0, y, 0])
            bar = Rectangle(width=(np.log10(ns) + 0.6) * 0.6, height=0.46, stroke_width=0).set_fill(color, 1)
            bar.move_to([x0, y, 0], aligned_edge=LEFT)
            v = txt(val, 18, color).next_to(bar, RIGHT, buff=0.18)
            hm = txt(human, 18, WHITE).move_to([5.4, y, 0])
            rows.append((lab, bar, v, hm))

        def show(r, rt=1.0):
            lab, bar, v, hm = r
            self.play(FadeIn(lab), GrowFromEdge(bar, LEFT), run_time=rt)
            self.play(FadeIn(v), FadeIn(hm, shift=LEFT * 0.15), run_time=0.6)

        self.play(FadeIn(h), FadeIn(heads))
        show(rows[0])
        for k in range(1, 5):
            self.cue(k)
            show(rows[k])
        self.cue(5)
        b = banner("That gap is why caches and CDNs exist")
        self.play(FadeIn(b, shift=UP * 0.2))
        self.finish()

    # ---------------------------------------------------------------- 9 interview method
    def s9(self):
        self.wipe()
        self.seg(9)
        h = header("A four-step interview plan")
        self.play(FadeIn(h))
        spec = [("Scope it", "Who uses it?\nHow many requests?\nHow much data?\nRead/write ratio?", HI),
                ("High-level design", "Sketch the main\ncomponents and\nhow they connect", NEG),
                ("Core components", "APIs and schemas\nHashing\nSQL or NoSQL", PURPLE_),
                ("Scale it", "Find bottlenecks\nLoad balancers\nCaches, sharding", GOOD)]
        cards = []
        for k, (title, body, color) in enumerate(spec):
            x = -5.1 + k * 3.4
            c = card(3.15, 4.7, [x, -0.25, 0]).set_stroke(color, 2)
            num = VGroup(Circle(0.32, stroke_width=0).set_fill(color, 1), txt(str(k + 1), 24, SCREEN, weight=BOLD))
            num.move_to([x, 1.55, 0])
            t = fit(txt(title, 21, color), 2.8).move_to([x, 0.85, 0])
            b = txt(body, 16, WHITE, line_spacing=1.1)
            b = fit(b, 2.75).next_to(t, DOWN, buff=0.35)
            cards.append(VGroup(c, num, t, b))
        for k in range(4):
            self.cue(k + 1)
            self.play(FadeIn(cards[k], shift=UP * 0.25), run_time=0.8)
        self.cue(5)
        b = banner("Everything is a trade-off", y=-3.3)
        self.play(FadeIn(b, shift=UP * 0.2))
        self.finish()

    # ---------------------------------------------------------------- 10 practice + outro
    def s10(self):
        self.wipe()
        self.seg(10)
        h = header("Practice with worked solutions")
        self.play(FadeIn(h))
        self.cue(1)
        qs = ["Pastebin / Bit.ly", "Twitter timeline", "Web crawler", "Mint.com",
              "Social network graph", "Key-value store", "Amazon sales rank", "Millions of users on AWS"]
        grid = VGroup(*[VGroup(card(3.05, 1.0, ORIGIN), fit(txt(q, 18), 2.75)) for q in qs])
        grid.arrange_in_grid(2, 4, buff=(0.25, 0.3)).move_to([0, 1.0, 0])
        self.play(LaggedStart(*[FadeIn(g, shift=UP * 0.2) for g in grid], lag_ratio=0.15), run_time=2.4)

        self.cue(2)
        tags = VGroup(chip("discussion", HI, 18), chip("code", GOOD, 18), chip("diagrams", NEG, 18))
        tags.arrange(RIGHT, buff=0.35).move_to([0, -1.0, 0])
        self.play(LaggedStart(*[FadeIn(t, scale=0.8) for t in tags], lag_ratio=0.3))

        self.cue(3)
        self.play(FadeOut(grid), FadeOut(tags), FadeOut(h), run_time=0.5)
        deck = VGroup(*[RoundedRectangle(corner_radius=0.15, width=2.6, height=1.7, stroke_color=PURPLE_,
                                         stroke_width=2).set_fill(PANEL, 1).rotate(a).shift(RIGHT * 0.12 * i)
                        for i, a in enumerate((-0.12, -0.04, 0.05))])
        deck.add(txt("Anki", 26, PURPLE_, weight=BOLD).move_to(deck[-1]))
        deck.move_to([-3.9, 0.6, 0])
        dl = txt("flashcards", 18, DIM).next_to(deck, DOWN, buff=0.3)
        table = VGroup()
        cols = ["", "Short", "Medium", "Long"]
        rows = [("Breadth", "✓", "✓", "✓"), ("Depth", "–", "some", "more"), ("Questions", "some", "many", "most")]
        for r, row in enumerate([cols] + rows):
            for c, cell in enumerate(row):
                color = HI if r == 0 else (DIM if c == 0 else WHITE)
                table.add(txt(cell, 18, color).move_to([1.5 + c * 1.5, 1.6 - r * 0.6, 0]))
        tbg = panel(table, buff=0.3)
        tl = txt("study guide", 18, DIM).next_to(tbg, DOWN, buff=0.3)
        self.play(FadeIn(deck, shift=UP * 0.2), FadeIn(dl))
        self.play(FadeIn(tbg), LaggedStart(*[FadeIn(t) for t in table], lag_ratio=0.03), FadeIn(tl), run_time=1.5)

        self.cue(4)
        b = banner("Start broad, then go deep in a few areas", y=-2.6)
        self.play(FadeIn(b, shift=UP * 0.2))

        self.cue(5)
        self.wipe(0.4)
        title = txt("The System Design Primer", 48, WHITE, weight=BOLD).shift(UP * 0.5)
        url = txt("github.com/donnemartin/system-design-primer", 26, HI).next_to(title, DOWN, buff=0.4)
        self.play(FadeIn(title, shift=UP * 0.2))
        self.play(Write(url), run_time=1.5)
        self.finish(1.5)
