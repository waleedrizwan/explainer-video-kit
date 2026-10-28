"""Episode 1 — One server, many users.

    python narration.py
    manim -ql scene.py Explainer      # preview
    manim -qh scene.py Explainer      # final

Running example: one server, 0.1 s per request, so a capacity of 10 requests per second.
The latency curve is the M/M/1 queue: average time in system = 0.1 / (1 - busy fraction).
The queue animation is a real simulation (random arrivals and service times), slowed down 5x.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import *  # noqa: E402,F403

SERVICE = 0.1          # seconds of work per request
SLOW = 5.0             # video seconds per simulated second in the queue animation


def latency(rho):
    return SERVICE / (1 - rho)


def simulate(rate, seconds, seed):
    """FIFO single server. Returns (arrive, start, finish) in simulated seconds."""
    rng = np.random.default_rng(seed)
    t, free, out = 0.0, 0.0, []
    while True:
        t += rng.exponential(1 / rate)
        if t > seconds:
            return out
        s = max(t, free)
        free = s + rng.exponential(SERVICE)
        out.append((t, s, free))


def max_waiting(reqs, upto):
    ts = np.linspace(0, upto, 400)
    return max(sum(1 for a, s, f in reqs if a <= t < s) for t in ts)


class Explainer(EpisodeScene):
    project_dir = Path(__file__).parent

    def construct(self):
        for i in range(12):
            getattr(self, f"s{i}")()

    # ---------------------------------------------------------------- 0 hook
    def s0(self):
        self.seg(0)
        rng = np.random.default_rng(3)
        phones = VGroup(*[phone(s=0.32) for _ in range(40)]).arrange_in_grid(5, 8, buff=(0.35, 0.25))
        phones.move_to(LEFT * 2.6)
        rack = VGroup(*[server(s=0.55) for _ in range(6)]).arrange_in_grid(2, 3, buff=0.3).move_to(RIGHT * 4.3)
        self.play(LaggedStart(*[FadeIn(p, scale=0.6) for p in phones], lag_ratio=0.02), FadeIn(rack), run_time=1.8)
        trips = [self.trip([pos(p), pos(rack[int(rng.integers(6))])], HI, 0.05) for p in phones]
        self.play(LaggedStart(*trips, lag_ratio=0.04), run_time=4.0)

        self.cue(1)
        q = txt("How are these systems built?", 34)
        self.play(FadeOut(phones), FadeOut(rack), FadeIn(q))

        self.cue(2)
        title = txt(SERIES, 44, WHITE, weight=BOLD).shift(UP * 0.6)
        ep = txt("Episode 1 · One server, many users", 28, HI).next_to(title, DOWN, buff=0.35)
        src = chip("based on The System Design Primer · github.com/donnemartin", DIM, 16).next_to(ep, DOWN, buff=0.5)
        self.play(FadeOut(q), Write(title), run_time=1.6)
        self.play(FadeIn(ep, shift=UP * 0.1), FadeIn(src))

        self.cue(3)
        self.play(FadeOut(VGroup(title, ep, src), shift=UP * 0.3), run_time=0.8)

        self.cue(4)
        need = VGroup(chip("a computer", NEG, 22), chip("the internet", NEG, 22)).arrange(RIGHT, buff=0.4)
        need.shift(DOWN * 0.6)
        lab = txt("all you need", 20, DIM).next_to(need, DOWN, buff=0.3)
        self.play(LaggedStart(*[FadeIn(c, shift=UP * 0.15) for c in need], lag_ratio=0.3), FadeIn(lab))
        self.finish()

    # ---------------------------------------------------------------- 1 a request
    def s1(self):
        self.wipe()
        self.seg(1)
        h = header("A request")
        self.play(FadeIn(h))

        self.cue(1)
        ph = VGroup(phone(), txt("you", 18, DIM)).arrange(DOWN, buff=0.15).move_to([-5.0, 0.2, 0])
        sv = server(s=1.2, label="server").move_to([5.0, 0.2, 0])
        self.play(FadeIn(ph, shift=RIGHT * 0.2), FadeIn(sv, shift=LEFT * 0.2))

        self.cue(2)
        net = DashedLine(ph.get_right() + RIGHT * 0.3, sv.get_left() + LEFT * 0.3, color=EDGE, dash_length=0.12)
        cloud = txt("the internet", 18, DIM).next_to(net, UP, buff=0.15)
        self.play(Create(net), FadeIn(cloud))
        self.play(self.trip([net.get_start(), net.get_end()], HI, 0.12), run_time=1.6)

        self.cue(3)
        req = term("request", "a message asking the server for something").move_to([0, -1.6, 0])
        self.play(FadeIn(req, shift=UP * 0.15))

        self.cue(4)
        work = VGroup(chip("run code", PURPLE_, 18), chip("read data from disk", GOOD, 18)).arrange(DOWN, buff=0.2)
        work.next_to(sv, UP, buff=0.35)
        self.play(FadeIn(work[0], shift=DOWN * 0.1), Indicate(sv, color=PURPLE_), run_time=1.0)
        self.play(FadeIn(work[1], shift=DOWN * 0.1), run_time=0.6)

        self.cue(5)
        self.play(self.trip([net.get_end(), net.get_start()], GOOD, 0.12), run_time=1.6)
        resp = term("response", "the server's answer", GOOD).move_to([0, -1.6, 0])
        self.play(FadeOut(req), FadeIn(resp, shift=UP * 0.15), run_time=0.5)

        self.cue(6)
        loop = CurvedArrow(net.get_start() + DOWN * 0.3, net.get_end() + DOWN * 0.3, angle=0.5, color=HI)
        loop2 = CurvedArrow(net.get_end() + UP * 0.3, net.get_start() + UP * 0.3, angle=0.5, color=GOOD)
        self.play(FadeOut(resp), FadeOut(cloud), Create(loop), Create(loop2), run_time=1.2)

        self.cue(7)
        goals = VGroup(chip("fast", HI, 24), chip("cheap", GOOD, 24), chip("reliable", NEG, 24))
        goals.arrange(RIGHT, buff=0.4).move_to([0, -2.3, 0])
        many = txt("× millions of people, at once", 22, DIM).next_to(goals, DOWN, buff=0.3)
        self.play(LaggedStart(*[FadeIn(g, shift=UP * 0.15) for g in goals], lag_ratio=0.3))
        self.play(FadeIn(many))
        self.finish()

    # ---------------------------------------------------------------- 2 latency
    def s2(self):
        self.wipe()
        self.seg(2)
        h = header("Latency: how long one request takes")
        ph = phone(s=0.8).move_to([-5.6, 1.2, 0])
        sv = server(s=0.8).move_to([5.6, 1.2, 0])
        self.play(FadeIn(h), FadeIn(ph), FadeIn(sv))

        self.cue(2)
        lat = term("latency", "time from sending a request to getting the answer").move_to([0, 1.2, 0])
        self.play(FadeIn(lat, shift=UP * 0.15))

        self.cue(3)
        parts = [("network there", 30, NEG), ("run code", 20, PURPLE_), ("read disk", 20, GOOD), ("network back", 30, NEG)]
        unit = 10.0 / 100
        x = -5.0
        blocks = []
        for name, ms, color in parts:
            r = Rectangle(width=ms * unit, height=0.7, stroke_color=SCREEN, stroke_width=3).set_fill(color, 1)
            r.move_to([x, -0.8, 0], aligned_edge=LEFT)
            x += ms * unit
            lab = VGroup(txt(name, 16, color), txt(f"{ms} ms", 16, DIM)).arrange(DOWN, buff=0.06).next_to(r, DOWN, buff=0.15)
            blocks.append(VGroup(r, lab))
        axis_note = txt("one request, start to finish", 16, DIM).move_to([0, -0.05, 0])
        self.play(lat.animate.shift(UP * 0.3).scale(0.85), FadeIn(axis_note))
        for k in range(4):
            self.cue(k + 4)
            self.play(GrowFromEdge(blocks[k][0], LEFT), FadeIn(blocks[k][1]), run_time=0.7)

        self.cue(8)
        br = BraceBetweenPoints([-5.0, -2.05, 0], [5.0, -2.05, 0], DOWN, color=HI)
        tot = txt("latency = 100 ms = 0.1 s", 24, HI).next_to(br, DOWN, buff=0.12)
        self.play(GrowFromCenter(br), FadeIn(tot))

        self.cue(9)
        self.play(Indicate(tot, color=GOOD, scale_factor=1.08))
        self.finish()

    # ---------------------------------------------------------------- 3 throughput
    def s3(self):
        self.wipe()
        self.seg(3)
        h = header("Throughput: how many per second")
        self.play(FadeIn(h))

        self.cue(2)
        thr = term("throughput", "how many requests get finished each second").move_to([0, 1.9, 0])
        self.play(FadeIn(thr, shift=UP * 0.15))

        self.cue(3)
        sv = server(s=0.9).move_to([0, 0.15, 0])
        self.play(FadeIn(sv))
        slots = VGroup(*[Rectangle(width=0.9, height=0.55, stroke_color=EDGE, stroke_width=2).set_fill(PANEL, 1)
                         for _ in range(10)]).arrange(RIGHT, buff=0).move_to([0, -1.5, 0])
        ticks = VGroup(txt("0 s", 16, DIM).next_to(slots, LEFT, buff=0.2), txt("1 s", 16, DIM).next_to(slots, RIGHT, buff=0.2))
        each = txt("0.1 s each", 16, DIM).next_to(slots[0], DOWN, buff=0.15)
        self.play(FadeIn(slots), FadeIn(ticks), FadeIn(each))

        self.cue(4)
        n = ValueTracker(0)
        count = always_redraw(lambda: txt(f"{int(round(n.get_value()))} finished", 30, HI).next_to(slots, DOWN, buff=0.6))
        self.play(FadeIn(count), FadeOut(each), run_time=0.3)
        for k in range(10):
            self.play(slots[k].animate.set_fill(HI, 1), n.animate.set_value(k + 1),
                      self.trip([sv.get_left() + LEFT * 1.6, pos(sv), sv.get_right() + RIGHT * 1.6], HI, 0.08),
                      run_time=0.24)

        self.cue(5)
        cap = banner("capacity = 10 requests per second", HI, y=-3.3)
        self.play(FadeIn(cap, shift=UP * 0.15))

        self.cue(6)
        self.play(Indicate(cap, color=BAD))
        self.finish()

    # ---------------------------------------------------------------- 4 queues
    def s4(self):
        self.wipe()
        self.seg(4)
        h = header("Many users: the queue")
        sv = server(s=1.0).move_to([2.6, 0.3, 0])
        exitx = 6.6
        self.play(FadeIn(h), FadeIn(sv))

        self.cue(1)
        note = txt("arrivals are random, like customers at a coffee shop", 18, DIM).move_to([0, 2.5, 0])
        self.play(FadeIn(note))

        def run_phase(rate, until_video, seed_base, label, busy):
            dur_sim = until_video / SLOW
            # pick the first seed whose run shows what that rate usually looks like
            target = (0, 1) if rate < 7 else (4, 7)
            for seed in range(seed_base, seed_base + 200):
                reqs = simulate(rate, dur_sim, seed)
                mw = max_waiting(reqs, dur_sim)
                if target[0] <= mw <= target[1]:
                    break
            tr = ValueTracker(0)
            front = sv.get_left() + LEFT * 0.45

            def draw():
                t = tr.get_value()
                g = VGroup()
                waiting = [r for r in reqs if r[0] <= t < r[1]]
                for k in range(len(waiting)):
                    g.add(Dot(front + LEFT * 0.42 * k, radius=0.13, color=HI))
                for _, s, f in reqs:
                    if s <= t < f:
                        g.add(Dot(pos(sv), radius=0.15, color=HI))
                    elif f <= t < f + 0.12:
                        x = interpolate(sv.get_right()[0], exitx, (t - f) / 0.12)
                        g.add(Dot([x, sv.get_center()[1], 0], radius=0.13, color=GOOD))
                return g

            def stats():
                t = tr.get_value()
                n = sum(1 for r in reqs if r[0] <= t < r[1])
                return txt(f"waiting in line: {n}", 22, BAD if n else DIM).move_to([-3.3, -1.6, 0])

            dots = always_redraw(draw)
            st = always_redraw(stats)
            info = VGroup(txt(label, 24, WHITE), txt(busy, 18, DIM)).arrange(DOWN, buff=0.1).move_to([-3.3, -0.6, 0])
            self.add(dots, st)
            self.play(FadeIn(info), run_time=0.3)
            self.play(tr.animate.set_value(dur_sim), run_time=until_video, rate_func=linear)
            self.play(FadeOut(info), FadeOut(st), FadeOut(dots), run_time=0.3)
            self.remove(dots, st)

        self.cue(2)
        run_phase(5, self.until(4) - 0.7, 10, "5 requests / second", "server busy 50% of the time")
        self.cue(4)
        run_phase(9, self.until(7) - 0.7, 40, "9 requests / second", "server busy 90% of the time")

        self.cue(7)
        line = VGroup(*[Dot(radius=0.13, color=HI).move_to(sv.get_left() + LEFT * (0.45 + 0.42 * k)) for k in range(5)])
        qterm = term("queue", "requests waiting their turn").move_to([-3.3, -1.4, 0])
        self.play(FadeIn(line), FadeIn(qterm, shift=UP * 0.15))

        self.cue(8)
        lat = txt("latency = waiting + being served", 24, HI).move_to([0, -3.0, 0])
        self.play(FadeIn(lat, shift=UP * 0.15))
        self.finish()

    # ---------------------------------------------------------------- 5 the curve
    def latency_axes(self, xlen=9.0, ylen=4.6):
        ax = Axes(x_range=[0, 1.05, 0.25], y_range=[0, 2.5, 0.5], x_length=xlen, y_length=ylen,
                  axis_config={"color": DIM, "stroke_width": 2, "include_ticks": True},
                  tips=False)
        xl = VGroup(*[txt(f"{int(v * 100)}%", 16, DIM).next_to(ax.c2p(v, 0), DOWN, buff=0.15) for v in (0, .25, .5, .75, 1)])
        yl = VGroup(*[txt(f"{v:g} s", 16, DIM).next_to(ax.c2p(0, v), LEFT, buff=0.15) for v in (0.5, 1, 1.5, 2, 2.5)])
        xt = txt("how busy the server is", 18, DIM).next_to(ax.x_axis, DOWN, buff=0.55)
        yt = txt("average latency", 18, DIM).rotate(PI / 2).next_to(yl, LEFT, buff=0.2)
        curve = ax.plot(latency, x_range=[0, 0.96, 0.005], color=HI, stroke_width=4)
        return ax, VGroup(xl, yl, xt, yt), curve

    def s5(self):
        self.wipe()
        self.seg(5)
        h = header("Latency explodes near capacity")
        self.play(FadeIn(h))

        self.cue(1)
        ax, labels, curve = self.latency_axes()
        VGroup(ax, labels, curve).move_to([-0.7, 0.0, 0]).scale(0.95)
        self.play(Create(ax), FadeIn(labels), run_time=1.2)

        pts = [(0.0, "0.1 s"), (0.5, "0.2 s"), (0.9, "1 s"), (0.95, "2 s")]
        tracker = ValueTracker(0.0)
        partial = always_redraw(lambda: ax.plot(latency, x_range=[0, max(tracker.get_value(), 0.001), 0.005],
                                                color=HI, stroke_width=4))
        self.add(partial)
        for k, (rho, lab) in enumerate(pts):
            self.cue(k + 2)
            self.play(tracker.animate.set_value(rho), run_time=0.9 if k else 0.3)
            d = Dot(ax.c2p(rho, latency(rho)), radius=0.08, color=WHITE)
            t = txt(lab, 20, WHITE).next_to(d, LEFT if rho > 0.8 else UP, buff=0.15)
            if rho == 0.9:
                t = VGroup(t, txt("10× slower", 16, BAD)).arrange(DOWN, buff=0.05).next_to(d, LEFT, buff=0.15)
            self.play(FadeIn(d, scale=1.5), FadeIn(t), run_time=0.5)

        self.cue(6)
        self.play(tracker.animate.set_value(0.96), run_time=0.5)
        cap = DashedLine(ax.c2p(1, 0), ax.c2p(1, 2.5), color=BAD)
        inf = txt("100%: the queue\nnever drains", 18, BAD).next_to(cap, RIGHT, buff=0.15).shift(UP * 1.2)
        self.play(Create(cap), FadeIn(inf))

        self.cue(7)
        b = banner("Latency doesn't rise gently. It explodes near the limit.")
        self.play(FadeIn(b, shift=UP * 0.15))
        self.finish()

    # ---------------------------------------------------------------- 6 performance vs scalability
    def s6(self):
        self.wipe()
        self.seg(6)
        h = header("Two kinds of slow")
        lc = card(6.4, 3.75, [-3.45, 0.75, 0]).set_stroke(BAD, 2)
        rc = card(6.4, 3.75, [3.45, 0.75, 0]).set_stroke(POS, 2)
        self.play(FadeIn(h))

        self.cue(1)
        lt = txt("Performance problem", 24, BAD).move_to([-3.45, 2.05, 0])
        p1 = person(s=0.8).move_to([-5.6, 1.1, 0])
        bar1 = Rectangle(width=3.6, height=0.35, stroke_width=0).set_fill(BAD, 1).move_to([-4.9, 1.1, 0], aligned_edge=LEFT)
        l1 = txt("slow even for 1 user", 18).move_to([-3.45, 0.45, 0])
        self.play(FadeIn(lc), FadeIn(lt), FadeIn(p1), GrowFromEdge(bar1, LEFT))
        self.play(FadeIn(l1))

        self.cue(2)
        fix1 = txt("fix: make each request cheaper", 18, GOOD).move_to([-3.45, -0.3, 0])
        how1 = txt("faster code · fewer disk reads", 16, DIM).next_to(fix1, DOWN, buff=0.12)
        self.play(FadeIn(fix1), FadeIn(how1), bar1.animate.stretch_to_fit_width(0.8, about_edge=LEFT).set_fill(GOOD))

        self.cue(3)
        rt = txt("Scalability problem", 24, POS).move_to([3.45, 2.05, 0])
        p2 = person(s=0.8).move_to([1.3, 1.25, 0])
        bar2 = Rectangle(width=0.8, height=0.3, stroke_width=0).set_fill(GOOD, 1).move_to([2.0, 1.25, 0], aligned_edge=LEFT)
        crowd = VGroup(*[person(s=0.4) for _ in range(6)]).arrange_in_grid(2, 3, buff=0.06).move_to([1.3, 0.45, 0])
        bar3 = Rectangle(width=3.6, height=0.3, stroke_width=0).set_fill(BAD, 1).move_to([2.0, 0.45, 0], aligned_edge=LEFT)
        l2 = txt("fast for 1, slow under load", 18).move_to([3.45, -0.3, 0])
        self.play(FadeIn(rc), FadeIn(rt), FadeIn(p2), GrowFromEdge(bar2, LEFT))
        self.play(FadeIn(crowd), GrowFromEdge(bar3, LEFT))
        self.play(FadeIn(l2))

        self.cue(4)
        fix2 = txt("fix: add capacity", 18, GOOD).move_to([3.45, -0.85, 0])
        self.play(FadeIn(fix2))

        self.cue(5)
        ax = Axes(x_range=[0, 4.5, 1], y_range=[0, 45, 10], x_length=4.6, y_length=1.5, tips=False,
                  axis_config={"color": DIM, "stroke_width": 2}).move_to([0, -2.75, 0])
        good = ax.plot(lambda m: 10 * m, x_range=[0, 4.2], color=GOOD, stroke_width=4)
        bad = ax.plot(lambda m: 22 * (1 - np.exp(-m / 1.3)), x_range=[0, 4.2], color=BAD, stroke_width=4)
        gx = txt("machines", 14, DIM).next_to(ax.x_axis, RIGHT, buff=0.1)
        gy = txt("work handled", 14, DIM).next_to(ax.y_axis, UP, buff=0.05)
        lg = txt("scalable", 16, GOOD).next_to(ax.c2p(4.2, 42), RIGHT, buff=0.1)
        lb = txt("not scalable", 16, BAD).next_to(ax.c2p(4.2, 21), RIGHT, buff=0.1)
        defn = txt("scalable: adding resources\nincreases performance\nin proportion", 17, WHITE).move_to([-4.6, -2.75, 0])
        self.play(FadeIn(defn), Create(ax), FadeIn(gx), FadeIn(gy))
        self.play(Create(good), FadeIn(lg), run_time=1.2)
        self.play(Create(bad), FadeIn(lb), run_time=1.2)

        self.cue(6)
        dbl = txt("2× machines\n→ 2× work", 18, HI).move_to([5.3, -2.75, 0])
        self.play(FadeIn(dbl), Indicate(good, color=HI))
        self.finish()

    # ---------------------------------------------------------------- 7 trade-off
    def s7(self):
        self.wipe()
        self.seg(7)
        h = header("Throughput vs latency")
        ax, labels, curve = self.latency_axes(8.4, 4.3)
        VGroup(ax, labels, curve).move_to([0.3, -0.2, 0])
        self.play(FadeIn(h), FadeIn(ax), FadeIn(labels), Create(curve))

        self.cue(1)
        rho = ValueTracker(0.3)
        d = always_redraw(lambda: Dot(ax.c2p(rho.get_value(), latency(rho.get_value())), radius=0.1, color=WHITE))
        self.add(d)
        self.play(rho.animate.set_value(0.95), run_time=2.5)

        self.cue(2)
        self.play(Flash(d, color=BAD), run_time=0.8)

        self.cue(3)
        ok = 0.3
        best = 1 - SERVICE / ok
        line = DashedLine(ax.c2p(0, ok), ax.c2p(1.02, ok), color=GOOD)
        lt = txt("acceptable latency: 0.3 s", 16, GOOD).next_to(ax.c2p(0, ok), UR, buff=0.08).shift(RIGHT * 0.2)
        self.play(Create(line), FadeIn(lt))
        self.play(rho.animate.set_value(best), run_time=1.2)
        drop = DashedLine(ax.c2p(best, ok), ax.c2p(best, 0), color=HI)
        bl = txt(f"max throughput that meets it:\n{best:.0%} busy ≈ {10 * best:.1f} requests/s", 18, HI)
        bl.next_to(ax.c2p(best, ok), UP, buff=0.9).shift(LEFT * 1.2)
        self.play(Create(drop), FadeIn(bl))

        self.cue(4)
        zone = Rectangle(width=ax.c2p(1, 0)[0] - ax.c2p(best, 0)[0], height=ax.c2p(0, 2.5)[1] - ax.c2p(0, 0)[1],
                         stroke_width=0).set_fill(BAD, 0.12).move_to(ax.c2p(best, 0), aligned_edge=DL)
        zl = txt("headroom", 18, BAD).move_to(zone).shift(UP * 1.3 + RIGHT * 0.05)
        self.play(FadeIn(zone), FadeIn(zl))
        self.finish()

    # ---------------------------------------------------------------- 8 vertical scaling
    def s8(self):
        self.wipe()
        self.seg(8)
        h = header("Option 1: a bigger machine")
        sv = server(s=0.8).move_to([-3.5, -0.3, 0])
        self.play(FadeIn(h), FadeIn(sv))

        self.cue(1)
        for s in (1.4, 1.9):
            self.play(sv.animate.scale_to_fit_height(0.9 * s).move_to([-3.5, -0.3, 0]), run_time=0.8)
        specs = VGroup(chip("more processors", NEG, 16), chip("more memory", NEG, 16), chip("faster disks", NEG, 16))
        specs.arrange(DOWN, buff=0.2).next_to(sv, RIGHT, buff=0.6)
        self.play(LaggedStart(*[FadeIn(s, shift=LEFT * 0.1) for s in specs], lag_ratio=0.3))

        self.cue(2)
        vt = term("vertical scaling", "“scaling up”: make one machine bigger").move_to([3.6, 1.6, 0])
        self.play(FadeIn(vt, shift=UP * 0.15))

        self.cue(3)
        pro = txt("+ simple: your code doesn't change", 20, GOOD).move_to([3.6, 0.2, 0])
        self.play(FadeIn(pro))

        self.cue(4)
        ceil = DashedLine([-6.2, 1.55, 0], [-0.8, 1.55, 0], color=BAD)
        cl = txt("biggest machine you can buy", 16, BAD).next_to(ceil, UP, buff=0.1)
        self.play(FadeOut(specs), sv.animate.scale_to_fit_height(2.6).move_to([-3.5, 0.2, 0]), run_time=0.8)
        self.play(Create(ceil), FadeIn(cl), run_time=0.8)
        c1 = txt("− there is a ceiling", 20, BAD).move_to([3.6, -0.4, 0])
        self.play(FadeIn(c1), run_time=0.6)

        self.cue(5)
        c2 = txt("− high-end hardware costs more per unit of power", 20, BAD)
        c2 = fit(c2, 6.2).move_to([3.6, -1.0, 0])
        self.play(FadeIn(c2))

        self.cue(6)
        x = Cross(stroke_color=BAD, stroke_width=8).scale(0.8).move_to(sv)
        users = VGroup(*[person(s=0.45) for _ in range(7)]).arrange(RIGHT, buff=0.2).move_to([-3.5, -2.6, 0])
        self.play(FadeIn(users))
        self.play(Create(x), users.animate.set_color(BAD), run_time=0.8)

        self.cue(7)
        c3 = txt("− a single point of failure", 20, BAD).move_to([3.6, -1.6, 0])
        self.play(FadeIn(c3))
        self.finish()

    # ---------------------------------------------------------------- 9 horizontal scaling
    def s9(self):
        self.wipe()
        self.seg(9)
        h = header("Option 2: many ordinary machines")
        self.play(FadeIn(h))
        grid = VGroup(*[server(s=0.5) for _ in range(10)]).arrange_in_grid(2, 5, buff=(0.45, 0.5)).move_to([-2.4, 0.6, 0])
        self.play(FadeIn(grid[0]))
        self.play(LaggedStart(*[TransformFromCopy(grid[0], g) for g in grid[1:]], lag_ratio=0.1), run_time=1.6)

        self.cue(1)
        ht = term("horizontal scaling", "“scaling out”: add more machines").move_to([3.9, 1.6, 0])
        self.play(FadeIn(ht, shift=UP * 0.15))

        self.cue(2)
        math = txt("10 servers × 10 per second = 100 per second", 22, HI).move_to([-2.4, -1.15, 0])
        self.play(*[Indicate(g, color=HI) for g in grid], FadeIn(math), run_time=1.2)

        self.cue(3)
        more = VGroup(*[server(s=0.5).set_opacity(0.5) for _ in range(2)]).arrange(DOWN, buff=0.45).next_to(grid, RIGHT, buff=0.45)
        self.play(FadeIn(more, shift=LEFT * 0.2), run_time=0.7)

        self.cue(4)
        x = Cross(stroke_color=BAD, stroke_width=6).scale(0.35).move_to(grid[6])
        self.play(Create(x), grid[6].animate.set_opacity(0.3), run_time=0.5)
        ok = txt("the other nine keep working", 18, GOOD).move_to([-2.4, -1.7, 0])
        self.play(FadeIn(ok), run_time=0.4)

        self.cue(5)
        pros = VGroup(*[txt(s, 20, GOOD) for s in ("+ cheaper", "+ more available", "+ easier to hire for")])
        pros.arrange(DOWN, aligned_edge=LEFT, buff=0.2).move_to([3.9, -0.4, 0])
        self.play(LaggedStart(*[FadeIn(p, shift=LEFT * 0.1) for p in pros], lag_ratio=0.4), run_time=2.0)

        self.cue(6)
        b = banner("Almost every large system scales out")
        self.play(FadeIn(b, shift=UP * 0.15))
        self.finish()

    # ---------------------------------------------------------------- 10 statelessness
    def s10(self):
        self.wipe()
        self.seg(10)
        h = header("The catch: state")
        user = VGroup(phone(s=0.8), txt("you", 16, DIM)).arrange(DOWN, buff=0.1).move_to([-5.8, 0.5, 0])
        svs = VGroup(*[server(s=0.55, label=f"server {k + 1}") for k in range(3)]).arrange(DOWN, buff=0.35).move_to([-0.6, 0.5, 0])
        self.play(FadeIn(h), FadeIn(user), FadeIn(svs))

        self.cue(1)
        login = txt("log in", 16, HI).next_to(user, UP, buff=0.15)
        self.play(FadeIn(login), self.trip([pos(user[0]), pos(svs[0][0])], HI), run_time=1.3)

        self.cue(2)
        sess = chip("session: you ✓", HI, 14).next_to(svs[0], RIGHT, buff=0.25)
        st = term("state", "what a server remembers about you").move_to([3.95, 2.1, 0])
        self.play(FadeIn(sess, scale=0.8))
        self.play(FadeIn(st, shift=UP * 0.15))

        self.cue(3)
        self.play(FadeOut(login), self.trip([pos(user[0]), pos(svs[2][0])], HI), run_time=1.3)

        self.cue(4)
        who = chip("who are you?", BAD, 14).next_to(svs[2], RIGHT, buff=0.25)
        self.play(FadeIn(who, scale=0.8), Indicate(svs[2], color=BAD))

        self.cue(5)
        fix = term("stateless servers", "keep no user data of their own", GOOD).move_to([3.95, 0.2, 0])
        self.play(FadeIn(fix, shift=UP * 0.15))

        self.cue(6)
        ex = txt("no sessions, no profile pictures", 16, DIM).next_to(fix, DOWN, buff=0.15)
        self.play(FadeIn(ex))

        self.cue(7)
        store = VGroup(db("shared store", PURPLE_, 2.3, 0.7, 16),
                       txt("a database or a cache", 15, DIM)).arrange(DOWN, buff=0.15).move_to([-0.6, -2.55, 0])
        self.play(FadeOut(who), FadeIn(store[0]), FadeIn(store[1]),
                  sess.animate.scale(0.9).move_to(store[0].get_right() + RIGHT * 1.1), run_time=1.0)
        links = VGroup(*[DashedLine(s[0].get_left() + LEFT * 0.15, store[0].get_left() + LEFT * 0.05, color=PURPLE_,
                                    stroke_width=2) for s in svs])
        self.play(Create(links), run_time=0.8)

        self.cue(8)
        self.play(self.trip([pos(user[0]), pos(svs[2][0]), store[0].get_left()], GOOD), run_time=1.2)
        okc = chip("hi again ✓", GOOD, 14).next_to(svs[2], RIGHT, buff=0.25)
        self.play(FadeIn(okc, scale=0.8))
        anyreq = txt("any server can handle any request", 18, GOOD).next_to(fix, DOWN, buff=0.75)
        self.play(FadeIn(anyreq))
        self.finish()

    # ---------------------------------------------------------------- 11 roadmap
    def s11(self):
        self.wipe()
        self.seg(11)
        h = header("New questions, and the rest of the series")
        self.play(FadeIn(h))
        qs = [("How does a phone find your servers?", "Ep 2 · DNS and CDNs", PURPLE_),
              ("How do you split requests between them?", "Ep 3 · Load balancers", NEG),
              ("What happens when parts break?", "Ep 4–5 · Availability, CAP", BAD),
              ("How do you keep shared data fast and correct?", "Ep 6–9 · Databases, caches, queues", GOOD)]
        cards = []
        for k, (q, ep, color) in enumerate(qs):
            x = -3.35 if k % 2 == 0 else 3.35
            y = 1.25 if k < 2 else -0.95
            c = card(6.3, 1.8, [x, y, 0]).set_stroke(color, 2)
            t = fit(txt(q, 19), 5.9).move_to([x, y + 0.3, 0])
            e = txt(ep, 17, color).move_to([x, y - 0.35, 0])
            cards.append(VGroup(c, t, e))
        for k in range(4):
            self.cue(k + 1)
            self.play(FadeIn(cards[k], shift=UP * 0.2), run_time=0.7)

        self.cue(5)
        nxt = banner("Next: how your phone finds one server among billions", HI, y=-2.85)
        self.play(*[c.animate.set_opacity(0.35) for c in cards[1:]], Indicate(cards[0], color=PURPLE_, scale_factor=1.04))
        self.play(FadeIn(nxt, shift=UP * 0.15))
        self.finish(1.2)
