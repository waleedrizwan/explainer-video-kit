"""How an electric motor works, as a 2D cross-section: magnets, field, a loop seen end on,
the push on each side, and the split ring that flips the current every half turn.

    python narration.py
    manim -pql --fps 120 scene.py Explainer     # fast preview (480p, 120 fps)
    manim -qh --fps 120 scene.py Explainer      # final -> media/videos/scene/1080p120/Explainer.mp4
"""
import sys
from pathlib import Path

import numpy as np
from manim import *

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if (p / "vidkit").is_dir())))
from vidkit import NarratedScene, vertical  # noqa: E402,F401

N_RED, S_BLUE = "#E5484D", "#3E7BFA"
COPPER, FIELD = "#D9822B", "#8A93A6"
C = np.array([-1.7, -0.25, 0])     # rotor centre
R = 1.45                           # rotor radius
RING = np.array([4.7, 0.2, 0])     # split-ring inset centre


def current_mark(sign):
    """Wire end seen head on: dot = current out of the screen, cross = into it."""
    disc = Circle(0.3, color=WHITE, stroke_width=4).set_fill(COPPER, 1)
    if sign > 0:
        return VGroup(disc, Dot(radius=0.08, color=WHITE))
    d = 0.17
    return VGroup(disc, Line([-d, -d, 0], [d, d, 0], stroke_width=5), Line([-d, d, 0], [d, -d, 0], stroke_width=5))


class Explainer(NarratedScene):
    project_dir = Path(__file__).parent

    def caption(self, text, color=WHITE):
        """Swap the one-line explanation at the bottom of the frame."""
        new = Text(text, font_size=32, color=color).to_edge(DOWN, buff=0.4)
        anims = [FadeIn(new, shift=UP * 0.2)]
        if self.cap is not None:
            anims.append(FadeOut(self.cap))
        self.cap = new
        return AnimationGroup(*anims)

    def construct(self):
        self.cap = None
        theta = ValueTracker(0)        # clockwise rotation of the rotor, radians
        state = {"signA": 1, "auto": False, "current": False}

        def pos_a():
            a = PI - theta.get_value()
            return C + R * np.array([np.cos(a), np.sin(a), 0])

        def pos_b():
            return 2 * C - pos_a()

        def sign_a():
            if state["auto"]:
                return 1 if pos_a()[0] < C[0] else -1
            return state["signA"]

        # ---- parts
        title = Text("How an electric motor works", font_size=46, weight=BOLD).to_edge(UP, buff=0.35)
        mag_n = VGroup(Rectangle(width=1.2, height=3.4, color=N_RED).set_fill(N_RED, 0.85),
                       Text("N", font_size=60, weight=BOLD)).move_to(C + LEFT * 3.35)
        mag_s = VGroup(Rectangle(width=1.2, height=3.4, color=S_BLUE).set_fill(S_BLUE, 0.85),
                       Text("S", font_size=60, weight=BOLD)).move_to(C + RIGHT * 3.35)
        path = DashedVMobject(Circle(R, color=GREY_B, stroke_width=2).move_to(C), num_dashes=40)
        axle = Dot(C, radius=0.12, color=GREY_A)
        field = VGroup(*[
            Arrow(C + np.array([-2.7, y, 0]), C + np.array([2.7, y, 0]), buff=0, color=FIELD,
                  stroke_width=3, max_tip_length_to_length_ratio=0.04).set_opacity(0.55)
            for y in np.linspace(-1.35, 1.35, 6)])
        field_label = Text("field: N → S", font_size=26, color=FIELD).next_to(mag_s, UP, buff=0.2)

        loop = always_redraw(lambda: Line(pos_a(), pos_b(), color=COPPER, stroke_width=8))
        wire_a = always_redraw(lambda: current_mark(sign_a()).move_to(pos_a()) if state["current"]
                               else Circle(0.3, color=WHITE, stroke_width=4).set_fill(COPPER, 1).move_to(pos_a()))
        wire_b = always_redraw(lambda: current_mark(-sign_a()).move_to(pos_b()) if state["current"]
                               else Circle(0.3, color=WHITE, stroke_width=4).set_fill(COPPER, 1).move_to(pos_b()))

        def force_arrow(get_pos, sign):
            p = get_pos()
            return Arrow(p + UP * 0.3 * sign, p + UP * 1.05 * sign, buff=0, color=YELLOW, stroke_width=8,
                         max_tip_length_to_length_ratio=0.35)
        force_a = always_redraw(lambda: force_arrow(pos_a, sign_a()))
        force_b = always_redraw(lambda: force_arrow(pos_b, -sign_a()))

        # split-ring inset: two copper halves turn with the rotor, brushes stay put
        def ring():
            t = theta.get_value()
            gap = 0.22
            halves = VGroup(*[
                Arc(radius=0.6, start_angle=centre - (PI / 2 - gap), angle=PI - 2 * gap,
                    color=COPPER, stroke_width=14, arc_center=RING)
                for centre in (PI - t, -t)])
            return halves
        ring_m = always_redraw(ring)
        brush_p = Rectangle(width=0.35, height=0.3, color=GREY_A).set_fill(GREY_D, 1).move_to(RING + LEFT * 0.88)
        brush_m = Rectangle(width=0.35, height=0.3, color=GREY_A).set_fill(GREY_D, 1).move_to(RING + RIGHT * 0.88)
        batt = VGroup(Line(LEFT * 0.35, RIGHT * 0.35, stroke_width=6), Line(LEFT * 0.2, RIGHT * 0.2, stroke_width=6)
                      .shift(DOWN * 0.2)).move_to(RING + DOWN * 1.9)
        plus = Text("+", font_size=44, weight=BOLD, color=N_RED).next_to(batt, LEFT, buff=0.5).shift(UP * 0.35)
        minus = Text("−", font_size=44, weight=BOLD, color=S_BLUE).next_to(batt, RIGHT, buff=0.5).shift(UP * 0.35)
        wires = VGroup(
            VMobject(stroke_width=3, color=GREY_B).set_points_as_corners(
                [brush_p.get_left(), brush_p.get_left() + LEFT * 0.4, RING + np.array([-1.28, -1.9, 0]),
                 batt[0].get_left()]),
            VMobject(stroke_width=3, color=GREY_B).set_points_as_corners(
                [brush_m.get_right(), brush_m.get_right() + RIGHT * 0.4, RING + np.array([1.28, -2.1, 0]),
                 batt[1].get_right()]))
        ring_title = Text("split ring", font_size=28).next_to(RING, UP, buff=0.95)
        ring_note = Text("(on the same axle)", font_size=20, color=GREY_B).next_to(ring_title, DOWN, buff=0.08)
        inset = VGroup(ring_title, ring_note, brush_p, brush_m, wires, batt, plus, minus)

        legend = VGroup(
            VGroup(current_mark(1).scale(0.7), Text("out of screen", font_size=22)).arrange(RIGHT, buff=0.15),
            VGroup(current_mark(-1).scale(0.7), Text("into screen", font_size=22)).arrange(RIGHT, buff=0.15),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.15).to_corner(UL, buff=0.4).shift(DOWN * 0.9)

        # ---- scene 0: title, draw the motor
        self.seg(0)
        self.play(Write(title), run_time=1.2)
        self.cue(1)
        self.play(FadeIn(mag_n, mag_s), Create(path), FadeIn(axle), run_time=1.5)
        self.finish()

        # ---- scene 1: magnets and field
        self.seg(1)
        self.play(Indicate(mag_n, scale_factor=1.08), Indicate(mag_s, scale_factor=1.08), self.caption("Two permanent magnets"))
        self.cue(1)
        self.play(Indicate(mag_n[1], color=N_RED), run_time=0.8)
        self.play(Indicate(mag_s[1], color=S_BLUE), run_time=0.8)
        self.cue(2)
        self.play(LaggedStart(*[GrowArrow(a) for a in field], lag_ratio=0.12), FadeIn(field_label),
                  self.caption("Field points from N to S"), run_time=1.8)
        self.finish()

        # ---- scene 2: the loop
        self.seg(2)
        self.add(loop)
        self.play(GrowFromCenter(loop), FadeIn(wire_a, wire_b), self.caption("A copper loop on a spinning axle"))
        self.add(wire_a, wire_b, axle)
        self.cue(1)
        self.play(Circumscribe(wire_a, Circle, color=YELLOW), Circumscribe(wire_b, Circle, color=YELLOW),
                  self.caption("Seen end on: we only see its two sides"), run_time=1.5)
        self.finish()

        # ---- scene 3: battery and current
        self.seg(3)
        self.play(FadeIn(inset), FadeIn(ring_m), self.caption("Connect a battery"))
        self.add(ring_m)
        self.cue(1)
        state["current"] = True
        self.play(Flash(pos_a(), color=YELLOW), Flash(pos_b(), color=YELLOW), FadeIn(legend),
                  self.caption("Current: out on the left, in on the right"))
        self.finish()

        # ---- scene 4: the push
        self.seg(4)
        self.play(self.caption("Current + magnetic field = a push", color=YELLOW))
        self.cue(1)
        self.play(GrowArrow(force_a), run_time=0.8)
        self.add(force_a)
        self.play(GrowArrow(force_b), run_time=0.8)
        self.add(force_b)
        self.cue(2)
        self.play(theta.animate.set_value(PI * 0.4), self.caption("Up on one side, down on the other: it turns"),
                  run_time=1.4, rate_func=rate_functions.ease_in_quad)
        self.finish()

        # ---- scene 5: dead spot and the split ring
        self.seg(5)
        self.play(theta.animate.set_value(PI / 2), run_time=1.2, rate_func=rate_functions.ease_out_quad)
        self.play(self.caption("Upright: the pushes only stretch it", color=GREY_A))
        self.cue(1)
        self.play(Indicate(ring_m, color=YELLOW), self.caption("The split ring flips the current", color=YELLOW))
        state["signA"] = -1
        self.play(Flash(pos_a(), color=YELLOW), Flash(pos_b(), color=YELLOW),
                  Flash(brush_p.get_center(), color=YELLOW), Flash(brush_m.get_center(), color=YELLOW))
        self.finish()

        # ---- scene 6: keeps spinning
        self.seg(6)
        state["auto"] = True
        self.play(theta.animate.set_value(PI * 1.0), self.caption("Pushes turn it the same way again"),
                  run_time=1.6, rate_func=rate_functions.ease_in_quad)
        speed = 2 * PI * 0.6                    # rad/s once it is up to speed
        theta.add_updater(lambda m, dt: m.increment_value(speed * dt))
        self.add(theta)
        self.cue(1)
        self.play(self.caption("Flip every half turn = it keeps spinning", color=YELLOW))
        self.finish()

        # ---- scene 7: outro, rotor keeps going
        self.seg(7)
        self.play(self.caption("No fuel."))
        self.cue(1)
        self.play(self.caption("No explosions."))
        self.cue(2)
        self.play(self.caption("Just magnets + a current that flips on time", color=YELLOW))
        self.finish(1.2)
