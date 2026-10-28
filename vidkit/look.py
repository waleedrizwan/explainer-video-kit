"""The house look for Manim videos: dark screen, rounded cards, Avenir Next, a soft palette.

    from vidkit.look import *      # colours, txt(), panel(), card(); sets the background

    title = txt("biggest so far", 28, HI)
    self.add(panel(group))          # rounded card behind a group
    self.add(card(6, 3, LEFT * 3))  # card of a fixed size at a spot
"""
from manim import MEDIUM, WHITE, RoundedRectangle, Text, config

POS, NEG = "#F2C230", "#4C8DFF"                               # warm yellow / clear blue
HI, GOOD, BAD, DIM = "#FFD166", "#5BD38A", "#FF6B6B", "#8A93A6"  # highlight, right, wrong, secondary text
SCREEN, PANEL, EDGE = "#0E1014", "#171A21", "#2A2F3A"        # background, card fill, card border
FONT = "Avenir Next"
TEXT_SCALE = 1.25        # every txt() size is scaled by this
config.background_color = SCREEN


def txt(s, size=24, color=WHITE, **kw):
    kw.setdefault("font", FONT)
    kw.setdefault("weight", MEDIUM)
    return Text(s, font_size=round(size * TEXT_SCALE), color=color, **kw)


def panel(mob, buff=0.3, **kw):
    """Rounded dark card behind a group."""
    return RoundedRectangle(corner_radius=0.2, width=mob.width + 2 * buff, height=mob.height + 2 * buff,
                            stroke_color=EDGE, stroke_width=1.5, **kw).set_fill(PANEL, 1).move_to(mob).set_z_index(-5)


def card(w, h, center):
    """Rounded dark card of a fixed size, centred at `center`."""
    return RoundedRectangle(corner_radius=0.2, width=w, height=h, stroke_color=EDGE, stroke_width=1.5) \
        .set_fill(PANEL, 1).move_to(center).set_z_index(-5)


__all__ = ["POS", "NEG", "HI", "GOOD", "BAD", "DIM", "SCREEN", "PANEL", "EDGE", "FONT", "TEXT_SCALE",
           "txt", "panel", "card"]
