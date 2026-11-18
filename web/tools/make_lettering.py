"""Make the hand-written "Take a Pic" title as an SVG of plain paths (one per letter + an underline swash).

    .venv/bin/python web/tools/make_lettering.py ["Other text"]

Writes web/site/assets/lettering/take-a-pic.svg. The page animates each letter's outline being drawn,
then fills it, so it looks written by hand. Font: Nothing You Could Do (SIL Open Font License, web/tools/fonts/).
"""
import os
import sys

from fontTools.pens.boundsPen import BoundsPen
from fontTools.pens.svgPathPen import SVGPathPen
from fontTools.pens.transformPen import TransformPen
from fontTools.ttLib import TTFont

HERE = os.path.dirname(os.path.abspath(__file__))
FONT = os.path.join(HERE, "fonts", "NothingYouCouldDo.ttf")
OUT = os.path.join(os.path.dirname(HERE), "site", "assets", "lettering", "take-a-pic.svg")
TEXT = sys.argv[1] if len(sys.argv) > 1 else "Take a Pic"
TRACKING = -70          # font units; negative pulls the cursive letters into each other
PAD = 60


def main():
    font = TTFont(FONT)
    gs = font.getGlyphSet()
    cmap = font.getBestCmap()
    hmtx = font["hmtx"]
    upm = font["head"].unitsPerEm

    letters, x = [], 0
    for ch in TEXT:
        gname = cmap.get(ord(ch))
        adv = hmtx[gname][0]
        if ch != " ":
            pen = SVGPathPen(gs)
            # font units are y-up; SVG is y-down
            gs[gname].draw(TransformPen(pen, (1, 0, 0, -1, x, 0)))
            bp = BoundsPen(gs)
            gs[gname].draw(TransformPen(bp, (1, 0, 0, -1, x, 0)))
            letters.append((ch, pen.getCommands(), bp.bounds))
        x += adv + TRACKING

    xs = [b[0] for _, _, b in letters] + [b[2] for _, _, b in letters]
    ys = [b[1] for _, _, b in letters] + [b[3] for _, _, b in letters]
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    # a quick hand-drawn swash under the words (like the reference lettering's zig-zag underline)
    w = x1 - x0
    by = y1 + 0.10 * upm
    sx = lambda f: f"{x0 + f * w:.0f}"
    swash = (f"M{sx(0.10)},{by + 10:.0f} C{sx(0.30)},{by + 60:.0f} {sx(0.50)},{by - 10:.0f} {sx(0.66)},{by - 70:.0f} "
             f"L{sx(0.50)},{by + 40:.0f} L{sx(0.66)},{by + 15:.0f} L{sx(0.56)},{by + 110:.0f}")
    vb = (x0 - PAD, y0 - PAD, w + 2 * PAD, (by + 150) - (y0 - PAD))

    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="{vb[0]:.0f} {vb[1]:.0f} {vb[2]:.0f} {vb[3]:.0f}" '
             f'role="img" aria-labelledby="lettering-title" class="lettering">',
             f'<title id="lettering-title">{TEXT}</title>']
    for i, (ch, d, _) in enumerate(letters):
        parts.append(f'<path class="letter" style="--i:{i}" data-char="{ch}" d="{d}"/>')
    parts.append(f'<path class="swash" style="--i:{len(letters)}" d="{swash}"/>')
    parts.append("</svg>")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        f.write("\n".join(parts) + "\n")
    print("wrote", OUT, f"({len(letters)} letters)")


if __name__ == "__main__":
    main()
