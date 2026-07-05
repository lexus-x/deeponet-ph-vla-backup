"""
svgkit.py — tiny, dependency-free helper to hand-author clean, EDITABLE SVG
block-and-arrow architecture diagrams for the project report.

Every shape is emitted as a separate SVG element (rect / line / text / path), so
the resulting file is fully editable in Inkscape/Illustrator/Word (Word 2016+
renders SVG and can convert it to shapes; we also convert to EMF for native
Word editing). No external libraries required.

Usage pattern (see make_diagrams.py):
    c = Canvas(width, height, title="...")
    c.box(x, y, w, h, "Title", "subtitle", fill=BLUE)
    c.arrow(x1, y1, x2, y2, "label")
    c.save("out.svg")
"""
from __future__ import annotations

import html

# A calm, consistent palette (light fills + darker strokes).
BLUE   = ("#eaf2fb", "#2f6db0")   # backbones / inputs
ORANGE = ("#fdf1e6", "#c2772e")   # action heads
GREEN  = ("#e9f6ec", "#3a9457")   # losses / objectives
GREY   = ("#f2f3f5", "#7a7f87")   # data / misc
PURPLE = ("#f1ecfa", "#7a52c0")   # operators / maths
YELLOW = ("#fdf8e3", "#b79a1f")   # outputs / actions
RED    = ("#fdecec", "#c0392b")   # perturbations / OOD


class Canvas:
    def __init__(self, width: int, height: int, title: str = "", font: str = "DejaVu Sans, Arial, sans-serif"):
        self.w = width
        self.h = height
        self.font = font
        self.title = title
        self.els: list[str] = []
        self._marker = False

    # ---- primitives ---------------------------------------------------------
    def _esc(self, s: str) -> str:
        return html.escape(str(s))

    # --- text fitting helpers (keep every label inside its box) --------------
    @staticmethod
    def _char_w(size, bold):
        return size * (0.62 if bold else 0.57)   # conservative DejaVu Sans estimate

    def _text_w(self, s, size, bold):
        return len(s) * self._char_w(size, bold)

    def _fit_block(self, text, avail_w, base_size, bold):
        """Return (lines, size): word-wrap `text` to fit avail_w at base_size;
        if a single word is still too wide, shrink the font until it fits."""
        if not text:
            return [], base_size
        # honor explicit newlines first, then word-wrap each piece
        size = base_size
        while size >= 8:
            ok = True
            out = []
            for piece in text.split("\n"):
                words = piece.split()
                if not words:
                    out.append("")
                    continue
                if any(self._text_w(w, size, bold) > avail_w for w in words):
                    ok = False
                    break
                cur = ""
                for w in words:
                    t = (cur + " " + w).strip()
                    if self._text_w(t, size, bold) <= avail_w or not cur:
                        cur = t
                    else:
                        out.append(cur)
                        cur = w
                if cur:
                    out.append(cur)
            if ok:
                return out, size
            size -= 1
        return text.split("\n"), 8

    def box(self, x, y, w, h, title="", subtitle="", fill=BLUE, rx=10,
            title_size=15, sub_size=11, dashed=False, title_dy=None):
        f, s = fill
        dash = ' stroke-dasharray="6 4"' if dashed else ""
        self.els.append(
            f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" ry="{rx}" '
            f'fill="{f}" stroke="{s}" stroke-width="2"{dash}/>')
        cx = x + w / 2
        avail = w - 14                       # inner padding
        tlines, tsize = self._fit_block(title, avail, title_size, bold=True)
        slines, ssize = self._fit_block(subtitle, avail, sub_size, bold=False)
        th = tsize * 1.24
        sh = ssize * 1.26
        block = len(tlines) * th + len(slines) * sh
        if title_dy is not None:
            cur = y + title_dy
        else:
            cur = y + (h - block) / 2 + tsize * 0.8   # vertically center block
        for ln in tlines:
            self.text(cx, cur, ln, size=tsize, weight="bold")
            cur += th
        for ln in slines:
            self.text(cx, cur, ln, size=ssize, fill="#555")
            cur += sh

    def region(self, x, y, w, h, label="", color="#7a52c0", label_fill="#7a52c0"):
        """Dashed rounded boundary with a small label chip on the top-left edge.
        Used to outline the DeepONet head (inner) and the whole model (outer)."""
        self.els.append(
            f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="14" ry="14" '
            f'fill="none" stroke="{color}" stroke-width="2" stroke-dasharray="8 5"/>')
        if label:
            tw = self._text_w(label, 12, True) + 14
            self.els.append(
                f'<rect x="{x+16}" y="{y-11}" width="{tw}" height="22" rx="6" '
                f'fill="#ffffff" stroke="{color}" stroke-width="1.2"/>')
            self.text(x + 16 + tw / 2, y + 4, label, size=12, weight="bold", fill=label_fill)

    def text(self, x, y, s, size=13, weight="normal", fill="#222", anchor="middle", italic=False):
        """Render text; supports multi-line via '\\n' (lines stack downward from y)."""
        style = 'font-style:italic;' if italic else ''
        lines = str(s).split("\n")
        lh = size * 1.25
        for i, ln in enumerate(lines):
            self.els.append(
                f'<text x="{x}" y="{y + i*lh}" text-anchor="{anchor}" font-family="{self.font}" '
                f'font-size="{size}" font-weight="{weight}" fill="{fill}" style="{style}">{self._esc(ln)}</text>')

    def _ensure_marker(self):
        if not self._marker:
            self.els.insert(0,
                '<defs><marker id="arw" markerWidth="12" markerHeight="12" refX="9" refY="3.5" '
                'orient="auto"><path d="M0,0 L9,3.5 L0,7 z" fill="#444"/></marker>'
                '<marker id="arwo" markerWidth="12" markerHeight="12" refX="9" refY="3.5" '
                'orient="auto"><path d="M0,0 L9,3.5 L0,7 z" fill="#7a52c0"/></marker></defs>')
            self._marker = True

    def arrow(self, x1, y1, x2, y2, label="", color="#444", dashed=False, label_dy=-6, label_dx=0, curve=None):
        self._ensure_marker()
        marker = "arw" if color == "#444" else "arwo"
        dash = ' stroke-dasharray="6 4"' if dashed else ""
        if curve is None:
            self.els.append(
                f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" '
                f'stroke-width="2"{dash} marker-end="url(#{marker})"/>')
        else:
            self.els.append(
                f'<path d="M{x1},{y1} Q{curve[0]},{curve[1]} {x2},{y2}" fill="none" '
                f'stroke="{color}" stroke-width="2"{dash} marker-end="url(#{marker})"/>')
        if label:
            mx, my = (x1 + x2) / 2 + label_dx, (y1 + y2) / 2 + label_dy
            self.text(mx, my, label, size=11, fill=color)

    def line(self, x1, y1, x2, y2, color="#bbb", dashed=False, width=1.5):
        dash = ' stroke-dasharray="5 4"' if dashed else ""
        self.els.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="{width}"{dash}/>')

    def small_tokens(self, x, y, n, w=13, h=20, gap=4, fill=GREY):
        """Draw a little row of n token squares (for illustrating token sequences)."""
        f, s = fill
        for i in range(n):
            self.els.append(
                f'<rect x="{x + i*(w+gap)}" y="{y}" width="{w}" height="{h}" rx="3" '
                f'fill="{f}" stroke="{s}" stroke-width="1.3"/>')
        return x + n * (w + gap) - gap  # right edge

    def brace_label(self, x, y, s, size=11, fill="#666"):
        self.text(x, y, s, size=size, fill=fill)

    # ---- output -------------------------------------------------------------
    def save(self, path: str, margin: int = 28):
        # Pad the frame with a transparent margin so no content sits on the edge.
        # (LibreOffice's SVG->EMF import clips a few % at the right/bottom edge; the
        # margin keeps all boxes/text comfortably inside the visible area.)
        W, H = self.w + 2 * margin, self.h + 2 * margin
        head = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
                f'viewBox="{-margin} {-margin} {W} {H}" font-family="{self.font}">')
        bg = f'<rect x="{-margin}" y="{-margin}" width="{W}" height="{H}" fill="#ffffff"/>'
        title = ""
        if self.title:
            title = (f'<text x="{self.w/2}" y="28" text-anchor="middle" font-family="{self.font}" '
                     f'font-size="17" font-weight="bold" fill="#111">{self._esc(self.title)}</text>')
        body = "\n".join(self.els)
        with open(path, "w") as fh:
            fh.write(f"{head}\n{bg}\n{title}\n{body}\n</svg>\n")
        return path
