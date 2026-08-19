"""Shared building blocks for the PDF guide.

``make_usage_guide_pdf.py`` needs two things, and getting either of them subtly wrong is
easy:

* **tables that wrap.** ReportLab will not wrap a bare string in a table cell -- it
  happily runs it off the right edge of the page. Every cell must be a ``Paragraph``.
  :func:`table` does that for you, so callers can keep passing plain strings.
* **figures scaled to a column width**, degrading to a visible placeholder rather than a
  crash when the image has not been generated yet.
"""

from __future__ import annotations

import os
import sys

from reportlab.lib import colors
from reportlab.lib.fonts import addMapping
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Image, KeepTogether, Paragraph, Spacer, Table, TableStyle

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from figures import IMG  # noqa: E402


def _register_fonts():
    """Embed DejaVu so Greek renders everywhere, falling back to the base-14 fonts.

    ReportLab's built-in Helvetica is WinAnsi-encoded and has no Greek, so a stray
    ``&#952;`` gets silently re-routed to the Symbol font -- which is neither embedded
    nor reliably rendered by every viewer. DejaVu ships with matplotlib, which is
    already a hard dependency, so we can embed a proper Unicode font at no extra cost
    and produce a PDF that looks the same everywhere.

    Returns ``(body_font, mono_font)``.
    """
    try:
        import matplotlib
        ttf = os.path.join(os.path.dirname(matplotlib.__file__), "mpl-data", "fonts",
                           "ttf")
        faces = {
            "DejaVuSans": "DejaVuSans.ttf",
            "DejaVuSans-Bold": "DejaVuSans-Bold.ttf",
            "DejaVuSans-Oblique": "DejaVuSans-Oblique.ttf",
            "DejaVuSans-BoldOblique": "DejaVuSans-BoldOblique.ttf",
            "DejaVuSansMono": "DejaVuSansMono.ttf",
            "DejaVuSansMono-Bold": "DejaVuSansMono-Bold.ttf",
        }
        for name, fname in faces.items():
            path = os.path.join(ttf, fname)
            if not os.path.exists(path):
                raise FileNotFoundError(path)
            pdfmetrics.registerFont(TTFont(name, path))
        # (family, bold, italic) -> face, so <b> and <i> resolve inside a Paragraph.
        addMapping("DejaVuSans", 0, 0, "DejaVuSans")
        addMapping("DejaVuSans", 1, 0, "DejaVuSans-Bold")
        addMapping("DejaVuSans", 0, 1, "DejaVuSans-Oblique")
        addMapping("DejaVuSans", 1, 1, "DejaVuSans-BoldOblique")
        addMapping("DejaVuSansMono", 0, 0, "DejaVuSansMono")
        addMapping("DejaVuSansMono", 1, 0, "DejaVuSansMono-Bold")
        return "DejaVuSans", "DejaVuSansMono"
    except Exception as exc:                                  # pragma: no cover
        print(f"  note: DejaVu unavailable ({exc}); falling back to Helvetica")
        return "Helvetica", "Courier"


BODY_FONT, MONO_FONT = _register_fonts()
BOLD_FONT = BODY_FONT + "-Bold" if BODY_FONT == "DejaVuSans" else "Helvetica-Bold"

_ss = getSampleStyleSheet()

CELL = ParagraphStyle("CELL", parent=_ss["BodyText"], fontName=BODY_FONT,
                      fontSize=7.6, leading=10.0, spaceAfter=0, spaceBefore=0)
CELL_HEAD = ParagraphStyle("CELL_HEAD", parent=CELL, fontName=BOLD_FONT)
CELL_MONO = ParagraphStyle("CELL_MONO", parent=CELL, fontName=MONO_FONT, fontSize=7.2,
                           leading=9.8)
CAPTION = ParagraphStyle("CAPTION", parent=_ss["BodyText"], fontName=BODY_FONT,
                         fontSize=7.9, leading=10.6, alignment=1,
                         textColor=colors.HexColor("#444444"),
                         spaceBefore=3, spaceAfter=10)


def restyle(style, **kw):
    """Clone a sample-sheet style onto the embedded fonts."""
    name = kw.pop("name", style.name + "_dv")
    mono = kw.pop("mono", False)
    bold = kw.pop("bold", False)
    font = MONO_FONT if mono else (BOLD_FONT if bold else BODY_FONT)
    return ParagraphStyle(name, parent=style, fontName=font, **kw)

_COMMON = [
    ("LINEBELOW", (0, -1), (-1, -1), 0.4, colors.black),
    ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ("LEFTPADDING", (0, 0), (-1, -1), 4),
    ("RIGHTPADDING", (0, 0), (-1, -1), 4),
    ("TOPPADDING", (0, 0), (-1, -1), 3),
    ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
]
#: With a header row: rule beneath it, zebra striping from the first body row.
TABLE_STYLE = TableStyle(_COMMON + [
    ("LINEBELOW", (0, 0), (-1, 0), 0.6, colors.black),
    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f2f2f2")]),
])
#: Headerless (a label/value list): no header rule, striping from row zero.
TABLE_STYLE_NOHEAD = TableStyle(_COMMON + [
    ("LINEABOVE", (0, 0), (-1, 0), 0.4, colors.black),
    ("ROWBACKGROUNDS", (0, 0), (-1, -1), [colors.white, colors.HexColor("#f2f2f2")]),
])


def table(rows, widths, mono_first_col: bool = False, header: bool = True):
    """A table whose cells wrap.

    ``rows[0]`` is the header unless ``header=False``, in which case the table reads as a
    label/value list and the first column is set in bold instead. Cells may be plain
    strings (with ``<br/>`` or ``\\n`` for a hard break, and the usual ``<b>``/``<i>``
    markup) or ready-made flowables, which are passed through untouched.
    """
    body = []
    for r, row in enumerate(rows):
        out = []
        for c, cellv in enumerate(row):
            if not isinstance(cellv, str):
                out.append(cellv)
                continue
            if (header and r == 0) or (not header and c == 0):
                style = CELL_HEAD
            elif mono_first_col and c == 0:
                style = CELL_MONO
            else:
                style = CELL
            out.append(Paragraph(cellv.replace("\n", "<br/>"), style))
        body.append(out)
    t = Table(body, colWidths=widths, hAlign="LEFT", repeatRows=1 if header else 0)
    t.setStyle(TABLE_STYLE if header else TABLE_STYLE_NOHEAD)
    return t


def fig(name, width_mm, caption=None, placeholder_style=None):
    """Place ``docs/img/<name>`` scaled to ``width_mm``, optionally with a caption."""
    path = os.path.join(IMG, name)
    if not os.path.exists(path):
        return [Paragraph(f"[figure {name} missing -- run: python docs/figures.py]",
                          placeholder_style or CAPTION)]
    img = Image(path)
    img.drawHeight = img.imageHeight * (width_mm * mm) / img.imageWidth
    img.drawWidth = width_mm * mm
    img.hAlign = "CENTER"
    if not caption:
        return [Spacer(1, 3), img]
    # Bind the caption to its figure, or reportlab will happily strand it on the
    # next page and leave a half-empty one behind.
    return [Spacer(1, 3), KeepTogether([img, Paragraph(caption, CAPTION)])]


def keep(*flowables):
    """Keep a heading with what follows it, so headings never end a page alone."""
    return KeepTogether(list(flowables))
