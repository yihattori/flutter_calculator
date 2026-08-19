"""Build a PanelAero ``aerogrid`` dict from a wing planform.

PanelAero (DLR, BSD 3-Clause -- see THIRD_PARTY_NOTICES.md) represents the lifting
surface as a set of boxes. For each box it needs:

    offset_j   collocation / receiving point at 3/4 of the *box* chord, mid-span
    offset_l   sending point at 1/4 of the box chord, mid-span
    offset_P1  inner end of the 1/4-chord bound-vortex line (left, smaller y)
    offset_P3  outer end of the 1/4-chord bound-vortex line (right, larger y)
    offset_k   load reference (1/4 chord, mid-span) -- same as offset_l here
    N          unit normal (n, 3); must have N_z > 0 ("panels left to right")
    A          box area (n,)
    l          box streamwise chord (n,)
    n          number of boxes

Geometry convention: x streamwise (aft +), y spanwise (tip +), z up. The wing lies in
the z = 0 plane (planar). Sweep shifts the leading edge by ``y * tan(sweep)``; taper is
handled by a chord distribution ``c(y)``. Boxes are uniform in span and chord fraction.
"""

from __future__ import annotations

import numpy as np

from ..geometry import WingGeometry


def build_aerogrid(geometry: WingGeometry, n_span: int = 12, n_chord: int = 6,
                   cosine_chord: bool = False) -> dict:
    """Construct an ``aerogrid`` dict and panel bookkeeping for ``geometry``.

    Returns the dict with the PanelAero keys plus helper arrays ``y_collocation`` and
    ``x_collocation`` (collocation-point coordinates) used by the coupling.
    """
    s = geometry.semi_span
    tanL = np.tan(np.deg2rad(geometry.sweep_deg))

    y_edges = np.linspace(0.0, s, n_span + 1)
    # chord-fraction edges (0..1), optionally cosine-clustered toward LE/TE
    if cosine_chord:
        theta = np.linspace(0.0, np.pi, n_chord + 1)
        f_edges = 0.5 * (1.0 - np.cos(theta))
    else:
        f_edges = np.linspace(0.0, 1.0, n_chord + 1)

    def chord(y):
        return float(geometry.section_properties(np.atleast_1d(y))["chord"][0])

    def x_le(y):
        return y * tanL  # leading edge swept by sweep_deg

    offset_j, offset_l, offset_P1, offset_P3 = [], [], [], []
    N, A, ll = [], [], []

    for i in range(n_span):
        y1, y2 = y_edges[i], y_edges[i + 1]
        ym = 0.5 * (y1 + y2)
        c1, c2, cm = float(chord(y1)), float(chord(y2)), float(chord(ym))
        xle1, xle2, xlem = x_le(y1), x_le(y2), x_le(ym)
        for jc in range(n_chord):
            fle, fte = f_edges[jc], f_edges[jc + 1]
            box_cm = (fte - fle) * cm                       # box chord at mid-span
            # bound vortex (1/4 of the box) endpoints at the span edges
            q1x = xle1 + fle * c1 + 0.25 * (fte - fle) * c1
            q3x = xle2 + fle * c2 + 0.25 * (fte - fle) * c2
            offset_P1.append([q1x, y1, 0.0])
            offset_P3.append([q3x, y2, 0.0])
            # sending (1/4) and collocation (3/4) at mid-span
            xj_le = xlem + fle * cm
            offset_l.append([xj_le + 0.25 * box_cm, ym, 0.0])
            offset_j.append([xj_le + 0.75 * box_cm, ym, 0.0])
            N.append([0.0, 0.0, 1.0])
            A.append(0.5 * (c1 + c2) * (fte - fle) * (y2 - y1))
            ll.append(box_cm)

    offset_j = np.array(offset_j)
    aerogrid = {
        "offset_j": offset_j,
        "offset_l": np.array(offset_l),
        "offset_k": np.array(offset_l).copy(),
        "offset_P1": np.array(offset_P1),
        "offset_P3": np.array(offset_P3),
        "N": np.array(N),
        "A": np.array(A),
        "l": np.array(ll),
        "n": n_span * n_chord,
        # helpers for coupling (not used by PanelAero itself):
        "x_collocation": offset_j[:, 0].copy(),
        "y_collocation": offset_j[:, 1].copy(),
    }
    return aerogrid
