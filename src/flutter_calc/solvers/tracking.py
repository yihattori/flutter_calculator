"""Eigenvalue branch tracking (mode matching) across the velocity sweep.

The p-k eigenproblem is solved independently at each airspeed, producing an unordered
set of complex eigenpairs. To draw continuous V-g / V-f curves we must decide which
eigenvalue at velocity ``U_{i+1}`` continues which branch from ``U_i``. We match by the
Modal Assurance Criterion (MAC) between complex eigenvectors, which is robust to the
frequency coalescence that is the hallmark of flutter (where nearest-frequency matching
fails).
"""

from __future__ import annotations

import numpy as np


def mac(v1: np.ndarray, v2: np.ndarray) -> float:
    """Modal Assurance Criterion between two complex mode shapes, in [0, 1]."""
    num = np.abs(np.vdot(v1, v2)) ** 2
    den = np.real(np.vdot(v1, v1) * np.vdot(v2, v2))
    return float(num / den) if den > 0 else 0.0


def best_match(reference: np.ndarray, candidates: np.ndarray,
               exclude: np.ndarray | None = None, exclude_threshold: float = 0.9) -> int:
    """Index of the column of ``candidates`` (n, m) best matching ``reference`` (n,).

    With ``exclude`` given, columns nearly parallel to it (MAC > ``exclude_threshold``)
    are skipped -- used to repair two branches that collapsed onto the same eigenvector.
    If every column would be excluded the plain best match is returned.
    """
    scores = [mac(reference, candidates[:, j]) for j in range(candidates.shape[1])]
    if exclude is not None:
        masked = [-1.0 if mac(exclude, candidates[:, j]) > exclude_threshold else sc
                  for j, sc in enumerate(scores)]
        if max(masked) >= 0.0:
            scores = masked
    return int(np.argmax(scores))
