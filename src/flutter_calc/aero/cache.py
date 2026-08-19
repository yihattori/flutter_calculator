"""Tabulate-and-interpolate wrapper for expensive aero models.

The p-k solver evaluates ``Qhh(k)`` thousands of times while iterating reduced
frequency over the airspeed sweep. For the DLM that means inverting a panel-sized
matrix each call -- far too slow to use directly, and the bottleneck in any parameter
study.

``TabulatedAero`` precomputes ``Qhh`` on a fixed reduced-frequency grid *once* and
linearly interpolates (real and imaginary parts) for any query ``k``. It honours the
:class:`~flutter_calc.aero.base.AeroModel` contract, so the solver cannot tell the
difference. In a parameter sweep, structural-only ("aero-invariant") changes reuse the
*same* table; only planform changes (sweep, AR, taper) require re-tabulation.
"""

from __future__ import annotations

import numpy as np


def default_k_grid(k_max: float = 20.0, n: int = 40) -> np.ndarray:
    """A reduced-frequency grid clustered near zero, where C(k) varies fastest."""
    return np.concatenate([[0.0], np.logspace(-3, np.log10(k_max), n)])


class TabulatedAero:
    """Wrap an aero model, caching ``Qhh`` on a ``k`` grid and interpolating."""

    def __init__(self, aero, k_grid=None, Ma: float = 0.0):
        self.n_dof = aero.n_dof
        self.b_ref = aero.b_ref
        self.Ma = Ma
        self.k_grid = np.asarray(default_k_grid() if k_grid is None else k_grid, dtype=float)
        self.k_grid = np.unique(self.k_grid)
        self.table = np.array([aero.Qhh(float(k), Ma) for k in self.k_grid])  # (nk, n, n)
        #: queries above the grid top so far -- they are clamped to ``k_grid[-1]``, which
        #: freezes ``Qhh`` for high-frequency branches at low speed. A nonzero count means
        #: the grid's ``k_max`` should be raised for fully resolved branch damping.
        self.n_clamped_high = 0

    def Qhh(self, k: float, Ma: float = 0.0) -> np.ndarray:
        """Interpolated generalized aero matrix at reduced frequency ``k``."""
        kg = self.k_grid
        if k > kg[-1]:
            self.n_clamped_high += 1
        k = float(np.clip(k, kg[0], kg[-1]))
        j = int(np.clip(np.searchsorted(kg, k), 1, len(kg) - 1))
        k0, k1 = kg[j - 1], kg[j]
        t = 0.0 if k1 == k0 else (k - k0) / (k1 - k0)
        return (1.0 - t) * self.table[j - 1] + t * self.table[j]
