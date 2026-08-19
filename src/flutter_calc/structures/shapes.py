"""Assumed-shape library for the Rayleigh-Ritz beam.

Shapes are functions of the dimensionless spanwise coordinate ``eta = y / semi_span``
in ``[0, 1]``. They are admissible functions: they must satisfy the *geometric*
(essential) boundary conditions of a clamped root, but not the natural ones.

    Bending (clamped: h(0) = h'(0) = 0):  eta^2, eta^3, eta^4, ...
    Torsion (clamped: theta(0) = 0):      eta^1, eta^2, eta^3, ...

Each :class:`Shape` carries analytic first and second derivatives (with respect to
``eta``), which the energy integrals in ``ritz.py`` rescale to derivatives in ``y``.
Polynomials are used because they integrate exactly under Gauss-Legendre quadrature
and converge to the analytic cantilever frequencies from above as terms are added.
"""

from __future__ import annotations

import numpy as np


class Shape:
    """A 1-D assumed shape in ``eta in [0, 1]`` with analytic derivatives."""

    def __init__(self, coef, label: str = ""):
        self.poly = np.polynomial.Polynomial(np.asarray(coef, dtype=float))
        self._d1 = self.poly.deriv(1)
        self._d2 = self.poly.deriv(2)
        self.label = label

    def __call__(self, eta):
        return self.poly(eta)

    def d1(self, eta):
        """First derivative d/d(eta)."""
        return self._d1(eta)

    def d2(self, eta):
        """Second derivative d^2/d(eta)^2."""
        return self._d2(eta)

    def __repr__(self):
        return f"Shape({self.label or self.poly})"


def _monomial(power: int) -> np.ndarray:
    coef = np.zeros(power + 1)
    coef[power] = 1.0
    return coef


def polynomial_bending_shapes(n: int) -> list[Shape]:
    """``n`` bending shapes eta^2 .. eta^(n+1) (clamped-root admissible)."""
    return [Shape(_monomial(p), f"eta^{p}") for p in range(2, n + 2)]


def polynomial_torsion_shapes(n: int) -> list[Shape]:
    """``n`` torsion shapes eta^1 .. eta^n (clamped-root admissible)."""
    return [Shape(_monomial(p), f"eta^{p}") for p in range(1, n + 1)]
