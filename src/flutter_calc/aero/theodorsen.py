"""Theodorsen strip-theory aerodynamics backend (incompressible, unswept).

This is the validation oracle: a closed-form 2-D unsteady aero model integrated across
the span ("strip theory"). It reproduces Wright & Cooper's binary-flutter behaviour and,
at zero sweep and low Mach, it anchors the DLM backend: the two must agree in the
high-aspect-ratio incompressible limit.

Sectional model (per unit span)
-------------------------------
For a 2-D section of semichord ``b`` with the elastic axis at ``a`` semichords aft of
midchord, heave ``w`` (positive up) and twist ``alpha`` (nose-up), the Theodorsen lift
``L`` (up) and moment about the EA ``M`` (nose-up) for harmonic motion are linear in
``(w, alpha)``. Written nondimensionally (divided by ``q_dyn = 1/2 rho U^2``) the
2x2 sectional matrix ``Asec`` such that ``[L, M] = q_dyn * Asec @ [w, alpha]`` has
entries (with ``C = C(k)`` Theodorsen's function, ``k`` the *local* reduced frequency):

    Asec00 = 2 pi k^2 - 4 pi i k C
    Asec01 = 2 pi i k b + 2 pi a b k^2 + 4 pi b C [1 + i (1/2 - a) k]
    Asec10 = 2 pi a b k^2 - 4 pi i k b (a + 1/2) C
    Asec11 = 2 pi b^2 [-i (1/2 - a) k + (1/8 + a^2) k^2] + 4 pi b^2 (a + 1/2) C [1 + i (1/2 - a) k]

These follow from the standard apparent-mass + circulatory decomposition with the
heave-up / nose-up sign convention of ``conventions.py`` (the usual heave-down forms
with ``h_dot -> -w_dot`` substituted). They reduce to the steady result
``L = 2 pi q_dyn b * (2 alpha)`` as ``k -> 0``.

Generalized projection
----------------------
Because the assumed shapes are analytic, the structure->aero coupling is a direct
evaluation: with heave ``w(y) = sum_i phi_i qb_i`` and twist ``alpha(y) = sum_j psi_j qt_j``,
the generalized force blocks are span integrals of the sectional matrix weighted by the
shapes, e.g. ``Qhh_bb[i,l] = integral phi_i Asec00 phi_l dy``. No surface spline needed.
For a tapered wing each strip uses its *local* reduced frequency ``k_local = k b(y)/b_ref``.
"""

from __future__ import annotations

import numpy as np
from scipy.special import hankel2

from ..geometry import WingGeometry


def _theodorsen_vec(k: np.ndarray) -> np.ndarray:
    """Vectorised Theodorsen C(k) over an array of (positive) reduced frequencies."""
    k = np.asarray(k, dtype=float)
    out = np.ones(k.shape, dtype=complex)        # C(0) = 1
    nz = k > 1e-9
    h1 = hankel2(1, k[nz])
    h0 = hankel2(0, k[nz])
    out[nz] = h1 / (h1 + 1j * h0)
    return out


def _leggauss_01(n: int):
    x, w = np.polynomial.legendre.leggauss(n)
    return 0.5 * (x + 1.0), 0.5 * w


class TheodorsenStripAero:
    """Strip-theory ``AeroModel`` for a cantilever wing with assumed shapes."""

    def __init__(self, geometry: WingGeometry, bending_shapes, torsion_shapes,
                 b_ref: float | None = None, n_quad: int = 48):
        self.geometry = geometry
        self.bending_shapes = bending_shapes
        self.torsion_shapes = torsion_shapes
        self.nb = len(bending_shapes)
        self.nt = len(torsion_shapes)
        self.n_dof = self.nb + self.nt

        eta, w = _leggauss_01(n_quad)
        self.eta = eta
        self.w = w
        self.L = geometry.semi_span
        y = eta * self.L
        props = geometry.section_properties(y)
        self.b = props["b"]                                    # local semichord (nq,)
        self.a = (props["ea_frac"] - 0.5) / 0.5                # Theodorsen a (nq,)
        self.b_ref = float(self.b[0]) if b_ref is None else float(b_ref)

        # Shape values at the quadrature nodes (rows = shapes, cols = nodes).
        self.Phi = np.array([s(eta) for s in bending_shapes])  # (nb, nq)
        self.Psi = np.array([s(eta) for s in torsion_shapes])  # (nt, nq)

    def _sectional(self, kl: np.ndarray):
        """Nondimensional sectional matrix entries at local reduced frequencies ``kl``."""
        b, a = self.b, self.a
        Ck = _theodorsen_vec(kl)
        half_minus_a = 0.5 - a
        A00 = 2 * np.pi * kl ** 2 - 4j * np.pi * kl * Ck
        A01 = (2j * np.pi * kl * b + 2 * np.pi * a * b * kl ** 2
               + 4 * np.pi * b * Ck * (1 + 1j * half_minus_a * kl))
        A10 = 2 * np.pi * a * b * kl ** 2 - 4j * np.pi * kl * b * (a + 0.5) * Ck
        A11 = (2 * np.pi * b ** 2 * (-1j * half_minus_a * kl + (0.125 + a ** 2) * kl ** 2)
               + 4 * np.pi * b ** 2 * (a + 0.5) * Ck * (1 + 1j * half_minus_a * kl))
        return A00, A01, A10, A11

    def Qhh(self, k: float, Ma: float = 0.0) -> np.ndarray:
        """Nondimensional generalized aero matrix at reference reduced frequency ``k``.

        ``Ma`` is ignored (Theodorsen is incompressible); it is accepted only to satisfy
        the :class:`~flutter_calc.aero.base.AeroModel` contract.
        """
        k = float(k)
        kl = k * self.b / self.b_ref                # local reduced frequency per strip
        A00, A01, A10, A11 = self._sectional(kl)
        jac = self.L * self.w                        # span integration weights
        Phi, Psi = self.Phi, self.Psi

        Qbb = (Phi * (jac * A00)) @ Phi.T
        Qbt = (Phi * (jac * A01)) @ Psi.T
        Qtb = (Psi * (jac * A10)) @ Phi.T
        Qtt = (Psi * (jac * A11)) @ Psi.T

        nb, n = self.nb, self.n_dof
        Q = np.zeros((n, n), dtype=complex)
        Q[:nb, :nb] = Qbb
        Q[:nb, nb:] = Qbt
        Q[nb:, :nb] = Qtb
        Q[nb:, nb:] = Qtt
        return Q
