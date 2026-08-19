"""Rayleigh-Ritz assembly of the generalized mass and stiffness matrices.

Given a :class:`~flutter_calc.geometry.WingGeometry` and sets of bending and torsion
assumed shapes, this builds the generalized matrices ``M`` and ``K`` for the
generalized coordinates ``q = [q_bending ; q_torsion]``.

Energy (straight, unswept beam)
-------------------------------
With heave ``h(y) = sum_i phi_i(eta) qb_i`` (positive up) and twist
``theta(y) = sum_j psi_j(eta) qt_j`` (nose-up), and a material point at chordwise
offset ``xi`` aft of the elastic axis displacing as ``z = h - xi*theta``:

    T = 1/2 ∫ [ m h_dot^2 - 2 S h_dot theta_dot + I_theta theta_dot^2 ] dy
    U = 1/2 ∫ EI (h'')^2 dy  +  1/2 ∫ GJ (theta')^2 dy

where ``S = m * x_theta`` is the static unbalance (CG aft of EA). This yields

    M_bb[i,k] =  ∫ m       phi_i  phi_k  dy
    M_bt[i,j] = -∫ S       phi_i  psi_j  dy
    M_tt[j,l] =  ∫ I_theta psi_j  psi_l  dy
    K_bb[i,k] =  ∫ EI      phi_i'' phi_k'' dy
    K_tt[j,l] =  ∫ GJ      psi_j'  psi_l'  dy

with derivatives taken with respect to ``y``. Using ``eta = y/L`` and exact
Gauss-Legendre quadrature on ``[0, 1]``: ``dy = L deta``, ``d/dy = (1/L) d/deta``,
``d^2/dy^2 = (1/L^2) d^2/deta^2``, giving the ``L`` powers below. Bending and torsion
are elastically uncoupled here (``K_bt = 0``): the structure is modelled as a
streamwise beam, so the elastic bend-twist coupling a swept box beam really has is NOT
represented -- a stated limitation. Sweep enters through the aerodynamics only (see
``aero/dlm.py``).

Concentrated masses (engine/pylon) add ``1/2 m_e (h_dot - xi theta_dot)^2`` plus their
own pitch inertia, evaluated at the attachment station.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..geometry import WingGeometry


def _leggauss_01(n: int):
    """Gauss-Legendre nodes/weights mapped from [-1, 1] to [0, 1]."""
    x, w = np.polynomial.legendre.leggauss(n)
    return 0.5 * (x + 1.0), 0.5 * w


@dataclass
class StructuralModel:
    """Assembled generalized matrices and the bookkeeping to interpret them.

    Structural damping is not modelled: the p-k solve assumes ``g = 0`` (conservative,
    the standard certification-analysis baseline), so there is no ``C`` matrix.
    """

    M: np.ndarray
    K: np.ndarray
    n_bending: int
    n_torsion: int
    bending_shapes: list
    torsion_shapes: list
    geometry: WingGeometry

    @property
    def n_dof(self) -> int:
        return self.n_bending + self.n_torsion

    def free_vibration(self):
        """Solve the symmetric generalized eigenproblem ``K v = omega^2 M v``.

        Returns ``(omega, V)`` with natural frequencies ``omega`` [rad/s] ascending
        and mass-normalised mode shapes in the columns of ``V``.
        """
        from scipy.linalg import eigh

        w2, V = eigh(self.K, self.M)
        omega = np.sqrt(np.clip(w2, 0.0, None))
        return omega, V


def assemble(geometry: WingGeometry, bending_shapes, torsion_shapes,
             n_quad: int = 48) -> StructuralModel:
    """Assemble the generalized ``M`` and ``K`` for the given shapes."""
    L = geometry.semi_span
    nb, nt = len(bending_shapes), len(torsion_shapes)
    eta, w = _leggauss_01(n_quad)
    y = eta * L
    p = geometry.section_properties(y)
    m, Ith, S, EI, GJ = p["mass"], p["inertia"], p["S"], p["EI"], p["GJ"]

    # Shape values / derivatives at the quadrature nodes: rows = shapes, cols = nodes.
    Phi = np.array([s(eta) for s in bending_shapes])        # (nb, nq)
    Phi2 = np.array([s.d2(eta) for s in bending_shapes])    # d^2/deta^2
    Psi = np.array([s(eta) for s in torsion_shapes])        # (nt, nq)
    Psi1 = np.array([s.d1(eta) for s in torsion_shapes])    # d/deta

    Mbb = L * (Phi * (w * m)) @ Phi.T
    Kbb = L ** -3 * (Phi2 * (w * EI)) @ Phi2.T
    Mtt = L * (Psi * (w * Ith)) @ Psi.T
    Ktt = L ** -1 * (Psi1 * (w * GJ)) @ Psi1.T
    Mbt = -L * (Phi * (w * S)) @ Psi.T

    # Concentrated masses.
    for pm in geometry.point_masses:
        pe = np.array([s(pm.eta) for s in bending_shapes])
        te = np.array([s(pm.eta) for s in torsion_shapes])
        Mbb = Mbb + pm.mass * np.outer(pe, pe)
        Mbt = Mbt - pm.mass * pm.xi * np.outer(pe, te)
        Mtt = Mtt + (pm.mass * pm.xi ** 2 + pm.pitch_inertia) * np.outer(te, te)

    n = nb + nt
    M = np.zeros((n, n))
    K = np.zeros((n, n))
    M[:nb, :nb] = Mbb
    M[:nb, nb:] = Mbt
    M[nb:, :nb] = Mbt.T
    M[nb:, nb:] = Mtt
    K[:nb, :nb] = Kbb
    K[nb:, nb:] = Ktt

    return StructuralModel(M, K, nb, nt, list(bending_shapes), list(torsion_shapes), geometry)
