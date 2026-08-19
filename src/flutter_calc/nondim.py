"""Dimensionless parameters for parameterising and reporting flutter.

These are the groups a parameter study perturbs and the quantities used to present
results in a scale- and Mach-independent way. Working in dimensionless parameters is
what lets a result obtained on one wing generalise to a whole class of wings rather
than being specific to one set of dimensional properties.

Notation follows the classic Theodorsen / Bisplinghoff sectional model:

    mu        = m / (pi * rho * b**2)     mass ratio
    x_alpha   = (CG aft of EA) / b        static unbalance, in semichords
    r_alpha2  = I_theta / (m * b**2)      squared radius of gyration about EA
    a         in [-1, 1]                  EA position relative to midchord, in semichords
    omega_h / omega_alpha                 bending-to-torsion frequency ratio
    V_index   = V / (b * omega_alpha * sqrt(mu))   reduced (flutter) speed index
"""

from __future__ import annotations

import numpy as np


def mass_ratio(mass_per_span: float, rho: float, b: float) -> float:
    """mu = m / (pi * rho * b**2)."""
    return mass_per_span / (np.pi * rho * b * b)


def static_unbalance(x_theta: float, b: float) -> float:
    """x_alpha = x_theta / b, with x_theta the distance of the CG aft of the EA."""
    return x_theta / b


def radius_of_gyration_sq(inertia_per_span: float, mass_per_span: float, b: float) -> float:
    """r_alpha**2 = I_theta / (m * b**2)."""
    return inertia_per_span / (mass_per_span * b * b)


def ea_position_param(ea_frac: float) -> float:
    """Theodorsen ``a``: EA relative to midchord in semichords, positive aft.

    ``ea_frac`` is the EA position as a fraction of chord aft of the leading edge,
    so midchord is 0.5 and ``a = (ea_frac - 0.5) / 0.5``.
    """
    return (ea_frac - 0.5) / 0.5


def frequency_ratio(omega_h: float, omega_alpha: float) -> float:
    """Bending-to-torsion uncoupled natural frequency ratio."""
    return omega_h / omega_alpha


def flutter_speed_index(V_flutter: float, b: float, omega_alpha: float, mu: float) -> float:
    """Reduced flutter-speed index V_f / (b * omega_alpha * sqrt(mu)).

    This is the standard nondimensional flutter speed (e.g. the AGARD 445.6
    flutter-speed index), making flutter results comparable across scale and density.
    """
    return V_flutter / (b * omega_alpha * np.sqrt(mu))
