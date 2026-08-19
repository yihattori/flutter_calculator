"""Central definitions and conventions for Flutter Calculator.

This module is the ONE source of truth for reference lengths, the reduced-frequency
definition, sign conventions, and the damping definition. Every other module imports
these helpers rather than re-deriving them. Centralising this is how we avoid the
classic flutter bugs: semichord vs chord, ``k = w*b/U`` vs ``w*c/U``, sign of the
downwash, and ``g = 2*zeta`` vs hysteretic structural ``g``.

Conventions used throughout
---------------------------
Axes (right-handed):
    x : streamwise, positive aft (downstream)
    y : spanwise,   positive toward the tip, ``y = 0`` at the root
    z : vertical,   positive up

Structural degrees of freedom:
    h(y)      bending heave of the elastic axis, positive UP.
    theta(y)  twist about the elastic axis, positive NOSE-UP (leading edge up).
    A material point at chordwise offset ``xi`` measured positive AFT of the
    elastic axis has vertical displacement ``z = h - xi * theta``.

Reference length:
    b : reference SEMICHORD (half the streamwise chord). All reduced frequencies
        and Theodorsen quantities are defined with ``b``.

Reduced frequency:
    k = omega * b / U          (omega in rad/s, U true airspeed in m/s)

Aerodynamic chordwise position (Theodorsen ``a``):
    a in [-1, 1] : elastic-axis position relative to MIDCHORD, in semichords,
                   positive aft. a = -1 leading edge, 0 midchord, +1 trailing edge.

Damping (p-k):
    The p-k eigenvalue is written ``p = (gamma + 1j) * omega`` so that motion
    ~ exp(p t) at the matched frequency ``omega``. The reported damping is the
    structural-damping-equivalent ``g = 2 * gamma`` used for V-g plots in
    Wright & Cooper. Flutter is the lowest airspeed at which any branch's ``g``
    crosses from negative to positive.
"""

from __future__ import annotations

import numpy as np

#: Sea-level ISA air density [kg/m^3] (convenience default; see envelope.py for ISA).
RHO_SEA_LEVEL = 1.225
#: Sea-level speed of sound [m/s].
A_SEA_LEVEL = 340.294


def reduced_frequency(omega: float, b: float, U: float) -> float:
    """k = omega * b / U."""
    return omega * b / U


def omega_from_k(k: float, b: float, U: float) -> float:
    """Inverse of :func:`reduced_frequency`: omega = k * U / b."""
    return k * U / b


def dynamic_pressure(rho: float, U: float) -> float:
    """q = 1/2 * rho * U**2."""
    return 0.5 * rho * U * U


def semichord(chord: float | np.ndarray) -> float | np.ndarray:
    """b = chord / 2."""
    return 0.5 * np.asarray(chord)


def mach(U: float, a_sound: float = A_SEA_LEVEL) -> float:
    """Flight Mach number U / a."""
    return U / a_sound
