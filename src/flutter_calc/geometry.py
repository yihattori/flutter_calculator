"""Wing geometry and inertia description for the Rayleigh-Ritz beam model.

A :class:`WingGeometry` holds the spanwise distributions of structural and inertial
properties of a single cantilever wing (treated as a beam). Each distribution may be a
constant (uniform wing) or a callable ``f(y) -> value`` of the spanwise coordinate
``y`` in metres (``0`` at root, ``semi_span`` at tip), enabling tapered / non-uniform
wings without changing any downstream code.

Sectional quantities consumed by the structural assembly (see ``structures.ritz``):

    chord(y)            streamwise chord c [m]
    EI(y)               bending stiffness [N m^2]
    GJ(y)               torsional stiffness [N m^2]
    mass_per_span(y)    m, mass per unit span [kg/m]
    inertia_per_span(y) I_theta, mass moment of inertia per unit span about the
                        ELASTIC AXIS [kg m^2 / m]
    ea_frac(y)          elastic-axis position, fraction of chord aft of the LE
    mass_axis_frac(y)   centre-of-mass position, fraction of chord aft of the LE

From these the static unbalance ``S = m * x_theta`` is derived, where
``x_theta = (mass_axis_frac - ea_frac) * chord`` is the distance of the CG aft of the
elastic axis (the bending-torsion inertial coupler). See conventions.py for signs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional, Union

import numpy as np

Distribution = Union[float, Callable[[np.ndarray], np.ndarray]]


def _call(d: Distribution, y: np.ndarray) -> np.ndarray:
    """Evaluate a constant-or-callable distribution at spanwise stations ``y``."""
    if callable(d):
        return np.asarray(d(y), dtype=float)
    return np.full(np.shape(y), float(d))


@dataclass
class PointMass:
    """A concentrated mass (e.g. an engine/pylon) attached to the wing.

    Contributes to kinetic energy via ``z = h(y_e) - xi * theta(y_e)`` at its
    attachment station, exactly like a beam section but lumped at a point.
    """

    mass: float                 #: [kg]
    eta: float                  #: spanwise location as a fraction y/semi_span in [0, 1]
    xi: float = 0.0             #: chordwise offset aft of the elastic axis [m]
    pitch_inertia: float = 0.0  #: mass moment of inertia about its own CG [kg m^2]


@dataclass
class WingGeometry:
    """Spanwise description of a single cantilever wing.

    Distributions are constants or callables of ``y`` [m]. ``mass_axis_frac`` defaults
    to ``ea_frac`` (a mass-balanced section, CG on the elastic axis, no inertial
    bending-torsion coupling) which is the configuration used for free-vibration
    validation.
    """

    semi_span: float                       #: L [m], root to tip
    chord: Distribution                    #: c(y) [m]
    EI: Distribution                       #: bending stiffness [N m^2]
    GJ: Distribution                       #: torsional stiffness [N m^2]
    mass_per_span: Distribution            #: m(y) [kg/m]
    inertia_per_span: Distribution         #: I_theta(y) about EA [kg m]
    ea_frac: Distribution = 0.4            #: elastic axis, fraction of chord aft of LE
    mass_axis_frac: Optional[Distribution] = None  #: CG, fraction aft of LE; None -> ea_frac
    sweep_deg: float = 0.0                 #: elastic-axis sweep, positive aft-swept
    point_masses: list[PointMass] = field(default_factory=list)

    @classmethod
    def uniform(cls, semi_span, chord, EI, GJ, mass_per_span, inertia_per_span,
                ea_frac=0.4, mass_axis_frac=None, sweep_deg=0.0, point_masses=None):
        """Convenience constructor for a prismatic (uniform) wing."""
        return cls(semi_span, chord, EI, GJ, mass_per_span, inertia_per_span,
                   ea_frac, mass_axis_frac, sweep_deg, point_masses or [])

    def section_properties(self, y: np.ndarray) -> dict:
        """Evaluate all sectional properties at spanwise stations ``y`` [m].

        Returns a dict of arrays including the derived static unbalance ``S`` and
        the distance of the CG aft of the elastic axis ``x_theta``.
        """
        y = np.asarray(y, dtype=float)
        c = _call(self.chord, y)
        eaf = _call(self.ea_frac, y)
        maf = _call(self.ea_frac if self.mass_axis_frac is None else self.mass_axis_frac, y)
        m = _call(self.mass_per_span, y)
        x_theta = (maf - eaf) * c          # CG aft of EA [m]
        return {
            "chord": c,
            "b": 0.5 * c,
            "EI": _call(self.EI, y),
            "GJ": _call(self.GJ, y),
            "mass": m,
            "inertia": _call(self.inertia_per_span, y),
            "ea_frac": eaf,
            "mass_axis_frac": maf,
            "x_theta": x_theta,
            "S": m * x_theta,              # static unbalance per span [kg]
        }
