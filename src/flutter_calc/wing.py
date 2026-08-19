"""Canonical wing builder: physical parameters -> (structure, aero, references).

One place to turn a small set of physically meaningful parameters into the validated
structure -> aero chain. Stiffnesses ``EI``/``GJ`` are *not* asked for directly (they are
rarely known up front); instead the caller gives target first bending/torsion frequencies
and EI/GJ are calibrated so the AS-BUILT wing -- CG offset applied, engines attached --
hits them. The targets are GVT-style frequencies, and a ground vibration test measures
the engined aircraft, so the engine must be on during calibration: calibrating the clean
wing and then hanging the engine (the original scheme) left every engined preset ~4-15%
below its own anchors, torsion especially, which deflates the flutter speed nearly
one-for-one and was the main driver of the spuriously low certification margins. The
default mass model spreads the half-wing mass in proportion to chord with a thin-plate
section inertia.

Used by :mod:`flutter_calc.cases.transport` (the A320-like baseline) and by the top-level
``flutter_sweep.py`` control script, so there is a single definition of "make a wing".
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import eigh

from .geometry import WingGeometry
from .nondim import mass_ratio as _mass_ratio
from .structures.ritz import StructuralModel, assemble
from .structures.shapes import polynomial_bending_shapes, polynomial_torsion_shapes


@dataclass
class WingBuild:
    """Everything ``flutter_sweep`` / the cases need from one built wing."""

    structure: StructuralModel
    aero: object                #: an AeroModel (Theodorsen or DLM backend)
    geometry: WingGeometry
    b_ref: float                #: reference semichord for k = omega b / U [m]
    omega_alpha: float          #: uncoupled torsion frequency [rad/s] = 2 pi f_torsion
    mean_mass_per_span: float   #: half-wing mass / semi-span [kg/m]
    EI: float                   #: calibrated bending stiffness [N m^2]
    GJ: float                   #: calibrated torsional stiffness [N m^2]
    natural_hz: np.ndarray      #: coupled natural frequencies [Hz], ascending

    def mass_ratio(self, rho: float) -> float:
        """Reference mass ratio mu = m / (pi rho b_ref^2) at air density ``rho``."""
        return _mass_ratio(self.mean_mass_per_span, rho, self.b_ref)


def _linear_chord(root_chord, tip_chord, semi_span):
    def chord(y):
        y = np.asarray(y, float)
        return root_chord + (tip_chord - root_chord) * (y / semi_span)
    return chord


def _first_classified(sm):
    """First bending- and torsion-dominated coupled frequencies [rad/s].

    Modes are classified by where their generalized-coordinate energy lives, so the
    calibration can follow the physical branches even when the engine and CG offset
    couple bending and torsion.
    """
    omega, V = sm.free_vibration()
    nb = sm.n_bending
    bend = tors = None
    for j in range(len(omega)):
        frac = np.sum(V[:nb, j] ** 2) / np.sum(V[:, j] ** 2)
        if frac > 0.5:
            if bend is None:
                bend = omega[j]
        elif tors is None:
            tors = omega[j]
        if bend is not None and tors is not None:
            break
    if bend is None or tors is None:
        raise RuntimeError("could not classify both a bending and a torsion mode")
    return float(bend), float(tors)


def build_wing(*, semi_span, root_chord, tip_chord, half_wing_mass,
               f_bending, f_torsion, sweep_deg=0.0, ea_frac=0.40, cg_frac=None,
               n_bending=2, n_torsion=2, point_masses=None,
               backend="dlm", b_ref=None, n_span=12, n_chord=5, n_quad=48) -> WingBuild:
    """Build a cantilever wing from physical parameters; calibrate EI/GJ to frequencies.

    Parameters
    ----------
    semi_span, root_chord, tip_chord : planform [m] (linear taper).
    half_wing_mass : structural+fuel mass of the half-wing [kg], spread proportional to chord.
    f_bending, f_torsion : target first bending / torsion frequencies [Hz] of the
        AS-BUILT wing (engines attached, CG offset applied) -- i.e. GVT-style anchors;
        EI/GJ are calibrated until the classified coupled modes hit them.
    sweep_deg : quarter-chord sweep, positive aft [deg].
    ea_frac, cg_frac : elastic-axis and section-CG positions as a fraction of chord aft of
        the LE. ``cg_frac=None`` puts the CG on the elastic axis (no inertial coupling).
    point_masses : list of :class:`~flutter_calc.geometry.PointMass` (e.g. an engine).
    backend : ``"dlm"`` (swept, subsonic-compressible) or ``"theodorsen"`` (incompressible
        oracle; aero backends are imported lazily so Theodorsen-only use needs no PanelAero).
    b_ref : reference semichord [m]; defaults to half the root chord.
    """
    if cg_frac is None:
        cg_frac = ea_frac
    if b_ref is None:
        b_ref = 0.5 * root_chord
    point_masses = list(point_masses or [])
    chord = _linear_chord(root_chord, tip_chord, semi_span)

    # Mass model: mass per span proportional to chord; thin-plate section inertia.
    mean_chord_integral = semi_span * (root_chord + tip_chord) / 2.0
    k_m = half_wing_mass / mean_chord_integral
    mass = lambda y: k_m * chord(y)
    inertia = lambda y: k_m * chord(y) * chord(y) ** 2 / 12.0

    bending = polynomial_bending_shapes(n_bending)
    torsion = polynomial_torsion_shapes(n_torsion)

    def make_geom(EI, GJ, balanced):
        return WingGeometry(
            semi_span=semi_span, chord=chord, EI=EI, GJ=GJ,
            mass_per_span=mass, inertia_per_span=inertia,
            ea_frac=ea_frac, mass_axis_frac=(ea_frac if balanced else cg_frac),
            sweep_deg=sweep_deg, point_masses=([] if balanced else point_masses),
        )

    # Calibrate EI/GJ so the AS-BUILT wing (CG offset applied, engines attached) hits
    # the target first bending/torsion frequencies. GVT anchors are measured on the
    # engined aircraft, so the engine must be on during calibration; the balanced clean
    # wing's closed-form scaling (omega ~ sqrt(stiffness)) seeds a fixed-point iteration
    # that tracks the coupled modes by dominant coordinate energy.
    sm_bal = assemble(make_geom(1.0, 1.0, balanced=True), bending, torsion)
    nb = sm_bal.n_bending
    wb0 = np.sqrt(eigh(sm_bal.K[:nb, :nb], sm_bal.M[:nb, :nb], eigvals_only=True))[0]
    wt0 = np.sqrt(eigh(sm_bal.K[nb:, nb:], sm_bal.M[nb:, nb:], eigvals_only=True))[0]
    w_b_target = 2 * np.pi * f_bending
    w_t_target = 2 * np.pi * f_torsion
    EI = (w_b_target / wb0) ** 2
    GJ = (w_t_target / wt0) ** 2
    for _ in range(40):
        geo = make_geom(EI, GJ, balanced=False)
        sm = assemble(geo, bending, torsion)
        w_b, w_t = _first_classified(sm)
        if (abs(w_b - w_b_target) <= 1e-9 * w_b_target
                and abs(w_t - w_t_target) <= 1e-9 * w_t_target):
            break
        EI *= (w_b_target / w_b) ** 2
        GJ *= (w_t_target / w_t) ** 2
    else:
        raise RuntimeError(
            "EI/GJ frequency calibration did not converge; the requested "
            "f_bending/f_torsion pair may be unreachable for this mass layout")

    if backend == "theodorsen":
        from .aero.theodorsen import TheodorsenStripAero
        aero = TheodorsenStripAero(geo, bending, torsion, b_ref=b_ref, n_quad=n_quad)
    elif backend == "dlm":
        from .aero.dlm import DLMAero
        aero = DLMAero(geo, bending, torsion, n_span=n_span, n_chord=n_chord, b_ref=b_ref)
    else:
        raise ValueError(f"unknown backend {backend!r}; use 'dlm' or 'theodorsen'")

    omega, _ = sm.free_vibration()
    return WingBuild(structure=sm, aero=aero, geometry=geo, b_ref=float(b_ref),
                     omega_alpha=2 * np.pi * f_torsion,
                     mean_mass_per_span=half_wing_mass / semi_span,
                     EI=float(EI), GJ=float(GJ), natural_hz=omega / (2 * np.pi))
