"""Flight-envelope analysis: ISA atmosphere + matched-point flutter.

The p-k solver (``solvers.pk``) answers "at this density and this *frozen* aero Mach,
what is the flutter speed?". That is not yet a flight condition: in real flight the
density, the speed of sound, and the flutter speed are all tied together by the
atmosphere. This module closes that loop.

It provides three things, all built on the validated structures -> aero -> p-k chain:

1. **ISA atmosphere** (:func:`isa`): the 1976 International Standard Atmosphere up to
   20 km (troposphere + lower stratosphere), giving density ``rho`` and speed of sound
   ``a`` versus altitude -- enough to cover any transport cruise altitude.

2. **Matched-point flutter** (:func:`matched_point_flutter`): at a chosen altitude the
   density is fixed by ISA, but the DLM aerodynamic-influence matrix depends on Mach,
   and the flight Mach at the flutter speed is itself ``M = V_f / a``. A *matched point*
   iterates the aero Mach to self-consistency, so the Mach used to build ``Qhh`` equals
   the actual flight Mach at the flutter speed. Sweeping altitude gives the
   **flutter boundary** (:func:`flutter_boundary`).

3. **Flutter-index vs Mach** (:func:`flutter_index_vs_mach`): the AGARD-style presentation
   of how the (compressible, subsonic) flutter-speed index varies with Mach. A linear
   panel method is valid only up to the onset of the transonic dip; beyond that we mark
   the validity boundary and bound the shock-induced dip from the literature rather than
   computing it (:data:`DLM_MACH_VALID_MAX`, :func:`transonic_dip_bound`).

Certification context: CS-25.629 / FAR 25.629 require the aircraft to be free of flutter
up to ``1.15 * V_D`` across the design envelope; :func:`clearance_speed` and
:func:`flutter_margin` provide that reference (the toolbox predicts a *boundary*; the
margin line is where it must sit relative to it).

All speeds are true airspeed (TAS) unless converted with :func:`eas_from_tas`. Mach is
true Mach ``V / a``. See ``conventions.py`` for ``b``, ``k`` and the damping sign.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

from .nondim import flutter_speed_index
from .solvers.pk import pk_flutter

# --- 1976 International Standard Atmosphere constants ---------------------------------
_G0 = 9.80665           #: standard gravity [m/s^2]
_R_AIR = 287.05287      #: specific gas constant for dry air [J/(kg K)]
_GAMMA = 1.4            #: ratio of specific heats
_T0 = 288.15            #: sea-level temperature [K]
_P0 = 101325.0          #: sea-level pressure [Pa]
_RHO0 = 1.225           #: sea-level density [kg/m^3]
_LAPSE = 0.0065         #: tropospheric temperature lapse rate [K/m]
_H_TROP = 11000.0       #: tropopause geopotential altitude [m]
_H_MAX = 20000.0        #: top of the modelled (isothermal) lower stratosphere [m]
# Derived tropopause conditions (continuity at 11 km).
_T_TROP = _T0 - _LAPSE * _H_TROP                                   # 216.65 K
_P_TROP = _P0 * (_T_TROP / _T0) ** (_G0 / (_LAPSE * _R_AIR))       # ~22632 Pa
_RHO_TROP = _P_TROP / (_R_AIR * _T_TROP)                           # ~0.3639 kg/m^3

#: Subsonic DLM validity ceiling. Above roughly this Mach the transonic dip sets in and a
#: linear panel method is no longer trustworthy (see the AGARD 445.6 case); results past
#: it are reported only to mark the boundary, never as predictions.
DLM_MACH_VALID_MAX = 0.85


@dataclass
class AtmosphereState:
    """ISA state at one altitude (SI units)."""

    altitude: float          #: geopotential altitude [m]
    temperature: float       #: T [K]
    pressure: float          #: p [Pa]
    density: float           #: rho [kg/m^3]
    speed_of_sound: float    #: a [m/s]

    @property
    def rho(self) -> float:
        return self.density

    @property
    def a(self) -> float:
        return self.speed_of_sound


def isa(altitude: float) -> AtmosphereState:
    """International Standard Atmosphere state at ``altitude`` [m], 0 <= h <= 20 km.

    Geometric and geopotential altitude differ by < 0.3 % over this range and are not
    distinguished here. Raises ``ValueError`` above 20 km (outside the modelled layers
    and well above any transport cruise altitude).
    """
    h = float(altitude)
    if h > _H_MAX:
        raise ValueError(f"altitude {h} m is above the modelled ISA ceiling ({_H_MAX} m)")
    if h <= _H_TROP:                      # troposphere: linear temperature lapse
        T = _T0 - _LAPSE * h
        p = _P0 * (T / _T0) ** (_G0 / (_LAPSE * _R_AIR))
    else:                                  # lower stratosphere: isothermal
        T = _T_TROP
        p = _P_TROP * np.exp(-_G0 * (h - _H_TROP) / (_R_AIR * _T_TROP))
    rho = p / (_R_AIR * T)
    a = np.sqrt(_GAMMA * _R_AIR * T)
    return AtmosphereState(h, T, p, rho, float(a))


@dataclass
class MatchedPoint:
    """A matched-point flutter solution at one altitude."""

    altitude: float
    atmosphere: AtmosphereState
    flutter: bool                  #: was a flutter crossing found in the speed range?
    converged: bool                #: did the aero Mach match the flight Mach?
    iterations: int
    V_flutter: float | None = None       #: true airspeed at flutter [m/s]
    mach: float | None = None            #: matched flight Mach = V_flutter / a
    omega_flutter: float | None = None   #: flutter frequency [rad/s]
    flutter_index: float | None = None   #: V_f / (b_ref omega_alpha sqrt(mu))
    beyond_validity: bool = False        #: flight Mach exceeded the subsonic-DLM ceiling
                                         #: (aero frozen at the ceiling; speed is a flagged
                                         #: estimate, not a converged matched point)

    @property
    def V_eas(self) -> float | None:
        """Equivalent airspeed at flutter [m/s] (for comparison with EAS-based V_D)."""
        if self.V_flutter is None:
            return None
        return eas_from_tas(self.V_flutter, self.atmosphere.rho)


# An aero factory turns a Mach number into an AeroModel (typically a freshly tabulated
# DLM at that Mach). Keeping it a callable lets the matched-point loop re-tabulate only
# when the Mach changes, and lets tests inject the cheap incompressible Theodorsen model.
AeroFactory = Callable[[float], object]


def _solve_with_expansion(structural_model, aero, rho, b_ref, mach,
                          v_min, v_max, n_v, auto_expand, omega_max=None, g_cross=0.0):
    """Run the p-k sweep, widening the speed window until a flutter crossing appears."""
    vmx = v_max
    for _ in range(auto_expand + 1):
        V = np.linspace(v_min, vmx, n_v)
        res = pk_flutter(structural_model, aero, V, rho=rho, b_ref=b_ref, mach=mach)
        crit = res.lowest_flutter(omega_max=omega_max, g_cross=g_cross)
        if crit is not None:
            return crit
        vmx *= 1.6
    return None


def matched_point_flutter(structural_model, aero_factory: AeroFactory, altitude: float, *,
                          b_ref: float, omega_alpha: float, mass_ratio: float,
                          v_min: float = 20.0, v_max: float = 400.0, n_v: int = 160,
                          mach0: float = 0.3, tol: float = 2e-3, max_iter: int = 15,
                          relax: float = 1.0, auto_expand: int = 3,
                          mach_ceiling: float = DLM_MACH_VALID_MAX,
                          omega_max: float | None = None,
                          g_cross: float = 0.0) -> MatchedPoint:
    """Matched-point flutter at ``altitude``: iterate aero Mach to ``M = V_f / a``.

    At fixed ISA density ``rho(h)`` the p-k solver returns a flutter speed ``V_f`` for a
    *given* aero Mach. The matched point is the fixed point where the Mach used to build
    the aerodynamics equals the flight Mach ``V_f / a(h)``. Each iteration calls
    ``aero_factory(mach)`` to obtain the aero model at the current Mach (for the DLM this
    re-tabulates ``Qhh``; for incompressible Theodorsen it simply ignores Mach and the
    loop converges in two passes).

    Subsonic-validity guard: the DLM is a subsonic panel method (its kernel goes singular
    as ``Ma -> 1``), so the aero is never built above ``mach_ceiling``. A point is
    declared transonic only when the aero is already AT the ceiling and the implied
    flight Mach ``V_f / a`` still exceeds it -- then no subsonic match exists and the
    point is reported with the aero frozen at the ceiling, flagged ``beyond_validity``
    (``converged=False``) rather than driving the DLM into NaNs. A single iterate
    overshooting the ceiling is not enough (``V_f`` falls as the aero Mach rises, so the
    fixed point can sit below the ceiling even when an early guess lands above it); the
    iteration walks the aero Mach up and only the ceiling solve makes the call. This
    enforces the policy declared by :data:`DLM_MACH_VALID_MAX` -- mark the boundary,
    never predict past it.

    Parameters
    ----------
    aero_factory : ``mach -> AeroModel`` providing ``Qhh(k, Ma)`` at that Mach.
    b_ref, omega_alpha, mass_ratio : reference semichord, uncoupled torsion frequency and
        mass ratio used to form the reported flutter-speed index.
    mach0 : initial Mach guess (warm-started across altitudes by :func:`flutter_boundary`).
    tol : convergence tolerance on the Mach mismatch.
    mach_ceiling : highest aero Mach the (subsonic) backend may be built at.
    omega_max : optional frequency guard [rad/s] on accepted flutter crossings (see
        :meth:`~flutter_calc.solvers.pk.FlutterResult.flutter_points`).
    g_cross : hysteretic structural damping available to oppose flutter (crossing level
        for the extraction; default 0 = conservative bare structure).
    """
    atm = isa(altitude)
    mach = min(float(mach0), mach_ceiling)
    crit = None
    last_iter = 0
    for it in range(1, max_iter + 1):
        last_iter = it
        aero_mach = min(mach, mach_ceiling)
        aero = aero_factory(aero_mach)
        crit = _solve_with_expansion(structural_model, aero, atm.rho, b_ref, aero_mach,
                                     v_min, v_max, n_v, auto_expand, omega_max, g_cross)
        if crit is None:
            return MatchedPoint(altitude, atm, flutter=False, converged=False, iterations=it)
        mach_new = crit["V_flutter"] / atm.a
        if mach_new > mach_ceiling:
            if aero_mach >= mach_ceiling:
                # Aero already AT the ceiling and the flight Mach still exceeds it:
                # genuinely transonic, no subsonic match exists. Report the ceiling-aero
                # speed, flagged -- not a converged matched point.
                return _finish_matched(altitude, atm, crit, mach_new, b_ref, omega_alpha,
                                       mass_ratio, False, it, beyond_validity=True)
            # One iterate overshooting the ceiling does NOT prove the point is transonic:
            # Vf falls as the aero Mach rises, so a subsonic fixed point may still exist.
            # Keep iterating -- the update below walks the aero Mach up to the ceiling,
            # where the branch above makes the final call.
        elif abs(mach_new - mach) <= tol:
            return _finish_matched(altitude, atm, crit, mach_new, b_ref,
                                   omega_alpha, mass_ratio, True, it)
        mach = (1.0 - relax) * mach + relax * mach_new
    # Ran out of iterations: report the last (unconverged) estimate.
    mach_last = crit["V_flutter"] / atm.a
    return _finish_matched(altitude, atm, crit, mach_last, b_ref, omega_alpha, mass_ratio,
                           False, last_iter, beyond_validity=mach_last > mach_ceiling)


def _finish_matched(altitude, atm, crit, mach, b_ref, omega_alpha, mass_ratio,
                    converged, iterations, beyond_validity=False) -> MatchedPoint:
    fsi = flutter_speed_index(crit["V_flutter"], b_ref, omega_alpha, mass_ratio)
    return MatchedPoint(
        altitude=altitude, atmosphere=atm, flutter=True, converged=converged,
        iterations=iterations, V_flutter=crit["V_flutter"], mach=mach,
        omega_flutter=crit["omega_flutter"], flutter_index=fsi,
        beyond_validity=beyond_validity,
    )


def flutter_boundary(structural_model, aero_factory: AeroFactory,
                     altitudes: Sequence[float], *, b_ref: float, omega_alpha: float,
                     mass_ratio: float, warm_start: bool = True,
                     mach0: float = 0.3, **kw) -> list[MatchedPoint]:
    """Matched-point flutter at each altitude -> the flutter boundary.

    With ``warm_start`` the converged Mach at one altitude seeds the next, which keeps the
    (expensive) DLM re-tabulation count low because successive matched Machs are close.
    """
    points: list[MatchedPoint] = []
    guess = mach0
    for h in altitudes:
        mp = matched_point_flutter(structural_model, aero_factory, h, b_ref=b_ref,
                                   omega_alpha=omega_alpha, mass_ratio=mass_ratio,
                                   mach0=guess, **kw)
        if warm_start and mp.flutter and mp.mach:
            guess = mp.mach
        points.append(mp)
    return points


def flutter_index_vs_mach(structural_model, aero_factory: AeroFactory,
                          machs: Sequence[float], *, rho: float, b_ref: float,
                          omega_alpha: float, mass_ratio: float, v_min: float = 20.0,
                          v_max: float = 400.0, n_v: int = 160,
                          auto_expand: int = 3, omega_max: float | None = None,
                          g_cross: float = 0.0) -> dict:
    """Flutter-speed index versus aero Mach at a fixed density (constant-altitude sweep).

    This isolates the effect of subsonic compressibility on the flutter index: density is
    held fixed and the DLM Mach is stepped, so the curve is the compressible-linear trend
    only. It is *not* a matched-point boundary (use :func:`flutter_boundary` for that);
    matched ``(Mach, index)`` pairs are the points where this Mach equals ``V_f / a``.
    Past :data:`DLM_MACH_VALID_MAX` the values mark the validity boundary, not predictions.

    Returns a dict of arrays: ``mach``, ``fsi``, ``V_flutter`` [m/s], ``freq_hz``
    (``nan`` where no flutter was found in the speed range).
    """
    mach_arr, fsi, vf, fhz = [], [], [], []
    for Ma in machs:
        aero = aero_factory(float(Ma))
        crit = _solve_with_expansion(structural_model, aero, rho, b_ref, float(Ma),
                                     v_min, v_max, n_v, auto_expand, omega_max, g_cross)
        mach_arr.append(float(Ma))
        if crit is None:
            fsi.append(np.nan); vf.append(np.nan); fhz.append(np.nan)
            continue
        vf.append(crit["V_flutter"])
        fhz.append(crit["omega_flutter"] / (2.0 * np.pi))
        fsi.append(flutter_speed_index(crit["V_flutter"], b_ref, omega_alpha, mass_ratio))
    return {"mach": np.array(mach_arr), "fsi": np.array(fsi),
            "V_flutter": np.array(vf), "freq_hz": np.array(fhz)}


def transonic_dip_bound(subsonic_fsi: float, dip_factor: float = 0.65,
                        mach_bottom: float = 0.90) -> tuple[float, float]:
    """Literature bound on the transonic flutter dip -- a marker, NOT a computed result.

    A linear DLM cannot capture the shock-induced transonic dip. For the AGARD 445.6
    weakened model the measured flutter-speed index bottoms near ``M ~ 0.9`` at roughly
    ``0.65`` of its subsonic level; comparable transport wings dip by a similar fraction
    near their drag-divergence Mach. Returns ``(mach_bottom, fsi_bottom)`` for shading the
    region the subsonic toolbox cannot reach. Both numbers are intentionally configurable
    so the assumed bound is explicit and citable.
    """
    return mach_bottom, dip_factor * subsonic_fsi


# --- CS-25.629 / FAR 25.629 flutter-clearance margin reference ------------------------
def clearance_speed(V_D: float, factor: float = 1.15) -> float:
    """Required flutter-free speed ``factor * V_D`` (CS-25.629(b)(2): ``factor = 1.15``)."""
    return factor * V_D


def eas_from_tas(V_tas: float, rho: float, rho0: float = _RHO0) -> float:
    """Equivalent airspeed from true airspeed: ``V_EAS = V_TAS sqrt(rho / rho0)``."""
    return V_tas * np.sqrt(rho / rho0)


def tas_from_eas(V_eas: float, rho: float, rho0: float = _RHO0) -> float:
    """True airspeed from equivalent airspeed: ``V_TAS = V_EAS / sqrt(rho / rho0)``."""
    return V_eas / np.sqrt(rho / rho0)


def flutter_margin(V_flutter: float, V_D: float, factor: float = 1.15) -> float:
    """Ratio of predicted flutter speed to the required clearance speed.

    ``> 1`` means the flutter boundary sits above ``factor * V_D`` (compliant); ``< 1``
    means it does not. Pass both speeds in the *same* airspeed measure (both TAS or both
    EAS) -- see :func:`eas_from_tas`.
    """
    return V_flutter / (factor * V_D)
