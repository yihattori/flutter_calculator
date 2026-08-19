"""AGARD 445.6 weakened wing -- the swept-wing DLM validation benchmark.

The standard swept-wing flutter benchmark: a 45 deg quarter-chord-swept, tapered,
mahogany wing tested in the NASA TDT. We use it as an ORDER-OF-MAGNITUDE check of the
swept, compressible DLM -- not to reproduce the transonic flutter dip (a linear panel
method cannot: that is the documented validity boundary of this toolbox).

Reference data (weakened model 3, experiment):
  semi-span s       = 0.762 m
  root chord c_r    = 0.5587 m
  tip chord c_t     = 0.3682 m      (taper 0.659)
  quarter-chord sweep = 45 deg
  airfoil NACA 65A004 (thickness ignored; flat-plate DLM)
  f (1st bending)   = 9.60 Hz
  f (1st torsion)   = 38.17 Hz      -> reference omega_alpha
  b_s (root semichord) = 0.27935 m
  Subsonic flutter point M = 0.499: air density 0.4277 kg/m^3,
      flutter-speed index FSI = V/(b_s omega_alpha sqrt(mu)) ~ 0.446,
      which corresponds to a dimensional flutter speed of ~173 m/s.

Approximations here (all order-of-magnitude appropriate): beam Rayleigh-Ritz (not a
plate); EI/GJ calibrated to match the first bending & torsion frequencies rather than
taken from the orthotropic layup; streamwise-beam sweep model (swept planform aero,
structural bend-twist coupling from sweep neglected); LE swept by 45 deg ~ quarter-chord
sweep.
"""

from __future__ import annotations

import numpy as np

from ..aero.dlm import DLMAero
from ..geometry import WingGeometry
from ..structures.ritz import assemble
from ..structures.shapes import polynomial_bending_shapes, polynomial_torsion_shapes

# Reference geometry / data.
S = 0.762
C_ROOT = 0.5587
C_TIP = 0.3682
SWEEP_DEG = 45.0
B_S = C_ROOT / 2.0
F_BENDING = 9.60
F_TORSION = 38.17
M_HALFWING = 0.845                      # half-span panel mass [kg] (approx, weakened model)

# Subsonic validation point.
MACH = 0.499
RHO = 0.4277
TARGET_VF = 173.0                       # m/s, from published FSI ~ 0.446
TARGET_FSI = 0.446


def _chord(y):
    return C_ROOT + (C_TIP - C_ROOT) * (np.asarray(y, float) / S)


def _build_geometry(EI, GJ):
    # mass per span proportional to chord (plate of fixed t/c); inertia ~ m c^2/12.
    mean_chord_integral = S * (C_ROOT + C_TIP) / 2.0
    k_m = M_HALFWING / mean_chord_integral
    mass = lambda y: k_m * _chord(y)
    inertia = lambda y: k_m * _chord(y) * _chord(y) ** 2 / 12.0
    return WingGeometry(
        semi_span=S, chord=_chord, EI=EI, GJ=GJ,
        mass_per_span=mass, inertia_per_span=inertia,
        ea_frac=0.5, mass_axis_frac=0.5,        # balanced; aero drives the coupling
        sweep_deg=SWEEP_DEG,
    )


def _classify(sm):
    omega, V = sm.free_vibration()
    nb = sm.n_bending
    bend, tors = [], []
    for j in range(len(omega)):
        v = V[:, j]
        frac = np.sum(v[:nb] ** 2) / np.sum(v ** 2)
        (bend if frac > 0.5 else tors).append(omega[j])
    return sorted(bend), sorted(tors)


def build(n_bending=2, n_torsion=2, n_span=16, n_chord=6):
    """Return ``(structural_model, aero_model, info)`` with EI/GJ calibrated to AGARD."""
    bending = polynomial_bending_shapes(n_bending)
    torsion = polynomial_torsion_shapes(n_torsion)

    # Calibrate: balanced wing -> bending depends only on EI, torsion only on GJ,
    # and omega ~ sqrt(stiffness), so one assembly at EI=GJ=1 fixes the scale factors.
    sm1 = assemble(_build_geometry(1.0, 1.0), bending, torsion)
    bend1, tors1 = _classify(sm1)
    EI = (2 * np.pi * F_BENDING / bend1[0]) ** 2
    GJ = (2 * np.pi * F_TORSION / tors1[0]) ** 2

    geo = _build_geometry(EI, GJ)
    sm = assemble(geo, bending, torsion)
    aero = DLMAero(geo, bending, torsion, n_span=n_span, n_chord=n_chord, b_ref=B_S)
    info = {
        "EI": EI, "GJ": GJ, "b_ref": B_S,
        "omega_alpha": 2 * np.pi * F_TORSION,
        "mass_ratio": M_HALFWING / (RHO * B_S ** 2 * S),   # ~ AGARD volume-based mu
        "mach": MACH, "rho": RHO,
        "target_Vf": TARGET_VF, "target_FSI": TARGET_FSI,
    }
    return sm, aero, info


def main():
    from ..aero.cache import TabulatedAero
    from ..nondim import flutter_speed_index
    from ..solvers.pk import pk_flutter

    sm, aero, info = build()
    bend, tors = _classify(sm)
    print("calibrated EI = %.3e, GJ = %.3e" % (info["EI"], info["GJ"]))
    print("bending freqs [Hz]:", np.round(np.array(bend) / (2 * np.pi), 2))
    print("torsion freqs [Hz]:", np.round(np.array(tors) / (2 * np.pi), 2))

    tab = TabulatedAero(aero, Ma=info["mach"])
    V = np.linspace(20.0, 450.0, 200)
    res = pk_flutter(sm, tab, V, rho=info["rho"], b_ref=info["b_ref"], mach=info["mach"])
    crit = res.lowest_flutter()
    if crit is None:
        print("no flutter found")
        return
    fsi = flutter_speed_index(crit["V_flutter"], info["b_ref"],
                              info["omega_alpha"], info["mass_ratio"])
    print(f"\nmass ratio mu = {info['mass_ratio']:.2f}")
    print(f"DLM flutter:  Vf = {crit['V_flutter']:.1f} m/s   "
          f"f = {crit['omega_flutter']/(2*np.pi):.1f} Hz   FSI = {fsi:.3f}")
    print(f"AGARD M=0.499: Vf ~ {info['target_Vf']:.0f} m/s            "
          f"             FSI ~ {info['target_FSI']:.3f}")


if __name__ == "__main__":
    main()
