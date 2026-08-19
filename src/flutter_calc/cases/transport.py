"""A representative civil-transport half-wing (A320-like), ready to run.

This is a *representative* transport wing, not a validated replica of any aircraft. It
exists so that envelope studies and parameter sweeps have a plausible
narrow-body wing to start from without anyone having to invent one. The dimensional
anchors are documented below; EI and GJ are calibrated (inside
:func:`flutter_calc.wing.build_wing`) to hit representative first bending and torsion
frequencies, so the absolute stiffnesses follow from the chosen frequencies rather than
from a structural layup.

Used as a convenient anchor for envelope studies and parameter sweeps: the
module-level constants are the single source of those anchors, so edit them here.
"""

from __future__ import annotations

import numpy as np

from ..geometry import PointMass
from ..nondim import mass_ratio
from ..wing import build_wing

# --- Representative A320-like half-wing (documented anchors) --------------------------
SEMI_SPAN = 16.0          #: exposed semi-span s [m]
C_ROOT = 5.5              #: root chord [m]
C_TIP = 1.6              #: tip chord [m]  (taper ratio ~ 0.29)
SWEEP_DEG = 25.0          #: quarter-chord sweep [deg]
EA_FRAC = 0.40            #: elastic axis at 40% chord
CG_FRAC = 0.46            #: section CG at 46% chord (aft of EA -> inertial coupling)
F_BENDING = 2.0           #: target first-bending frequency [Hz]
F_TORSION = 5.5           #: target first-torsion frequency [Hz]
M_HALFWING = 6000.0       #: half-wing structural + fuel mass [kg]
ENGINE_MASS = 2300.0      #: engine + pylon mass [kg]
ENGINE_ETA = 0.34         #: engine spanwise location, fraction of semi-span
ENGINE_XI = -0.4          #: engine CG ahead of the EA [m] (mass-balancing, stabilising)
ENGINE_PITCH_INERTIA = 1500.0   #: engine pitch inertia about its own CG [kg m^2]

# Reference flight condition and certification speed (representative).
RHO_REF = 0.3639          #: ISA density at 11 km cruise [kg/m^3]
MACH_REF = 0.45           #: representative cruise-ish Mach for the boundary warm-start
V_D_EAS = 180.0           #: design dive speed, EAS [m/s] (~350 kt) -- CS-25.629 reference

B_REF = C_ROOT / 2.0      #: reference semichord [m]


def build(n_bending=2, n_torsion=2, n_span=12, n_chord=5):
    """Return ``(structural_model, dlm_aero, info)`` for the transport baseline.

    Thin wrapper over :func:`flutter_calc.wing.build_wing` with the A320-like anchors; the
    returned ``info`` carries the reference quantities (``b_ref``, ``omega_alpha``,
    ``mass_ratio``, ``rho_ref``, ``mach_ref``, ``V_D_eas``) needed by the envelope and
    sensitivity layers.
    """
    engines = [PointMass(mass=ENGINE_MASS, eta=ENGINE_ETA, xi=ENGINE_XI,
                         pitch_inertia=ENGINE_PITCH_INERTIA)]
    wb = build_wing(
        semi_span=SEMI_SPAN, root_chord=C_ROOT, tip_chord=C_TIP,
        half_wing_mass=M_HALFWING, f_bending=F_BENDING, f_torsion=F_TORSION,
        sweep_deg=SWEEP_DEG, ea_frac=EA_FRAC, cg_frac=CG_FRAC,
        n_bending=n_bending, n_torsion=n_torsion, point_masses=engines,
        backend="dlm", b_ref=B_REF, n_span=n_span, n_chord=n_chord,
    )
    info = {
        "EI": wb.EI, "GJ": wb.GJ, "b_ref": wb.b_ref,
        "omega_alpha": wb.omega_alpha,
        "mass_ratio": mass_ratio(M_HALFWING / SEMI_SPAN, RHO_REF, B_REF),
        "rho_ref": RHO_REF, "mach_ref": MACH_REF, "V_D_eas": V_D_EAS,
    }
    return wb.structure, wb.aero, info


def main():
    from ..aero.cache import TabulatedAero
    from ..nondim import flutter_speed_index
    from ..solvers.pk import pk_flutter

    sm, aero, info = build()
    omega, _ = sm.free_vibration()
    print("first natural frequencies [Hz]:", np.round(omega[:4] / (2 * np.pi), 2))
    print(f"reference mu = {info['mass_ratio']:.1f}, b_ref = {info['b_ref']:.2f} m")

    tab = TabulatedAero(aero, Ma=info["mach_ref"])
    V = np.linspace(40.0, 360.0, 180)
    crit = pk_flutter(sm, tab, V, rho=info["rho_ref"], b_ref=info["b_ref"],
                      mach=info["mach_ref"]).lowest_flutter()
    if crit is None:
        print("no flutter found in the speed range")
        return
    fsi = flutter_speed_index(crit["V_flutter"], info["b_ref"],
                              info["omega_alpha"], info["mass_ratio"])
    print(f"cruise-density flutter: Vf = {crit['V_flutter']:.1f} m/s, "
          f"f = {crit['omega_flutter']/(2*np.pi):.2f} Hz, FSI = {fsi:.3f}")


if __name__ == "__main__":
    main()
