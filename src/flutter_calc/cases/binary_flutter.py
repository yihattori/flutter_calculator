"""Wright & Cooper style binary flutter case (1 bending + 1 torsion shape).

A representative uniform rectangular cantilever wing with the classic assumed shapes
``phi = (y/s)^2`` (bending) and ``psi = (y/s)`` (torsion), the CG aft of the elastic
axis so that bending and torsion couple. This is the smallest model that flutters and
is used to validate the full structures -> aero -> p-k pipeline.

The numerical properties are representative textbook values (not tied to one edition's
example); the script reports the predicted flutter speed/frequency for cross-checking.
"""

from __future__ import annotations

from ..aero.theodorsen import TheodorsenStripAero
from ..geometry import WingGeometry
from ..structures.ritz import assemble
from ..structures.shapes import polynomial_bending_shapes, polynomial_torsion_shapes

# Representative uniform rectangular wing (SI units).
PARAMS = dict(
    semi_span=6.0,        # s [m]
    chord=2.0,            # c [m]
    EI=6.6e5,             # bending stiffness [N m^2]   -> ~2.0 Hz first bending
    GJ=2.4e5,             # torsional stiffness [N m^2] -> ~5.0 Hz first torsion
    mass_per_span=40.0,   # m [kg/m]
    inertia_per_span=17.0,  # I_theta about EA [kg m]
    ea_frac=0.35,         # elastic axis at 35% chord
    mass_axis_frac=0.45,  # CG at 45% chord (aft of EA -> inertial coupling)
)


def build(n_quad: int = 48):
    """Return ``(structural_model, aero_model, b_ref)`` for the binary case."""
    geo = WingGeometry.uniform(**PARAMS)
    bending = polynomial_bending_shapes(1)   # (y/s)^2
    torsion = polynomial_torsion_shapes(1)   # (y/s)
    sm = assemble(geo, bending, torsion, n_quad=n_quad)
    aero = TheodorsenStripAero(geo, bending, torsion, n_quad=n_quad)
    return sm, aero, aero.b_ref


def main():
    import numpy as np

    from ..postprocess.flutter_point import summarize
    from ..solvers.pk import pk_flutter

    sm, aero, b_ref = build()
    omega_vac, _ = sm.free_vibration()
    velocities = np.linspace(5.0, 250.0, 246)
    result = pk_flutter(sm, aero, velocities, rho=1.225, b_ref=b_ref)
    print("in-vacuo frequencies [Hz]:", np.round(omega_vac / (2 * np.pi), 3))
    print(summarize(result))


if __name__ == "__main__":
    main()
