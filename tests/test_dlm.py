"""Validation: PanelAero DLM backend.

At high aspect ratio and low Mach the 3-D DLM must reduce to 2-D Theodorsen
strip theory (validates the mesh, downwash sign, and load-point force projection).
The 45 deg swept AGARD 445.6 wing reproduces the subsonic flutter-speed index
to order of magnitude (a linear panel method cannot capture the transonic dip).
"""

import numpy as np

from flutter_calc.aero.cache import TabulatedAero
from flutter_calc.aero.dlm import DLMAero
from flutter_calc.aero.theodorsen import TheodorsenStripAero
from flutter_calc.geometry import WingGeometry
from flutter_calc.solvers.pk import pk_flutter
from flutter_calc.structures.ritz import assemble
from flutter_calc.structures.shapes import (
    polynomial_bending_shapes,
    polynomial_torsion_shapes,
)


def _high_ar_model(L=24.0):
    """Unswept high-AR wing with fixed sectional props / target frequencies."""
    chord, m, Ith = 2.0, 40.0, 17.0
    w_b, w_t = 15.77, 36.34
    EI = (w_b / 1.8751041 ** 2) ** 2 * m * L ** 4
    GJ = (w_t / (np.pi / 2)) ** 2 * Ith * L ** 2
    geo = WingGeometry.uniform(semi_span=L, chord=chord, EI=EI, GJ=GJ,
                               mass_per_span=m, inertia_per_span=Ith,
                               ea_frac=0.35, mass_axis_frac=0.45)
    bending = polynomial_bending_shapes(1)
    torsion = polynomial_torsion_shapes(1)
    return assemble(geo, bending, torsion), geo, bending, torsion


def test_dlm_moment_term_does_not_collapse():
    """Regression: the torsion-torsion Qhh term must match Theodorsen at high AR.

    The 1/4-chord load-point bug drove Qhh[1,1] toward zero; guard against it.
    """
    _, geo, bending, torsion = _high_ar_model(L=24.0)
    theo = TheodorsenStripAero(geo, bending, torsion)
    dlm = DLMAero(geo, bending, torsion, n_span=36, n_chord=5)
    ratio = dlm.Qhh(0.05).real[1, 1] / theo.Qhh(0.05).real[1, 1]
    assert 0.8 < ratio < 1.2


def test_dlm_converges_to_theodorsen_flutter():
    sm, geo, bending, torsion = _high_ar_model(L=24.0)
    theo = TheodorsenStripAero(geo, bending, torsion)
    dlm = TabulatedAero(DLMAero(geo, bending, torsion, n_span=36, n_chord=5), Ma=0.0)
    V = np.linspace(10.0, 400.0, 120)
    vt = pk_flutter(sm, theo, V, rho=1.225, b_ref=theo.b_ref).lowest_flutter()["V_flutter"]
    vd = pk_flutter(sm, dlm, V, rho=1.225, b_ref=dlm.b_ref).lowest_flutter()["V_flutter"]
    assert abs(vd - vt) / vt < 0.08


def test_sweep_washout_vanishes_unswept_active_swept():
    """The swept-wing washout must be exactly zero at zero sweep (so every unswept
    validation is unchanged) and non-zero once the wing is swept."""
    import dataclasses

    _, geo, bending, torsion = _high_ar_model(L=24.0)   # geo is unswept (sweep_deg=0)
    dlm0 = DLMAero(geo, bending, torsion, n_span=20, n_chord=4)
    assert np.allclose(dlm0.dZdx[:, : dlm0.nb], 0.0)    # bending slope zero when unswept

    geo_swept = dataclasses.replace(geo, sweep_deg=30.0)
    dlm30 = DLMAero(geo_swept, bending, torsion, n_span=20, n_chord=4)
    assert np.any(np.abs(dlm30.dZdx[:, : dlm30.nb]) > 1e-6)


def test_swept_bending_washout_is_load_alleviating():
    """Steady-limit sign regression for the swept elastic-axis kinematics.

    On a swept-BACK wing the trailing edge of a streamwise section is structurally
    outboard of its leading edge, so up-bending twists the section nose-down and sheds
    lift: the steady bending-bending generalized force must be NEGATIVE (restoring).
    Anchor: nose-up torsion must still produce up-lift (positive bending-row term).
    Guards against the anti-kinematic sign a calibrated washout once carried.
    """
    import dataclasses

    _, geo, bending, torsion = _high_ar_model(L=24.0)
    geo_swept = dataclasses.replace(geo, sweep_deg=30.0)
    dlm = DLMAero(geo_swept, bending, torsion, n_span=20, n_chord=4)
    Q0 = dlm.Qhh(1e-4).real                              # steady limit
    assert Q0[0, 0] < 0.0                                # washout = load alleviation
    assert Q0[0, dlm.nb] > 0.0                           # nose-up twist -> up lift


def test_agard445_flutter_order_of_magnitude():
    from flutter_calc.cases import agard445
    from flutter_calc.nondim import flutter_speed_index

    sm, aero, info = agard445.build(n_span=14, n_chord=5)
    tab = TabulatedAero(aero, Ma=info["mach"])
    V = np.linspace(20.0, 450.0, 160)
    crit = pk_flutter(sm, tab, V, rho=info["rho"], b_ref=info["b_ref"],
                      mach=info["mach"]).lowest_flutter()
    assert crit is not None
    fsi = flutter_speed_index(crit["V_flutter"], info["b_ref"],
                              info["omega_alpha"], info["mass_ratio"])
    # published subsonic FSI ~ 0.446; accept within a factor of ~2 (order of magnitude)
    assert 0.22 < fsi < 0.9
    # flutter frequency between first bending and first torsion
    assert 2 * np.pi * 9.6 < crit["omega_flutter"] < 2 * np.pi * 38.17
