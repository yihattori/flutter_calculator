"""Tests for the canonical wing builder (flutter_calc.wing.build_wing).

Uses the incompressible Theodorsen backend throughout so the tests are fast and need no
PanelAero. Covers the EI/GJ frequency calibration and that the build feeds the p-k solver.
"""

import numpy as np

from flutter_calc.solvers.pk import pk_flutter
from flutter_calc.wing import build_wing


def test_calibration_recovers_target_frequencies():
    """On a mass-balanced wing the coupled modes equal the uncoupled targets exactly."""
    wb = build_wing(semi_span=12.0, root_chord=3.0, tip_chord=1.2, half_wing_mass=2500.0,
                    f_bending=2.5, f_torsion=7.0, cg_frac=None,  # balanced -> decoupled
                    backend="theodorsen")
    # First two modes are first bending then first torsion, hitting the targets.
    assert abs(wb.natural_hz[0] - 2.5) < 1e-6
    assert abs(wb.natural_hz[1] - 7.0) < 1e-6
    assert wb.EI > 0 and wb.GJ > 0


def test_calibration_with_engine_hits_as_built_targets():
    """The frequency anchors are GVT-style values of the engined aircraft, so the
    AS-BUILT coupled model (engine attached, CG aft of EA) must hit them exactly.

    Guards the clean-wing-calibration bug that left engined wings ~12% soft in torsion
    and deflated every certification margin nearly one-for-one.
    """
    from flutter_calc.geometry import PointMass
    from flutter_calc.wing import _first_classified

    wb = build_wing(semi_span=16.0, root_chord=5.5, tip_chord=1.6, half_wing_mass=6000.0,
                    f_bending=2.0, f_torsion=5.5, sweep_deg=25.0, ea_frac=0.40,
                    cg_frac=0.46,
                    point_masses=[PointMass(mass=2300.0, eta=0.34, xi=-0.4,
                                            pitch_inertia=1500.0)],
                    backend="theodorsen")
    w_b, w_t = _first_classified(wb.structure)
    assert abs(w_b / (2 * np.pi) - 2.0) < 1e-6
    assert abs(w_t / (2 * np.pi) - 5.5) < 1e-6


def test_cg_offset_introduces_coupling_and_runs_pk():
    """A CG aft of the EA couples bending/torsion; the build runs through the p-k solver."""
    wb = build_wing(semi_span=12.0, root_chord=3.0, tip_chord=1.2, half_wing_mass=2500.0,
                    f_bending=2.5, f_torsion=7.0, ea_frac=0.40, cg_frac=0.48,
                    backend="theodorsen")
    # As-built calibration holds the classified firsts on target even with coupling.
    assert wb.natural_hz[0] < wb.natural_hz[1]
    V = np.linspace(10.0, 300.0, 160)
    result = pk_flutter(wb.structure, wb.aero, V, rho=1.225, b_ref=wb.b_ref)
    assert result.omega.shape[0] == wb.structure.n_dof
    assert np.all(np.isfinite(result.damping)) and np.all(np.isfinite(result.omega))


def test_mass_ratio_scales_inversely_with_density():
    wb = build_wing(semi_span=10.0, root_chord=2.5, tip_chord=1.0, half_wing_mass=1500.0,
                    f_bending=3.0, f_torsion=8.0, backend="theodorsen")
    assert wb.mass_ratio(0.4) > wb.mass_ratio(1.225) > 0.0


def test_unknown_backend_raises():
    import pytest
    with pytest.raises(ValueError):
        build_wing(semi_span=10.0, root_chord=2.5, tip_chord=1.0, half_wing_mass=1500.0,
                   f_bending=3.0, f_torsion=8.0, backend="nonsense")
