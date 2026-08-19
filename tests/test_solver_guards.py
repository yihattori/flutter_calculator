"""Solver-hardening guards: divergence detection and the flutter-frequency filter.

These protect unattended parameter sweeps: a diverging sample must be labeled
(the p-k sweep reports a real positive ``p^2`` as neutral ``g = 0``), and a spurious
high-frequency crossing must be droppable without losing a later plausible one.
"""

import numpy as np

from flutter_calc.aero.theodorsen import TheodorsenStripAero
from flutter_calc.geometry import WingGeometry
from flutter_calc.solvers.pk import FlutterResult, divergence_speed
from flutter_calc.structures.ritz import assemble
from flutter_calc.structures.shapes import (
    polynomial_bending_shapes,
    polynomial_torsion_shapes,
)

RHO = 1.225


def _uniform(ea_frac):
    geo = WingGeometry.uniform(
        semi_span=6.0, chord=2.0, EI=2.0e4, GJ=1.5e4,
        mass_per_span=8.0, inertia_per_span=0.8,
        ea_frac=ea_frac, mass_axis_frac=ea_frac,       # balanced: static problem only
    )
    bending = polynomial_bending_shapes(1)
    torsion = polynomial_torsion_shapes(1)
    sm = assemble(geo, bending, torsion)
    aero = TheodorsenStripAero(geo, bending, torsion)
    return sm, aero


def test_divergence_speed_matches_analytic_torsion():
    """EA aft of the aerodynamic centre -> torsional divergence at the strip-theory value.

    For one torsion shape ``psi = eta`` on a uniform wing the Ritz problem factorises
    exactly: ``q_div = K_tt / Q_tt(0) = 3 GJ / (4 pi b^2 (a + 1/2) L^2)``.
    """
    sm, aero = _uniform(ea_frac=0.60)                   # a = +0.2, EA aft of the AC
    L, b, GJ, a = 6.0, 1.0, 1.5e4, 0.2
    q_div = 3.0 * GJ / (4.0 * np.pi * b ** 2 * (a + 0.5) * L ** 2)
    V_analytic = np.sqrt(2.0 * q_div / RHO)
    V = divergence_speed(sm, aero, rho=RHO)
    assert V is not None
    assert abs(V - V_analytic) / V_analytic < 1e-6


def test_no_divergence_with_ea_ahead_of_ac():
    """EA ahead of the aerodynamic centre -> nose-down feedback, no divergence."""
    sm, aero = _uniform(ea_frac=0.20)                   # a = -0.6, a + 1/2 < 0
    assert divergence_speed(sm, aero, rho=RHO) is None


def test_flutter_points_omega_max_skips_to_next_crossing():
    """The frequency guard must drop a high-frequency crossing but keep scanning the
    same branch for a later plausible one (not just discard the branch)."""
    res = FlutterResult(
        velocity=np.array([1.0, 2.0, 3.0, 4.0]),
        omega=np.array([[100.0, 100.0, 10.0, 10.0]]),
        damping=np.array([[-1.0, 1.0, -1.0, 1.0]]),
        rho=1.225, b_ref=1.0, mach=0.0,
    )
    unguarded = res.lowest_flutter()
    assert abs(unguarded["V_flutter"] - 1.5) < 1e-12
    assert abs(unguarded["omega_flutter"] - 100.0) < 1e-12
    guarded = res.lowest_flutter(omega_max=50.0)
    assert abs(guarded["V_flutter"] - 3.5) < 1e-12
    assert abs(guarded["omega_flutter"] - 10.0) < 1e-12
    assert res.lowest_flutter(omega_max=5.0) is None


def test_g_cross_moves_the_flutter_point_aft():
    """With structural damping available, flutter is where g rises through +g_cross --
    later than the bare-structure g = 0 crossing on a rising branch."""
    res = FlutterResult(
        velocity=np.array([1.0, 2.0, 3.0]),
        omega=np.array([[10.0, 10.0, 10.0]]),
        damping=np.array([[-0.02, 0.0, 0.02]]),
        rho=1.225, b_ref=1.0, mach=0.0,
    )
    bare = res.lowest_flutter()
    damped = res.lowest_flutter(g_cross=0.01)
    assert abs(bare["V_flutter"] - 2.0) < 1e-12
    assert abs(damped["V_flutter"] - 2.5) < 1e-12
    assert res.lowest_flutter(g_cross=0.03) is None
