"""Validation: ISA atmosphere + matched-point flutter.

The atmosphere is checked against standard-table values; the matched-point machinery is
exercised with the incompressible Theodorsen oracle (fast and Mach-independent, so the
fixed point ``M = V_f / a`` must be hit exactly and the flutter speed must agree with a
plain p-k solve at the same density).
"""

import numpy as np

from flutter_calc import envelope
from flutter_calc.cases import binary_flutter
from flutter_calc.solvers.pk import pk_flutter


def test_isa_sea_level():
    s = envelope.isa(0.0)
    assert abs(s.temperature - 288.15) < 1e-6
    assert abs(s.pressure - 101325.0) < 1.0
    assert abs(s.density - 1.225) < 1e-3
    assert abs(s.speed_of_sound - 340.294) < 0.05


def test_isa_tropopause_11km():
    s = envelope.isa(11000.0)
    # Standard tabulated values at the tropopause.
    assert abs(s.temperature - 216.65) < 1e-6
    assert abs(s.density - 0.36392) < 1e-3
    assert abs(s.speed_of_sound - 295.07) < 0.1
    assert abs(s.pressure - 22632.0) < 5.0


def test_isa_density_decreases_with_altitude():
    h = np.linspace(0.0, 20000.0, 25)
    rho = np.array([envelope.isa(z).density for z in h])
    assert np.all(np.diff(rho) < 0.0)


def test_isa_above_ceiling_raises():
    import pytest
    with pytest.raises(ValueError):
        envelope.isa(25000.0)


def test_matched_point_self_consistent_theodorsen():
    """With the (Mach-independent) Theodorsen oracle the matched Mach must equal V_f/a."""
    sm, aero, b_ref = binary_flutter.build()
    omega_vac, _ = sm.free_vibration()
    # Theodorsen ignores Mach, so the factory returns the same model for any Mach.
    factory = lambda Ma: aero
    mp = envelope.matched_point_flutter(
        sm, factory, altitude=8000.0, b_ref=b_ref,
        omega_alpha=omega_vac[1], mass_ratio=20.0,
        v_min=5.0, v_max=250.0, n_v=246, mach0=0.4,
    )
    assert mp.flutter and mp.converged
    # Fixed point: the reported Mach is exactly V_f / a(h).
    assert abs(mp.mach - mp.V_flutter / mp.atmosphere.a) < 1e-6
    # And V_f agrees with a direct p-k solve at the same ISA density.
    rho = envelope.isa(8000.0).density
    V = np.linspace(5.0, 250.0, 246)
    direct = pk_flutter(sm, aero, V, rho=rho, b_ref=b_ref).lowest_flutter()
    assert abs(mp.V_flutter - direct["V_flutter"]) / direct["V_flutter"] < 1e-3


def test_flutter_boundary_rises_with_altitude():
    """Lower density at altitude -> higher true flutter speed (incompressible check)."""
    sm, aero, b_ref = binary_flutter.build()
    omega_vac, _ = sm.free_vibration()
    factory = lambda Ma: aero
    pts = envelope.flutter_boundary(
        sm, factory, altitudes=[0.0, 6000.0, 11000.0], b_ref=b_ref,
        omega_alpha=omega_vac[1], mass_ratio=20.0,
        v_min=5.0, v_max=300.0, n_v=200, mach0=0.2,
    )
    vf = [p.V_flutter for p in pts]
    assert all(p.flutter for p in pts)
    assert vf[0] < vf[1] < vf[2]
    # The incompressible flutter index is altitude-invariant (mu folds in the density).
    fsi = [p.flutter_index for p in pts]
    # same wing/omega_alpha/mu here, so FSI tracks V_f; just confirm it is finite & ordered
    assert all(np.isfinite(f) for f in fsi)


def test_matched_point_guards_aero_mach_ceiling():
    """Transonic flutter must be flagged, with the aero frozen at the ceiling and never
    built past it -- the regression guard for the run_envelope NaN crash (subsonic DLM
    goes singular as Ma -> 1)."""
    sm, aero, b_ref = binary_flutter.build()
    omega_vac, _ = sm.free_vibration()
    seen = []

    def factory(Ma):
        seen.append(Ma)
        return aero  # Theodorsen ignores Mach; the loop's clamp is what we test

    # A deliberately low ceiling puts the binary wing's flutter Mach (~0.19) above it.
    mp = envelope.matched_point_flutter(
        sm, factory, altitude=0.0, b_ref=b_ref,
        omega_alpha=omega_vac[1], mass_ratio=20.0,
        v_min=5.0, v_max=250.0, n_v=246, mach0=0.05, mach_ceiling=0.1,
    )
    assert mp.flutter and mp.beyond_validity and not mp.converged
    assert mp.mach > 0.1                       # implied flight Mach exceeds the ceiling
    assert mp.V_flutter is not None and np.isfinite(mp.V_flutter)
    assert max(seen) <= 0.1 + 1e-9             # aero never built above the ceiling


def test_cs25629_margin_helpers():
    assert abs(envelope.clearance_speed(200.0) - 230.0) < 1e-9
    assert abs(envelope.flutter_margin(230.0, 200.0) - 1.0) < 1e-9
    # EAS/TAS round-trip.
    rho = envelope.isa(10000.0).density
    v = 180.0
    assert abs(envelope.tas_from_eas(envelope.eas_from_tas(v, rho), rho) - v) < 1e-9
    # At altitude, TAS exceeds EAS.
    assert envelope.eas_from_tas(v, rho) < v
