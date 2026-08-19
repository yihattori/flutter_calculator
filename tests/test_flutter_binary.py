"""Validation: binary flutter via the full structures -> aero -> p-k pipeline.

Checks the qualitative signatures of classical bending-torsion flutter rather than a
single magic number (the exact speed depends on the aero model; Theodorsen here):

* a flutter speed exists and is physically sensible;
* the flutter frequency lies between the two in-vacuo natural frequencies;
* the branch frequencies coalesce approaching flutter;
* the wing is stable well below the flutter speed.
"""

import numpy as np

from flutter_calc.cases import binary_flutter
from flutter_calc.solvers.pk import pk_flutter

RHO = 1.225


def _run():
    sm, aero, b_ref = binary_flutter.build()
    omega_vac, _ = sm.free_vibration()
    V = np.linspace(5.0, 250.0, 246)
    res = pk_flutter(sm, aero, V, rho=RHO, b_ref=b_ref)
    return sm, res, omega_vac, V


def test_flutter_exists_and_sensible():
    _, res, omega_vac, _ = _run()
    crit = res.lowest_flutter()
    assert crit is not None, "no flutter found in the swept range"
    assert 40.0 < crit["V_flutter"] < 100.0
    # flutter frequency between the in-vacuo bending and torsion frequencies
    assert omega_vac[0] < crit["omega_flutter"] < omega_vac[1]


def test_stable_below_flutter():
    _, res, _, V = _run()
    crit = res.lowest_flutter()
    Vf = crit["V_flutter"]
    below = V < 0.5 * Vf
    assert np.all(res.damping[:, below] < 0.0)   # comfortably stable at low speed


def test_frequency_coalescence():
    _, res, _, V = _run()
    crit = res.lowest_flutter()
    Vf = crit["V_flutter"]
    sep_low = abs(res.omega[0, 0] - res.omega[1, 0])
    i_near = int(np.argmin(np.abs(V - Vf)))
    sep_near = abs(res.omega[0, i_near] - res.omega[1, i_near])
    assert sep_near < sep_low   # branches draw together approaching flutter
