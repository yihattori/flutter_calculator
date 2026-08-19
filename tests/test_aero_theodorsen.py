"""Validation: Theodorsen's function C(k) limits, trends, and definition."""

import numpy as np
from scipy.special import j0, j1, y0, y1

from flutter_calc.aero.base import theodorsen_function


def _C_via_bessel(k):
    """C(k) from the J/Y Bessel combination (independent of the Hankel form)."""
    return (j1(k) - 1j * y1(k)) / ((j1(k) + y0(k)) + 1j * (j0(k) - y1(k)))


def test_limits():
    assert theodorsen_function(0.0) == 1.0 + 0j
    assert abs(theodorsen_function(1e-5) - 1.0) < 1e-2          # quasi-steady limit
    c_inf = theodorsen_function(80.0)
    assert abs(c_inf.real - 0.5) < 0.02                        # C -> 1/2 as k -> inf
    assert abs(c_inf.imag) < 0.02


def test_trends():
    ks = np.array([0.05, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2])
    F = np.array([theodorsen_function(k).real for k in ks])
    G = np.array([theodorsen_function(k).imag for k in ks])
    assert np.all(np.diff(F) < 0)        # F(k) monotonically decreasing
    assert np.all(G < 0)                 # G(k) negative (phase lag) over this range
    assert np.all(F > 0.5) and np.all(F < 1.0)


def test_definition_matches_bessel():
    for k in (0.25, 0.5, 1.0, 2.0, 5.0):
        assert np.isclose(theodorsen_function(k), _C_via_bessel(k), rtol=1e-10)
