"""Validation: Rayleigh-Ritz free vibration vs the exact uniform-cantilever beam.

For a prismatic clamped-free beam the analytic natural frequencies are known in closed
form, so this is a rigorous check of the generalized M and K assembly:

    bending:  omega_n = (beta_n L)^2 * sqrt(EI / (m L^4)),
              beta_n L = 1.8751041, 4.6940911, 7.8547574, ...
    torsion:  omega_n = ((2n-1) pi / 2) * sqrt(GJ / (I_theta L^2))

The section is mass-balanced (CG on the elastic axis) so bending and torsion decouple
and the two analytic families can be compared independently. Polynomial Ritz converges
to these values from above, so a handful of shapes gives the first two modes tightly.
"""

import numpy as np

from flutter_calc.geometry import WingGeometry
from flutter_calc.structures.ritz import assemble
from flutter_calc.structures.shapes import (
    polynomial_bending_shapes,
    polynomial_torsion_shapes,
)

# Representative uniform-wing properties (SI).
L = 6.0          # semi-span [m]
CHORD = 1.0      # chord [m]
EI = 2.0e4       # bending stiffness [N m^2]
GJ = 1.5e4       # torsional stiffness [N m^2]
M_SPAN = 8.0     # mass per span [kg/m]
I_SPAN = 0.8     # inertia per span about EA [kg m]

BETA_L = (1.8751041, 4.6940911)  # first two clamped-free bending eigenvalues


def _uniform_model(nb=6, nt=5):
    geo = WingGeometry.uniform(
        semi_span=L, chord=CHORD, EI=EI, GJ=GJ,
        mass_per_span=M_SPAN, inertia_per_span=I_SPAN,
        ea_frac=0.4, mass_axis_frac=0.4,  # balanced -> bending/torsion decoupled
    )
    return assemble(geo, polynomial_bending_shapes(nb), polynomial_torsion_shapes(nt))


def _split_modes(sm):
    """Classify each free-vibration mode as bending- or torsion-dominated."""
    omega, V = sm.free_vibration()
    nb = sm.n_bending
    bending, torsion = [], []
    for j in range(len(omega)):
        v = V[:, j]
        frac_bending = np.sum(v[:nb] ** 2) / np.sum(v ** 2)
        (bending if frac_bending > 0.5 else torsion).append(omega[j])
    return np.sort(bending), np.sort(torsion)


def test_bending_frequencies():
    sm = _uniform_model()
    bending, _ = _split_modes(sm)
    scale = np.sqrt(EI / (M_SPAN * L ** 4))
    analytic = [(bl ** 2) * scale for bl in BETA_L]
    assert abs(bending[0] - analytic[0]) / analytic[0] < 0.005
    assert abs(bending[1] - analytic[1]) / analytic[1] < 0.02


def test_torsion_frequencies():
    sm = _uniform_model()
    _, torsion = _split_modes(sm)
    scale = np.sqrt(GJ / (I_SPAN * L ** 2))
    analytic = [((2 * n - 1) * np.pi / 2) * scale for n in (1, 2)]
    assert abs(torsion[0] - analytic[0]) / analytic[0] < 0.005
    assert abs(torsion[1] - analytic[1]) / analytic[1] < 0.03


def test_matrices_symmetric_and_pd():
    sm = _uniform_model()
    assert np.allclose(sm.M, sm.M.T)
    assert np.allclose(sm.K, sm.K.T)
    # M positive definite, K positive semi-definite (clamped-free has no rigid modes).
    assert np.all(np.linalg.eigvalsh(sm.M) > 0)
    assert np.min(np.linalg.eigvalsh(sm.K)) > -1e-6 * np.max(np.linalg.eigvalsh(sm.K))


def test_static_unbalance_couples_modes():
    """Moving the CG aft of the EA must introduce inertial coupling (M_bt != 0)."""
    geo = WingGeometry.uniform(
        semi_span=L, chord=CHORD, EI=EI, GJ=GJ,
        mass_per_span=M_SPAN, inertia_per_span=I_SPAN,
        ea_frac=0.4, mass_axis_frac=0.5,  # CG aft of EA
    )
    sm = assemble(geo, polynomial_bending_shapes(4), polynomial_torsion_shapes(3))
    nb = sm.n_bending
    assert np.linalg.norm(sm.M[:nb, nb:]) > 0
