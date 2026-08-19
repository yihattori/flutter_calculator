"""The AeroModel contract and Theodorsen's lift-deficiency function.

Every aerodynamic backend (Theodorsen strip theory now; PanelAero DLM later) exposes
the *same* method::

    Qhh(k, Ma) -> ndarray[n_dof, n_dof], complex

``Qhh`` is the **nondimensional generalized aerodynamic force matrix**: the modal
aerodynamic force is ``F_aero = q_dyn * Qhh(k, Ma) @ q`` with ``q_dyn = 1/2 rho U^2``
and ``q`` the Ritz generalized coordinates. Pulling ``q_dyn`` out front makes ``Qhh``
a function of reduced frequency ``k`` (and Mach) only — independent of ``U`` and
``rho`` — which is exactly what the p-k solver needs to iterate on ``k`` and what makes
the DLM influence coefficients tabulatable. The flutter solver therefore depends only
on ``M, K`` and this one method, so the backend is swappable.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np
from scipy.special import hankel2


def theodorsen_function(k: float) -> complex:
    """Theodorsen's lift-deficiency function ``C(k) = F(k) + i G(k)``.

    ``C(k) = H1(2)(k) / [H1(2)(k) + i H0(2)(k)]`` with ``H(2)`` the Hankel functions
    of the second kind. Limits: ``C(0) = 1`` (quasi-steady) and ``C -> 1/2`` as
    ``k -> inf``.
    """
    k = float(k)
    if k <= 1e-9:
        return 1.0 + 0.0j
    h1 = hankel2(1, k)
    h0 = hankel2(0, k)
    return h1 / (h1 + 1j * h0)


@runtime_checkable
class AeroModel(Protocol):
    """Structural-typing contract for an unsteady-aerodynamics backend."""

    n_dof: int

    def Qhh(self, k: float, Ma: float = 0.0) -> np.ndarray:
        """Nondimensional generalized aero matrix at reduced frequency ``k``, Mach ``Ma``."""
        ...
