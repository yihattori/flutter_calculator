"""The p-k flutter solver.

Aeroelastic equation of motion in Ritz generalized coordinates ``q``:

    M q_ddot + K q = q_dyn * Qhh(k) q ,     q_dyn = 1/2 rho U^2 ,  k = omega b_ref / U

The p-k method seeks complex eigenvalues ``p`` of the frozen-``k`` system

    [ M p^2 + (K - q_dyn Qhh(k)) ] q = 0

i.e. ``p^2 = eig( M^{-1} (q_dyn Qhh(k) - K) )``. Because ``Qhh`` is evaluated at a
guessed ``k``, each branch is iterated to self-consistency: solve, read its oscillatory
frequency ``omega = Im(p) > 0``, update ``k = omega b_ref / U``, repeat. From the
converged ``p`` we report the frequency ``omega`` and the structural-damping-equivalent

    g = 2 Re(p) / Im(p)

so that ``g < 0`` is stable and the flutter speed is the lowest ``U`` at which any
branch's ``g`` crosses zero upward (see ``conventions.py``). Branches are kept coherent
across the sweep by MAC matching of eigenvectors (``tracking.py``); when two trackers
collapse onto the same eigenpair the loser is re-iterated with that eigenvector excluded
so no branch is silently dropped.

The p-k sweep only sees oscillatory roots: a static divergence appears as a real
positive ``p^2`` and is reported with ``g = 0`` (neutral), not as an instability. Use
:func:`divergence_speed` to check for it separately.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import eig, solve

from .tracking import best_match, mac


@dataclass
class FlutterResult:
    """Tracked p-k branches over an airspeed sweep."""

    velocity: np.ndarray          #: (nv,) airspeeds [m/s]
    omega: np.ndarray             #: (n_branch, nv) frequency [rad/s]
    damping: np.ndarray           #: (n_branch, nv) g = 2 Re(p)/Im(p)
    rho: float
    b_ref: float
    mach: float

    def flutter_points(self, omega_max: float | None = None, g_cross: float = 0.0):
        """Lowest upward crossing of ``g`` through ``g_cross`` per branch.

        ``g_cross`` is the hysteretic structural damping available to oppose the
        aerodynamic excitation: flutter is declared where a branch's required damping
        ``g`` rises through it. The default ``0`` is the conservative bare-structure
        criterion; measured transport airframes typically justify ``0.02-0.03`` (the
        classic V-g reading at the g = const line). ``omega_max`` [rad/s], if given,
        drops crossings whose interpolated flutter frequency lies above it -- a guard
        against spurious high-frequency artefacts when extracting unattended (the
        branch keeps being scanned for a later, plausible crossing). Returns a list of
        ``dict(branch, V_flutter, omega_flutter)`` sorted by speed.
        """
        points = []
        V = self.velocity
        for r in range(self.omega.shape[0]):
            g = self.damping[r]
            for i in range(len(V) - 1):
                if g[i] < g_cross <= g[i + 1]:
                    # linear interpolation of the crossing
                    t = (g_cross - g[i]) / (g[i + 1] - g[i])
                    Vf = V[i] + t * (V[i + 1] - V[i])
                    wf = self.omega[r, i] + t * (self.omega[r, i + 1] - self.omega[r, i])
                    if omega_max is not None and wf > omega_max:
                        continue
                    points.append({"branch": r, "V_flutter": Vf, "omega_flutter": wf})
                    break
        return sorted(points, key=lambda p: p["V_flutter"])

    def lowest_flutter(self, omega_max: float | None = None, g_cross: float = 0.0):
        """The critical (lowest-speed) flutter point, or ``None`` if stable throughout."""
        pts = self.flutter_points(omega_max=omega_max, g_cross=g_cross)
        return pts[0] if pts else None


def _iterate_branch(M, K, aero_model, q_dyn, mach, b_ref, U, ref_vec, k0,
                    max_iter, tol, relax, exclude_vec=None):
    """k-iterate one branch, tracking the eigenpair best MAC-matching ``ref_vec``.

    With ``exclude_vec`` the candidate eigenvectors nearly parallel to it are skipped
    (collision repair). Returns ``(p_sel, w_sel, k)`` at the self-consistent ``k``.
    """
    k = float(k0)
    p_sel = None
    w_sel = None
    for _ in range(max_iter):
        Q = aero_model.Qhh(k, mach)
        D = solve(M, q_dyn * Q - K)            # p^2 = eig(D)
        mu, W = eig(D)
        p = np.sqrt(mu.astype(complex))
        p = np.where(p.imag < 0, -p, p)        # take the +omega representative
        s = best_match(ref_vec, W, exclude=exclude_vec)
        p_sel, w_sel = p[s], W[:, s]
        omega_new = float(p_sel.imag)
        k_new = omega_new * b_ref / U if U > 0 else k
        if abs(k_new - k) <= tol * max(k, 1.0):
            k = k_new
            break
        k = (1 - relax) * k + relax * k_new
    return p_sel, w_sel, k


def pk_flutter(structural_model, aero_model, velocities, rho, b_ref=None,
               mach=0.0, max_iter=60, tol=1e-7, relax=0.6):
    """Run the p-k method over ``velocities`` and return a :class:`FlutterResult`.

    Parameters
    ----------
    structural_model : has ``.M``, ``.K``, ``.n_dof``, ``.free_vibration()``.
    aero_model : provides ``Qhh(k, Ma)`` (see :class:`~flutter_calc.aero.base.AeroModel`).
    velocities : ascending array of airspeeds [m/s], starting above zero.
    rho : air density [kg/m^3].
    b_ref : reference semichord for ``k = omega b_ref / U`` (defaults to the aero model's).
    """
    M = np.asarray(structural_model.M, dtype=float)
    K = np.asarray(structural_model.K, dtype=float)
    n = structural_model.n_dof
    if b_ref is None:
        b_ref = getattr(aero_model, "b_ref", None)
    if b_ref is None:
        raise ValueError("b_ref must be given or available on the aero model")

    velocities = np.asarray(velocities, dtype=float)
    nv = len(velocities)

    # Initialise branches from the in-vacuo modes.
    omega_vac, vecs_vac = structural_model.free_vibration()
    prev_vecs = vecs_vac.astype(complex).copy()
    prev_k = np.maximum(omega_vac * b_ref / max(velocities[0], 1e-6), 1e-3)

    omega_out = np.zeros((n, nv))
    damp_out = np.zeros((n, nv))

    for iv, U in enumerate(velocities):
        q_dyn = 0.5 * rho * U * U
        k_start = prev_k.copy()
        new_vecs = np.zeros((n, n), dtype=complex)
        p_new = np.zeros(n, dtype=complex)
        for r in range(n):
            p_sel, w_sel, k = _iterate_branch(M, K, aero_model, q_dyn, mach, b_ref, U,
                                              prev_vecs[:, r], k_start[r],
                                              max_iter, tol, relax)
            p_new[r] = p_sel
            new_vecs[:, r] = w_sel
            prev_k[r] = max(k, 1e-4)

        # Collision repair: each branch iterates its own k independently, so two
        # trackers can converge onto the SAME eigenpair, silently orphaning a mode.
        # Detect true duplicates (near-identical eigenvector AND eigenvalue -- tight
        # thresholds so genuine frequency coalescence, where damping splits the p's,
        # does not trigger) and re-iterate the worse-matching tracker with the
        # duplicated eigenvector excluded. The repair is kept only if the re-pick still
        # plausibly continues that branch (MAC to its predecessor >= 0.2).
        for r in range(n):
            for c in range(r + 1, n):
                same_vec = mac(new_vecs[:, r], new_vecs[:, c]) > 0.98
                same_p = abs(p_new[r] - p_new[c]) <= 5e-3 * max(abs(p_new[r]), 1e-9)
                if not (same_vec and same_p):
                    continue
                loser = r if (mac(prev_vecs[:, r], new_vecs[:, r])
                              < mac(prev_vecs[:, c], new_vecs[:, c])) else c
                winner = c if loser == r else r
                p_fix, w_fix, k_fix = _iterate_branch(
                    M, K, aero_model, q_dyn, mach, b_ref, U,
                    prev_vecs[:, loser], k_start[loser], max_iter, tol, relax,
                    exclude_vec=new_vecs[:, winner])
                if mac(prev_vecs[:, loser], w_fix) >= 0.2:
                    p_new[loser] = p_fix
                    new_vecs[:, loser] = w_fix
                    prev_k[loser] = max(k_fix, 1e-4)

        for r in range(n):
            omega_out[r, iv] = abs(p_new[r].imag)
            damp_out[r, iv] = (2.0 * p_new[r].real / p_new[r].imag
                               if abs(p_new[r].imag) > 1e-9 else 0.0)
        prev_vecs = new_vecs

    return FlutterResult(velocities, omega_out, damp_out, rho, b_ref, mach)


def divergence_speed(structural_model, aero_model, rho, mach=0.0, k_steady=0.0):
    """Static divergence speed [m/s], or ``None`` if the wing cannot diverge.

    Divergence is the static aeroelastic instability ``K q = q_dyn Qhh(0) q``: the
    smallest positive real generalized eigenvalue ``q_dyn`` of ``(K, Re Qhh(0))`` gives
    ``V_div = sqrt(2 q_dyn / rho)``. The p-k sweep cannot flag it (a real positive
    ``p^2`` is reported as ``g = 0``), so unattended parameter studies should check
    this alongside the flutter speed and label whichever instability comes first.

    Evaluate at exactly ``k_steady = 0`` (and, for a tabulated aero, a grid containing
    ``k = 0``): there the non-lifting steady columns vanish identically and drop out as
    infinite eigenvalues instead of polluting the spectrum with near-singular roots.
    """
    K = np.asarray(structural_model.K, dtype=float)
    Q0 = np.real(aero_model.Qhh(k_steady, mach))
    lam = eig(K, Q0, right=False)
    lam = lam[np.isfinite(lam)]
    real_pos = lam[(np.abs(lam.imag) <= 1e-6 * np.maximum(np.abs(lam.real), 1e-30))
                   & (lam.real > 0.0)]
    if len(real_pos) == 0:
        return None
    q_div = float(np.min(real_pos.real))
    return float(np.sqrt(2.0 * q_div / rho))
