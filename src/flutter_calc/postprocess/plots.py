"""V-g and V-f diagram plotting (matplotlib imported lazily)."""

from __future__ import annotations

import numpy as np


def plot_vg_vf(result, title="", savepath=None, show=False):
    """Plot V-g (damping) and V-f (frequency) diagrams for a FlutterResult.

    Returns the matplotlib Figure. Matplotlib is imported here so the core solver has
    no hard plotting dependency at import time.
    """
    import matplotlib.pyplot as plt

    V = result.velocity
    n_branch = result.omega.shape[0]
    fig, (ax_g, ax_f) = plt.subplots(2, 1, sharex=True, figsize=(7, 7))

    for r in range(n_branch):
        ax_g.plot(V, result.damping[r], label=f"branch {r}")
        ax_f.plot(V, result.omega[r] / (2 * 3.141592653589793), label=f"branch {r}")

    ax_g.axhline(0.0, color="k", lw=0.8, ls="--")
    crit = result.lowest_flutter()
    if crit is not None:
        ax_g.axvline(crit["V_flutter"], color="r", lw=0.8, ls=":")
        ax_f.axvline(crit["V_flutter"], color="r", lw=0.8, ls=":",
                     label=f"Vf={crit['V_flutter']:.1f} m/s")

    ax_g.set_ylabel("damping  g = 2 Re(p)/Im(p)")
    ax_f.set_ylabel("frequency [Hz]")
    ax_f.set_xlabel("airspeed V [m/s]")
    ax_g.set_title(title)
    ax_g.legend(fontsize=8)
    ax_f.legend(fontsize=8)
    fig.tight_layout()

    if savepath:
        fig.savefig(savepath, dpi=130)
    if show:
        plt.show()
    return fig


def plot_flutter_boundary(points, title="", V_D=None, eas=False, savepath=None, show=False):
    """Plot a matched-point flutter boundary (altitude vs flutter speed).

    ``points`` is a list of :class:`~flutter_calc.envelope.MatchedPoint`. With ``eas=True``
    the speed axis is equivalent airspeed (the natural frame for a ``V_D`` reference,
    which collapses the altitude variation of the certification line). If ``V_D`` is given
    the ``1.15 * V_D`` CS-25.629 clearance line is drawn for comparison.
    """
    import matplotlib.pyplot as plt

    flut = [p for p in points if p.flutter and p.V_flutter is not None]
    spd = (lambda p: p.V_eas) if eas else (lambda p: p.V_flutter)
    valid = [p for p in flut if not getattr(p, "beyond_validity", False)]
    beyond = [p for p in flut if getattr(p, "beyond_validity", False)]

    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.plot([spd(p) for p in valid], [p.altitude / 1e3 for p in valid], "-o", ms=4,
            color="C0", label="flutter boundary (subsonic-valid)")
    if beyond:
        ax.plot([spd(p) for p in beyond], [p.altitude / 1e3 for p in beyond], "--s", ms=5,
                mfc="none", color="C0", alpha=0.6,
                label="transonic flutter (beyond DLM validity)")
    for p in flut:
        if p.mach is not None:
            ax.annotate(f"M{p.mach:.2f}", (spd(p), p.altitude / 1e3), fontsize=7,
                        xytext=(4, 2), textcoords="offset points")
    if V_D is not None:
        ax.axvline(1.15 * V_D, color="r", lw=1.0, ls="--",
                   label=f"1.15 V_D = {1.15 * V_D:.0f} m/s (CS-25.629)")
        ax.axvline(V_D, color="r", lw=0.7, ls=":", label=f"V_D = {V_D:.0f} m/s")
    ax.set_xlabel("flutter speed  " + ("V_EAS" if eas else "V_TAS") + "  [m/s]")
    ax.set_ylabel("altitude [km]")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=130)
    if show:
        plt.show()
    return fig


def plot_fsi_vs_mach(result, title="", validity_mach=0.85, dip=None,
                     savepath=None, show=False):
    """Plot the flutter-speed index versus Mach (output of ``flutter_index_vs_mach``).

    The subsonic-valid range is drawn solid; beyond ``validity_mach`` the curve is dashed
    and shaded to mark where the linear DLM stops being trustworthy. If ``dip`` is given
    as ``(mach_bottom, fsi_bottom)`` (from ``transonic_dip_bound``) the literature-bounded
    transonic dip is sketched -- explicitly a bound, not a computed value.
    """
    import matplotlib.pyplot as plt

    m = result["mach"]
    fsi = result["fsi"]
    valid = m <= validity_mach

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(m[valid], fsi[valid], "-o", ms=4, color="C0", label="DLM (subsonic, valid)")
    if (~valid).any():
        ax.plot(m[~valid], fsi[~valid], "--o", ms=4, color="C0", alpha=0.5,
                label="DLM (beyond validity)")
        ax.axvspan(validity_mach, m.max(), color="grey", alpha=0.12)
    ax.axvline(validity_mach, color="grey", lw=0.8, ls=":")
    if dip is not None:
        mb, fb = dip
        sub = np.nanmax(fsi[valid]) if valid.any() else np.nanmax(fsi)
        ax.plot([validity_mach, mb], [sub, fb], "r:", lw=1.2)
        ax.plot([mb], [fb], "rv", ms=7,
                label=f"transonic dip bound (~{fb:.2f}, literature)")
    ax.set_xlabel("Mach number")
    ax.set_ylabel(r"flutter-speed index  $V_f / (b\,\omega_\alpha\sqrt{\mu})$")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=130)
    if show:
        plt.show()
    return fig
