"""Generate every figure used by the README and the usage guide.

    python docs/figures.py              # all figures
    python docs/figures.py --fast       # skip the one that needs PanelAero

Everything lands in ``docs/img/``. Figures are committed to the repository (see the
negation rule in ``.gitignore``) so a reader who only clones the repo still sees them,
but they are all reproducible from this one file.

One function per figure, each returning the path it wrote.

The last figure builds a doublet-lattice model and therefore needs PanelAero
installed; without it it is skipped with a warning and any previously generated file
is left in place.
"""

from __future__ import annotations

import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                       # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

IMG = os.path.join(ROOT, "docs", "img")

# House style: readable at print size inside an A4 PDF.
plt.rcParams.update({
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "legend.fontsize": 8,
    "figure.dpi": 160,
    "savefig.dpi": 160,
    "savefig.bbox": "tight",
    "axes.grid": True,
    "grid.alpha": 0.25,
})

INK = "#222222"
ACCENT = "#c0392b"
COOL = "#1f6fb4"
WARM = "#e08214"


def _out(name: str) -> str:
    os.makedirs(IMG, exist_ok=True)
    return os.path.join(IMG, name)


def _save(fig, name: str) -> str:
    path = _out(name)
    fig.savefig(path)
    plt.close(fig)
    print(f"  wrote docs/img/{name}")
    return path


# ----------------------------------------------------------------------------------
# 1. Real p-k output for the binary case
# ----------------------------------------------------------------------------------
def _binary_result():
    from flutter_calc.cases import binary_flutter
    from flutter_calc.solvers.pk import pk_flutter

    sm, aero, b_ref = binary_flutter.build()
    omega_vac, _ = sm.free_vibration()
    V = np.linspace(5.0, 250.0, 246)
    res = pk_flutter(sm, aero, V, rho=1.225, b_ref=b_ref)
    return res, omega_vac


def fig_vg_vf() -> str:
    """Annotated V-g / V-f diagram of the binary-flutter case (real solver output)."""
    res, omega_vac = _binary_result()
    crit = res.lowest_flutter()
    V = res.velocity

    fig, (ax_g, ax_f) = plt.subplots(2, 1, sharex=True, figsize=(6.4, 5.4))
    names = ["branch 0 (bending-dominated)", "branch 1 (torsion-dominated)"]
    cols = [COOL, ACCENT]
    for r in range(res.omega.shape[0]):
        lbl = names[r] if r < len(names) else f"branch {r}"
        ax_g.plot(V, res.damping[r], color=cols[r % 2], lw=1.7, label=lbl)
        ax_f.plot(V, res.omega[r] / (2 * np.pi), color=cols[r % 2], lw=1.7, label=lbl)

    ax_g.axhline(0.0, color=INK, lw=1.0, ls="--")
    ax_g.text(V[-1], 0.04, "g = 0: neutral stability", ha="right", va="bottom",
              fontsize=7.6, color=INK)
    ax_g.set_ylim(-1.6, 0.9)
    ax_g.set_ylabel("damping  $g = 2\\,\\mathrm{Re}(p)/\\mathrm{Im}(p)$")
    ax_g.set_title("V-g / V-f diagram -- binary flutter, Theodorsen strip theory")

    if crit is not None:
        Vf = crit["V_flutter"]
        for ax in (ax_g, ax_f):
            ax.axvline(Vf, color="#2c7a3f", lw=1.2, ls=":")
        ax_g.annotate(f"FLUTTER\n$V_f$ = {Vf:.0f} m/s",
                      xy=(Vf, 0.0), xytext=(Vf - 62, 0.55),
                      fontsize=8.4, color="#2c7a3f", fontweight="bold",
                      arrowprops=dict(arrowstyle="->", color="#2c7a3f", lw=1.2))
        ax_f.annotate(f"$f_f$ = {crit['omega_flutter']/(2*np.pi):.2f} Hz",
                      xy=(Vf, crit["omega_flutter"] / (2 * np.pi)),
                      xytext=(Vf - 52, 3.1), fontsize=8.4, color="#2c7a3f",
                      ha="center",
                      arrowprops=dict(arrowstyle="->", color="#2c7a3f", lw=1.2))

    for w in omega_vac:
        ax_f.axhline(w / (2 * np.pi), color="#999999", lw=0.8, ls=":")
    ax_f.text(V[-1], omega_vac[1] / (2 * np.pi) + 0.10, "in-vacuo natural frequencies",
              ha="right", va="bottom", fontsize=7.6, color="#666666")

    # Point at where the two branches are actually closest together.
    f = res.omega / (2 * np.pi)
    i_close = int(np.argmin(np.abs(f[1] - f[0])))
    ax_f.annotate("branches drawn together:\nCOALESCENCE",
                  xy=(V[i_close], 0.5 * (f[0, i_close] + f[1, i_close])),
                  xytext=(0.42 * V[-1], 5.1), fontsize=8.2, color=INK, ha="center",
                  arrowprops=dict(arrowstyle="->", color=INK, lw=1.0,
                                  connectionstyle="arc3,rad=-0.2"))
    ax_f.set_ylim(1.2, 6.5)
    ax_f.set_ylabel("frequency [Hz]")
    ax_f.set_xlabel("airspeed  $V$  [m/s]")
    ax_g.legend(loc="lower left", framealpha=0.92)
    fig.tight_layout()
    return _save(fig, "vg_vf_binary.png")


# ----------------------------------------------------------------------------------
# 2. Figure that needs the doublet-lattice backend
# ----------------------------------------------------------------------------------
def fig_flutter_sweep_example() -> str | None:
    """The figure `flutter_sweep.py` produces for the A320 preset, as shipped."""
    from flutter_calc.aero.cache import TabulatedAero, default_k_grid
    from flutter_calc.geometry import PointMass
    from flutter_calc.postprocess.plots import plot_vg_vf
    from flutter_calc.solvers.pk import pk_flutter
    from flutter_calc.wing import build_wing

    wing = dict(semi_span=16.0, root_chord=5.5, tip_chord=1.6, sweep_deg=25.0,
                ea_frac=0.40, cg_frac=0.46, half_wing_mass=6000.0,
                f_bending=2.0, f_torsion=5.5, n_bending=2, n_torsion=2)
    engine = PointMass(mass=2300.0, eta=0.34, xi=-0.4, pitch_inertia=1500.0)
    wb = build_wing(point_masses=[engine], backend="dlm", n_span=12, n_chord=5, **wing)
    aero = TabulatedAero(wb.aero, k_grid=default_k_grid(k_max=12.0, n=28), Ma=0.45)
    V = np.linspace(40.0, 400.0, 200)
    res = pk_flutter(wb.structure, aero, V, rho=1.225, b_ref=wb.b_ref, mach=0.45)

    fig = plot_vg_vf(res, title="flutter sweep -- Airbus A320 family  [preset A320]")
    path = _out("flutter_sweep_example.png")
    fig.savefig(path)
    plt.close(fig)
    print("  wrote docs/img/flutter_sweep_example.png")
    return path


FAST = [fig_vg_vf]
SLOW = [fig_flutter_sweep_example]


def main(fast_only: bool = False) -> None:
    os.makedirs(IMG, exist_ok=True)
    print(f"figures -> {IMG}")
    for f in FAST:
        f()
    if fast_only:
        print("  (--fast: skipped the doublet-lattice figure)")
        return
    try:
        import panelaero  # noqa: F401
    except ImportError:
        print("  WARNING: PanelAero is not installed -- skipping the DLM figure.\n"
              "           pip install \"git+https://github.com/DLR-AE/PanelAero.git\"")
        return
    for f in SLOW:
        f()


if __name__ == "__main__":
    main(fast_only="--fast" in sys.argv)
