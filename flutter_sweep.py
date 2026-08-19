#!/usr/bin/env python
"""Flutter Calculator -- single-file flutter sweep.

WHAT THIS DOES
    Builds one cantilever wing (an aircraft PRESET, or your own CUSTOM geometry) and runs a
    p-k flutter sweep across airspeed V at a fixed altitude and Mach. It prints the flutter
    speed, frequency and flutter-speed index, and saves a V-g / V-f diagram.

HOW TO RUN  (from the repository folder -- no other setup, no PYTHONPATH needed)
    python flutter_sweep.py

    Edit ONLY the CONFIG block. Everything below "END CONFIG" is machinery.
    See flutter_calculator_usage_guide.pdf for a walkthrough of every parameter.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

import numpy as np

# =================================== CONFIG ===================================
# 1) PICK AN AIRCRAFT --------------------------------------------------------------
#    Choose one of the preset keys listed under PRESETS below (the ten most numerous
#    airliners in service), or "custom" to use CUSTOM_WING / CUSTOM_ENGINE instead.
PRESET = "A320"     # A320 A330 A350 ATR72 B737 B767 B777 B787 CRJ900 E190  |  or "custom"

# 2) CUSTOM WING (used only when PRESET == "custom") -------------------------------
CUSTOM_WING = dict(
    semi_span      = 16.0,    # exposed half-span, root to tip             [m]
    root_chord     = 5.5,     # chord at the root                          [m]
    tip_chord      = 1.6,     # chord at the tip                           [m]
    sweep_deg      = 25.0,    # quarter-chord sweep, positive aft          [deg]
    ea_frac        = 0.40,    # elastic axis, fraction of chord aft of LE  [-]
    cg_frac        = 0.46,    # section CG,  fraction of chord aft of LE   [-]
    half_wing_mass = 6000.0,  # structural + fuel mass of the half-wing    [kg]
    f_bending      = 2.0,     # target 1st bending frequency               [Hz]
    f_torsion      = 5.5,     # target 1st torsion frequency               [Hz]
    n_bending      = 2,       # number of bending assumed-shapes           [-]
    n_torsion      = 2,       # number of torsion assumed-shapes           [-]
)
CUSTOM_ENGINE = dict(         # set CUSTOM_ENGINE = None for a clean wing (no engine)
    mass          = 2300.0,   # engine + pylon mass                        [kg]
    eta           = 0.34,     # spanwise location, fraction of semi-span   [-]
    xi            = -0.4,     # CG offset aft(+) / forward(-) of the EA    [m]
    pitch_inertia = 1500.0,   # pitch inertia about the engine's own CG    [kg m^2]
)

# 3) AERODYNAMICS ------------------------------------------------------------------
AERO = dict(
    backend = "dlm",          # "dlm" (swept, compressible) | "theodorsen" (fast, unswept)
    mach    = 0.45,           # flight Mach for the DLM (theodorsen ignores it)  [-]
    n_span  = 12,             # DLM spanwise panels                        [-]
    n_chord = 5,              # DLM chordwise panels                       [-]
)

# 4) FLIGHT CONDITION (sets the air density) ---------------------------------------
FLIGHT = dict(
    altitude_m   = 0.0,       # ISA altitude -> density (0 = sea level)    [m]
    density_kgm3 = None,      # set a number here to override the ISA density [kg/m^3]
)

# 5) AIRSPEED SWEEP ----------------------------------------------------------------
SWEEP = dict(
    v_min = 40.0,             # lowest airspeed in the sweep               [m/s]
    v_max = 400.0,            # highest airspeed in the sweep              [m/s]
    n_v   = 200,              # number of speed points (>= 10)             [-]
    g_structural = 0.0,       # hysteretic structural damping available to oppose
                              # flutter: 0 = conservative bare structure (default);
                              # measured transport airframes justify ~0.02-0.03  [-]
)

# 6) OUTPUT ------------------------------------------------------------------------
OUTPUT = dict(
    plot_path = "flutter_sweep.png",   # saved next to this script unless absolute
    title     = None,                  # None -> auto title from the aircraft name
    show      = False,                 # True to also pop up the plot window
)

# --- PRESET LIBRARY (top-10 airliners by fleet size, sorted alphabetically) -------
#     Geometry/mass scale from each type's real span and MTOW; structural frequencies
#     and EA/CG fractions are REPRESENTATIVE ESTIMATES, not certified data. Edit freely.
PRESETS = {
    "A320": dict(name="Airbus A320 family",
        wing=dict(semi_span=16.0, root_chord=5.5, tip_chord=1.6, sweep_deg=25.0,
                  ea_frac=0.40, cg_frac=0.46, half_wing_mass=6000.0,
                  f_bending=2.0, f_torsion=5.5, n_bending=2, n_torsion=2),
        engine=dict(mass=2300.0, eta=0.34, xi=-0.4, pitch_inertia=1500.0)),
    "A330": dict(name="Airbus A330",
        wing=dict(semi_span=28.0, root_chord=10.5, tip_chord=2.7, sweep_deg=30.0,
                  ea_frac=0.40, cg_frac=0.46, half_wing_mass=18000.0,
                  f_bending=1.4, f_torsion=3.6, n_bending=2, n_torsion=2),
        engine=dict(mass=6500.0, eta=0.30, xi=-0.6, pitch_inertia=6000.0)),
    "A350": dict(name="Airbus A350",
        wing=dict(semi_span=30.0, root_chord=9.0, tip_chord=2.0, sweep_deg=32.0,
                  ea_frac=0.40, cg_frac=0.46, half_wing_mass=21000.0,
                  f_bending=1.3, f_torsion=3.3, n_bending=2, n_torsion=2),
        engine=dict(mass=7500.0, eta=0.32, xi=-0.7, pitch_inertia=7000.0)),
    "ATR72": dict(name="ATR 72 (turboprop)",
        wing=dict(semi_span=12.5, root_chord=2.6, tip_chord=1.6, sweep_deg=3.0,
                  ea_frac=0.40, cg_frac=0.45, half_wing_mass=1800.0,
                  f_bending=3.5, f_torsion=8.0, n_bending=2, n_torsion=2),
        engine=dict(mass=1100.0, eta=0.35, xi=-0.5, pitch_inertia=800.0)),
    "B737": dict(name="Boeing 737 (NG/MAX)",
        wing=dict(semi_span=16.0, root_chord=6.0, tip_chord=1.5, sweep_deg=25.0,
                  ea_frac=0.40, cg_frac=0.46, half_wing_mass=5800.0,
                  f_bending=2.2, f_torsion=5.8, n_bending=2, n_torsion=2),
        engine=dict(mass=2500.0, eta=0.34, xi=-0.4, pitch_inertia=1500.0)),
    "B767": dict(name="Boeing 767",
        wing=dict(semi_span=22.0, root_chord=8.5, tip_chord=2.1, sweep_deg=31.5,
                  ea_frac=0.40, cg_frac=0.46, half_wing_mass=14000.0,
                  f_bending=1.6, f_torsion=4.0, n_bending=2, n_torsion=2),
        engine=dict(mass=5500.0, eta=0.31, xi=-0.6, pitch_inertia=5000.0)),
    "B777": dict(name="Boeing 777",
        wing=dict(semi_span=30.0, root_chord=12.0, tip_chord=3.0, sweep_deg=31.6,
                  ea_frac=0.40, cg_frac=0.46, half_wing_mass=27000.0,
                  f_bending=1.2, f_torsion=3.0, n_bending=2, n_torsion=2),
        engine=dict(mass=9000.0, eta=0.30, xi=-0.8, pitch_inertia=9000.0)),
    "B787": dict(name="Boeing 787",
        wing=dict(semi_span=28.0, root_chord=9.0, tip_chord=2.0, sweep_deg=32.0,
                  ea_frac=0.40, cg_frac=0.46, half_wing_mass=19000.0,
                  f_bending=1.2, f_torsion=3.2, n_bending=2, n_torsion=2),
        engine=dict(mass=6800.0, eta=0.32, xi=-0.7, pitch_inertia=6500.0)),
    "CRJ900": dict(name="Bombardier CRJ900",
        wing=dict(semi_span=11.6, root_chord=3.6, tip_chord=1.5, sweep_deg=24.75,
                  ea_frac=0.40, cg_frac=0.46, half_wing_mass=2900.0,
                  f_bending=2.8, f_torsion=6.5, n_bending=2, n_torsion=2),
        engine=None),   # CRJ engines are aft-fuselage mounted, not on the wing
    "E190": dict(name="Embraer E190 (E-Jet)",
        wing=dict(semi_span=13.4, root_chord=5.0, tip_chord=1.5, sweep_deg=24.0,
                  ea_frac=0.40, cg_frac=0.46, half_wing_mass=3900.0,
                  f_bending=2.5, f_torsion=6.0, n_bending=2, n_torsion=2),
        engine=dict(mass=2000.0, eta=0.34, xi=-0.4, pitch_inertia=1200.0)),
}
# ================================= END CONFIG ================================


def _fail(msg):
    print("ERROR: " + msg)
    sys.exit(1)


def _resolve_preset():
    """Return (wing_dict, engine_dict_or_None, display_name) for the chosen PRESET."""
    if PRESET == "custom":
        return dict(CUSTOM_WING), (dict(CUSTOM_ENGINE) if CUSTOM_ENGINE else None), "custom wing"
    if PRESET not in PRESETS:
        _fail(f"PRESET '{PRESET}' is not known. Choose one of: "
              f"{', '.join(sorted(PRESETS))}  (or \"custom\").")
    p = PRESETS[PRESET]
    eng = p.get("engine")
    return dict(p["wing"]), (dict(eng) if eng else None), f"{p['name']}  [preset {PRESET}]"


def _check_config(wing, engine):
    if not (0.0 < SWEEP["v_min"] < SWEEP["v_max"]):
        _fail("SWEEP needs 0 < v_min < v_max.")
    if SWEEP["n_v"] < 10:
        _fail("SWEEP n_v should be >= 10 for a usable sweep.")
    if not (0.0 <= SWEEP.get("g_structural", 0.0) < 0.10):
        _fail("SWEEP g_structural must be in [0, 0.1) -- typical measured values "
              "are 0.02-0.03.")
    if AERO["backend"] not in ("dlm", "theodorsen"):
        _fail("AERO backend must be 'dlm' or 'theodorsen'.")
    if not (0.0 <= AERO["mach"] < 0.95):
        _fail("AERO mach must be in [0, 0.95) -- the DLM is a subsonic method.")
    if AERO["backend"] == "dlm" and AERO["mach"] > 0.85:
        # keep in step with flutter_calc.envelope.DLM_MACH_VALID_MAX
        print("WARNING: Mach > 0.85 is beyond the declared subsonic-DLM validity "
              "ceiling (transonic dip not captured); results mark a boundary, they "
              "are not predictions.")
    for key in ("semi_span", "root_chord", "tip_chord", "half_wing_mass",
                "f_bending", "f_torsion"):
        if wing.get(key, 0) <= 0:
            _fail(f"wing['{key}'] must be positive.")
    if engine is not None and engine.get("mass", 0) <= 0:
        _fail("engine['mass'] must be positive (or set the engine to None).")
    if wing["f_bending"] >= wing["f_torsion"]:
        print("WARNING: f_bending >= f_torsion is unusual (bending is normally the lower "
              "mode). Continuing anyway.")


def _resolve(path):
    if os.path.isabs(path):
        return path
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), path)


def main():
    wing, engine, label = _resolve_preset()
    _check_config(wing, engine)

    from flutter_calc import envelope
    from flutter_calc.aero.cache import TabulatedAero, default_k_grid
    from flutter_calc.geometry import PointMass
    from flutter_calc.nondim import flutter_speed_index
    from flutter_calc.postprocess.plots import plot_vg_vf
    from flutter_calc.solvers.pk import pk_flutter
    from flutter_calc.wing import build_wing

    # --- build the wing ------------------------------------------------------
    point_masses = [] if engine is None else [PointMass(**engine)]
    wb = build_wing(point_masses=point_masses, backend=AERO["backend"],
                    n_span=AERO["n_span"], n_chord=AERO["n_chord"], **wing)

    # --- flight condition (density) ------------------------------------------
    if FLIGHT["density_kgm3"] is not None:
        rho = float(FLIGHT["density_kgm3"])
        flight_note = f"density set directly = {rho:.4f} kg/m^3"
    else:
        atm = envelope.isa(FLIGHT["altitude_m"])
        rho = atm.rho
        flight_note = f"ISA {FLIGHT['altitude_m']:.0f} m -> rho {rho:.4f} kg/m^3"

    # --- aero for the solve --------------------------------------------------
    if AERO["backend"] == "dlm":
        # The raw DLM is far too slow inside the p-k loop; tabulate Qhh on a k-grid once.
        # k_max 12 keeps even the highest tracked branch (second torsion at the lowest
        # sweep speed) inside the table so no branch has its aero frozen by clamping.
        aero = TabulatedAero(wb.aero, k_grid=default_k_grid(k_max=12.0, n=28), Ma=AERO["mach"])
        mach = AERO["mach"]
    else:
        aero = wb.aero            # Theodorsen is analytic and cheap; no tabulation needed
        mach = 0.0

    # --- run the sweep -------------------------------------------------------
    V = np.linspace(SWEEP["v_min"], SWEEP["v_max"], SWEEP["n_v"])
    result = pk_flutter(wb.structure, aero, V, rho=rho, b_ref=wb.b_ref, mach=mach)
    g_s = SWEEP.get("g_structural", 0.0)
    crit = result.lowest_flutter(g_cross=g_s)
    mu = wb.mass_ratio(rho)

    # --- report --------------------------------------------------------------
    bar = "=" * 64
    freqs = ", ".join(f"{f:.2f}" for f in wb.natural_hz[:4])
    eng = "none" if engine is None else f"{engine['mass']:.0f} kg at {engine['eta']*100:.0f}% span"
    print(bar)
    print(" Flutter Calculator -- flutter sweep")
    print(bar)
    print(f" Aircraft  : {label}")
    print(f" Wing      : semi-span {wing['semi_span']:.1f} m, chords "
          f"{wing['root_chord']:.1f}->{wing['tip_chord']:.1f} m, sweep {wing['sweep_deg']:.0f} deg")
    print(f" Structure : EI={wb.EI:.3e}  GJ={wb.GJ:.3e} N m^2  "
          f"(calibrated to f_bend={wing['f_bending']:.1f}, f_tors={wing['f_torsion']:.1f} Hz)")
    print(f" Nat. freqs: {freqs} Hz")
    print(f" Engine    : {eng}")
    print(f" Aero      : {AERO['backend']}"
          + (f", Mach {AERO['mach']:.2f}, {AERO['n_span']}x{AERO['n_chord']} panels"
             if AERO['backend'] == 'dlm' else " (incompressible, unswept)"))
    print(f" Flight    : {flight_note}")
    print(f" Sweep     : V {SWEEP['v_min']:.0f}->{SWEEP['v_max']:.0f} m/s ({SWEEP['n_v']} pts), "
          f"b_ref {wb.b_ref:.2f} m, mu {mu:.1f}"
          + (f", flutter at g = {g_s:.3f}" if g_s else ""))
    print("-" * 64)
    if crit is None:
        print(f" RESULT: no flutter found in {SWEEP['v_min']:.0f}-{SWEEP['v_max']:.0f} m/s.")
        print("   -> raise SWEEP['v_max'], or make the wing more flutter-prone")
        print("      (e.g. CG aft of the EA: cg_frac > ea_frac, or lower f_torsion).")
    else:
        fsi = flutter_speed_index(crit["V_flutter"], wb.b_ref, wb.omega_alpha, mu)
        print(f" RESULT: FLUTTER at V = {crit['V_flutter']:.1f} m/s,  "
              f"f = {crit['omega_flutter']/(2*np.pi):.2f} Hz,  "
              f"index = {fsi:.3f}   (branch {crit['branch']})")
    if getattr(aero, "n_clamped_high", 0):
        print(f" NOTE: {aero.n_clamped_high} aero queries exceeded the k-grid top "
              "(branch damping there uses frozen Qhh; raise k_max if it matters).")
    print("-" * 64)

    title = OUTPUT["title"] or f"flutter sweep -- {label}"
    plot_path = _resolve(OUTPUT["plot_path"])
    plot_vg_vf(result, title=title, savepath=plot_path, show=OUTPUT["show"])
    print(f" V-g / V-f diagram saved to: {plot_path}")
    print(bar)


if __name__ == "__main__":
    main()
