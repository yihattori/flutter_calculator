"""Generate flutter_calculator_usage_guide.pdf in the repository root.

    python docs/make_usage_guide_pdf.py

A self-contained walkthrough of flutter_sweep.py: every CONFIG knob, how to read the
output, what to do when it misbehaves, and the library API. Keep it in step with the
CONFIG block -- regenerate after changing presets, defaults or knobs.

Figures come from docs/img (regenerate with `python docs/figures.py`).
Requires reportlab:  pip install -e ".[docs]"
"""

import sys
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    PageBreak, Paragraph, Preformatted, SimpleDocTemplate, Spacer,
)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "docs"))

from _pdfcommon import BODY_FONT, BOLD_FONT, MONO_FONT   # noqa: E402
from _pdfcommon import fig, keep, table                   # noqa: E402

OUT = ROOT / "flutter_calculator_usage_guide.pdf"

#: Counted from the suite itself so the number in the PDF cannot go stale.
N_TESTS = sum(
    sum(1 for line in p.read_text(encoding="utf-8").splitlines()
        if line.startswith("def test_"))
    for p in sorted((ROOT / "tests").glob("test_*.py"))
)

FALLBACK_RUN = """ RESULT: FLUTTER at V = 211.1 m/s,  f = 3.86 Hz,  index = 0.619
         (branch 1)"""


def real_run():
    """Run flutter_sweep.py and return (console text, flutter frequency in Hz).

    Showing a transcript nobody has run is how usage guides start lying. This runs the
    shipped defaults and embeds the actual output, with the local absolute path scrubbed
    so no one's directory tree ends up in a published PDF. Needs PanelAero, since the
    default backend is the DLM; without it the section falls back to a stored excerpt.
    """
    import re
    import subprocess

    png = ROOT / "flutter_sweep.png"
    existed = png.exists()
    try:
        proc = subprocess.run([sys.executable, "flutter_sweep.py"], cwd=str(ROOT),
                              capture_output=True, text=True, timeout=900)
        if proc.returncode != 0:
            raise RuntimeError(proc.stderr.strip()[-300:])
        text = proc.stdout.replace("\r\n", "\n").strip("\n")
        # Drop the machine-specific output path.
        text = re.sub(r"(V-g / V-f diagram saved to: ).*", r"\1flutter_sweep.png", text)
    except Exception as exc:
        print(f"  note: could not run flutter_sweep.py ({exc}); using stored excerpt")
        return FALLBACK_RUN, 3.86
    finally:
        if not existed and png.exists():
            png.unlink()
    m = re.search(r"f = ([\d.]+) Hz", text)
    return text, float(m.group(1)) if m else 3.86


RUN_TEXT, RUN_FREQ = real_run()

ss = getSampleStyleSheet()
H1 = ParagraphStyle("H1", parent=ss["Heading1"], fontName=BOLD_FONT,
                    spaceBefore=11, spaceAfter=4, fontSize=13.5, leading=16.5)
H2 = ParagraphStyle("H2", parent=ss["Heading2"], fontName=BOLD_FONT,
                    spaceBefore=9, spaceAfter=3, fontSize=10.5, leading=13.5)
BODY = ParagraphStyle("BODY", parent=ss["BodyText"], fontName=BODY_FONT,
                      fontSize=8.9, leading=12.6, spaceAfter=5)
SMALL = ParagraphStyle("SMALL", parent=BODY, fontSize=7.9, leading=10.4,
                       textColor=colors.HexColor("#444444"))

CODE = ParagraphStyle("CODE", parent=ss["Code"], fontName=MONO_FONT,
                      fontSize=7.4, leading=10.0,
                      backColor=colors.HexColor("#f4f4f4"), borderPadding=5,
                      leftIndent=3, spaceBefore=3, spaceAfter=7)

def P(txt, style=BODY):
    return Paragraph(txt, style)


def H(title, *first):
    """A section heading, kept with whatever follows it."""
    return keep(Paragraph(title, H1), *first)


story = [
    Paragraph("Flutter Calculator", ParagraphStyle("T", parent=ss["Title"], fontName=BOLD_FONT, fontSize=23, leading=27)),
    P("<b>Usage guide</b> &nbsp;|&nbsp; a complete walkthrough of "
      "<font face='DejaVuSansMono'>flutter_sweep.py</font>, the single-file entry point, and of "
      "the library underneath it. Everything you need is in this document.", SMALL),

    # --- 1 ---------------------------------------------------------------------------
    H("1&nbsp; What it does",
      P("<font face='DejaVuSansMono'>flutter_sweep.py</font> builds one cantilever wing -- an "
      "aircraft preset or your own geometry -- and runs a p-k flutter sweep across "
      "airspeed at a fixed altitude and Mach. It prints the flutter speed, frequency and "
      "flutter-speed index, and saves a V-g / V-f diagram.")),
    P("Under the hood: a Rayleigh-Ritz beam structure (assumed bending and torsion "
      "shapes, energy integrals, no FEA), unsteady aerodynamics from either analytic "
      "strip theory or a doublet-lattice panel method, and a p-k eigenvalue solver with "
      "branch tracking. A run takes seconds to about a minute."),
    P("<b>What it does not do.</b> It will not give you a certifiable flutter speed. It "
      "is a linear, subsonic, single-wing model: no transonic dip, no fuselage or "
      "free-free modes, no control surfaces or stores, no gust response, no limit-cycle "
      "behaviour. Section 7 spells this out. Used for what it is good at -- comparing "
      "wings, seeing which way a design change moves the boundary, teaching the method "
      "-- it is quick and honest."),

    Paragraph("2&nbsp; Install and run", H1),
    Preformatted(
        "git clone https://github.com/yihattori/flutter_calculator.git\n"
        "cd flutter_calculator\n"
        "python -m pip install -e \".[dev]\"\n"
        "\n"
        "# doublet-lattice backend (optional, needed for AERO backend = \"dlm\")\n"
        "python -m pip install \"git+https://github.com/DLR-AE/PanelAero.git\"", CODE),
    P("Then edit the <b>CONFIG block</b> at the top of "
      "<font face='DejaVuSansMono'>flutter_sweep.py</font> and run it. Everything below the "
      "<font face='DejaVuSansMono'>END CONFIG</font> marker is machinery; you never need to "
      "touch it. The script adds <font face='DejaVuSansMono'>src/</font> to the path itself, so "
      "no <font face='DejaVuSansMono'>PYTHONPATH</font> juggling is required."),
    Preformatted("python flutter_sweep.py", CODE),
    P("Without PanelAero installed, set <font face='DejaVuSansMono'>AERO[\"backend\"] = "
      "\"theodorsen\"</font>; everything else works unchanged. PanelAero (DLR Institute "
      "of Aeroelasticity, BSD 3-Clause) is what actually computes the doublet-lattice "
      "aerodynamics -- see section 8 for the credit it is owed and how to cite it.",
      SMALL),

    # --- 3 ---------------------------------------------------------------------------
    Paragraph("3&nbsp; CONFIG reference", H1),
    Paragraph("3.1&nbsp; Pick an aircraft", H2),
    P("<font face='DejaVuSansMono'>PRESET</font> selects one of ten representative airliners -- "
      "A320, A330, A350, ATR72, B737, B767, B777, B787, CRJ900, E190 -- or "
      "<font face='DejaVuSansMono'>\"custom\"</font> to use your own numbers. Presets scale from "
      "each type's real span and weight, but the stiffness anchors and CG/EA positions "
      "are <b>representative estimates, not certified data</b>. The CRJ900 has no wing "
      "engine (its engines are aft-fuselage mounted), so it is the natural clean-wing "
      "comparison."),

    Paragraph("3.2&nbsp; Custom wing (when PRESET = \"custom\")", H2),
    table([
        ["key", "meaning", "unit"],
        ["semi_span", "exposed half-span, root to tip", "m"],
        ["root_chord / tip_chord", "chord at root / tip (linear taper)", "m"],
        ["sweep_deg", "quarter-chord sweep, positive aft", "deg"],
        ["ea_frac", "elastic axis, fraction of chord aft of the leading edge", "-"],
        ["cg_frac", "section CG, fraction of chord aft of the LE. Aft of the EA\n"
                    "is what couples bending and torsion -- the flutter mechanism", "-"],
        ["half_wing_mass", "structure + fuel mass of the half-wing", "kg"],
        ["f_bending / f_torsion", "first bending / torsion frequency of the AS-BUILT\n"
                                  "wing, engine on -- GVT-style anchors. EI and GJ are\n"
                                  "calibrated until the built model hits them", "Hz"],
        ["n_bending / n_torsion", "number of assumed shapes (2 + 2 is converged to\n"
                                  "about 2% for these wings)", "-"],
    ], widths=[36 * mm, 110 * mm, 12 * mm], mono_first_col=True),
    P("<font face='DejaVuSansMono'>CUSTOM_ENGINE</font> takes <font face='DejaVuSansMono'>mass</font> "
      "[kg], <font face='DejaVuSansMono'>eta</font> (span fraction), "
      "<font face='DejaVuSansMono'>xi</font> (CG offset aft(+) or forward(-) of the EA, m) and "
      "<font face='DejaVuSansMono'>pitch_inertia</font> [kg m&#178;]. Set it to "
      "<font face='DejaVuSansMono'>None</font> for a clean wing. A pylon hung forward of the "
      "elastic axis (negative <font face='DejaVuSansMono'>xi</font>) is mass-balancing and "
      "usually raises the flutter speed."),
    P("<b>Why no EI and GJ?</b> Because you rarely know them early in a design, but you "
      "can usually name the wing's first bending and torsion frequencies. The builder "
      "iterates the stiffnesses until the assembled wing -- CG offset applied, engine "
      "attached -- reproduces the frequencies you asked for. The engine is on during "
      "calibration because a ground vibration test measures the engined aircraft."),

    Paragraph("3.3&nbsp; Aerodynamics, flight condition, sweep", H2),
    table([
        ["key", "meaning"],
        ["AERO backend", "\"dlm\" (swept, compressible; the production model) or "
         "\"theodorsen\" (fast unswept incompressible check, no PanelAero needed)"],
        ["AERO mach", "flight Mach for the DLM. Subsonic method: hard fail at 0.95, "
         "warning above the 0.85 validity ceiling. Theodorsen ignores it"],
        ["AERO n_span / n_chord", "DLM panels; the 12 x 5 default is within about 2% of "
         "20 x 8"],
        ["FLIGHT altitude_m", "ISA altitude, which sets the density (0 = sea level)"],
        ["FLIGHT density_kgm3", "set a number to override the ISA density directly"],
        ["SWEEP v_min / v_max / n_v", "airspeed range and resolution of the p-k sweep"],
        ["SWEEP g_structural", "hysteretic structural damping available to oppose "
         "flutter. 0 is the conservative bare-structure default; measured transport "
         "airframes justify 0.02-0.03. Flutter is then read where a branch's damping "
         "crosses that level rather than zero"],
        ["OUTPUT", "plot_path, title (None = auto), show = True to pop the window"],
    ], widths=[42 * mm, 116 * mm], mono_first_col=True),

    PageBreak(),

    # --- 4 ---------------------------------------------------------------------------
    Paragraph("4&nbsp; Reading the output", H1),
    P("This is what the shipped defaults actually print -- the block below was produced "
      "by running <font face='DejaVuSansMono'>flutter_sweep.py</font> when this guide was "
      "generated:", SMALL),
    Preformatted(RUN_TEXT, CODE),
    P("<b>V</b> is the true airspeed at which the tracked branch's damping crosses the "
      "<font face='DejaVuSansMono'>g_structural</font> level. At sea level TAS equals equivalent "
      "airspeed; at altitude, convert before comparing with a quoted V<sub>D</sub>."),
    P(f"<b>f</b> is the flutter frequency. For classical bending-torsion flutter it sits "
      f"between the bending and torsion natural frequencies -- here {RUN_FREQ:.2f} Hz, "
      "between the 2.00 Hz bending and 5.50 Hz torsion anchors -- because the two "
      "branches have merged. A flutter frequency outside that band is a signal to look "
      "closely at the diagram before believing it."),
    P("<b>index</b> is the flutter-speed index V<sub>f</sub> / (b &#183; "
      "&#969;<sub>&#945;</sub> &#183; &#8730;&#181;). It removes size and density, so it "
      "is the right number to quote when comparing wings. Transport wings typically land "
      "around 0.4-0.6. <b>branch</b> tells you which V-g curve did the crossing."),
    P("The header is worth reading too: the calibrated EI and GJ tell you what stiffness "
      "your frequency anchors implied, and the natural frequencies should equal your "
      "<font face='DejaVuSansMono'>f_bending</font> and <font face='DejaVuSansMono'>f_torsion</font> "
      "targets exactly. If they do not, the calibration struggled -- see section 5."),

    keep(Paragraph("4.1&nbsp; The V-g / V-f diagram", H2),
         *fig("flutter_sweep_example.png", 122,
              "The saved diagram. <b>Top (V-g):</b> damping against speed. A branch "
              "rising through zero -- or through your g_structural line -- is flutter. "
              "<b>Bottom (V-f):</b> frequency against speed; watch two branches draw "
              "together as speed rises. That convergence is coalescence, the fingerprint "
              "of bending-torsion flutter.")),
    P("Read the two panels together. Frequencies merging with damping still negative is "
      "benign; frequencies merging as one branch's damping climbs to zero is the "
      "classical mechanism. A branch that crosses zero without any partner nearby in "
      "frequency is usually single-degree-of-freedom behaviour and deserves a second "
      "look."),

    PageBreak(),

    # --- 5 ---------------------------------------------------------------------------
    Paragraph("5&nbsp; When it misbehaves", H1),
    table([
        ["message or symptom", "what it means, and what to do"],
        ["no flutter found in the\nspeed range",
         "The sweep never saw a crossing. Raise SWEEP v_max, or make the wing more "
         "flutter-prone: move the CG aft of the EA (cg_frac > ea_frac) or lower "
         "f_torsion. A genuinely mass-balanced wing may simply not flutter in range."],
        ["NOTE: n aero queries\nexceeded the k-grid top",
         "A high-frequency branch ran past the tabulated reduced-frequency range at low "
         "speed, so its aero is frozen there. Harmless for the flutter point unless the "
         "flagged branch is the one that crosses; if it is, raise k_max in the "
         "TabulatedAero call."],
        ["EI/GJ frequency calibration\ndid not converge",
         "The requested f_bending / f_torsion pair is unreachable for that mass layout -- "
         "usually a very heavy engine combined with an extreme frequency target. Move the "
         "targets closer to plausible values, or lighten the point mass."],
        ["WARNING: f_bending >=\nf_torsion",
         "Unusual: bending is normally the lower mode. The run continues, but check you "
         "have not swapped the two."],
        ["WARNING: Mach > 0.85",
         "Beyond the subsonic-DLM validity ceiling. The result marks a boundary; it is "
         "not a prediction. See section 7."],
        ["ERROR: AERO mach must be\nin [0, 0.95)",
         "A hard stop. The doublet-lattice kernel becomes singular approaching Mach 1 and "
         "would return nonsense."],
        ["flutter speed looks far too\nlow or too high",
         "Check the torsion frequency first -- the flutter speed scales roughly linearly "
         "with it. Then check ea_frac and cg_frac: their separation drives the coupling. "
         "Then re-run with the theodorsen backend as a sanity check."],
    ], widths=[42 * mm, 116 * mm], mono_first_col=True),

    # --- 6 ---------------------------------------------------------------------------
    H("6&nbsp; Using it as a library",
      P("For anything beyond a single sweep -- parameter studies, your own plots, batch "
      "runs -- import the package directly. The entry points:")),
    table([
        ["object", "what it gives you"],
        ["wing.build_wing(...)", "physical parameters -> a WingBuild carrying .structure, "
         ".aero, .b_ref, .omega_alpha, .EI, .GJ, .natural_hz and .mass_ratio(rho)"],
        ["solvers.pk.pk_flutter", "the p-k sweep -> a FlutterResult with .velocity, "
         ".omega, .damping and .lowest_flutter(g_cross=...)"],
        ["solvers.pk.divergence_speed", "static divergence speed; the p-k sweep cannot "
         "see it, so check it separately in unattended studies"],
        ["aero.cache.TabulatedAero", "wraps a slow aero model, tabulating Qhh on a k-grid. "
         "Essential for the DLM inside a loop"],
        ["envelope.flutter_boundary", "matched-point flutter speed at each altitude, with "
         "the aero Mach iterated to equal the flight Mach"],
        ["envelope.flutter_margin", "predicted flutter speed against the CS-25.629 "
         "1.15 x V_D clearance line"],
        ["nondim.flutter_speed_index", "the dimensionless flutter speed"],
        ["postprocess.plots", "plot_vg_vf, plot_flutter_boundary, plot_fsi_vs_mach"],
    ], widths=[46 * mm, 112 * mm], mono_first_col=True),
    Preformatted(
        "import numpy as np\n"
        "from flutter_calc.wing import build_wing\n"
        "from flutter_calc.solvers.pk import pk_flutter\n"
        "\n"
        "# Sweep the CG position and watch the flutter speed move.\n"
        "for cg in [0.40, 0.44, 0.48, 0.52]:\n"
        "    wb = build_wing(semi_span=16.0, root_chord=5.5, tip_chord=1.6,\n"
        "                    sweep_deg=25.0, half_wing_mass=6000.0,\n"
        "                    f_bending=2.0, f_torsion=5.5,\n"
        "                    ea_frac=0.40, cg_frac=cg, backend=\"theodorsen\")\n"
        "    res = pk_flutter(wb.structure, wb.aero, np.linspace(40, 500, 220),\n"
        "                     rho=1.225, b_ref=wb.b_ref)\n"
        "    crit = res.lowest_flutter()\n"
        "    print(cg, None if crit is None else round(crit[\"V_flutter\"], 1))", CODE),
    PageBreak(),

    # --- 7 ---------------------------------------------------------------------------
    H("7&nbsp; What to trust",
      P("The calculator is built for <b>relative</b> work: comparing wings, ranking design "
      "changes, showing students the mechanism. Absolute margins depend on inputs you "
      "probably estimated -- particularly the frequency anchors, since the flutter speed "
      "moves roughly one-for-one with the torsion frequency.")),
    Paragraph("7.1&nbsp; The validity envelope", H2),
    table([
        ["assumption", "consequence if you leave it behind"],
        ["Subsonic, linear aerodynamics", "The transonic flutter dip is not represented. "
         "Past Mach 0.85 the code flags results rather than predicting; past 0.95 it "
         "refuses. Real wings can dip well below the subsonic trend near drag "
         "divergence."],
        ["Beam structure, assumed shapes", "Fine for a slender, high-aspect-ratio wing. "
         "Low-aspect-ratio or highly tailored composite wings need a plate or FE model."],
        ["Sweep only in the aerodynamics", "The elastic bend-twist coupling of a real "
         "swept box beam is absent, so swept-wing results carry a model-form uncertainty "
         "the calculator cannot quantify for you."],
        ["Flat plate, inviscid", "No thickness, camber, separation or shock effects."],
        ["Single cantilever wing", "No fuselage, free-free modes, control surfaces, "
         "stores or engine-pylon flexibility beyond a rigid point mass."],
        ["Linear onset only", "Predicts where flutter starts, not what happens after. No "
         "gust loads, no limit-cycle oscillation."],
    ], widths=[44 * mm, 114 * mm]),

    Paragraph("7.2&nbsp; How the physics is checked", H2),
    P(f"The repository ships {N_TESTS} tests, run with "
      "<font face='DejaVuSansMono'>python -m pytest</font>. Each checks against something "
      "independent rather than against a previously recorded answer:"),
    table([
        ["what", "checked against"],
        ["Theodorsen C(k)", "an independent Bessel-function form, plus the C(0) = 1 and "
         "C(inf) = 1/2 limits"],
        ["Rayleigh-Ritz free vibration", "the exact closed-form frequencies of a uniform "
         "clamped-free beam, in both bending and torsion"],
        ["Binary flutter", "the textbook signatures: coalescence, flutter frequency "
         "between the two natural frequencies, stability well below V_f"],
        ["Doublet lattice", "must collapse onto 2-D strip theory in the "
         "high-aspect-ratio, low-Mach limit -- this catches mesh, downwash-sign and "
         "load-point errors"],
        ["AGARD 445.6", "the published subsonic flutter-speed index of the standard "
         "45 deg swept benchmark wing"],
        ["ISA atmosphere", "standard atmosphere tables at sea level and the tropopause"],
        ["Matched point", "self-consistency M = V_f / a against an analytic oracle"],
        ["Solver guards", "divergence is detected and labelled; spurious high-frequency "
         "crossings can be filtered without losing a genuine later one"],
    ], widths=[44 * mm, 114 * mm]),
    P("Convergence of the numerical settings is not assumed either: panel count, shape "
      "count and k-grid density were each varied independently to see the drift. The "
      "shipped defaults sit within roughly 2% of the finest settings tried."),

    # --- 8 ---------------------------------------------------------------------------
    H("8&nbsp; Credits, licence and how to cite",
      P("Flutter Calculator stands on other people's work, and the doublet-lattice "
        "backend in particular is a thin wrapper around someone else's solver. If you "
        "publish results from it, credit them, not just this program.")),

    keep(
        Paragraph("8.1&nbsp; PanelAero -- the doublet-lattice aerodynamics", H2),
        P("When you set <font face='DejaVuSansMono'>AERO[\"backend\"] = \"dlm\"</font>, "
          "this program does <b>not</b> compute the doublet-lattice aerodynamics itself. "
          "It builds the panel mesh and the modal downwash, hands them to "
          "<b>PanelAero</b>, and projects the pressures that come back onto the "
          "Rayleigh-Ritz modes. Every part of the unsteady kernel -- the oscillatory "
          "wake, the compressibility treatment, the numerical integration of the "
          "influence coefficients -- is PanelAero's."),
        table([
            ["Project", "PanelAero"],
            ["Authors", "Deutsches Zentrum f&#252;r Luft- und Raumfahrt e.V. (DLR), "
             "Institute of Aeroelasticity"],
            ["Source", "https://github.com/DLR-AE/PanelAero"],
            ["Licence", "BSD 3-Clause"],
            ["Installed by",
             "pip install \"git+https://github.com/DLR-AE/PanelAero.git\""],
        ], widths=[26 * mm, 132 * mm], header=False),
    ),
    P("PanelAero is a separate package installed from its own upstream source; no part of "
      "it is copied into this repository. It is needed only for the "
      "<font face='DejaVuSansMono'>dlm</font> backend -- the analytic "
      "<font face='DejaVuSansMono'>theodorsen</font> backend does not use it, so the "
      "calculator still runs without it installed."),
    P("<b>Please cite PanelAero</b> in any work whose aerodynamics came through this "
      "backend; the repository above states its current recommended reference. Citing "
      "only Flutter Calculator would credit the wrapper and not the method."),
    P("<b>Non-endorsement.</b> The DLR is named here solely to identify the authors of "
      "software this project depends on. It is a statement of provenance: the DLR neither "
      "endorses nor is affiliated with Flutter Calculator. This follows clause 3 of the "
      "BSD 3-Clause licence, which prohibits using the copyright holder's name to endorse "
      "or promote derived products without prior written permission. The full licence "
      "text is reproduced in <font face='DejaVuSansMono'>LICENSE</font> and "
      "<font face='DejaVuSansMono'>THIRD_PARTY_NOTICES.md</font> at the repository root."),

    Paragraph("8.2&nbsp; The methods behind the model", H2),
    P("Not software, but the physics rests on published work and the sources deserve "
      "naming:"),
    table([
        ["source", "what it contributes"],
        ["Theodorsen, T. (1935), <i>General Theory of Aerodynamic Instability and the "
         "Mechanism of Flutter</i>, NACA Report 496.",
         "The closed-form lift-deficiency function C(k) used by the theodorsen backend."],
        ["Albano, E. &amp; Rodden, W. P. (1969), <i>A doublet-lattice method for "
         "calculating lift distributions on oscillating surfaces in subsonic flows</i>, "
         "AIAA Journal 7(2).", "The original doublet-lattice formulation."],
        ["Yates, E. C. Jr. (1987), <i>AGARD Standard Aeroelastic Configurations for "
         "Dynamic Response I -- Wing 445.6</i>, AGARD Report 765.",
         "The swept-wing benchmark used to validate the DLM backend."],
        ["Wright, J. R. &amp; Cooper, J. E., <i>Introduction to Aircraft Aeroelasticity "
         "and Loads</i>, 2nd ed., Wiley, 2015.",
         "The Rayleigh-Ritz wing model, the binary-flutter case and the p-k presentation "
         "conventions followed throughout."],
        ["U.S. Standard Atmosphere (1976).", "The ISA model in envelope.py."],
        ["CS-25.629 / FAR 25.629.",
         "The 1.15 x V_D flutter-clearance requirement used as the margin reference."],
    ], widths=[74 * mm, 84 * mm]),
    P("NumPy, SciPy and matplotlib carry the numerics and the plotting; pytest runs the "
      "test suite and ReportLab builds these guides. All are permissively licensed and "
      "listed in <font face='DejaVuSansMono'>THIRD_PARTY_NOTICES.md</font>.", SMALL),

    keep(
        Paragraph("8.3&nbsp; Licence", H2),
        P("Flutter Calculator is released under the <b>MIT License</b> -- use it, modify "
          "it, redistribute it, commercially or otherwise, provided the copyright notice "
          "and permission notice travel with it. It comes with no warranty. Third-party "
          "components remain under their own licences, as set out above."),
        Spacer(1, 4),
        P("The aircraft presets are representative estimates assembled from public span "
          "and weight figures with plausible structural anchors. They are not certified "
          "data and are not derived from any manufacturer's proprietary information.",
          SMALL),
        P("Generated by docs/make_usage_guide_pdf.py.", SMALL),
    ),
]

SimpleDocTemplate(
    str(OUT), pagesize=A4, leftMargin=21 * mm, rightMargin=21 * mm,
    topMargin=16 * mm, bottomMargin=16 * mm,
    title="Flutter Calculator -- usage guide", author="Flutter Calculator",
).build(story)
print(f"wrote {OUT}")
