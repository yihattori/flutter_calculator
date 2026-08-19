"""Flutter-point extraction and summary helpers."""

from __future__ import annotations

from ..nondim import flutter_speed_index


def summarize(result, omega_alpha=None, mass_ratio=None):
    """Human-readable summary of the critical flutter point of a FlutterResult.

    If ``omega_alpha`` (uncoupled torsion frequency) and ``mass_ratio`` are supplied,
    the nondimensional flutter-speed index is included.
    """
    crit = result.lowest_flutter()
    if crit is None:
        return {"flutter": False}
    out = {
        "flutter": True,
        "V_flutter": crit["V_flutter"],
        "omega_flutter": crit["omega_flutter"],
        "freq_flutter_hz": crit["omega_flutter"] / (2 * 3.141592653589793),
        "branch": crit["branch"],
    }
    if omega_alpha is not None and mass_ratio is not None:
        out["flutter_index"] = flutter_speed_index(
            crit["V_flutter"], result.b_ref, omega_alpha, mass_ratio
        )
    return out
