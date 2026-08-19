"""Doublet-Lattice aerodynamics backend (swept, subsonic-compressible).

This is the production aero model. It wraps the DLM of **PanelAero** to provide the same
``Qhh(k, Ma)`` contract as the Theodorsen backend, so it drops straight into the p-k
solver. The cantilever wing sits on the fuselage symmetry plane, so we use PanelAero's
``xz_symmetry`` to include the mirror-image wing.

Credit where it is due: none of the unsteady doublet-lattice kernel mathematics is
implemented here. PanelAero (Deutsches Zentrum fuer Luft- und Raumfahrt e.V. -- DLR,
Institute of Aeroelasticity; BSD 3-Clause; https://github.com/DLR-AE/PanelAero) computes
the aerodynamic influence coefficients ``Qjj``. This module supplies the panel mesh and
the modal downwash, and projects the returned pressures onto the Ritz modes. See
THIRD_PARTY_NOTICES.md at the repository root for the full licence text.

Coupling (direct evaluation, no spline)
---------------------------------------
The wing displaces with beam kinematics about the SWEPT elastic axis: modal displacement
is constant along cross-sections perpendicular to the EA, so a panel point ``(x, y)``
takes the modal value of the EA station whose perpendicular cross-section passes through
it,

    eta* = [ y + (x - x_ea(y)) sin(L_ea) cos(L_ea) ] / s

with ``L_ea`` the sweep of the elastic-axis line itself (equal to the planform sweep for
a constant chord; slightly less for a tapered swept wing). Evaluating the analytic shapes
at that station gives

    bending i:  z = phi_i(eta*),                dz/dx = sinL cosL phi_i'(eta*) / s
    torsion m:  z = -(x - x_ea(y)) psi_m(eta*), dz/dx = -psi_m(eta*)
                                                        - (x - x_ea) sinL cosL psi_m'(eta*)/s

The bending slope is the swept-wing WASHOUT: on a swept-back wing the trailing edge of a
streamwise section lies structurally outboard of its leading edge, so up-bending twists
the section nose-down (load alleviation). Every sweep term vanishes at zero EA sweep, so
unswept results are identical to the plain strip coupling; the torsion DOF remains the
streamwise twist (the structural definition), so its leading arm and slope match the
unswept form exactly. Structural sweep coupling (bend-twist stiffness) is still NOT
modelled -- the structure stays a streamwise beam with calibrated EI/GJ.

The DLM normalwash (downwash) boundary condition, calibrated
against steady VLM and consistent with conventions.py, is

    w = -( dz/dx + i (omega/U) z ) = -( dz/dx + i k_pa z ),   k_pa = omega/U = k / b_ref

(PanelAero's ``k`` is the dimensional ``omega/U``, *not* ``omega b/U``.) PanelAero returns
``Qjj`` mapping ``w -> cp``; the up-force per panel is ``q_dyn cp A``. The generalized
matrix is therefore

    Qhh = Z^T diag(A) Qjj Dw ,    F_aero = q_dyn Qhh q

where ``Z`` holds the modal displacements, ``Dw`` the modal downwash, and ``A`` the panel
areas. ``Z`` and ``dz/dx`` are geometry-only (k-independent) and precomputed once.
"""

from __future__ import annotations

import logging

import numpy as np
from panelaero import DLM

from ..geometry import WingGeometry
from .mesh import build_aerogrid


class _SuppressFlippedPanels(logging.Filter):
    """Drop PanelAero's 'flipped panels' warning.

    Under ``xz_symmetry`` PanelAero mirrors the wing and negates the mirrored panels'
    z-normals, which trips its own ``N_z < 0`` check every call. The symmetric result
    is correct (validated against steady VLM lift slope), so this warning is noise.
    """

    def filter(self, record):
        return "upside down" not in record.getMessage()


if not any(isinstance(f, _SuppressFlippedPanels) for f in logging.getLogger().filters):
    logging.getLogger().addFilter(_SuppressFlippedPanels())


class DLMAero:
    """Doublet-Lattice ``AeroModel`` for a cantilever wing with assumed shapes."""

    def __init__(self, geometry: WingGeometry, bending_shapes, torsion_shapes,
                 n_span: int = 16, n_chord: int = 6, b_ref: float | None = None,
                 xz_symmetry: bool = True):
        self.geometry = geometry
        self.bending_shapes = bending_shapes
        self.torsion_shapes = torsion_shapes
        self.nb = len(bending_shapes)
        self.nt = len(torsion_shapes)
        self.n_dof = self.nb + self.nt
        self.xz_symmetry = xz_symmetry

        self.aerogrid = build_aerogrid(geometry, n_span=n_span, n_chord=n_chord)
        self.A = self.aerogrid["A"]
        # Two chordwise stations per box matter and they are NOT the same point:
        #   - boundary condition (downwash) is enforced at the 3/4-chord collocation
        #     point (offset_j);
        #   - the box lift acts at the 1/4-chord load reference / bound vortex
        #     (offset_l), so the virtual-work force projection must use that point.
        # The torsion moment arm -(x - x_ea) and (when swept) the EA station eta* both
        # depend on x, so getting this wrong corrupts the pitching-moment generalized
        # forces.
        x_colloc = self.aerogrid["offset_j"][:, 0]        # 3/4 box chord (downwash)
        x_load = self.aerogrid["offset_l"][:, 0]          # 1/4 box chord (lift acts)
        yj = self.aerogrid["y_collocation"]
        s = geometry.semi_span

        props = geometry.section_properties(yj)
        x_le = yj * np.tan(np.deg2rad(geometry.sweep_deg))
        x_ea = x_le + props["ea_frac"] * props["chord"]

        self.b_ref = float(props["b"][0]) if b_ref is None else float(b_ref)

        npan = self.aerogrid["n"]
        Z_proj = np.zeros((npan, self.n_dof))             # modal displ at the load point
        Z_down = np.zeros((npan, self.n_dof))             # modal displ at collocation
        dZdx = np.zeros((npan, self.n_dof))
        # Swept elastic-axis kinematics: every shape is evaluated at the EA station eta*
        # whose perpendicular cross-section passes through the panel point (module
        # docstring). eta* = eta at zero EA sweep, so unswept results are unchanged. The
        # bending washout slope (swept-back = load-alleviating: bends up, twists nose-
        # down) and the torsion station shift both fall out of the same substitution --
        # no tuned factor. L_ea is the sweep of the EA line itself, from its endpoints
        # (exact for a linear chord/ea_frac distribution). Near the tip trailing edge
        # eta* can slightly exceed 1; the polynomial shapes extend smoothly.
        p_rt = geometry.section_properties(np.array([0.0, s]))
        x_ea_rt = (np.array([0.0, s]) * np.tan(np.deg2rad(geometry.sweep_deg))
                   + p_rt["ea_frac"] * p_rt["chord"])
        tan_ea = (x_ea_rt[1] - x_ea_rt[0]) / s
        sc = tan_ea / (1.0 + tan_ea ** 2)                 # sin(L_ea) cos(L_ea)
        eta_load = (yj + (x_load - x_ea) * sc) / s        # EA station feeding each point
        eta_coll = (yj + (x_colloc - x_ea) * sc) / s
        for i, sh in enumerate(bending_shapes):
            Z_proj[:, i] = sh(eta_load)                   # heave of the swept section
            Z_down[:, i] = sh(eta_coll)
            dZdx[:, i] = sc * sh.d1(eta_coll) / s         # bending washout slope
        for mi, sh in enumerate(torsion_shapes):
            col = self.nb + mi                            # twist about the EA
            Z_proj[:, col] = -(x_load - x_ea) * sh(eta_load)
            Z_down[:, col] = -(x_colloc - x_ea) * sh(eta_coll)
            dZdx[:, col] = -sh(eta_coll) - (x_colloc - x_ea) * sc * sh.d1(eta_coll) / s
        self.Z_proj = Z_proj
        self.Z_down = Z_down
        self.dZdx = dZdx

    def Qhh(self, k: float, Ma: float = 0.0) -> np.ndarray:
        """Nondimensional generalized aero matrix at reduced frequency ``k``, Mach ``Ma``."""
        k = float(k)
        k_pa = k / self.b_ref                            # PanelAero uses omega/U
        Qjj = DLM.calc_Qjjs(self.aerogrid, [Ma], [k_pa],
                            xz_symmetry=self.xz_symmetry)[0, 0]
        Dw = -(self.dZdx + 1j * k_pa * self.Z_down)       # downwash at collocation point
        forces = self.A[:, None] * (Qjj @ Dw)            # per-panel up-force / q_dyn
        return self.Z_proj.T @ forces                     # project at the load point
