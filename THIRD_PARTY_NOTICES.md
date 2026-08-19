# Third-party notices

Flutter Calculator itself is released under the MIT License (see [LICENSE](LICENSE)).
It depends on the third-party software listed below. None of it is vendored into this
repository — each package is installed from its own upstream source — but the licences
and attributions are reproduced here so the provenance of the method is clear.

The PanelAero attribution and its BSD 3-Clause text also appear in
[LICENSE](LICENSE) and in section 8 of the usage guide; this file is the fullest
account and adds the packages and published methods behind the rest of the model.

---

## PanelAero — doublet-lattice aerodynamics

The `dlm` backend (`src/flutter_calc/aero/dlm.py`) does not implement the doublet-lattice
method itself. It builds a panel mesh and a modal downwash, calls **PanelAero** to obtain
the aerodynamic influence coefficients `Qjj`, and projects the resulting pressures onto
the Ritz modes. All of the unsteady-kernel mathematics is PanelAero's.

- **Project:** PanelAero
- **Authors:** Deutsches Zentrum für Luft- und Raumfahrt e.V. (DLR),
  Institute of Aeroelasticity
- **Source:** https://github.com/DLR-AE/PanelAero
- **Licence:** BSD 3-Clause

Please cite PanelAero's own recommended reference in any work whose aerodynamics come
from it; the repository above states the current citation.

*The DLR is credited here as the author of software this project depends on. This is a
statement of provenance only — the DLR neither endorses nor is affiliated with Flutter
Calculator (BSD 3-Clause, clause 3).*

```
BSD 3-Clause License

Copyright (c) 2020-2022, Deutsches Zentrum für Luft- und Raumfahrt e.V.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```

---

## Core scientific stack

Required at runtime, all permissively licensed and installed from PyPI:

| Package | Licence | Used for |
|---|---|---|
| NumPy | BSD 3-Clause | arrays and linear algebra throughout |
| SciPy | BSD 3-Clause | generalized eigensolvers, Hankel functions, Gauss–Legendre quadrature |
| matplotlib | matplotlib licence (BSD-style, PSF-derived) | V-g / V-f and envelope plots, documentation figures |

Optional, development and documentation only:

| Package | Licence | Used for |
|---|---|---|
| pytest | MIT | the validation test suite |
| ReportLab | BSD 3-Clause | building the usage-guide PDF in `docs/` |

---

## Methods and published data

Not software, but the model rests on published work and the sources deserve naming:

- **Theodorsen, T. (1935),** *General Theory of Aerodynamic Instability and the Mechanism
  of Flutter*, NACA Report 496 — the closed-form `C(k)` used by the `theodorsen` backend.
- **Albano, E. & Rodden, W. P. (1969),** *A doublet-lattice method for calculating lift
  distributions on oscillating surfaces in subsonic flows*, AIAA Journal 7(2) — the
  original DLM formulation.
- **Yates, E. C. Jr. (1987),** *AGARD Standard Aeroelastic Configurations for Dynamic
  Response I — Wing 445.6*, AGARD Report 765 — the swept-wing benchmark used in
  `cases/agard445.py` and the corresponding test.
- **U.S. Standard Atmosphere (1976)** — the ISA model in `envelope.py`.
- **CS-25.629 / FAR 25.629** — the `1.15 · V_D` flutter-clearance requirement referenced
  by `envelope.clearance_speed` and `envelope.flutter_margin`.

The aircraft presets in `flutter_sweep.py` are **representative estimates** assembled from
public span/weight figures and plausible structural anchors. They are not certified data
and are not derived from any manufacturer's proprietary information.
