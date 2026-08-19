"""Flutter Calculator: Rayleigh-Ritz + doublet-lattice flutter prediction."""

from __future__ import annotations

from . import conventions, nondim
from .geometry import PointMass, WingGeometry
from .structures.ritz import StructuralModel, assemble
from .structures.shapes import (
    Shape,
    polynomial_bending_shapes,
    polynomial_torsion_shapes,
)

__version__ = "0.1.0"

__all__ = [
    "conventions",
    "nondim",
    "WingGeometry",
    "PointMass",
    "StructuralModel",
    "assemble",
    "Shape",
    "polynomial_bending_shapes",
    "polynomial_torsion_shapes",
]
