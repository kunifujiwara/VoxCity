"""
Shared utilities for simulator subpackages.

Currently exposes lightweight 3D geometry helpers used by both
`visibility` and `solar`.
"""

from .geometry import (  # noqa: F401
    _generate_ray_directions_grid,
    _generate_ray_directions_fibonacci,
    rotate_vector_axis_angle,
    _build_face_basis,
)

__all__ = [
    "_generate_ray_directions_grid",
    "_generate_ray_directions_fibonacci",
    "rotate_vector_axis_angle",
    "_build_face_basis",
]


