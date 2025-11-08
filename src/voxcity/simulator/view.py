"""Compatibility wrapper for the legacy view module.

The implementation has been split into the `visibility` package:
  - voxcity.simulator.visibility.raytracing
  - voxcity.simulator.visibility.geometry
  - voxcity.simulator.visibility.view
  - voxcity.simulator.visibility.landmark

Import the new API from `voxcity.simulator.visibility`.
This module re-exports the main public functions for backward compatibility.
"""

from .visibility.view import (
    get_view_index,
    get_sky_view_factor_map,
    get_surface_view_factor,
)
from .visibility.landmark import (
    mark_building_by_id,
    compute_landmark_visibility,
    get_landmark_visibility_map,
    get_surface_landmark_visibility,
)
from .common.geometry import rotate_vector_axis_angle

__all__ = [
    "get_view_index",
    "get_sky_view_factor_map",
    "get_surface_view_factor",
    "mark_building_by_id",
    "compute_landmark_visibility",
    "get_landmark_visibility_map",
    "get_surface_landmark_visibility",
    "rotate_vector_axis_angle",
]

