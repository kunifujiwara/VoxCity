"""Geometry helpers for VoxCity compatibility.

This module is intended to satisfy imports like:
    from simulator.common.geometry import rotate_vector_axis_angle

It forwards to the implementation used by `simulator_gpu.visibility`.
"""

from ..visibility.geometry import rotate_vector_axis_angle

__all__ = ["rotate_vector_axis_angle"]
