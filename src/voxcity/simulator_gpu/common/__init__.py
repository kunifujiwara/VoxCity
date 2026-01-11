"""VoxCity-style `common` namespace.

VoxCity exposes helpers under `voxcity.simulator.common.*`.
`simulator_gpu` implements only the subset needed for drop-in compatibility.
"""

from .geometry import rotate_vector_axis_angle

__all__ = ["rotate_vector_axis_angle"]
