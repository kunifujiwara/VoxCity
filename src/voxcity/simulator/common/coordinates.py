"""Coordinate helpers shared by simulator backends.

VoxCity simulation arrays use uv-domain coordinates: x/u = north,
y/v = east, z = up. Visualization meshes use scene coordinates:
x = v/east, y = u/north, z = up.
"""

from __future__ import annotations

import numpy as np


def uv_domain_points_to_scene(points: np.ndarray) -> np.ndarray:
    """Map uv-domain points/vectors (u, v, z) to scene (x=v, y=u, z)."""
    result = np.asarray(points).copy()
    result[..., [0, 1]] = result[..., [1, 0]]
    return result


def scene_points_to_uv_domain(points: np.ndarray) -> np.ndarray:
    """Map scene points (x=v, y=u, z) to uv-domain points (u, v, z)."""
    return uv_domain_points_to_scene(points)


def scene_vectors_to_uv_domain(vectors: np.ndarray) -> np.ndarray:
    """Map scene vectors (v, u, z) to uv-domain vectors (u, v, z)."""
    return uv_domain_points_to_scene(vectors)
