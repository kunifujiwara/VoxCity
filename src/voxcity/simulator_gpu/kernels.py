"""VoxCity-style `kernels` module (toplevel) for compatibility.

Implements a minimal subset of `voxcity.simulator.solar.kernels` so code that
imports these symbols continues to run when using:

    import simulator_gpu as simulator
"""

from __future__ import annotations

from typing import Tuple

import numpy as np


def compute_direct_solar_irradiance_map_binary(
    voxel_data,
    sun_direction,
    view_point_height,
    hit_values,
    meshsize,
    tree_k,
    tree_lad,
    inclusion_mode,
):
    """Approximate VoxCity kernel: return direct-beam transmittance map (0..1).

    Signature matches `voxcity.simulator.solar.kernels.compute_direct_solar_irradiance_map_binary`.

    Notes:
        - `hit_values` and `inclusion_mode` are accepted for compatibility but
          are not currently used in the GPU path.
        - Output is flipped with `np.flipud`, matching VoxCity.
    """
    from .solar.integration import _compute_direct_transmittance_map_gpu

    sd = np.array(sun_direction, dtype=np.float64)
    L = float(np.sqrt((sd * sd).sum()))
    if L == 0.0:
        nx, ny = voxel_data.shape[0], voxel_data.shape[1]
        return np.flipud(np.full((nx, ny), np.nan, dtype=np.float64))
    sd /= L

    trans = _compute_direct_transmittance_map_gpu(
        voxel_data=np.asarray(voxel_data),
        sun_direction=(float(sd[0]), float(sd[1]), float(sd[2])),
        view_point_height=float(view_point_height),
        meshsize=float(meshsize),
        tree_k=float(tree_k),
        tree_lad=float(tree_lad),
    )

    return np.flipud(trans)


__all__ = ["compute_direct_solar_irradiance_map_binary"]
