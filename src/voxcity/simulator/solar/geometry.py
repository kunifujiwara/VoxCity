"""
Stage 1: Visibility & geometric analysis.
"""

from numba import njit, prange
import numpy as np
import pandas as pd
from astral import Observer
from astral.sun import elevation, azimuth
from ..view import trace_ray_generic

# Geometry kernel: direct visibility with tree transmittance
@njit(parallel=True)
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
    """
    Return 2D transmittance map (0..1, NaN invalid) for direct beam along sun_direction.
    """

    view_height_voxel = int(view_point_height / meshsize)
    nx, ny, nz = voxel_data.shape
    irradiance_map = np.full((nx, ny), np.nan, dtype=np.float64)

    sd = np.array(sun_direction, dtype=np.float64)
    sd_len = np.sqrt(sd[0] ** 2 + sd[1] ** 2 + sd[2] ** 2)
    if sd_len == 0.0:
        return np.flipud(irradiance_map)
    sd /= sd_len

    for x in prange(nx):
        for y in range(ny):
            found_observer = False
            for z in range(1, nz):
                if voxel_data[x, y, z] in (0, -2) and voxel_data[x, y, z - 1] not in (0, -2):
                    if (voxel_data[x, y, z - 1] in (7, 8, 9)) or (voxel_data[x, y, z - 1] < 0):
                        irradiance_map[x, y] = np.nan
                        found_observer = True
                        break
                    else:
                        observer_location = np.array([x, y, z + view_height_voxel], dtype=np.float64)
                        hit, transmittance = trace_ray_generic(
                            voxel_data,
                            observer_location,
                            sd,
                            hit_values,
                            meshsize,
                            tree_k,
                            tree_lad,
                            inclusion_mode,
                        )
                        irradiance_map[x, y] = transmittance if not hit else 0.0
                        found_observer = True
                        break
            if not found_observer:
                irradiance_map[x, y] = np.nan
    return np.flipud(irradiance_map)


def get_solar_positions_astral(times, lon, lat):
    """
    Compute solar azimuth and elevation using Astral for given times and location.
    """
    observer = Observer(latitude=lat, longitude=lon)
    df_pos = pd.DataFrame(index=times, columns=['azimuth', 'elevation'], dtype=float)
    for t in times:
        el = elevation(observer=observer, dateandtime=t)
        az = azimuth(observer=observer, dateandtime=t)
        df_pos.at[t, 'elevation'] = el
        df_pos.at[t, 'azimuth'] = az
    return df_pos


