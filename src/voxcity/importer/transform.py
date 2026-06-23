"""Build the affine that maps OBJ model coordinates to VoxCity voxel indices.

Convention (rotation=0, axis-aligned domain):
    model +X -> +v (east  / array axis 1)
    model +Y -> +u (north / array axis 0)
    model +Z -> +k (up)
Positive `rotation` (degrees) rotates the model counter-clockwise in the
(east, north) plane so that, at rotation=90, model +X points north (+u).

The composed transform (right-to-left, applied to a column vector
``[x, y, z, 1]``)::

    M = Toff @ Sv @ T1 @ R @ S @ T0

  T0  translate model space so ``anchor_model_point`` becomes the origin.
  S   scale model units -> metres (``unit_scale(units)``).
  R   rotate the now-metric (x=east, y=north) vector by ``rotation`` degrees,
      then re-express it in the VoxCity domain's own (u, v) axes -- the domain
      itself may be rotated relative to true north (``domain_rotation``).
  T1  translate to the anchor's position in domain metres (``u_a, v_a``),
      apply the horizontal/vertical ``move`` offset, and shift the vertical
      datum so the DEM minimum sits at z=0 (matching the voxel grid's ground
      datum).
  Sv  scale metres -> voxel indices (divide by ``meshsize``) and add the +1
      vertical offset because the voxelizer seats buildings one voxel above
      the ground voxel (``ground_level = int(dem/voxel_size + 0.5) + 1``).
"""
from __future__ import annotations

import math
import numpy as np

from ..geoprocessor.raster.core import compute_grid_geometry
from ..utils.projector import GridProjector
from .units import unit_scale


def grid_geom_from_voxcity(voxcity):
    """Recover grid geometry from a VoxCity object via its rectangle_vertices."""
    rv = (voxcity.extras or {}).get("rectangle_vertices")
    if not rv:
        raise ValueError(
            "VoxCity object has no extras['rectangle_vertices']; cannot georeference "
            "the import. Regenerate the model with get_voxcity (which stores it) or "
            "set voxcity.extras['rectangle_vertices'] manually."
        )
    meshsize = float(voxcity.voxels.meta.meshsize)
    geom = compute_grid_geometry(rv, meshsize)
    if geom is None:
        raise ValueError("Could not compute grid geometry from rectangle_vertices.")
    return geom


def _domain_rotation_deg(geom) -> float:
    """Bearing (deg, clockwise from north) of the domain +u axis (side_1)."""
    u = np.asarray(geom["u_vec"], dtype=float)  # (dlon, dlat) per metre
    # clockwise-from-north bearing of (east=dlon, north=dlat)
    return math.degrees(math.atan2(u[0], u[1]))


def build_placement_transform(
    voxcity,
    anchor_lonlat,
    anchor_elevation,
    anchor_model_point=(0.0, 0.0, 0.0),
    rotation=0.0,
    move=(0.0, 0.0, 0.0),
    units="m",
):
    """Return a 4x4 affine mapping model coords -> voxel index space (i, j, k)."""
    geom = grid_geom_from_voxcity(voxcity)
    meshsize = float(geom["meshsize_m"])
    scale = unit_scale(units)

    # 1. translate model so anchor_model_point is the origin
    T0 = np.eye(4)
    T0[:3, 3] = -np.asarray(anchor_model_point, dtype=float)

    # 2. scale model units -> meters
    S = np.eye(4)
    S[0, 0] = S[1, 1] = S[2, 2] = scale

    # 3. horizontal rotation: model (x=east, y=north) -> domain (u, v) metres.
    #
    # Step A: rotate the model's (x, y) by `rotation` (theta) counter-clockwise
    # in the (east, north) plane (to_en), giving each model basis vector's
    # (e, n) = (east, north) components.
    #
    # Step B: re-express those (e, n) vectors in the domain's own (u, v) axes.
    # The domain's u-axis (side_1) has bearing `phi` (clockwise from true
    # north), so the domain (u, v) basis vectors expressed in (east, north)
    # are:
    #   u_dir = (sin(phi),  cos(phi))   # (east, north) components of +u
    #   v_dir = (cos(phi), -sin(phi))   # (east, north) components of +v
    # Projecting an (e, n) vector onto those axes (dot product, since u_dir
    # and v_dir are orthonormal) gives its (u, v) coordinates:
    #   u = e*sin(phi) + n*cos(phi)
    #   v = e*cos(phi) - n*sin(phi)
    phi = math.radians(_domain_rotation_deg(geom))
    theta = math.radians(float(rotation))
    cos_t, sin_t = math.cos(theta), math.sin(theta)

    def to_en(x, y):
        return (x * cos_t - y * sin_t, x * sin_t + y * cos_t)

    sin_p, cos_p = math.sin(phi), math.cos(phi)

    # (e_x, n_x) = (east, north) components of the rotated model +X axis;
    # (e_y, n_y) = (east, north) components of the rotated model +Y axis.
    e_x, n_x = to_en(1.0, 0.0)
    e_y, n_y = to_en(0.0, 1.0)
    u_from_x = e_x * sin_p + n_x * cos_p
    v_from_x = e_x * cos_p - n_x * sin_p
    u_from_y = e_y * sin_p + n_y * cos_p
    v_from_y = e_y * cos_p - n_y * sin_p

    R = np.eye(4)
    R[0, 0] = u_from_x
    R[0, 1] = u_from_y
    R[1, 0] = v_from_x
    R[1, 1] = v_from_y
    R[2, 2] = 1.0

    # 4. translate to anchor position in domain metres + move + vertical datum
    proj = GridProjector(geom)
    u_a, v_a = proj.lon_lat_to_uv_m(float(anchor_lonlat[0]), float(anchor_lonlat[1]))
    move_e, move_n, move_up = (float(m) for m in move)
    dem_min = float(np.min(voxcity.dem.elevation))
    z_a = float(anchor_elevation) - dem_min + move_up

    T1 = np.eye(4)
    T1[0, 3] = u_a + move_n   # north -> u axis (i)
    T1[1, 3] = v_a + move_e   # east  -> v axis (j)
    T1[2, 3] = z_a

    # 5. metres -> voxel index, plus +1 ground voxel offset on k
    Sv = np.eye(4)
    Sv[0, 0] = Sv[1, 1] = Sv[2, 2] = 1.0 / meshsize
    Toff = np.eye(4)
    Toff[2, 3] = 1.0  # ground offset (matches voxelizer ground_level = ... + 1)

    # compose: index = Toff @ Sv @ T1 @ R @ S @ T0
    M = Toff @ Sv @ T1 @ R @ S @ T0
    return M
