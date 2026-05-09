"""Lon/lat polygon utilities for VoxCity grids.

Bridges geographic polygons (closed lon/lat rings, GeoJSON-style) into
the grid's (i, j) cell frame. Algorithm: project every cell centre to
lon/lat via GridProjector, then test point-in-polygon with MplPath.
"""

from typing import List, Sequence, Tuple

import numpy as np
from matplotlib.path import Path as MplPath

from .projector import GridGeom, GridProjector


def mask_from_lonlat_ring(
    ring: Sequence[Sequence[float]],
    grid_geom: GridGeom,
) -> np.ndarray:
    """Rasterize a closed lon/lat ring to a (nx, ny) boolean mask.

    A cell is True iff its centre lies inside the polygon.
    """
    nx, ny = grid_geom["grid_size"]
    if len(ring) < 3:
        return np.zeros((nx, ny), dtype=bool)
    proj = GridProjector(grid_geom)
    ii, jj = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
    lons, lats = proj.cell_to_lon_lat(ii.ravel(), jj.ravel())
    centres = np.stack([np.asarray(lons), np.asarray(lats)], axis=1)
    inside = MplPath(np.asarray(ring, dtype=float)).contains_points(centres)
    return inside.reshape(nx, ny)


def polygon_lonlat_to_cells(
    ring: Sequence[Sequence[float]],
    grid_geom: GridGeom,
) -> List[Tuple[int, int]]:
    """Rasterize a lon/lat ring to a list of (i, j) cell indices.

    Equivalent to ``np.argwhere(mask_from_lonlat_ring(...))`` returned as
    a list of Python tuples. Provided for parity with the helper that has
    historically lived in app/backend/zoning.py; new code should prefer
    ``mask_from_lonlat_ring``.
    """
    mask = mask_from_lonlat_ring(ring, grid_geom)
    if not mask.any():
        return []
    cells = np.argwhere(mask)
    return [(int(i), int(j)) for i, j in cells]


def points_in_polygon_lonlat(
    points_lonlat: np.ndarray,                # (N, 2)
    ring: Sequence[Sequence[float]],
) -> np.ndarray:
    """Vectorized point-in-polygon test (boolean array of length N)."""
    if len(ring) < 3 or points_lonlat.size == 0:
        return np.zeros(len(points_lonlat), dtype=bool)
    return MplPath(np.asarray(ring, dtype=float)).contains_points(points_lonlat)
