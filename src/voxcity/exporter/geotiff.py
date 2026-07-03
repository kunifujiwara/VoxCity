"""GeoTIFF export utilities for VoxCity 2D data layers.

Exports land cover, building height, DEM, and canopy height grids as
conventional north-up, single-band GeoTIFF files in EPSG:4326.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
from affine import Affine

from ..geoprocessor.raster.core import compute_grid_geometry

__all__ = [
    "export_grid_geotiff",
    "export_geotiffs",
    "GeoTIFFExporter",
]


def _north_up_affine_and_array(grid, rectangle_vertices, meshsize):
    """Convert a VoxCity (nx, ny) east/north grid into a north-up raster array
    and its rasterio Affine.

    VoxCity grids are indexed grid[i, j] with i along u_vec (east) and j along
    v_vec (north). A conventional north-up GeoTIFF wants shape (rows=ny, cols=nx)
    with row 0 = north. Returns (array, transform).
    """
    geom = compute_grid_geometry(rectangle_vertices, meshsize)
    if geom is None:
        raise ValueError(
            "Could not compute grid geometry; need at least 4 rectangle_vertices"
        )
    origin = np.asarray(geom["origin"], dtype=float)
    u_vec = np.asarray(geom["u_vec"], dtype=float)
    v_vec = np.asarray(geom["v_vec"], dtype=float)
    nx, ny = geom["grid_size"]
    dx, dy = geom["adj_mesh"]

    grid = np.asarray(grid)
    if grid.shape != (nx, ny):
        raise ValueError(
            f"grid shape {grid.shape} does not match expected (nx, ny) = {(nx, ny)}"
        )

    nw = origin + ny * dy * v_vec  # NW corner = top-left in north-up
    transform = Affine(
        dx * u_vec[0], -dy * v_vec[0], float(nw[0]),
        dx * u_vec[1], -dy * v_vec[1], float(nw[1]),
    )
    array = np.ascontiguousarray(np.flipud(grid.T))
    return array, transform
