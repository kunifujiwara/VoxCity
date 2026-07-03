"""GeoTIFF export utilities for VoxCity 2D data layers.

Exports land cover, building height, DEM, and canopy height grids as
conventional north-up, single-band GeoTIFF files in EPSG:4326.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import rasterio
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


def export_grid_geotiff(
    grid,
    rectangle_vertices,
    meshsize,
    output_path,
    *,
    crs="EPSG:4326",
    dtype=None,
    nodata=None,
    color_table=None,
    category_names=None,
):
    """Write a single 2D grid to a georeferenced, north-up GeoTIFF.

    Parameters
    ----------
    grid : 2D array, VoxCity (nx, ny) east/north layout.
    rectangle_vertices, meshsize : AOI geometry used for georeferencing.
    output_path : destination .tif path (parent dirs created as needed).
    crs : output CRS string (default "EPSG:4326").
    dtype : optional output dtype; grid is cast to it before writing.
    nodata : optional nodata value written into the file.
    color_table : optional {int index: (r, g, b)} palette (categorical layers).
    category_names : optional {int index: str} or list[str] of class names.

    Returns
    -------
    str : the written path.
    """
    grid = np.asarray(grid)
    if grid.ndim != 2:
        raise ValueError(f"grid must be 2D; got shape {grid.shape}")

    array, transform = _north_up_affine_and_array(grid, rectangle_vertices, meshsize)
    if dtype is not None:
        array = array.astype(dtype)

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    profile = {
        "driver": "GTiff",
        "height": array.shape[0],
        "width": array.shape[1],
        "count": 1,
        "dtype": array.dtype,
        "crs": crs,
        "transform": transform,
        "compress": "lzw",
        "tiled": True,
    }
    if nodata is not None:
        profile["nodata"] = nodata

    with rasterio.open(path, "w", **profile) as dst:
        dst.write(array, 1)
        if color_table:
            dst.write_colormap(
                1, {int(i): (int(r), int(g), int(b), 255) for i, (r, g, b) in color_table.items()}
            )
        if category_names:
            if isinstance(category_names, dict):
                items = {str(k): str(v) for k, v in category_names.items()}
            else:
                items = {str(i): str(v) for i, v in enumerate(category_names)}
            dst.update_tags(1, **items)

    return str(path)
