"""GeoTIFF export utilities for VoxCity 2D data layers.

Exports land cover, building height, DEM, and canopy height grids as
conventional north-up, single-band GeoTIFF files in EPSG:4326.
"""
from __future__ import annotations

import warnings
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
        Requires an integer `dtype` (e.g. "uint8") -- GDAL colormaps only
        support Byte/UInt16 bands.
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


DEFAULT_LAYERS = ("land_cover", "building_height", "dem", "canopy_height")


def _get_layer_grid(city, layer):
    if layer == "land_cover":
        return city.land_cover.classes
    if layer == "building_height":
        return city.buildings.heights
    if layer == "dem":
        return city.dem.elevation
    if layer == "canopy_height":
        return city.tree_canopy.top
    raise ValueError(f"Unknown layer: {layer!r}")


def _land_cover_color_table(city):
    """Return (color_table, names) built from the active land-cover source,
    or (None, None) if unavailable."""
    source = city.extras.get("land_cover_source")
    if not source:
        warnings.warn("No 'land_cover_source' in extras; "
                      "land cover written without a color table")
        return None, None
    try:
        from ..utils.lc import get_land_cover_classes
        classes = get_land_cover_classes(source)
    except Exception:
        warnings.warn(f"Could not load land cover classes for source {source!r}; "
                      "writing land cover without a color table")
        return None, None
    color_table, names = {}, {}
    for idx, ((r, g, b), name) in enumerate(classes.items()):
        color_table[idx] = (r, g, b)
        names[idx] = name
    return color_table, names


def export_geotiffs(city, output_directory, base_filename="voxcity", *,
                    layers=DEFAULT_LAYERS, **kwargs):
    """Export a VoxCity object's 2D layers as one GeoTIFF per layer.

    Writes ``{base_filename}_{layer}.tif`` for each requested layer. Land cover
    is written as uint8 with an embedded color table + class-name tags; building
    height, DEM, and canopy height are float32 with NaN nodata (0 stays valid
    data). Missing layers are skipped with a warning.

    Returns a dict mapping written layer name -> path.
    """
    from ..models import VoxCity
    if not isinstance(city, VoxCity):
        raise TypeError("export_geotiffs expects a VoxCity instance")

    rect = city.extras.get("rectangle_vertices")
    if rect is None:
        raise ValueError(
            "city.extras['rectangle_vertices'] is required for GeoTIFF export"
        )
    meshsize = city.land_cover.meta.meshsize

    written = {}
    for layer in layers:
        grid = _get_layer_grid(city, layer)
        if grid is None:
            warnings.warn(f"Layer {layer!r} is missing; skipping")
            continue
        out_path = Path(output_directory) / f"{base_filename}_{layer}.tif"
        if layer == "land_cover":
            color_table, names = _land_cover_color_table(city)
            export_grid_geotiff(
                np.asarray(grid).astype("uint8"), rect, meshsize, out_path,
                dtype="uint8", color_table=color_table, category_names=names,
            )
        else:
            export_grid_geotiff(
                np.asarray(grid, dtype="float32"), rect, meshsize, out_path,
                dtype="float32", nodata=float("nan"),
            )
        written[layer] = str(out_path)
    return written
