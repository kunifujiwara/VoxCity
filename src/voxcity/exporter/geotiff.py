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

from ..geoprocessor.raster.core import compute_grid_geometry, compute_cell_center_coords

__all__ = [
    "export_grid_geotiff",
    "export_geotiffs",
    "GeoTIFFExporter",
]


def _north_up_affine_and_array(grid, rectangle_vertices, meshsize):
    """Convert a VoxCity 2D grid into a raster array + rasterio ``Affine``.

    The grid is indexed ``grid[i, j]`` with ``i`` along ``u_vec`` (side_1) and
    ``j`` along ``v_vec`` (side_2). The cell centre of ``grid[i, j]`` is at
    ``(cc.lons[i, j], cc.lats[i, j])`` — this holds for *any* rectangle-vertex
    order, so we orient the output from those cell centres rather than assuming
    which axis is east/north.

    For an axis-aligned AOI (the common case, incl. all real generated models,
    whose canonical order ``[SW, NW, NE, SE]`` makes ``u_vec=north, v_vec=east``)
    the result is a genuinely **north-up** raster: ``array[0]`` = north edge,
    ``array[:, 0]`` = west edge, with a diagonal affine (``b == d == 0``).

    For a rotated AOI a north-up raster is not representable without resampling,
    so we emit a rotated — but still georeferenced-correct — affine (an
    affine-aware GIS tool places every pixel correctly).

    Returns ``(array, transform)``.
    """
    geom = compute_grid_geometry(rectangle_vertices, meshsize)
    if geom is None:
        raise ValueError(
            "Could not compute grid geometry; need at least 4 rectangle_vertices"
        )
    nx, ny = geom["grid_size"]

    grid = np.asarray(grid)
    if grid.shape != (nx, ny):
        raise ValueError(
            f"grid shape {grid.shape} does not match expected (nx, ny) = {(nx, ny)}"
        )

    u_vec = np.asarray(geom["u_vec"], dtype=float)
    v_vec = np.asarray(geom["v_vec"], dtype=float)

    # An axis-aligned AOI has each side vector along a pure lon or lat axis.
    axis_aligned = (
        (abs(u_vec[0]) < 1e-12 or abs(u_vec[1]) < 1e-12)
        and (abs(v_vec[0]) < 1e-12 or abs(v_vec[1]) < 1e-12)
    )

    if axis_aligned:
        cc = compute_cell_center_coords(rectangle_vertices, meshsize)
        lons = np.asarray(cc["lons"], dtype=float)  # shape (nx, ny), indexed like grid
        lats = np.asarray(cc["lats"], dtype=float)

        uniq_lon = np.unique(np.round(lons, 9))
        uniq_lat = np.unique(np.round(lats, 9))
        n_cols, n_rows = uniq_lon.size, uniq_lat.size

        if n_cols * n_rows == grid.size:  # exact axis-aligned tiling
            dlon = (
                (uniq_lon[-1] - uniq_lon[0]) / (n_cols - 1)
                if n_cols > 1
                else _axis_spacing_deg(u_vec, v_vec, geom["adj_mesh"], lon=True)
            )
            dlat = (
                (uniq_lat[-1] - uniq_lat[0]) / (n_rows - 1)
                if n_rows > 1
                else _axis_spacing_deg(u_vec, v_vec, geom["adj_mesh"], lon=False)
            )
            min_lon = float(uniq_lon[0])
            max_lat = float(uniq_lat[-1])

            out = np.empty((n_rows, n_cols), dtype=grid.dtype)
            col_idx = np.rint((lons - min_lon) / dlon).astype(int)
            row_idx = np.rint((max_lat - lats) / dlat).astype(int)
            out[row_idx, col_idx] = grid  # bijective for an axis-aligned grid

            transform = Affine(
                dlon, 0.0, min_lon - dlon / 2.0,
                0.0, -dlat, max_lat + dlat / 2.0,
            )
            return np.ascontiguousarray(out), transform

    # Rotated AOI (or a degenerate 1-cell tiling that isn't axis-aligned):
    # rotated but georeferenced-correct affine, matching the original behaviour.
    origin = np.asarray(geom["origin"], dtype=float)
    dx, dy = geom["adj_mesh"]
    nw = origin + ny * dy * v_vec
    transform = Affine(
        dx * u_vec[0], -dy * v_vec[0], float(nw[0]),
        dx * u_vec[1], -dy * v_vec[1], float(nw[1]),
    )
    array = np.ascontiguousarray(np.flipud(grid.T))
    return array, transform


def _axis_spacing_deg(u_vec, v_vec, adj_mesh, *, lon):
    """Degrees-per-cell along the lon (``lon=True``) or lat axis for an
    axis-aligned AOI. One of u_vec/v_vec is along lon, the other along lat;
    pick the non-zero contribution. Used only for degenerate 1-wide grids."""
    dx, dy = adj_mesh
    comp = 0 if lon else 1
    return max(abs(dx * u_vec[comp]), abs(dy * v_vec[comp]))


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
    grid : 2D array, VoxCity ``(nx, ny)`` grid indexed along the AOI's
        side_1/side_2 axes (either order supported for axis-aligned AOIs).
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
        # Band-level colormap / color-interpretation tags must be set before
        # the first write() call -- setting them after triggers a libtiff
        # "Cannot modify tag ... while writing" ERROR-severity GDAL log line.
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
        dst.write(array, 1)

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
            grid_arr = np.asarray(grid)
            if np.any(grid_arr < 0) or np.any(grid_arr > 255):
                warnings.warn(
                    "land_cover grid contains values outside the uint8 range "
                    "[0, 255] (e.g. -1 for unmatched classes); these will "
                    "wrap around silently when cast to uint8"
                )
            color_table, names = _land_cover_color_table(city)
            export_grid_geotiff(
                grid_arr.astype("uint8"), rect, meshsize, out_path,
                dtype="uint8", color_table=color_table, category_names=names,
            )
        else:
            export_grid_geotiff(
                np.asarray(grid, dtype="float32"), rect, meshsize, out_path,
                dtype="float32", nodata=float("nan"),
            )
        written[layer] = str(out_path)
    return written


class GeoTIFFExporter:
    """Exporter adapter to write a VoxCity object's 2D layers to GeoTIFF files."""

    def export(self, obj, output_directory, base_filename, **kwargs):
        from ..models import VoxCity
        if not isinstance(obj, VoxCity):
            raise TypeError("GeoTIFFExporter expects a VoxCity instance")
        return export_geotiffs(obj, output_directory, base_filename, **kwargs)
