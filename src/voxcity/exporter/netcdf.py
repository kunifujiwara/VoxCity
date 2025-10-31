"""
NetCDF export utilities for VoxCity.

This module provides functions to convert a 3D voxel grid produced by
`voxcity.generator.get_voxcity` into a NetCDF file for portable storage
and downstream analysis.

The voxel values follow VoxCity conventions (see generator.create_3d_voxel):
- -3: built structures (buildings)
- -2: vegetation canopy
- -1: subsurface/underground
- >= 1: ground-surface land cover code (offset by +1 from source classes)

Notes
-----
- This writer prefers xarray for NetCDF export. If xarray is not installed,
  a clear error is raised with installation hints.
- Coordinates are stored as index-based distances in meters from the grid
  origin along the y, x, and z axes. Geographic metadata such as the
  `rectangle_vertices` and `meshsize_m` are stored as global attributes to
  avoid making assumptions about map projection or geodesic conversions.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping, MutableMapping, Optional, Sequence, Tuple
import json

import numpy as np

try:  # xarray is the preferred backend for NetCDF writing
    import xarray as xr  # type: ignore
    XR_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    XR_AVAILABLE = False

__all__ = [
    "voxel_to_xarray_dataset",
    "save_voxel_netcdf",
    "NetCDFExporter",
]


def _ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def voxel_to_xarray_dataset(
    voxcity_grid: np.ndarray,
    voxel_size_m: float,
    rectangle_vertices: Optional[Sequence[Tuple[float, float]]] = None,
    extra_attrs: Optional[Mapping[str, Any]] = None,
) -> "xr.Dataset":
    """Create an xarray Dataset from a VoxCity voxel grid.

    Parameters
    ----------
    voxcity_grid
        3D numpy array with shape (rows, cols, levels) as returned by
        `get_voxcity` (first element of the returned tuple).
    voxel_size_m
        Voxel size (mesh size) in meters.
    rectangle_vertices
        Optional polygon vertices defining the area of interest in
        longitude/latitude pairs, typically the same list passed to
        `get_voxcity`.
    extra_attrs
        Optional mapping of additional global attributes to store in the
        dataset.

    Returns
    -------
    xr.Dataset
        Dataset containing one DataArray named "voxels" with dims (y, x, z)
        and coordinate variables in meters from origin.
    """
    if not XR_AVAILABLE:  # pragma: no cover - optional dependency
        raise ImportError(
            "xarray is required to export NetCDF. Install with: \n"
            "  pip install xarray netCDF4\n"
            "or: \n"
            "  pip install xarray h5netcdf"
        )

    if voxcity_grid.ndim != 3:
        raise ValueError(
            f"voxcity_grid must be 3D (rows, cols, levels); got shape={voxcity_grid.shape}"
        )

    rows, cols, levels = voxcity_grid.shape

    # Coordinate vectors in meters relative to the grid origin
    # y increases with row index, x increases with column index
    y_m = np.arange(rows, dtype=float) * float(voxel_size_m)
    x_m = np.arange(cols, dtype=float) * float(voxel_size_m)
    z_m = np.arange(levels, dtype=float) * float(voxel_size_m)

    ds_attrs: MutableMapping[str, Any] = {
        "title": "VoxCity voxel grid",
        "institution": "VoxCity",
        "source": "voxcity.generator.create_3d_voxel",
        "Conventions": "CF-1.10 (partial)",
        # NetCDF attributes must be basic types; serialize complex structures as strings
        "vox_value_meanings": [
            "-3: building",
            "-2: vegetation_canopy",
            "-1: subsurface",
            ">=1: surface_land_cover_code (offset +1)",
        ],
        "meshsize_m": float(voxel_size_m),
        # Store vertices as JSON string for portability
        "rectangle_vertices_lonlat_json": (
            json.dumps([[float(v[0]), float(v[1])] for v in rectangle_vertices])
            if rectangle_vertices is not None else ""
        ),
        "vertical_reference": "z=0 corresponds to min(DEM) as used in voxel construction",
    }
    if extra_attrs:
        ds_attrs.update(dict(extra_attrs))

    da = xr.DataArray(
        voxcity_grid,
        dims=("y", "x", "z"),
        coords={
            "y": ("y", y_m, {"units": "m", "long_name": "row_distance_from_origin"}),
            "x": ("x", x_m, {"units": "m", "long_name": "col_distance_from_origin"}),
            "z": ("z", z_m, {"units": "m", "positive": "up", "long_name": "height_above_vertical_origin"}),
        },
        name="voxels",
        attrs={
            "units": "category",
            "description": "VoxCity voxel values; see global attribute 'vox_value_meanings'",
        },
    )

    ds = xr.Dataset({"voxels": da}, attrs=ds_attrs)
    return ds


def save_voxel_netcdf(
    voxcity_grid: np.ndarray,
    output_path: str | Path,
    voxel_size_m: float,
    rectangle_vertices: Optional[Sequence[Tuple[float, float]]] = None,
    extra_attrs: Optional[Mapping[str, Any]] = None,
    engine: Optional[str] = None,
) -> str:
    """Save a VoxCity voxel grid to a NetCDF file.

    Parameters
    ----------
    voxcity_grid
        3D numpy array (rows, cols, levels) of voxel values.
    output_path
        Path to the NetCDF file to be written. Parent directories will be
        created as needed.
    voxel_size_m
        Voxel size in meters.
    rectangle_vertices
        Optional list of (lon, lat) pairs defining the area of interest.
        Stored as dataset metadata only.
    extra_attrs
        Optional additional global attributes to embed in the dataset.
    engine
        Optional xarray engine, e.g., "netcdf4" or "h5netcdf". If not provided,
        xarray will choose a default; on failure we retry alternate engines.

    Returns
    -------
    str
        The string path to the written NetCDF file.
    """
    if not XR_AVAILABLE:  # pragma: no cover - optional dependency
        raise ImportError(
            "xarray is required to export NetCDF. Install with: \n"
            "  pip install xarray netCDF4\n"
            "or: \n"
            "  pip install xarray h5netcdf"
        )

    path = Path(output_path)
    _ensure_parent_dir(path)

    ds = voxel_to_xarray_dataset(
        voxcity_grid=voxcity_grid,
        voxel_size_m=voxel_size_m,
        rectangle_vertices=rectangle_vertices,
        extra_attrs=extra_attrs,
    )

    # Attempt to save with the requested or default engine; on failure, try a fallback
    tried_engines = []
    try:
        ds.to_netcdf(path, engine=engine)  # type: ignore[call-arg]
    except Exception as e_first:  # pragma: no cover - I/O backend dependent
        tried_engines.append(engine or "default")
        for fallback in ("netcdf4", "h5netcdf"):
            try:
                ds.to_netcdf(path, engine=fallback)  # type: ignore[call-arg]
                break
            except Exception:
                tried_engines.append(fallback)
        else:
            raise RuntimeError(
                f"Failed to write NetCDF using engines: {tried_engines}. "
                f"Original error: {e_first}"
            )

    return str(path)


class NetCDFExporter:
    """Exporter adapter to write a VoxCity voxel grid to NetCDF."""

    def export(self, obj, output_directory: str, base_filename: str, **kwargs):
        from ..models import VoxCity
        path = Path(output_directory) / f"{base_filename}.nc"
        if not isinstance(obj, VoxCity):
            raise TypeError("NetCDFExporter expects a VoxCity instance")
        rect = obj.extras.get("rectangle_vertices")
        # Merge default attrs with user-provided extras
        user_extra = kwargs.get("extra_attrs") or {}
        attrs = {
            "land_cover_source": obj.extras.get("land_cover_source", ""),
            "building_source": obj.extras.get("building_source", ""),
            "dem_source": obj.extras.get("dem_source", ""),
        }
        attrs.update(user_extra)
        return save_voxel_netcdf(
            voxcity_grid=obj.voxels.classes,
            output_path=path,
            voxel_size_m=obj.voxels.meta.meshsize,
            rectangle_vertices=rect,
            extra_attrs=attrs,
            engine=kwargs.get("engine"),
        )

