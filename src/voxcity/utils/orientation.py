"""Grid orientation helpers.

Contract:
- Core Phase 3 processing uses uv_m/SOUTH_UP: axis 0 = u/north (row 0 = southern origin edge),
  increasing row index moves north. Columns increase eastward: column 0 is
  west/left and indices increase toward the east/right. All processing functions
  accept and return 2D grids in this orientation unless explicitly documented
  otherwise.
- Visualization utilities may flip vertically for display purposes only.
- 3D voxel arrays index (i, j, k) with the same horizontal orientation as the
  2D grids (row 0 = southern origin edge, i increases northward; j increases
  eastward) and k = ground→up. The voxelizer applies no flip between 2D
  grids and voxel layers.

Utilities here are intentionally minimal to avoid introducing hidden behavior.
They can be used at I/O boundaries (e.g., when reading rasters with south_up
conventions) to normalize to the internal orientation.

Coordinate-frame vocabulary (see also voxcity.utils.projector):
  uv cell — cell index (i, j) in uv_m/SOUTH_UP (Phase 3); row 0 is south/origin.

File format (voxcity_results.v3, written by voxcity.io):
  Every saved HDF5 file declares this contract: attribute
  ``axes = "north,east,up"`` (root, `voxcity` group, and
  `voxcity/voxel_grid` dataset), attribute ``rotation_angle`` (degrees
  clockwise; the axes tokens apply in the frame rotated by this angle), and
  root dataset ``rectangle_vertices`` (4x2 lon/lat, [SW, NW, NE, SE]).
  ``axes`` and ``rotation_angle`` are a documented pair: axis 0 points
  north *rotated clockwise by rotation_angle*. Use check_axes() to assert
  the contract; pre-v3 files are converted with voxcity.io.migrate_h5().

Boundary layouts handled by helpers in this module:
  north_up raster — row 0 = north (GeoTIFF, display, coastline masks).
      uv <-> north_up is a pure vertical flip: ensure_orientation().
  rasterio layout — shape (ny, nx): rows along v, cols along u, as produced/
      consumed by rasterio.features.rasterize with the uv affine.
      uv <-> rasterio is a pure transpose: to/from_rasterio_layout().
  rotated raster — legacy layout for rotated-AOI GeoTIFFs:
      grid_to_rotated_raster().
  MagicaVoxel dense axes — voxels_to_magicavoxel_axes().
  OBJ mesher (k, i, j) axes — voxels_to_kji().

Intentional exception: the ENVI-met exporter keeps SOUTH_UP internally and
writes north-first rows itself (arr[::-1]) at the file-format boundary. It
deliberately does NOT call ensure_orientation(); a guard test
(tests/test_exporter_envimet.py::TestEnvimetSouthUpProcessing) enforces this.
"""

from __future__ import annotations

import os
from typing import Literal
import numpy as np

# Public constants to reference orientation in docs and code
ORIENTATION_NORTH_UP: Literal["north_up"] = "north_up"
ORIENTATION_SOUTH_UP: Literal["south_up"] = "south_up"

# --- Axis contract ---------------------------------------------------------

AXES = ("north", "east", "up")
"""Geographic direction of increasing index for array axes 0, 1, 2, in the
un-rotated frame. Axis 0 points north rotated clockwise by the AOI's
``rotation_angle`` degrees (axis 1 likewise for east; axis 2 is up); for
axis-aligned AOIs (``rotation_angle == 0``) the tokens are literally true."""

AXES_ATTR = ",".join(AXES)
"""The exact string stored as the ``axes`` attribute in v3 HDF5 files
(root, ``voxcity`` group, and ``voxcity/voxel_grid`` dataset)."""


def direction_to_axis_vector(azimuth_deg, elevation_deg=0.0, rotation_angle_deg=0.0):
    """Unit direction ``(d_axis0, d_axis1, d_axis2)`` for a compass azimuth.

    Parameters
    ----------
    azimuth_deg : float or np.ndarray
        Compass azimuth in degrees, clockwise from north — a **toward**
        direction (where the ray points). Meteorological *from*-directions
        (wind direction, "the sun stands at azimuth X") must be converted by
        the caller: ``az_toward = az_from + 180``.
    elevation_deg : float or np.ndarray, default 0.0
        Elevation above the horizontal plane, degrees.
    rotation_angle_deg : float or np.ndarray, default 0.0
        AOI rotation in degrees clockwise, as stored in v3 files. The
        azimuth is geographic; subtracting this maps it into the rotated
        grid frame.

    Returns
    -------
    np.ndarray
        Scalar inputs → shape ``(3,)``; array inputs broadcast → shape
        ``broadcast + (3,)``. Components are **array-axis** deltas:
        component 0 is along axis 0 = **north** (not east!), component 1
        along axis 1 = east, component 2 along axis 2 = up. Unit-norm by
        construction; no re-normalization is applied (bit-parity with the
        historical inline call sites).
    """
    az_deg = np.asarray(azimuth_deg, dtype=np.float64) - np.asarray(
        rotation_angle_deg, dtype=np.float64
    )
    el_deg = np.asarray(elevation_deg, dtype=np.float64)
    az, el = np.broadcast_arrays(np.deg2rad(az_deg), np.deg2rad(el_deg))
    cos_el = np.cos(el)
    return np.stack((cos_el * np.cos(az), cos_el * np.sin(az), np.sin(el)), axis=-1)


def check_axes(file_or_attrs) -> None:
    """Raise ``ValueError`` unless *file_or_attrs* declares the v3 axis contract.

    Accepts an ``h5py`` File/Group/Dataset (anything with ``.attrs``), a
    plain attrs mapping, or a path to an HDF5 file. Passes silently when the
    ``axes`` attribute equals :data:`AXES_ATTR`.
    """
    if isinstance(file_or_attrs, (str, os.PathLike)):
        import h5py

        with h5py.File(file_or_attrs, "r") as f:
            return check_axes(f)
    attrs = getattr(file_or_attrs, "attrs", file_or_attrs)
    val = attrs.get("axes")
    if val is None:
        raise ValueError(
            "no 'axes' attribute: this is a pre-v3 VoxCity file or foreign "
            "data. VoxCity arrays are [i=north, j=east, k=up] only when the "
            "file declares it; convert pre-v3 VoxCity files once with "
            "voxcity.io.migrate_h5(src, dst)."
        )
    if isinstance(val, bytes):
        val = val.decode("utf-8")
    if str(val) != AXES_ATTR:
        raise ValueError(
            f"file declares axes={val!r}; this voxcity version expects "
            f"{AXES_ATTR!r} (axis 0 = north, row 0 = south edge; axis 1 = "
            "east; axis 2 = up — in the frame rotated clockwise by the "
            "file's rotation_angle)."
        )


def ensure_orientation(
    grid: np.ndarray,
    orientation_in: Literal["north_up", "south_up"],
    orientation_out: Literal["north_up", "south_up"] = ORIENTATION_NORTH_UP,
) -> np.ndarray:
    """Return ``grid`` converted from ``orientation_in`` to ``orientation_out``.

    Both orientations are defined for 2D arrays as:
    - north_up: row 0 = north/top, last row = south/bottom
    - south_up: row 0 = south/bottom, last row = north/top

    If orientations match, the input array is returned unchanged. When converting
    between north_up and south_up, a vertical flip is applied using ``np.flipud``.

    Notes
    -----
    - This function does not copy when no conversion is needed.
    - Use at data boundaries (read/write, interop) rather than deep in processing code.
    """
    if orientation_in == orientation_out:
        return grid
    # Only two orientations supported; converting between them is a vertical flip
    return np.flipud(grid)


def to_rasterio_layout(grid: np.ndarray) -> np.ndarray:
    """Convert an internal uv grid (nx, ny) to rasterio layout (ny, nx).

    Frames: uv_m/SOUTH_UP ``grid[i, j]`` (axis 0 = u, axis 1 = v) ->
    rasterio ``arr[row, col]`` with rows along v and cols along u, as
    consumed by ``rasterio.features.rasterize`` with the uv affine
    ``Affine(du*u_vec[0], dv*v_vec[0], ox, du*u_vec[1], dv*v_vec[1], oy)``.
    Pure transpose — no vertical flip. Returns a C-contiguous array
    (prevents silent Numba performance degradation downstream).
    """
    return np.ascontiguousarray(np.asarray(grid).T)


def from_rasterio_layout(arr: np.ndarray) -> np.ndarray:
    """Convert a rasterio-layout array (ny, nx) to an internal uv grid (nx, ny).

    Inverse of :func:`to_rasterio_layout`. Pure transpose — no vertical
    flip. Returns a C-contiguous array.
    """
    return np.ascontiguousarray(np.asarray(arr).T)


def grid_to_rotated_raster(grid: np.ndarray) -> np.ndarray:
    """Convert a uv grid to the legacy rotated-raster layout for GeoTIFF.

    ``flipud(grid.T)``: rows advance along -v, columns along +u. Only
    meaningful together with the rotated affine whose origin is the AOI's
    far-v corner (see the rotated-AOI fallback in ``exporter/geotiff.py``).
    Axis-aligned AOIs use the cell-centre scatter path instead.
    """
    return np.ascontiguousarray(np.flipud(np.asarray(grid).T))


def voxels_to_magicavoxel_axes(voxels: np.ndarray) -> np.ndarray:
    """(u/north, v/east, z/height) voxels -> pyvox dense (y=north, z=height, x=east).

    z is pre-flipped because pyvox inverts dense z when writing MagicaVoxel
    voxels.
    """
    return np.transpose(np.flip(voxels, axis=2), (0, 2, 1))


def voxels_to_kji(voxels: np.ndarray) -> np.ndarray:
    """(u, v, z) voxels -> (k=z, i=u, j=v) axis order used by the OBJ mesher."""
    return np.asarray(voxels).transpose(2, 0, 1)

