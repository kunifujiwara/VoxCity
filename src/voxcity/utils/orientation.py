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
  legacy ij_south — same layout as uv_m/SOUTH_UP; kept for backward compatibility.
  ensure_orientation() is the only legitimate place to convert between the two.

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

from typing import Literal
import numpy as np

# Public constants to reference orientation in docs and code
ORIENTATION_NORTH_UP: Literal["north_up"] = "north_up"
ORIENTATION_SOUTH_UP: Literal["south_up"] = "south_up"


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

