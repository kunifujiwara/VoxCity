"""Coordinate-frame projector for VoxCity grids.

Three frames this module understands:

  lon_lat    — geographic (lon°, lat°), WGS84
  ij_north   — continuous cell coordinate (i, j); i along u_vec, j along v_vec
  xy_world_m — Three.js world metres (x, y), Z-up, (0, 0) at SW voxel corner

The north→south orientation flip used by mesh builders is handled separately
by ensure_orientation() in orientation.py and is invisible here — it is already
folded into the (nx - u) term of lon_lat_to_xy_world_m.
"""

from __future__ import annotations

from typing import TypedDict

import numpy as np


class GridGeom(TypedDict):
    """Typed view of the dict returned by compute_grid_geometry()."""

    origin: np.ndarray
    """[lon, lat] of the reference corner (rectangle_vertices[0])."""

    side_1: np.ndarray
    """Vector v0 → v1 in lon/lat degrees."""

    side_2: np.ndarray
    """Vector v0 → v3 in lon/lat degrees."""

    u_vec: np.ndarray
    """lon/lat degrees per metre along the side_1 direction."""

    v_vec: np.ndarray
    """lon/lat degrees per metre along the side_2 direction."""

    grid_size: tuple[int, int]
    """(nx, ny) — number of cells along the u and v axes."""

    adj_mesh: tuple[float, float]
    """(dx_m, dy_m) — adjusted cell size in metres along each axis."""

    meshsize_m: float
    """Nominal cell size in metres; used by lon_lat_to_xy_world_m and xy_world_m_to_lon_lat."""


class GridProjector:
    """Projects coordinates between lon_lat and ij_north frames.

    The 2×2 affine matrix A maps cell coordinates (ij_north_i, ij_north_j)
    to lon/lat offsets (dlon, dlat) from the grid origin::

        A = [[u_vec[0]*dx_m, v_vec[0]*dy_m],
             [u_vec[1]*dx_m, v_vec[1]*dy_m]]

        [dlon, dlat]^T = A · [ij_north_i, ij_north_j]^T

    A⁻¹ is pre-computed in __init__ so each projection call is O(1).

    This mirrors the matrix inversion in lonLatToWorldXY (grid.ts) and the
    forward formula in _cell_centres_lonlat (zoning.py).
    """

    def __init__(self, geom: GridGeom) -> None:
        self._origin = np.asarray(geom["origin"], dtype=float)
        dx_m, dy_m = geom["adj_mesh"]
        u = np.asarray(geom["u_vec"], dtype=float)
        v = np.asarray(geom["v_vec"], dtype=float)
        self._nx = int(geom["grid_size"][0])
        self._meshsize_m = float(geom["meshsize_m"])

        # Forward affine coefficients (lon/lat offset per cell step)
        self._a = float(u[0] * dx_m)  # dlon per ij_north_i step
        self._b = float(v[0] * dy_m)  # dlon per ij_north_j step
        self._c = float(u[1] * dx_m)  # dlat per ij_north_i step
        self._d = float(v[1] * dy_m)  # dlat per ij_north_j step

        det = self._a * self._d - self._b * self._c
        if abs(det) < 1e-30:
            raise ValueError("GridGeom is degenerate (zero-area rectangle).")

        # Inverse affine coefficients
        self._inv00 = self._d / det
        self._inv01 = -self._b / det
        self._inv10 = -self._c / det
        self._inv11 = self._a / det

    def lon_lat_to_ij_north(
        self, lon: float, lat: float
    ) -> tuple[float, float]:
        """lon_lat → ij_north.

        Returns continuous (ij_north_i, ij_north_j) cell coordinates.
        The integer floor gives the cell index; (0.5, 0.5) is the centre of
        cell (0, 0). The origin corner maps to exactly (0.0, 0.0).
        """
        dlon = lon - float(self._origin[0])
        dlat = lat - float(self._origin[1])
        ij_north_i = self._inv00 * dlon + self._inv01 * dlat
        ij_north_j = self._inv10 * dlon + self._inv11 * dlat
        return ij_north_i, ij_north_j

    def ij_north_to_lon_lat(
        self, ij_north_i: float, ij_north_j: float
    ) -> tuple[float, float]:
        """ij_north → lon_lat. Exact inverse of lon_lat_to_ij_north."""
        dlon = self._a * ij_north_i + self._b * ij_north_j
        dlat = self._c * ij_north_i + self._d * ij_north_j
        return float(self._origin[0] + dlon), float(self._origin[1] + dlat)

    def lon_lat_to_xy_world_m(
        self, lon: float, lat: float
    ) -> tuple[float, float]:
        """lon_lat → xy_world_m.

        Mirrors lonLatToWorldXY in app/frontend/src/lib/grid.ts.
        The (nx - u) flip aligns the projector's ij_north u-axis with the
        x-axis emitted by build_voxel_buffers (which places voxel i at x=i*ms).
        """
        u, v = self.lon_lat_to_ij_north(lon, lat)
        return (self._nx - u) * self._meshsize_m, v * self._meshsize_m

    def xy_world_m_to_lon_lat(
        self, x_world_m: float, y_world_m: float
    ) -> tuple[float, float]:
        """xy_world_m → lon_lat. Exact inverse of lon_lat_to_xy_world_m."""
        u = self._nx - x_world_m / self._meshsize_m
        v = y_world_m / self._meshsize_m
        return self.ij_north_to_lon_lat(u, v)
