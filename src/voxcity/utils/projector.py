"""Coordinate-frame projector for VoxCity grids.

Two frames this module understands:

  lon_lat — geographic (lon°, lat°), WGS84
  uv_m    — local grid metres from the grid origin (rectangle_vertices[0]):
              u_m = metres along u_vec (side_1 direction)
              v_m = metres along v_vec (side_2 direction)

Key invariant:
  voxcity_grid[i, j, k]  ↔  scene position (i×meshsize_m, j×meshsize_m, k×meshsize_m)

No orientation flip, no (nx−u) compensation. The voxel array index IS the
scene coordinate (divided by meshsize_m).
"""

from __future__ import annotations

from typing import TypedDict, Union
import math

import numpy as np

Scalar = Union[float, int]
ArrayLike = Union[Scalar, np.ndarray]


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
    """(du_m, dv_m) — adjusted cell size in metres along u and v axes."""

    meshsize_m: float
    """Nominal cell size in metres."""


class GridProjector:
    """Projects between lon_lat (WGS84) and uv_m (local grid metres).

    The 2×2 affine A maps (u_cell, v_cell) → (dlon, dlat)::

        A = [[u_vec[0]*du_m, v_vec[0]*dv_m],
             [u_vec[1]*du_m, v_vec[1]*dv_m]]

    where (du_m, dv_m) = adj_mesh.  A⁻¹ is pre-computed for O(1) calls.
    Accepts both scalars and numpy arrays.
    """

    def __init__(self, geom: GridGeom) -> None:
        self._origin = np.asarray(geom["origin"], dtype=float)
        du_m, dv_m = geom["adj_mesh"]
        self._du_m = float(du_m)
        self._dv_m = float(dv_m)
        u = np.asarray(geom["u_vec"], dtype=float)
        v = np.asarray(geom["v_vec"], dtype=float)

        # Forward affine: (u_cell, v_cell) → (dlon, dlat)
        self._a = float(u[0] * du_m)
        self._b = float(v[0] * dv_m)
        self._c = float(u[1] * du_m)
        self._d = float(v[1] * dv_m)

        det = self._a * self._d - self._b * self._c
        if abs(det) < 1e-30:
            raise ValueError("GridGeom is degenerate (zero-area rectangle).")

        self._inv00 = self._d / det
        self._inv01 = -self._b / det
        self._inv10 = -self._c / det
        self._inv11 = self._a / det

    # ------------------------------------------------------------------
    # Primary: lon_lat ↔ uv_m
    # ------------------------------------------------------------------

    def lon_lat_to_uv_m(
        self,
        lon: ArrayLike,
        lat: ArrayLike,
    ) -> tuple[ArrayLike, ArrayLike]:
        """lon_lat → uv_m.  Accepts scalars or numpy arrays.

        Returns (u_m, v_m) where u_m = metres along u_vec from grid origin,
        v_m = metres along v_vec from grid origin.
        Cell (i, j) occupies uv_m in [i*du_m, (i+1)*du_m) × [j*dv_m, (j+1)*dv_m).
        """
        dlon = lon - self._origin[0]
        dlat = lat - self._origin[1]
        u_cell = self._inv00 * dlon + self._inv01 * dlat
        v_cell = self._inv10 * dlon + self._inv11 * dlat
        return u_cell * self._du_m, v_cell * self._dv_m

    def uv_m_to_lon_lat(
        self,
        u_m: ArrayLike,
        v_m: ArrayLike,
    ) -> tuple[ArrayLike, ArrayLike]:
        """uv_m → lon_lat. Exact inverse of lon_lat_to_uv_m."""
        u_cell = u_m / self._du_m
        v_cell = v_m / self._dv_m
        dlon = self._a * u_cell + self._b * v_cell
        dlat = self._c * u_cell + self._d * v_cell
        return self._origin[0] + dlon, self._origin[1] + dlat

    # ------------------------------------------------------------------
    # Convenience: integer cell index
    # ------------------------------------------------------------------

    def lon_lat_to_cell(
        self,
        lon: ArrayLike,
        lat: ArrayLike,
    ) -> tuple:
        """lon_lat → (i, j) integer cell index.

        Uses floor division — safe for boundary points and negative coordinates.
        Scalar inputs return Python ints; array inputs return int ndarrays.
        """
        u_m, v_m = self.lon_lat_to_uv_m(lon, lat)
        eps = 1e-12
        if isinstance(u_m, np.ndarray):
            return np.floor(u_m / self._du_m + eps).astype(int), np.floor(v_m / self._dv_m + eps).astype(int)
        return int(math.floor(u_m / self._du_m + eps)), int(math.floor(v_m / self._dv_m + eps))

    def cell_to_lon_lat(
        self,
        i: ArrayLike,
        j: ArrayLike,
    ) -> tuple[ArrayLike, ArrayLike]:
        """Cell centre (i+0.5, j+0.5) → lon_lat."""
        return self.uv_m_to_lon_lat((i + 0.5) * self._du_m, (j + 0.5) * self._dv_m)
