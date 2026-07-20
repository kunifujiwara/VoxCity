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
    # Factories
    # ------------------------------------------------------------------

    @classmethod
    def from_city(cls, city) -> "GridProjector":
        """Build a projector from a VoxCity instance.

        Uses ``extras['rectangle_vertices']`` and the voxel grid's meshsize;
        falls back to an axis-aligned rectangle from ``meta.bounds`` when the
        extras carry no vertices (mirrors the v3 save fallback).
        """
        # Lazy import: geoprocessor.raster.core imports GridGeom from this module.
        from ..geoprocessor.raster.core import compute_grid_geometry
        from ..geoprocessor.utils import normalize_rectangle_vertices

        extras = getattr(city, "extras", None) or {}
        rect = extras.get("rectangle_vertices")
        if rect is None:
            lon0, lat0, lon1, lat1 = city.voxels.meta.bounds
            rect = [(lon0, lat0), (lon0, lat1), (lon1, lat1), (lon1, lat0)]
        else:
            # Canonicalize to [SW,NW,NE,SE] as save_results_h5 does, since
            # compute_grid_geometry assumes that ordering (origin=v0, side_1=v1-v0).
            rect = normalize_rectangle_vertices(rect, warn=False)
        geom = compute_grid_geometry([tuple(p) for p in rect], city.voxels.meta.meshsize)
        if geom is None:
            raise ValueError("could not compute grid geometry from the city's rectangle_vertices")
        return cls(geom)

    @classmethod
    def from_h5(cls, path) -> "GridProjector":
        """Build a projector from a v3 VoxCity HDF5 file.

        Reads the first-class v3 geometry (``rectangle_vertices`` dataset,
        ``meshsize`` attr). Raises the migrate-pointing ``ValueError`` on
        pre-v3 files (no JSON parsing, no fallback ladder).
        """
        import h5py

        from .orientation import check_axes
        # Lazy import: geoprocessor.raster.core imports GridGeom from this module.
        from ..geoprocessor.raster.core import compute_grid_geometry

        with h5py.File(path, "r") as f:
            check_axes(f)
            if "rectangle_vertices" not in f or "meshsize" not in f.attrs:
                raise ValueError(
                    f"{path}: declares the v3 axes contract but is missing "
                    "rectangle_vertices/meshsize; the file may be truncated or "
                    "corrupted."
                )
            rect = [tuple(p) for p in f["rectangle_vertices"][:].tolist()]
            meshsize = float(f.attrs["meshsize"])
        geom = compute_grid_geometry(rect, meshsize)
        if geom is None:
            raise ValueError(f"{path}: could not compute grid geometry from rectangle_vertices")
        return cls(geom)

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
