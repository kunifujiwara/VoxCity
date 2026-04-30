"""Pure helpers for zone aggregation. No FastAPI / state imports here."""
from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np
from matplotlib.path import Path as MplPath

from .models import ZoneStat


def _cell_centres_lonlat(grid_geom: dict):
    """Return ((nx*ny, 2) centres, ii flat, jj flat)."""
    nx, ny = grid_geom["grid_size"]
    dx, dy = grid_geom["adj_mesh"]
    o = np.asarray(grid_geom["origin"], dtype=float)
    u = np.asarray(grid_geom["u_vec"], dtype=float)
    v = np.asarray(grid_geom["v_vec"], dtype=float)
    ii, jj = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
    centres = (
        o[None, None, :]
        + (ii[..., None] + 0.5) * dx * u[None, None, :]
        + (jj[..., None] + 0.5) * dy * v[None, None, :]
    )
    return centres.reshape(-1, 2), ii.ravel(), jj.ravel()


def polygon_lonlat_to_cells(
    ring: Sequence[Sequence[float]],
    grid_geom: dict,
) -> List[Tuple[int, int]]:
    """Rasterize a closed lon/lat ring to NORTH_UP (i, j) cells.

    Mirrors the JS `polygonToCells` in `app/frontend/src/lib/grid.ts`.
    """
    if len(ring) < 3:
        return []
    centres, ii, jj = _cell_centres_lonlat(grid_geom)
    path = MplPath(np.asarray(ring, dtype=float))
    inside = path.contains_points(centres)
    if not inside.any():
        return []
    return list(zip(ii[inside].tolist(), jj[inside].tolist()))


def points_in_polygon_lonlat(
    points_lonlat: np.ndarray,
    ring: Sequence[Sequence[float]],
) -> np.ndarray:
    """Vectorized point-in-polygon test. `points_lonlat` is (N, 2)."""
    if len(ring) < 3 or points_lonlat.size == 0:
        return np.zeros(len(points_lonlat), dtype=bool)
    return MplPath(np.asarray(ring, dtype=float)).contains_points(points_lonlat)


def stats_from_values(
    zone_id: str,
    cell_count: int,
    values: np.ndarray,
    weights: np.ndarray | None = None,
) -> ZoneStat:
    """Compute count/valid/mean/min/max/std. `weights` enables area-weighted mean."""
    if values.size == 0:
        return ZoneStat(zone_id=zone_id, cell_count=int(cell_count), valid_count=0)
    finite = np.isfinite(values)
    if not finite.any():
        return ZoneStat(zone_id=zone_id, cell_count=int(cell_count), valid_count=0)
    v = values[finite]
    if weights is not None:
        w = weights[finite]
        if w.sum() > 0:
            mean = float((v * w).sum() / w.sum())
        else:
            mean = float(v.mean())
    else:
        mean = float(v.mean())
    return ZoneStat(
        zone_id=zone_id,
        cell_count=int(cell_count),
        valid_count=int(finite.sum()),
        mean=mean,
        min=float(v.min()),
        max=float(v.max()),
        std=float(v.std()),
    )
