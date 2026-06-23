"""Voxelize a mesh into a VoxCity voxel grid via column z-ray casting.

The mesh is first mapped into voxel-index space (i, j, k) using the caller's
placement ``transform`` (e.g. produced by ``build_placement_transform`` in
``transform.py``). Occupancy is then determined per (i, j) column by casting
a single vertical ray through the column center and pairing up the ray's
hit z-values under the even-odd rule:

    sorted hits -> (z0, z1), (z2, z3), ...
    fill voxel k in [floor(z0 + 0.5), floor(z1 + 0.5)) for each pair

A non-watertight mesh can produce an odd number of hits for some column; in
that case we fall back to filling the single span from the first to the last
hit and log a warning, since this is the best approximation available for a
degenerate/open mesh.
"""
from __future__ import annotations

import numpy as np

from ..utils.logging import get_logger

_logger = get_logger(__name__)


def voxelize_mesh(mesh, transform, grid_shape):
    """Voxelize *mesh* into occupied (i, j, k) voxel indices.

    Args:
        mesh: a ``trimesh.Trimesh`` (or compatible) in original model
            coordinates. Not mutated.
        transform: 4x4 numpy affine mapping model coordinates -> voxel-index
            space (i, j, k), as produced by ``build_placement_transform``.
        grid_shape: ``(nx, ny, nz)`` voxel grid bounds used to clip results.

    Returns:
        ``(N, 3)`` numpy ``int64`` array of unique, sorted occupied (i, j, k)
        voxel indices. Empty ``(0, 3)`` array if there are no candidate
        columns or no ray hits anywhere.
    """
    nx, ny, nz = grid_shape

    m = mesh.copy()
    m.apply_transform(np.asarray(transform, dtype=float))

    if len(m.faces) == 0 or len(m.vertices) == 0:
        return np.empty((0, 3), dtype=np.int64)

    bounds = m.bounds  # (2, 3): [min, max] in transformed (i, j, k) space
    i_min = int(np.floor(bounds[0, 0]))
    i_max = int(np.ceil(bounds[1, 0]))
    j_min = int(np.floor(bounds[0, 1]))
    j_max = int(np.ceil(bounds[1, 1]))

    i_lo = max(i_min, 0)
    i_hi = min(i_max, nx - 1)
    j_lo = max(j_min, 0)
    j_hi = min(j_max, ny - 1)

    if i_lo > i_hi or j_lo > j_hi:
        return np.empty((0, 3), dtype=np.int64)

    columns = [(i, j) for i in range(i_lo, i_hi + 1) for j in range(j_lo, j_hi + 1)]
    if not columns:
        return np.empty((0, 3), dtype=np.int64)

    z_below = float(bounds[0, 2]) - 10.0
    ray_origins = np.array([[i + 0.5, j + 0.5, z_below] for i, j in columns], dtype=float)
    ray_directions = np.tile(np.array([0.0, 0.0, 1.0]), (len(columns), 1))

    locations, index_ray, _index_tri = m.ray.intersects_location(
        ray_origins=ray_origins, ray_directions=ray_directions, multiple_hits=True
    )

    if len(locations) == 0:
        return np.empty((0, 3), dtype=np.int64)

    voxel_rows = []
    n_odd_columns = 0
    for ray_idx in np.unique(index_ray):
        i, j = columns[int(ray_idx)]
        z_hits = np.sort(locations[index_ray == ray_idx, 2])
        n_hits = len(z_hits)

        if n_hits == 0:
            continue

        if n_hits % 2 == 0:
            for pair_start in range(0, n_hits, 2):
                a, b = z_hits[pair_start], z_hits[pair_start + 1]
                k0 = int(np.floor(a + 0.5))
                k1 = int(np.floor(b + 0.5))
                if k1 > k0:
                    voxel_rows.append((i, j, k0, k1))
        else:
            n_odd_columns += 1
            k0 = int(np.floor(z_hits[0] + 0.5))
            k1 = int(np.floor(z_hits[-1] + 0.5))
            if k1 > k0:
                voxel_rows.append((i, j, k0, k1))

    if n_odd_columns:
        _logger.warning(
            "voxelize_mesh: %d column(s) had an odd number of ray hits "
            "(non-watertight mesh); falling back to first-to-last-hit span "
            "for those columns.",
            n_odd_columns,
        )

    if not voxel_rows:
        return np.empty((0, 3), dtype=np.int64)

    ijk = np.concatenate(
        [
            np.stack(
                [
                    np.full(k1 - k0, i, dtype=np.int64),
                    np.full(k1 - k0, j, dtype=np.int64),
                    np.arange(k0, k1, dtype=np.int64),
                ],
                axis=1,
            )
            for (i, j, k0, k1) in voxel_rows
        ],
        axis=0,
    )

    in_bounds = (
        (ijk[:, 0] >= 0) & (ijk[:, 0] < nx)
        & (ijk[:, 1] >= 0) & (ijk[:, 1] < ny)
        & (ijk[:, 2] >= 0) & (ijk[:, 2] < nz)
    )
    n_clipped = int((~in_bounds).sum())
    if n_clipped:
        _logger.warning(
            "voxelize_mesh: clipped %d voxel(s) outside grid_shape=%s.",
            n_clipped, grid_shape,
        )

    ijk = ijk[in_bounds]
    if ijk.shape[0] == 0:
        return np.empty((0, 3), dtype=np.int64)

    return np.unique(ijk, axis=0).astype(np.int64)
