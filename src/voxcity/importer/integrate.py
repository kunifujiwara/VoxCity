"""Stamp voxelized buildings into a VoxCity object and update metadata grids."""
from __future__ import annotations

import numpy as np

from ..geoprocessor.raster import process_grid
from ..utils.logging import get_logger

_logger = get_logger(__name__)

BUILDING_CODE = -3
EMPTY_CODE = 0


def _spans_from_ks(ks):
    """Return list of [start_k, end_k_exclusive] contiguous spans from sorted ks."""
    ks = sorted(set(int(k) for k in ks))
    spans = []
    if not ks:
        return spans
    start = prev = ks[0]
    for k in ks[1:]:
        if k == prev + 1:
            prev = k
        else:
            spans.append([start, prev + 1])
            start = prev = k
    spans.append([start, prev + 1])
    return spans


def stamp_buildings(voxcity, occupied_by_name, building_value=BUILDING_CODE,
                    overwrite=True, source=None, manifest_extra=None):
    """Write occupied cells into voxcity and update derived metadata grids.

    Runs in two passes. Pass 1 writes voxel classes and assigns building ids.
    Pass 2 derives heights/min_heights -- but only after the *effective* ground
    datum has been computed, because that datum depends on the ids pass 1 just
    assigned (process_grid averages the DEM per building footprint).

    The datum MUST match generator/voxelizer.py's, since regenerate_voxels()
    rebuilds these columns from the metadata written here. Getting it wrong made
    imported buildings shift vertically on the first edit.
    """
    classes = voxcity.voxels.classes
    nx, ny, nz = classes.shape
    meshsize = float(voxcity.voxels.meta.meshsize)

    # 1. grow Z if needed
    max_k = -1
    for occ in occupied_by_name.values():
        if len(occ):
            max_k = max(max_k, int(occ[:, 2].max()))
    if max_k >= nz:
        pad = np.zeros((nx, ny, max_k + 1 - nz), dtype=classes.dtype)
        classes = np.concatenate([classes, pad], axis=2)
        voxcity.voxels.classes = classes
        nz = classes.shape[2]

    ids_grid = voxcity.buildings.ids
    heights_grid = voxcity.buildings.heights
    min_heights = voxcity.buildings.min_heights

    next_id = int(ids_grid.max()) + 1 if ids_grid.size else 1
    collisions = 0
    id_map = {}
    # Tracks which group last claimed each (i, j) column within *this* call,
    # so cross-group collisions inside a single stamp_buildings invocation
    # can be detected and logged instead of silently overwriting ids_grid.
    column_owner = {}
    # Pass-1 output, consumed by pass 2: [(bid, {(i, j): [absolute k, ...]}), ...]
    pending = []

    # --- Pass 1: voxel classes + building ids ---
    for name, occ in occupied_by_name.items():
        if not len(occ):
            continue

        # group occupied cells by column for metadata
        cols = {}
        for i, j, k in occ:
            i, j, k = int(i), int(j), int(k)
            if k < 0 or k >= nz or i < 0 or i >= nx or j < 0 or j >= ny:
                continue
            current = classes[i, j, k]
            if overwrite or current == EMPTY_CODE:
                if current != EMPTY_CODE and overwrite:
                    collisions += 1
                classes[i, j, k] = building_value
                cols.setdefault((i, j), []).append(k)

        if not cols:
            _logger.warning(
                "Group '%s' has no in-bounds occupied cells; skipping id assignment.",
                name,
            )
            continue

        bid = next_id
        next_id += 1
        id_map[name] = bid

        for i, j in cols:
            owner = column_owner.get((i, j))
            if owner is not None and owner[0] != name:
                _logger.warning(
                    "Column (%d, %d) claimed by group '%s' (id=%d) was already "
                    "stamped by group '%s' (id=%d) in this call; last group wins.",
                    i, j, name, bid, owner[0], owner[1],
                )
            column_owner[(i, j)] = (name, bid)
            ids_grid[i, j] = bid

        pending.append((bid, cols))

    # --- Effective ground datum, mirroring generator/voxelizer.py exactly:
    #     dem_grid = process_grid(building_id_grid, dem - dem.min())
    #     ground_level = int(dem_grid[i, j] / voxel_size + 0.5) + 1
    #     Computed here (after pass 1) because process_grid averages the DEM per
    #     building id and therefore needs the ids just assigned above.
    dem0 = np.asarray(voxcity.dem.elevation, dtype=float)
    eff_dem = process_grid(ids_grid, dem0 - dem0.min())

    # --- Pass 2: heights / min_heights, measured above the effective ground ---
    for _bid, cols in pending:
        for (i, j), ks in cols.items():
            spans = _spans_from_ks(ks)
            top_k = max(s[1] for s in spans)  # exclusive end
            ground_level = int(eff_dem[i, j] / meshsize + 0.5) + 1
            heights_grid[i, j] = max(float(heights_grid[i, j]), (top_k - ground_level) * meshsize)
            cell = min_heights[i, j]
            if not isinstance(cell, list):
                cell = []
            for a, b in spans:
                cell.append([(a - ground_level) * meshsize, (b - ground_level) * meshsize])
            min_heights[i, j] = cell

    if collisions:
        _logger.info("Imported buildings overwrote %d existing non-air voxel(s).", collisions)

    manifest = {"source": source, "id_map": id_map}
    if manifest_extra:
        manifest.update(manifest_extra)
    voxcity.extras.setdefault("imported_buildings", []).append(manifest)

    return voxcity
