"""Precise (geometry-intersection) building rasterization.

Split out of buildings.py so this path has no Earth Engine import and can
be unit-tested directly. Output-equivalent to the legacy implementation
except for one documented change: among equal-height buildings overlapping
the same cell, processing order is now deterministic (height desc, then
building index asc) instead of rtree-iteration-order dependent.
"""
import math

import numpy as np
import pandas as pd
from shapely.errors import GEOSException

from .core import create_cell_polygon
from ...utils.logging import get_logger

_logger = get_logger(__name__)

# Fraction of a grid cell's area that a building must cover to be assigned
# that building's height. Below this the cell is left as ground.
_CELL_INTERSECTION_THRESHOLD = 0.3

# Safety margin (in cells) added around each building's projected uv index
# range. Guarantees the per-cell candidate lists are a SUPERSET of what the
# legacy rtree bbox query produced (the exact bbox filter downstream then
# yields the identical qualifying set). 1 cell covers the parallelogram
# bbox slack for any grid rotation; +1 absorbs float rounding.
_CANDIDATE_CELL_MARGIN = 2


def _collect_building_polygons(filtered_gdf, complement_height):
    """Per-building preprocessing (verbatim legacy semantics)."""
    building_polygons = []
    for idx_b, row in filtered_gdf.iterrows():
        polygon = row.geometry
        height = row.get('height', None)
        if complement_height is not None and (height == 0 or height is None or pd.isna(height)):
            height = complement_height
        min_height = row.get('min_height', 0)
        if pd.isna(min_height):
            min_height = 0
        is_inner = row.get('is_inner', False)
        # NaN is truthy; treat it as "not inner" (legacy fix preserved).
        if pd.isna(is_inner):
            is_inner = False
        feature_id = row.get('id', idx_b)
        if not polygon.is_valid:
            try:
                polygon = polygon.buffer(0)
                if not polygon.is_valid:
                    polygon = polygon.simplify(1e-8)
            except (GEOSException, ValueError):
                _logger.debug("Could not repair invalid building polygon (id=%s)", feature_id)
        bounding_box = polygon.bounds
        building_polygons.append((
            polygon, bounding_box, height, min_height, is_inner, feature_id
        ))
    return building_polygons


def _candidate_cells_by_building(building_polygons, grid_size, adjusted_meshsize,
                                 origin, u_vec, v_vec):
    """Invert the enumeration: building bbox -> uv cell-index range.

    Maps each building's lon/lat bbox corners through the inverse of the
    2x2 grid affine, expands by _CANDIDATE_CELL_MARGIN, clamps, and
    registers the building index in each covered cell's candidate list.
    Returns {(i, j): [building indices, ascending]}. Cells never covered by
    any building do not appear (and are never touched downstream).
    """
    ni, nj = int(grid_size[0]), int(grid_size[1])
    n_buildings = len(building_polygons)
    du, dv = float(adjusted_meshsize[0]), float(adjusted_meshsize[1])
    a = du * float(u_vec[0]); b = dv * float(v_vec[0])
    c = du * float(u_vec[1]); d = dv * float(v_vec[1])
    det = a * d - b * c
    candidates = {}
    if abs(det) < 1e-30:
        # Degenerate affine (should not occur for real grids): fall back to
        # "every building is a candidate for every cell" — slow but the
        # exact downstream filters keep the output identical.
        full = list(range(n_buildings))
        for i in range(ni):
            for j in range(nj):
                candidates[(i, j)] = list(full)
        return candidates
    inv_a, inv_b = d / det, -b / det
    inv_c, inv_d = -c / det, a / det
    ox, oy = float(origin[0]), float(origin[1])
    m = _CANDIDATE_CELL_MARGIN
    for k, (_, bbox, _, _, _, _) in enumerate(building_polygons):
        if not bbox or len(bbox) < 4:
            # Empty geometry after failed repair: legacy code crashed here
            # (rtree insert of empty bounds); skipping is strictly safer.
            continue
        minx, miny, maxx, maxy = bbox
        us, vs = [], []
        for (px, py) in ((minx, miny), (minx, maxy), (maxx, miny), (maxx, maxy)):
            dx, dy = px - ox, py - oy
            us.append(inv_a * dx + inv_b * dy)
            vs.append(inv_c * dx + inv_d * dy)
        i_lo = max(0, int(math.floor(min(us))) - m)
        i_hi = min(ni - 1, int(math.ceil(max(us))) + m)
        j_lo = max(0, int(math.floor(min(vs))) - m)
        j_hi = min(nj - 1, int(math.ceil(max(vs))) + m)
        if i_lo > i_hi or j_lo > j_hi:
            continue
        for i in range(i_lo, i_hi + 1):
            for j in range(j_lo, j_hi + 1):
                candidates.setdefault((i, j), []).append(k)
    return candidates


def _process_with_geometry_intersection(filtered_gdf, grid_size, adjusted_meshsize,
                                        origin, u_vec, v_vec, complement_height):
    building_height_grid = np.zeros(grid_size)
    building_id_grid = np.zeros(grid_size)
    building_min_height_grid = np.empty(grid_size, dtype=object)
    for idx_flat in range(building_min_height_grid.size):
        building_min_height_grid.flat[idx_flat] = []

    building_polygons = _collect_building_polygons(filtered_gdf, complement_height)
    candidates = _candidate_cells_by_building(
        building_polygons, grid_size, adjusted_meshsize, origin, u_vec, v_vec
    )

    for (i, j), cand_ks in candidates.items():
        cell = create_cell_polygon(origin, i, j, adjusted_meshsize, u_vec, v_vec)
        if not cell.is_valid:
            cell = cell.buffer(0)
        cell_area = cell.area

        cell_buildings = []
        for k in cand_ks:
            bpoly, bbox, height, minh, inr, fid = building_polygons[k]
            sort_val = height if (height is not None) else -float('inf')
            cell_buildings.append((k, bpoly, bbox, height, minh, inr, fid, sort_val))
        # Height descending; ties broken by ascending building index (the
        # one documented divergence from the rtree-order legacy behavior).
        cell_buildings.sort(key=lambda x: (x[-1], -x[0]), reverse=True)

        found_intersection = False
        all_zero_or_nan = True
        for (k, polygon, bbox, height, min_height, is_inner, feature_id, _) in cell_buildings:
            try:
                minx_p, miny_p, maxx_p, maxy_p = bbox
                minx_c, miny_c, maxx_c, maxy_c = cell.bounds
                overlap_minx = max(minx_p, minx_c)
                overlap_miny = max(miny_p, miny_c)
                overlap_maxx = min(maxx_p, maxx_c)
                overlap_maxy = min(maxy_p, maxy_c)
                if (overlap_maxx <= overlap_minx) or (overlap_maxy <= overlap_miny):
                    continue
                bbox_intersect_area = (overlap_maxx - overlap_minx) * (overlap_maxy - overlap_miny)
                if bbox_intersect_area < _CELL_INTERSECTION_THRESHOLD * cell_area:
                    continue
                if not polygon.is_valid:
                    polygon = polygon.buffer(0)
                if not cell.intersects(polygon):
                    continue
                inter_area = cell.intersection(polygon).area
            except (GEOSException, ValueError):
                # Legacy fallback: retry with a simplified polygon. Note the
                # inner except deliberately catches ONLY GEOSException,
                # matching the original.
                try:
                    simplified_polygon = polygon.simplify(1e-8)
                    if not simplified_polygon.is_valid:
                        continue
                    inter_area = cell.intersection(simplified_polygon).area
                except GEOSException:
                    _logger.debug("Skipping geometry intersection at cell (%d,%d)", i, j)
                    continue

            if (inter_area / cell_area) > _CELL_INTERSECTION_THRESHOLD:
                found_intersection = True
                if not is_inner:
                    building_min_height_grid[i, j].append([min_height, height])
                    building_id_grid[i, j] = feature_id
                    if (height is not None and not np.isnan(height) and height > 0):
                        all_zero_or_nan = False
                        current_height = building_height_grid[i, j]
                        if (current_height == 0 or np.isnan(current_height) or current_height < height):
                            building_height_grid[i, j] = height
                else:
                    building_min_height_grid[i, j] = [[0, 0]]
                    building_height_grid[i, j] = 0
                    all_zero_or_nan = False
                    break

        if found_intersection and all_zero_or_nan:
            building_height_grid[i, j] = np.nan

    return building_height_grid, building_min_height_grid, building_id_grid, filtered_gdf
