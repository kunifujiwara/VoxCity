"""
PALM static driver export module for VoxCity.

Writes a PIDS-conformant NetCDF static driver (``<domain_name>_static``) for
the PALM model system (https://palm.muk.uni-hannover.de/).

Notes:
- Expects raw land cover grids as produced per-source by VoxCity (same
  contract as the CityLES exporter). Supported sources: 'OpenStreetMap'
  (alias 'Standard'), 'Urbanwatch', 'OpenEarthMapJapan', 'ESA WorldCover',
  'ESRI 10m Annual Land Cover', 'Dynamic World V1'.
- Grids are written with no vertical flip: VoxCity's axis contract
  (axis 0 = north, row 0 = south edge; axis 1 = east) equals PALM's (y, x).
- Automated validation covers the file format plus PALM's documented runtime
  consistency rules; an actual PALM run is a separate follow-up validation.
  Design: docs/superpowers/specs/2026-08-20-palm-exporter-design.md
"""

from __future__ import annotations

import datetime
import os
from collections import namedtuple
from pathlib import Path

import numpy as np
from netCDF4 import Dataset
from pyproj import Transformer

from ..geoprocessor.utils import compute_rotation_angle, normalize_rectangle_vertices
from ..models import VoxCity
from ..utils.lc import get_land_cover_classes
from ..utils.logging import get_logger

_logger = get_logger(__name__)

__all__ = ["PalmExporter", "export_palm"]  # noqa: F822 — defined in the final unit

# ---------------------------------------------------------------------------
# PIDS fill values
FILL_FLOAT = -9999.0
FILL_INT = -9999
FILL_BYTE = -127

# surface_fraction index order per PIDS
IDX_VEGETATION = 0
IDX_PAVEMENT = 1
IDX_WATER = 2

# Valid PIDS class ranges (used by the consistency validator)
BYTE_RANGES = {
    "vegetation_type": (1, 18),
    "pavement_type": (1, 15),
    "water_type": (1, 5),
    "soil_type": (1, 6),
    "building_type": (1, 6),
}

# ---------------------------------------------------------------------------
# Land-cover class name -> (category, PALM code) per source.
# Categories: 'vegetation' -> vegetation_type, 'pavement' -> pavement_type,
# 'water' -> water_type, 'building' -> handled by the building mask (code None).
#
# PALM codes used: vegetation_type 1 bare soil, 2 crops, 3 short grass,
# 7 deciduous broadleaf trees, 13 ice caps, 14 bogs and marshes,
# 16 deciduous shrubs; pavement_type 1 asphalt, 2 concrete;
# water_type 1 lake, 3 ocean.

OSM_CLASS_TO_PALM = {
    'Bareland': ('vegetation', 1),
    'Rangeland': ('vegetation', 3),
    'Shrub': ('vegetation', 16),
    'Moss and lichen': ('vegetation', 1),
    'Agriculture land': ('vegetation', 2),
    'Tree': ('vegetation', 7),
    'Wet land': ('vegetation', 14),
    'Mangroves': ('vegetation', 7),
    'Water': ('water', 1),
    'Snow and ice': ('vegetation', 13),
    'Developed space': ('pavement', 2),
    'Road': ('pavement', 1),
    'Building': ('building', None),
    'No Data': ('vegetation', 3),
}

URBANWATCH_CLASS_TO_PALM = {
    'Building': ('building', None),
    'Road': ('pavement', 1),
    'Parking Lot': ('pavement', 1),
    'Tree Canopy': ('vegetation', 7),
    'Grass/Shrub': ('vegetation', 3),
    'Agriculture': ('vegetation', 2),
    'Water': ('water', 1),
    'Barren': ('vegetation', 1),
    'Unknown': ('vegetation', 3),
    'Sea': ('water', 3),
}

OEMJ_CLASS_TO_PALM = {
    'Bareland': ('vegetation', 1),
    'Rangeland': ('vegetation', 3),
    'Developed space': ('pavement', 2),
    'Road': ('pavement', 1),
    'Tree': ('vegetation', 7),
    'Water': ('water', 1),
    'Agriculture land': ('vegetation', 2),
    'Building': ('building', None),
    'No Data': ('vegetation', 3),
}

ESA_CLASS_TO_PALM = {
    'Trees': ('vegetation', 7),
    'Shrubland': ('vegetation', 16),
    'Grassland': ('vegetation', 3),
    'Cropland': ('vegetation', 2),
    'Built-up': ('pavement', 2),
    'Barren / sparse vegetation': ('vegetation', 1),
    'Snow and ice': ('vegetation', 13),
    'Open water': ('water', 1),
    'Herbaceous wetland': ('vegetation', 14),
    'Mangroves': ('vegetation', 7),
    'Moss and lichen': ('vegetation', 1),
}

ESRI_CLASS_TO_PALM = {
    'No Data': ('vegetation', 3),
    'Water': ('water', 1),
    'Trees': ('vegetation', 7),
    'Grass': ('vegetation', 3),
    'Flooded Vegetation': ('vegetation', 14),
    'Crops': ('vegetation', 2),
    'Scrub/Shrub': ('vegetation', 16),
    'Built Area': ('pavement', 2),
    'Bare Ground': ('vegetation', 1),
    'Snow/Ice': ('vegetation', 13),
    'Clouds': ('vegetation', 3),
}

DYNAMIC_WORLD_CLASS_TO_PALM = {
    'Water': ('water', 1),
    'Trees': ('vegetation', 7),
    'Grass': ('vegetation', 3),
    'Flooded Vegetation': ('vegetation', 14),
    'Crops': ('vegetation', 2),
    'Shrub and Scrub': ('vegetation', 16),
    'Built': ('pavement', 2),
    'Bare': ('vegetation', 1),
    'Snow and Ice': ('vegetation', 13),
}

# Fallback assignment when a raw index has no name entry
DEFAULT_ASSIGNMENT = ('vegetation', 3)


def _get_source_name_mapping(land_cover_source):
    """Return the class-name -> (category, PALM code) table for a source."""
    if land_cover_source in ('OpenStreetMap', 'Standard'):
        return OSM_CLASS_TO_PALM
    if land_cover_source == 'Urbanwatch':
        return URBANWATCH_CLASS_TO_PALM
    if land_cover_source == 'OpenEarthMapJapan':
        return OEMJ_CLASS_TO_PALM
    if land_cover_source == 'ESA WorldCover':
        return ESA_CLASS_TO_PALM
    if land_cover_source == 'ESRI 10m Annual Land Cover':
        return ESRI_CLASS_TO_PALM
    if land_cover_source == 'Dynamic World V1':
        return DYNAMIC_WORLD_CLASS_TO_PALM
    _logger.warning(
        f"Unknown land cover source {land_cover_source!r}; "
        "falling back to the OpenStreetMap/Standard mapping"
    )
    return OSM_CLASS_TO_PALM


def _build_index_to_palm_map(land_cover_source):
    """Map raw per-source 0-based index -> (category, PALM code).

    Uses the source's class-name order from get_land_cover_classes, exactly
    like the CityLES exporter. Unknown sources use the OSM name list so raw
    indices still resolve deterministically.
    """
    try:
        class_names = list(get_land_cover_classes(land_cover_source).values())
    except Exception:
        # get_land_cover_classes has no else branch: an unrecognised source
        # raises rather than returning a default (UnboundLocalError today).
        # Fall back to the OSM class order so raw indices still resolve.
        class_names = list(get_land_cover_classes('OpenStreetMap').values())

    name_to_assignment = _get_source_name_mapping(land_cover_source)
    index_to_assignment = {
        idx: name_to_assignment.get(name, DEFAULT_ASSIGNMENT)
        for idx, name in enumerate(class_names)
    }
    return index_to_assignment, class_names


def _build_georeference(rectangle_vertices):
    """Georeferencing attributes from the AOI rectangle.

    Returns dict with origin_lon/origin_lat (SW corner), origin_x/origin_y
    (that corner in the auto-detected UTM zone), rotation_angle (degrees
    clockwise; VoxCity and PIDS conventions agree), and the EPSG code used.
    """
    rect = normalize_rectangle_vertices(rectangle_vertices, warn=False)
    origin_lon = float(rect[0][0])
    origin_lat = float(rect[0][1])
    rotation_angle = float(compute_rotation_angle(rect))

    zone = min(max(int((origin_lon + 180.0) // 6.0) + 1, 1), 60)
    epsg = (32600 if origin_lat >= 0.0 else 32700) + zone
    transformer = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
    origin_x, origin_y = transformer.transform(origin_lon, origin_lat)

    return {
        "origin_lon": origin_lon,
        "origin_lat": origin_lat,
        "origin_x": float(origin_x),
        "origin_y": float(origin_y),
        "rotation_angle": rotation_angle,
        "epsg": epsg,
    }


def _build_zt(dem_grid):
    """Terrain height, shifted so its minimum is exactly 0.

    Returns (zt float32 (y, x), origin_z) where origin_z is the subtracted
    minimum (recorded as the PIDS global attribute ``origin_z``). NaN/inf
    cells take the minimum of the finite cells (i.e. 0 after shifting).
    """
    zt = np.asarray(dem_grid, dtype=np.float64).copy()
    finite = np.isfinite(zt)
    if not finite.any():
        return np.zeros(zt.shape, dtype=np.float32), 0.0
    min_val = float(zt[finite].min())
    zt[~finite] = min_val
    zt -= min_val
    return zt.astype(np.float32), min_val


def _clean_heights(heights):
    """Heights with NaN/inf treated as 'no building'."""
    h = np.asarray(heights, dtype=np.float64)
    return np.where(np.isfinite(h), h, 0.0)


def _to_level(value, meshsize):
    """Convert a height in meters to a voxel level index.

    Uses the same rounding rule as the voxelizer
    (voxelizer.py:_flatten_building_segments uses
    ``int(seg[0] * inv_vs + 0.5)``): ``int(value / meshsize + 0.5)``. Every
    height-to-level conversion in this module routes through here so
    nz-sizing and slice bounds are derived from one rounding rule.
    """
    return int(float(value) / meshsize + 0.5)


def _clean_segment_bound(value):
    """A single LOD2 segment bound with NaN/inf treated as ground level (0.0).

    Segment bounds are otherwise the one input in this module not routed
    through finite-cleaning (heights via _clean_heights, ids via
    nan_to_num in _build_buildings): without this, _segment_top_m would
    silently drop a non-finite bound to 0 while _build_buildings_3d's
    _to_level call on the same raw value raised ValueError/OverflowError --
    the two builders disagreeing about whether the segment exists.
    """
    v = float(value)
    return v if np.isfinite(v) else 0.0


def _segment_top_m(min_heights, shape):
    """Per-cell tallest LOD2 segment top, in meters above ground.

    Each cell's value is ``max(0.0, seg[1])`` over its segments (0.0 if the
    cell has no segments or all segments are entirely below ground).
    """
    top = np.zeros(shape, dtype=np.float64)
    if min_heights is None:
        return top
    mh = np.asarray(min_heights, dtype=object)
    ny, nx = shape
    for i in range(ny):
        for j in range(nx):
            segs = mh[i, j]
            if not segs:
                continue
            cell_top = 0.0
            for seg in segs:
                # The max(0.0, ...) is redundant with the cell_top = 0.0
                # initialization above (t > cell_top already excludes any
                # t <= 0, so cell_top can never go negative either way);
                # kept so the floor survives a refactor of this loop.
                t = max(0.0, _clean_segment_bound(seg[1]))
                if t > cell_top:
                    cell_top = t
            top[i, j] = cell_top
    return top


def _build_building_mask(heights, min_heights):
    """Shared building-presence mask for the LOD1 and LOD2 builders.

    Returns ``(mask, segment_top_m)``, both (y, x): ``mask`` is
    ``(heights > 0) | (segment_top_m > 0)`` and ``segment_top_m`` is each
    cell's tallest LOD2 segment top (see _segment_top_m). Pass both to
    _build_buildings and _build_buildings_3d so the two outputs agree on
    which cells are buildings.
    """
    h = _clean_heights(heights)
    segment_top_m = _segment_top_m(min_heights, h.shape)
    mask = (h > 0.0) | (segment_top_m > 0.0)
    return mask, segment_top_m


def _build_buildings(heights, ids, building_type, segment_top_m, building_mask):
    """LOD1 building fields.

    Returns (buildings_2d float32, building_id int32, building_type int8),
    all (y, x) with PIDS fill values off ``building_mask`` (see
    _build_building_mask). ``buildings_2d`` is
    ``max(heights, segment_top_m)`` on the mask, so a cell with LOD2
    geometry but no recorded LOD1 height still gets a height. Cells with no
    positive id receive generated ids above the existing max.
    """
    h = _clean_heights(heights)
    effective_h = np.maximum(h, segment_top_m)

    buildings_2d = np.full(h.shape, FILL_FLOAT, dtype=np.float32)
    buildings_2d[building_mask] = effective_h[building_mask].astype(np.float32)

    repaired = building_mask & (segment_top_m > h)
    n_repaired = int(repaired.sum())
    if n_repaired:
        max_discrepancy = float((segment_top_m[repaired] - h[repaired]).max())
        _logger.warning(
            f"{n_repaired} cell(s) had LOD2 segment geometry taller than "
            "the recorded LOD1 height; buildings_2d was raised to match "
            f"(orphan-segment repair) (max discrepancy {max_discrepancy:.1f} m)."
        )

    if ids is None:
        ids_arr = np.zeros(h.shape, dtype=np.int64)
    else:
        # ids assumed small positive integers (VoxCity pipeline convention);
        # not validated here.
        ids_arr = np.nan_to_num(
            np.asarray(ids, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0
        ).astype(np.int64)

    building_id = np.full(h.shape, FILL_INT, dtype=np.int32)
    has_id = building_mask & (ids_arr > 0)
    building_id[has_id] = ids_arr[has_id].astype(np.int32)
    missing = building_mask & (ids_arr <= 0)
    if missing.any():
        next_id = int(ids_arr.max()) + 1
        building_id[missing] = np.arange(
            next_id, next_id + int(missing.sum()), dtype=np.int32
        )

    building_type_arr = np.full(h.shape, FILL_BYTE, dtype=np.int8)
    building_type_arr[building_mask] = np.int8(building_type)
    return buildings_2d, building_id, building_type_arr


def _has_elevated_segments(min_heights, meshsize):
    """True when any building segment starts above ground level after
    _to_level rounding — geometry buildings_2d cannot express."""
    if min_heights is None:
        return False
    for cell in np.asarray(min_heights, dtype=object).ravel():
        if not cell:
            continue
        for seg in cell:
            if _to_level(_clean_segment_bound(seg[0]), meshsize) > 0:
                return True
    return False


def _build_buildings_3d(heights, min_heights, meshsize, segment_top_m, building_mask):
    """LOD2 byte mask (z, y, x) from per-cell [min, max] segments.

    Cells without segments fall back to ground extrusion from ``heights``.
    Aligned to ``building_mask`` (see _build_building_mask): every masked
    cell gets at least one filled level.
    """
    h = _clean_heights(heights)
    ny, nx = h.shape
    mh = None if min_heights is None else np.asarray(min_heights, dtype=object)

    height_top = _to_level(float(h.max()), meshsize) if h.size else 0
    # segment_top_m already holds each cell's max segment top in meters, and
    # _to_level is monotonic on non-negative inputs, so converting the
    # array-wide max once is equivalent to converting every segment
    # individually and taking the max of the resulting levels -- no second
    # scan of min_heights needed here.
    segment_top_level = (
        _to_level(float(segment_top_m.max()), meshsize) if segment_top_m.size else 0
    )
    nz = max(height_top, segment_top_level, 1)

    b3d = np.zeros((nz, ny, nx), dtype=np.int8)
    for i in range(ny):
        for j in range(nx):
            segs = mh[i, j] if mh is not None else None
            filled = False
            if segs:
                for seg in segs:
                    # Clamp both bounds to [0, nz] independently -- see
                    # TestBuildBuildings3d.test_negative_segment_min_clamps_to_ground
                    # and .test_fully_below_ground_segment_does_not_spuriously_fill
                    # for the negative-index wraparound this avoids.
                    k0 = max(0, min(_to_level(_clean_segment_bound(seg[0]), meshsize), nz))
                    k1 = max(0, min(_to_level(_clean_segment_bound(seg[1]), meshsize), nz))
                    if k1 > k0:
                        b3d[k0:k1, i, j] = 1
                        filled = True
            if not filled and h[i, j] > 0.0:
                # Segments present but none of them filled (e.g. all
                # entirely below ground) must still fall back to extrusion
                # from heights, the same as a cell with no segments at all --
                # otherwise a recorded LOD1 height is silently dropped in
                # favor of the forced single-level fill below.
                top_level = _to_level(h[i, j], meshsize)
                if top_level > 0:
                    b3d[:top_level, i, j] = 1
                    filled = True
            if not filled and building_mask[i, j]:
                # Every masked cell must end up with >=1 level: PALM treats
                # buildings_3d as authoritative when present, so a sub-voxel
                # building (e.g. 0.4 m at meshsize=2.0) that rounds to an
                # empty level range must not vanish from the LOD2 mask while
                # still holding a building_id in buildings_2d -- that would
                # silently delete a building the user asked to export.
                # Rounding up overstates height by less than one dz (the
                # vertical resolution limit anyway): a bounded, explainable
                # error beats unbounded silent data loss.
                b3d[0, i, j] = 1
    return b3d


def _build_surface_types(land_cover_grid, land_cover_source, canopy_mask,
                         under_tree_vegetation_type, soil_type_code, building_mask):
    """Mutually exclusive surface classification per PIDS.

    Precedence per cell: building (all fields stay fill) > canopy-over-
    non-water (ground under trees becomes ``under_tree_vegetation_type``)
    > land-cover mapping. Canopy over a cell the land-cover mapping
    resolves to water does NOT override it: overhanging/riparian canopy
    keeps its water surface (water differs from short grass in albedo,
    heat capacity, and evaporation by margins that dominate a microclimate
    result), and the LAD field already represents the trees independently
    of the surface type below them. A 'building' land-cover class without
    building height becomes pavement 2 (concrete): the surface is sealed
    but there is no obstacle.

    Returns dict with vegetation_type/pavement_type/water_type/soil_type
    (int8 (y, x)) and surface_fraction (float32 (3, y, x)).
    """
    index_to_assignment, class_names = _build_index_to_palm_map(land_cover_source)
    ny, nx = land_cover_grid.shape
    if building_mask.shape != (ny, nx) or canopy_mask.shape != (ny, nx):
        # A shape mismatch (e.g. a stale mask from a different grid) is the
        # one place in this module that would otherwise produce a wrong
        # answer quietly: NumPy broadcasting/fancy-indexing on a
        # differently-shaped boolean mask silently ignores the extra rows
        # or raises somewhere unrelated, rather than failing at the point
        # that actually has the wrong input.
        raise ValueError(
            "_build_surface_types: building_mask "
            f"{building_mask.shape} and canopy_mask {canopy_mask.shape} "
            f"must both match land_cover_grid {(ny, nx)}"
        )

    vegetation_type = np.full((ny, nx), FILL_BYTE, dtype=np.int8)
    pavement_type = np.full((ny, nx), FILL_BYTE, dtype=np.int8)
    water_type = np.full((ny, nx), FILL_BYTE, dtype=np.int8)
    soil_type = np.full((ny, nx), FILL_BYTE, dtype=np.int8)
    surface_fraction = np.full((3, ny, nx), FILL_FLOAT, dtype=np.float32)

    category_index = {
        'vegetation': IDX_VEGETATION,
        'pavement': IDX_PAVEMENT,
        'water': IDX_WATER,
    }
    mapping_stats = {}

    for i in range(ny):
        for j in range(nx):
            # Tier 1: building footprint -- the cell stays all-fill (the
            # building surface takes over); nothing below this applies.
            if building_mask[i, j]:
                continue
            # Tier 3 (resolved first so tier 2 can check whether it landed
            # on water): the land-cover mapping. A raw 'building' class
            # without recorded height has no obstacle, so it becomes
            # pavement (concrete) rather than staying an unmapped category.
            raw_idx = int(land_cover_grid[i, j])
            category, code = index_to_assignment.get(raw_idx, DEFAULT_ASSIGNMENT)
            if category == 'building':
                category, code = 'pavement', 2
            # Tier 2: canopy over non-water -- ground under trees becomes
            # vegetation. Skipped when tier 3 resolved to water: overhanging
            # canopy keeps its water surface (see the docstring).
            if canopy_mask[i, j] and category != 'water':
                category, code = 'vegetation', int(under_tree_vegetation_type)

            if category == 'vegetation':
                vegetation_type[i, j] = np.int8(code)
                soil_type[i, j] = np.int8(soil_type_code)
            elif category == 'pavement':
                pavement_type[i, j] = np.int8(code)
                soil_type[i, j] = np.int8(soil_type_code)
            else:  # water
                water_type[i, j] = np.int8(code)

            surface_fraction[:, i, j] = 0.0
            surface_fraction[category_index[category], i, j] = 1.0

            key = (raw_idx, category, code)
            mapping_stats[key] = mapping_stats.get(key, 0) + 1

    _logger.info("Land cover -> PALM surface classification summary:")
    total = ny * nx
    n_building = int(building_mask.sum())
    if n_building:
        # mapping_stats only covers non-building cells, so building_mask's
        # share is logged as its own line -- otherwise the itemized
        # percentages below sum to ~80% on a 20%-built domain with nothing
        # explaining the remainder.
        _logger.info(
            f"  {n_building} cell(s) ({n_building / total * 100:.1f}%) are "
            "buildings (classified separately; not counted below)."
        )
    for (raw_idx, category, code), count in sorted(mapping_stats.items()):
        name = class_names[raw_idx] if 0 <= raw_idx < len(class_names) else 'Unknown'
        _logger.info(
            f"  {raw_idx}: {name} -> {category}_type {code}: "
            f"{count} cells ({count / total * 100:.1f}%)"
        )

    return {
        "vegetation_type": vegetation_type,
        "pavement_type": pavement_type,
        "water_type": water_type,
        "soil_type": soil_type,
        "surface_fraction": surface_fraction,
    }


def _build_lad(canopy_top, canopy_bottom, meshsize, lad_value, building_mask):
    """Leaf area density field lad(zlad, y, x).

    zlad levels are [0, (k - 0.5) * dz ...] up to the highest canopy top
    (palm_csd convention: surface level plus cell centres); zlad[0] is
    always exactly 0.0 by construction. Vegetated columns carry 0.0 below
    the crown, ``lad_value`` inside [bottom, top] (both boundaries
    inclusive), and fill above the top; non-vegetated columns are all
    fill. Canopy over buildings is cleared. ``bottom`` needs no clamping:
    a negative bottom already satisfies ``level >= bottom`` for every zlad
    level, since every level is >= 0 (see the empty-crown fallback below
    for the other edge case this invariant covers).

    Returns (lad, zlad) or (None, None) when no canopy remains.
    """
    # NaN/inf -> 0.0 ("no canopy here"), matching _clean_heights and the
    # ids cleaning in _build_buildings. Without posinf/neginf, nan_to_num's
    # default substitutes +-inf with the largest finite float64 rather than
    # literal infinity, so n_centres below ends up astronomically large but
    # finite, and np.arange(1, n_centres + 1) raises ValueError: Maximum
    # allowed size exceeded -- not the OverflowError int() would raise on
    # literal infinity. On the bottom side, an unsanitized +inf would leave
    # bottom far above every zlad level, forcing the single-level fallback
    # instead of filling from the ground.
    top = np.nan_to_num(
        np.asarray(canopy_top, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0
    )
    top = np.where(np.asarray(building_mask, dtype=bool), 0.0, top)
    bottom = np.nan_to_num(
        np.asarray(canopy_bottom, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0
    )

    max_top = float(top.max()) if top.size else 0.0
    if max_top <= 0.0:
        return None, None

    # Cell-centre convention (palm_csd), not the edge convention _to_level
    # implements: level k covers a slab centred on (k - 0.5) * dz, so the
    # centre count is ceil(max_top / dz - 0.5), not the int(x / dz + 0.5)
    # nearest-edge rounding used everywhere else in this module. This is a
    # genuinely different rule, not an inline duplicate of _to_level's -- do
    # not "fix" it to route through _to_level.
    n_centres = max(int(np.ceil(max_top / meshsize - 0.5)), 1)
    zlad = np.concatenate(
        ([0.0], (np.arange(1, n_centres + 1) - 0.5) * meshsize)
    ).astype(np.float32)

    ny, nx = top.shape
    lad = np.full((zlad.size, ny, nx), FILL_FLOAT, dtype=np.float32)
    for i, j in zip(*np.nonzero(top > 0.0)):
        t, b = top[i, j], bottom[i, j]
        col = np.zeros(zlad.size, dtype=np.float32)
        col[zlad > t] = FILL_FLOAT
        in_crown = (zlad >= b) & (zlad <= t)
        if not in_crown.any():
            # Empty crown: either too thin to catch a level (bottom and top
            # both between two levels) or inverted (bottom > top). Both
            # fall back to the single topmost level at/below top. `[-1]` is
            # safe because zlad[0] == 0.0 and t > 0 here (top > 0.0 is the
            # loop condition above), so `zlad <= t` always has >=1 match.
            in_crown[np.nonzero(zlad <= t)[0][-1]] = True
        col[in_crown] = np.float32(lad_value)
        lad[:, i, j] = col
    return lad, zlad


# Single source of truth for every static-driver field: the validator
# asserts dtype from this table, the writer takes dims/fill/units/
# long_name/lod from it too, so the two can no longer independently
# drift. nc_type is a netCDF4 createVariable type code, and numpy
# understands the same codes (np.dtype("f4") == float32, "i4" == int32,
# "b" == int8), so it also serves as the expected in-memory dtype.
_FieldSpec = namedtuple("_FieldSpec", "dims nc_type fill units long_name lod optional")

_FIELD_SPECS = {
    "zt": _FieldSpec(("y", "x"), "f4", FILL_FLOAT, "m", "terrain height", None, False),
    "buildings_2d": _FieldSpec(("y", "x"), "f4", FILL_FLOAT, "m", "building height", 1, False),
    "building_id": _FieldSpec(("y", "x"), "i4", FILL_INT, "1", "building id numbers", None, False),
    "building_type": _FieldSpec(("y", "x"), "b", FILL_BYTE, "1", "building type classification", None, False),
    "vegetation_type": _FieldSpec(("y", "x"), "b", FILL_BYTE, "1", "vegetation type classification", None, False),
    "pavement_type": _FieldSpec(("y", "x"), "b", FILL_BYTE, "1", "pavement type classification", None, False),
    "water_type": _FieldSpec(("y", "x"), "b", FILL_BYTE, "1", "water type classification", None, False),
    "soil_type": _FieldSpec(("y", "x"), "b", FILL_BYTE, "1", "soil type classification", 1, False),
    "surface_fraction": _FieldSpec(
        ("nsurface_fraction", "y", "x"), "f4", FILL_FLOAT, "1", "surface fraction", None, False),
    "buildings_3d": _FieldSpec(("z", "y", "x"), "b", FILL_BYTE, "1", "building structure in 3d", 2, True),
    "lad": _FieldSpec(("zlad", "y", "x"), "f4", FILL_FLOAT, "m2 m-3", "leaf area density", None, True),
}


def _validate_static_fields(fields):
    """Enforce PALM's documented runtime consistency rules before writing.

    A violation is an exporter bug (the builders should never produce one),
    so this raises RuntimeError rather than silently correcting.

    Enforced:
    - Every field's in-memory dtype matches its _FIELD_SPECS declaration
      (the same table _write_static_driver writes from).
    - zt is finite, and its minimum is (within 1e-4 m) exactly 0.
    - buildings_2d is finite: a NaN/inf height is otherwise invisible to
      every rule below it -- `bld = buildings_2d != FILL_FLOAT` is True
      for NaN (NaN != anything is True), so a non-finite height silently
      counts as "this cell is a building" and would go on to satisfy
      every downstream presence rule rather than getting caught.
    - Every non-building cell has exactly one of vegetation_type/
      pavement_type/water_type set; every building cell has none.
    - soil_type is set exactly where vegetation_type or pavement_type is,
      in both directions.
    - surface_fraction has exactly 3 leading slices (vegetation/pavement/
      water) and sums to 1 on every classified cell.
    - Every building cell has a building_id and a building_type, and,
      unconditionally (not only when buildings_3d is present -- the
      buildings_3d="auto" default means most exports never build one),
      neither is set on a non-building cell.
    - Byte-coded fields (BYTE_RANGES) fall within their documented PIDS
      class range, both bounds.
    - When buildings_3d is present, its per-cell column presence agrees
      with buildings_2d/building_id/building_type in both directions.
      This deliberately never compares magnitudes: a buildings_2d height
      may legitimately exceed a shorter 3D column's top (e.g. 20 m next
      to a single [0, 4] segment is valid LOD2 data).
    - lad values (where not fill) are finite and non-negative.

    Deliberately not enforced here (spec's validation section lists both;
    neither is silently dropped):
    - "LAD values only within [bottom, top] layers": this function
      receives only the final lad array, not the canopy top/bottom inputs
      _build_lad placed it from, so it has no independent reference to
      re-derive the expected layer range against -- that placement is
      _build_lad's own contract, not something a downstream consumer of
      its output can re-verify.
    """
    problems = []

    for name, spec in _FIELD_SPECS.items():
        if spec.optional:
            arr = fields.get(name)
            if arr is None:
                continue
        else:
            arr = fields[name]
        expected_dtype = np.dtype(spec.nc_type)
        if arr.dtype != expected_dtype:
            problems.append(
                f"{name} has dtype {arr.dtype}, expected {expected_dtype}"
            )

    zt = fields["zt"]
    if not np.isfinite(zt).all():
        problems.append("zt contains non-finite values")
    elif abs(float(zt.min())) > 1e-4:
        problems.append(f"zt minimum is {float(zt.min())}, expected 0")

    b2d_raw = fields["buildings_2d"]
    if not np.isfinite(b2d_raw).all():
        problems.append("buildings_2d contains non-finite values")

    veg = fields["vegetation_type"] != FILL_BYTE
    pav = fields["pavement_type"] != FILL_BYTE
    wat = fields["water_type"] != FILL_BYTE
    bld = b2d_raw != np.float32(FILL_FLOAT)
    has_id = fields["building_id"] != FILL_INT
    has_type = fields["building_type"] != FILL_BYTE

    n_types = veg.astype(int) + pav.astype(int) + wat.astype(int)
    if (n_types[~bld] != 1).any():
        problems.append("non-building cells must have exactly one surface type")
    if (n_types[bld] != 0).any():
        problems.append("building cells must have no surface type")

    soil = fields["soil_type"] != FILL_BYTE
    if ((veg | pav) != soil).any():
        problems.append("soil_type must be set exactly where vegetation or pavement is")

    sf = fields["surface_fraction"]
    if sf.shape[0] != 3:
        problems.append(
            f"surface_fraction leading dimension must be 3, got {sf.shape[0]}"
        )
    classified = veg | pav | wat
    if classified.any() and not np.allclose(sf.sum(axis=0)[classified], 1.0):
        problems.append("surface_fraction must sum to 1 on classified cells")

    # Both id/type presence directions are checked unconditionally (not
    # only inside the buildings_3d block below): a stray building_id or
    # building_type on a non-building cell is just as invalid whether or
    # not a 3D mask happens to be present, and buildings_3d="auto" means
    # most exports never build one -- LOD1-only is the common path, not
    # an edge case, so this cannot be deferred to the buildings_3d branch.
    if (bld & ~has_id).any():
        problems.append("building cells missing building_id")
    if (bld & ~has_type).any():
        problems.append("building cells missing building_type")
    if (~bld & has_id).any():
        problems.append("building_id present on a non-building cell")
    if (~bld & has_type).any():
        problems.append("building_type present on a non-building cell")

    for name, (lo, hi) in BYTE_RANGES.items():
        arr = fields[name]
        set_mask = arr != FILL_BYTE
        if set_mask.any():
            vals = arr[set_mask]
            if (vals < lo).any() or (vals > hi).any():
                problems.append(f"{name} values outside valid range [{lo}, {hi}]")

    # LOD1/LOD2 presence invariant is bidirectional: a cell is a building in
    # every building output or in none of them (see _build_building_mask).
    # buildings_2d may legitimately exceed a shorter 3D column's top (a 20 m
    # buildings_2d height next to a single [0, 4] segment is valid LOD2
    # data) -- this deliberately never compares magnitudes, only presence.
    b3d = fields.get("buildings_3d")
    if b3d is not None:
        col = b3d.astype(bool).any(axis=0)
        presence_checks = (
            (col & ~bld, "buildings_3d column present without a buildings_2d height"),
            (bld & ~col, "buildings_2d height present without a buildings_3d column"),
            (col & ~has_id, "buildings_3d column present without a building_id"),
            (has_id & ~col, "building_id present without a buildings_3d column"),
            (col & ~has_type, "buildings_3d column present without a building_type"),
            (has_type & ~col, "building_type present without a buildings_3d column"),
        )
        for mask, message in presence_checks:
            if mask.any():
                problems.append(message)

    lad = fields.get("lad")
    if lad is not None:
        data = lad[lad != np.float32(FILL_FLOAT)]
        if data.size and (not np.isfinite(data).all() or (data < 0).any()):
            problems.append("lad contains negative or non-finite values")

    if problems:
        raise RuntimeError(
            "PALM static driver consistency check failed: " + "; ".join(problems)
        )


def _write_static_driver(path, fields, coords, attrs):
    """Write the PIDS static driver NetCDF file (NETCDF4 format).

    Truncates and overwrites an existing file at ``path`` (netCDF4's
    ``"w"`` mode), matching how every other exporter in this package
    writes its output file.
    """
    with Dataset(str(path), "w", format="NETCDF4") as nc:
        for key, value in attrs.items():
            if value is not None:
                setattr(nc, key, value)

        def coord(name, values, long_name):
            nc.createDimension(name, values.size)
            var = nc.createVariable(name, "f4", (name,))
            var[:] = values
            var.units = "m"
            var.long_name = long_name

        coord("x", coords["x"], "distance to origin in x-direction")
        coord("y", coords["y"], "distance to origin in y-direction")
        if "z" in coords:
            coord("z", coords["z"], "height above origin")
        if "zlad" in coords:
            coord("zlad", coords["zlad"], "height above ground")

        nc.createDimension("nsurface_fraction", 3)
        nsf = nc.createVariable("nsurface_fraction", "i4", ("nsurface_fraction",))
        nsf[:] = [IDX_VEGETATION, IDX_PAVEMENT, IDX_WATER]

        def write(name, data, dims, dtype, fill, **var_attrs):
            var = nc.createVariable(name, dtype, dims, fill_value=fill, zlib=True)
            var[:] = data
            for k, v in var_attrs.items():
                setattr(var, k, v)

        # Declared from the same _FIELD_SPECS table the validator's dtype
        # check reads (defined above _validate_static_fields).
        for name, spec in _FIELD_SPECS.items():
            data = fields.get(name) if spec.optional else fields[name]
            if data is None:
                continue
            var_attrs = {"units": spec.units, "long_name": spec.long_name}
            if spec.lod is not None:
                var_attrs["lod"] = np.int32(spec.lod)
            fill = np.dtype(spec.nc_type).type(spec.fill)
            write(name, data, spec.dims, spec.nc_type, fill, **var_attrs)
