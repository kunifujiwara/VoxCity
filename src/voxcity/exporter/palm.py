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
from pathlib import Path

import numpy as np
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

    Uses the voxelizer's own rounding (voxelizer.py:_flatten_building_segments
    uses ``int(seg[0] * inv_vs + 0.5)``): ``int(value / meshsize + 0.5)``.
    Every height-to-level conversion in this module routes through here so
    nz-sizing and slice bounds are derived from one rounding rule.
    """
    return int(float(value) / meshsize + 0.5)


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
                # Floor at 0: a segment entirely below ground (seg[1] <= 0)
                # must not register as a building with a negative height.
                t = max(0.0, float(seg[1]))
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


def _build_buildings(heights, ids, building_type, mask, segment_top_m):
    """LOD1 building fields.

    Returns (buildings_2d float32, building_id int32, building_type int8),
    all (y, x) with PIDS fill values off ``mask`` (see
    _build_building_mask). ``buildings_2d`` is
    ``max(heights, segment_top_m)`` on the mask, so a cell with LOD2
    geometry but no recorded LOD1 height still gets a height. Cells with no
    positive id receive generated ids above the existing max.
    """
    h = _clean_heights(heights)
    effective_h = np.maximum(h, segment_top_m)

    buildings_2d = np.full(h.shape, FILL_FLOAT, dtype=np.float32)
    buildings_2d[mask] = effective_h[mask].astype(np.float32)

    repaired = mask & (segment_top_m > h)
    n_repaired = int(repaired.sum())
    if n_repaired:
        _logger.info(
            f"{n_repaired} cell(s) had LOD2 segment geometry taller than "
            "the recorded LOD1 height; buildings_2d was raised to match "
            "(orphan-segment repair)."
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
    has_id = mask & (ids_arr > 0)
    building_id[has_id] = ids_arr[has_id].astype(np.int32)
    missing = mask & (ids_arr <= 0)
    if missing.any():
        next_id = int(ids_arr.max()) + 1
        building_id[missing] = np.arange(
            next_id, next_id + int(missing.sum()), dtype=np.int32
        )

    building_type_arr = np.full(h.shape, FILL_BYTE, dtype=np.int8)
    building_type_arr[mask] = np.int8(building_type)
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
            if _to_level(seg[0], meshsize) > 0:
                return True
    return False


def _build_buildings_3d(heights, min_heights, meshsize, mask, segment_top_m):
    """LOD2 byte mask (z, y, x) from per-cell [min, max] segments.

    Cells without segments fall back to ground extrusion from ``heights``.
    Aligned to ``mask`` (see _build_building_mask): every masked cell gets
    at least one filled level.
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
                    # Clamp both bounds to [0, nz] independently: an
                    # unclamped k0 lets a below-ground minimum wrap to the
                    # top of the column via negative-index slicing, and an
                    # unclamped k1 lets a fully-below-ground segment (both
                    # bounds negative) spuriously fill from level 0 up to a
                    # wrapped stop index -- the `k1 > k0` check below is what
                    # then makes such a segment a no-op instead.
                    k0 = max(0, min(_to_level(seg[0], meshsize), nz))
                    k1 = max(0, min(_to_level(seg[1], meshsize), nz))
                    if k1 > k0:
                        b3d[k0:k1, i, j] = 1
                        filled = True
            elif h[i, j] > 0.0:
                top_level = _to_level(h[i, j], meshsize)
                if top_level > 0:
                    b3d[:top_level, i, j] = 1
                    filled = True
            if not filled and mask[i, j]:
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


def _build_surface_types(land_cover_grid, land_cover_source, building_mask,
                         canopy_mask, under_tree_vegetation_type, soil_type_code):
    """Mutually exclusive surface classification per PIDS.

    Precedence per cell: building (all fields stay fill) > canopy (ground
    under trees becomes ``under_tree_vegetation_type``) > land-cover mapping.
    A 'building' land-cover class without building height becomes pavement 2
    (concrete): the surface is sealed but there is no obstacle.

    Returns dict with vegetation_type/pavement_type/water_type/soil_type
    (int8 (y, x)) and surface_fraction (float32 (3, y, x)).
    """
    index_to_assignment, class_names = _build_index_to_palm_map(land_cover_source)
    ny, nx = land_cover_grid.shape

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
            if building_mask[i, j]:
                continue
            raw_idx = int(land_cover_grid[i, j])
            if canopy_mask[i, j]:
                category, code = 'vegetation', int(under_tree_vegetation_type)
            else:
                category, code = index_to_assignment.get(raw_idx, DEFAULT_ASSIGNMENT)
                if category == 'building':
                    category, code = 'pavement', 2

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
