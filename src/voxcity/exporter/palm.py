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
