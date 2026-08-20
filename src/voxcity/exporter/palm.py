"""
PALM static driver export module for VoxCity.

Writes a PIDS-conformant NetCDF static driver (``<domain_name>_static``) for
the PALM model system (https://palm.muk.uni-hannover.de/).

Notes:
- Expects raw land cover grids as produced per-source by VoxCity (same
  contract as the CityLES exporter). Supported sources: 'OpenStreetMap',
  'Urbanwatch', 'OpenEarthMapJapan', 'ESA WorldCover',
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

from ..models import VoxCity
from ..utils.logging import get_logger

_logger = get_logger(__name__)

__all__ = ["PalmExporter", "export_palm"]

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
