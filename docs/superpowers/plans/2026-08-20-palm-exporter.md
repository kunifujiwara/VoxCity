# PALM Static Driver Exporter Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Export a VoxCity model to a PIDS-conformant PALM static driver NetCDF file (`<domain_name>_static`).

**Architecture:** One new module `src/voxcity/exporter/palm.py` following the CityLES exporter pattern: per-source land-cover mapping tables → pure numpy array builders (no I/O) → an internal consistency validator encoding PALM's runtime rules → a direct netCDF4 writer → an `export_palm()` orchestrator plus `PalmExporter` adapter registered in the exporter package. Spec: `docs/superpowers/specs/2026-08-20-palm-exporter-design.md`.

**Tech Stack:** numpy, netCDF4 (hard dep), pyproj (hard dep), pytest. Test env: conda env `voxcity` (conda is NOT on PATH — always `& "C:\Users\kunih\miniconda3\Scripts\conda.exe" run -n voxcity ...`).

**Key codebase facts (verified — do not re-derive):**
- VoxCity 2D grids: axis 0 = north (row 0 = south edge), axis 1 = east → equals PALM `(y, x)`. **No flip anywhere.**
- `city.buildings.min_heights` is an object-dtype 2D array; each cell is a list of `[min_h, max_h]` segments in meters above ground. Voxelizer rounding is `int(h / meshsize + 0.5)`.
- `city.land_cover.classes` holds **raw per-source 0-based indices**; names come from `voxcity.utils.lc.get_land_cover_classes(source)` values in dict order. OSM/Standard order: 0 Bareland, 1 Rangeland, 2 Shrub, 3 Moss and lichen, 4 Agriculture land, 5 Tree, 6 Wet land, 7 Mangroves, 8 Water, 9 Snow and ice, 10 Developed space, 11 Road, 12 Building, 13 No Data.
- `normalize_rectangle_vertices(rect, warn=False)` → canonical `[SW, NW, NE, SE]` (lon, lat) pairs; `compute_rotation_angle(rect)` → degrees clockwise (matches PIDS `rotation_angle`). Both in `voxcity.geoprocessor.utils`.
- PIDS fill values: float `-9999.0`, int `-9999`, byte `-127`. `surface_fraction` index order: 0=vegetation, 1=pavement, 2=water.
- Test command: `& "C:\Users\kunih\miniconda3\Scripts\conda.exe" run -n voxcity python -m pytest tests/test_exporter_palm.py -v`

---

### Task 0: Branch

**Files:** none

- [ ] **Step 1: Create the feature branch (repo is currently on `main`)**

```powershell
git switch -c feat/palm-exporter
```

No worktree (user preference: plain branches only).

---

### Task 1: Module scaffold + land-cover mapping tables

**Files:**
- Create: `src/voxcity/exporter/palm.py`
- Create: `tests/test_exporter_palm.py`

- [ ] **Step 1: Write failing tests for the mapping tables**

Create `tests/test_exporter_palm.py`:

```python
"""Tests for voxcity.exporter.palm (PALM static driver exporter)."""

import numpy as np
import pytest
from pathlib import Path

from voxcity.exporter.palm import (
    FILL_FLOAT,
    FILL_INT,
    FILL_BYTE,
    OSM_CLASS_TO_PALM,
    URBANWATCH_CLASS_TO_PALM,
    OEMJ_CLASS_TO_PALM,
    ESA_CLASS_TO_PALM,
    ESRI_CLASS_TO_PALM,
    DYNAMIC_WORLD_CLASS_TO_PALM,
)

ALL_TABLES = [
    OSM_CLASS_TO_PALM,
    URBANWATCH_CLASS_TO_PALM,
    OEMJ_CLASS_TO_PALM,
    ESA_CLASS_TO_PALM,
    ESRI_CLASS_TO_PALM,
    DYNAMIC_WORLD_CLASS_TO_PALM,
]

VALID_RANGES = {"vegetation": (1, 18), "pavement": (1, 15), "water": (1, 5)}


class TestMappingTables:
    def test_fill_values(self):
        assert FILL_FLOAT == -9999.0
        assert FILL_INT == -9999
        assert FILL_BYTE == -127

    @pytest.mark.parametrize("table", ALL_TABLES)
    def test_entries_are_category_code_pairs(self, table):
        for name, (category, code) in table.items():
            assert category in ("vegetation", "pavement", "water", "building")
            if category == "building":
                assert code is None
            else:
                lo, hi = VALID_RANGES[category]
                assert lo <= code <= hi, f"{name}: {category} code {code} out of range"

    def test_osm_covers_all_standard_classes(self):
        from voxcity.utils.lc import get_land_cover_classes
        names = list(get_land_cover_classes("OpenStreetMap").values())
        assert set(names) == set(OSM_CLASS_TO_PALM.keys())

    def test_spec_pinned_codes(self):
        assert OSM_CLASS_TO_PALM["Road"] == ("pavement", 1)
        assert OSM_CLASS_TO_PALM["Developed space"] == ("pavement", 2)
        assert OSM_CLASS_TO_PALM["Tree"] == ("vegetation", 7)
        assert OSM_CLASS_TO_PALM["Shrub"] == ("vegetation", 16)
        assert OSM_CLASS_TO_PALM["Wet land"] == ("vegetation", 14)
        assert OSM_CLASS_TO_PALM["Snow and ice"] == ("vegetation", 13)
        assert OSM_CLASS_TO_PALM["Water"] == ("water", 1)
        assert OSM_CLASS_TO_PALM["Building"] == ("building", None)
        assert URBANWATCH_CLASS_TO_PALM["Sea"] == ("water", 3)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `& "C:\Users\kunih\miniconda3\Scripts\conda.exe" run -n voxcity python -m pytest tests/test_exporter_palm.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'voxcity.exporter.palm'`

- [ ] **Step 3: Create the module with constants and tables**

Create `src/voxcity/exporter/palm.py`:

```python
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
from pyproj import Transformer

from ..geoprocessor.utils import compute_rotation_angle, normalize_rectangle_vertices
from ..models import VoxCity
from ..utils.lc import get_land_cover_classes
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `& "C:\Users\kunih\miniconda3\Scripts\conda.exe" run -n voxcity python -m pytest tests/test_exporter_palm.py -v`
Expected: PASS (all TestMappingTables)

- [ ] **Step 5: Commit**

```powershell
git add src/voxcity/exporter/palm.py tests/test_exporter_palm.py
git commit -m "feat(palm): PALM exporter scaffold with land-cover mapping tables"
```

---

### Task 2: Source resolution and index map

**Files:**
- Modify: `src/voxcity/exporter/palm.py` (append after tables)
- Modify: `tests/test_exporter_palm.py` (append)

- [ ] **Step 1: Write failing tests**

Append to `tests/test_exporter_palm.py`:

```python
from voxcity.exporter.palm import (
    _get_source_name_mapping,
    _build_index_to_palm_map,
    DEFAULT_ASSIGNMENT,
)


class TestSourceResolution:
    def test_known_sources(self):
        assert _get_source_name_mapping('OpenStreetMap') is OSM_CLASS_TO_PALM
        assert _get_source_name_mapping('Standard') is OSM_CLASS_TO_PALM
        assert _get_source_name_mapping('Urbanwatch') is URBANWATCH_CLASS_TO_PALM
        assert _get_source_name_mapping('OpenEarthMapJapan') is OEMJ_CLASS_TO_PALM
        assert _get_source_name_mapping('ESA WorldCover') is ESA_CLASS_TO_PALM
        assert _get_source_name_mapping('ESRI 10m Annual Land Cover') is ESRI_CLASS_TO_PALM
        assert _get_source_name_mapping('Dynamic World V1') is DYNAMIC_WORLD_CLASS_TO_PALM

    def test_unknown_source_falls_back_to_osm(self):
        assert _get_source_name_mapping('SomeUnknownSource') is OSM_CLASS_TO_PALM

    def test_index_map_osm(self):
        index_to_assignment, class_names = _build_index_to_palm_map('OpenStreetMap')
        # OSM raw order: 0 Bareland ... 5 Tree ... 8 Water, 11 Road, 12 Building
        assert class_names[0] == 'Bareland'
        assert index_to_assignment[0] == ('vegetation', 1)
        assert index_to_assignment[5] == ('vegetation', 7)
        assert index_to_assignment[8] == ('water', 1)
        assert index_to_assignment[11] == ('pavement', 1)
        assert index_to_assignment[12] == ('building', None)

    def test_index_map_unknown_source_uses_osm_names(self):
        index_to_assignment, class_names = _build_index_to_palm_map('Nope')
        assert index_to_assignment[0] == ('vegetation', 1)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `& "C:\Users\kunih\miniconda3\Scripts\conda.exe" run -n voxcity python -m pytest tests/test_exporter_palm.py::TestSourceResolution -v`
Expected: FAIL — ImportError (`_get_source_name_mapping` not defined)

- [ ] **Step 3: Implement**

Append to `src/voxcity/exporter/palm.py`:

```python
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
```

`get_land_cover_classes` is imported once at module scope (`from ..utils.lc import get_land_cover_classes`, alongside the module's other imports) rather than locally inside this function — it defers nothing and breaks no import cycle, since `voxcity/utils/__init__.py` already does `from .lc import *` and `palm.py` already imports `..utils.logging`, so `utils.lc` is always already loaded by the time this module executes.

Note: `get_land_cover_classes` does **not** return `None` for an unknown source — verified by execution, it raises `UnboundLocalError` (`src/voxcity/utils/lc.py` is an if/elif chain with no else branch and no initializer for the local it returns). The `try/except` above is written against that real failure mode rather than a defensive `None` check, and is kept as a `try/except` (rather than checking source-name membership) so it stays correct if `lc.py` is later changed to return `None` or raise a different exception.

- [ ] **Step 4: Run tests to verify they pass**

Run: `& "C:\Users\kunih\miniconda3\Scripts\conda.exe" run -n voxcity python -m pytest tests/test_exporter_palm.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```powershell
git add src/voxcity/exporter/palm.py tests/test_exporter_palm.py
git commit -m "feat(palm): per-source raw-index to PALM assignment map"
```

---

### Task 3: Georeference builder

**Files:**
- Modify: `src/voxcity/exporter/palm.py` (append)
- Modify: `tests/test_exporter_palm.py` (append)

- [ ] **Step 1: Write failing tests**

Append to `tests/test_exporter_palm.py`:

```python
from voxcity.exporter.palm import _build_georeference

# Axis-aligned rectangle near Tokyo, canonical [SW, NW, NE, SE] (lon, lat)
RECT = [(139.7000, 35.6800), (139.7000, 35.6809),
        (139.7011, 35.6809), (139.7011, 35.6800)]


class TestGeoreference:
    def test_tokyo_rectangle(self):
        geo = _build_georeference(RECT)
        assert geo["origin_lon"] == pytest.approx(139.7000)
        assert geo["origin_lat"] == pytest.approx(35.6800)
        assert geo["epsg"] == 32654  # UTM zone 54N
        assert abs(geo["rotation_angle"]) < 0.5
        # Tokyo UTM54N easting ~ 380-390 km, northing ~ 3.94-3.96 Mm
        assert 300_000 < geo["origin_x"] < 500_000
        assert 3_900_000 < geo["origin_y"] < 4_000_000

    def test_southern_hemisphere_epsg(self):
        rect = [(151.2, -33.87), (151.2, -33.86), (151.21, -33.86), (151.21, -33.87)]
        geo = _build_georeference(rect)
        assert geo["epsg"] == 32756  # Sydney, UTM zone 56S
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `& "C:\Users\kunih\miniconda3\Scripts\conda.exe" run -n voxcity python -m pytest tests/test_exporter_palm.py::TestGeoreference -v`
Expected: FAIL — ImportError

- [ ] **Step 3: Implement**

Append to `src/voxcity/exporter/palm.py`:

```python
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
```

`pyproj.Transformer` and `..geoprocessor.utils.{compute_rotation_angle, normalize_rectangle_vertices}` are imported once at module scope (alongside the module's other imports, see Task 1) rather than locally inside this function — same rationale as the Task 2 hoist: they defer nothing, and `geoprocessor.utils` does not import from `exporter`, so there is no cycle.

- [ ] **Step 4: Run tests to verify they pass**

Run: `& "C:\Users\kunih\miniconda3\Scripts\conda.exe" run -n voxcity python -m pytest tests/test_exporter_palm.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```powershell
git add src/voxcity/exporter/palm.py tests/test_exporter_palm.py
git commit -m "feat(palm): georeference builder (UTM origin, rotation angle)"
```

---

### Task 4: Terrain builder

**Files:**
- Modify: `src/voxcity/exporter/palm.py` (append)
- Modify: `tests/test_exporter_palm.py` (append)

- [ ] **Step 1: Write failing tests**

Append to `tests/test_exporter_palm.py`:

```python
from voxcity.exporter.palm import _build_zt


class TestBuildZt:
    def test_shift_to_zero_min(self):
        dem = np.array([[3.0, 5.0], [4.0, 7.0]])
        zt, origin_z = _build_zt(dem)
        assert zt.dtype == np.float32
        assert origin_z == pytest.approx(3.0)
        assert zt.min() == pytest.approx(0.0)
        assert zt[1, 1] == pytest.approx(4.0)

    def test_nan_replaced_with_min_before_shift(self):
        dem = np.array([[np.nan, 5.0], [4.0, 7.0]])
        zt, origin_z = _build_zt(dem)
        assert origin_z == pytest.approx(4.0)
        assert zt[0, 0] == pytest.approx(0.0)
        assert np.isfinite(zt).all()

    def test_all_nan_becomes_flat_zero(self):
        dem = np.full((2, 2), np.nan)
        zt, origin_z = _build_zt(dem)
        assert origin_z == 0.0
        assert (zt == 0.0).all()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `& "C:\Users\kunih\miniconda3\Scripts\conda.exe" run -n voxcity python -m pytest tests/test_exporter_palm.py::TestBuildZt -v`
Expected: FAIL — ImportError

- [ ] **Step 3: Implement**

Append to `src/voxcity/exporter/palm.py`:

```python
def _build_zt(dem_grid):
    """Terrain height, shifted so its minimum is exactly 0.

    Returns (zt float32 (y, x), origin_z) where origin_z is the subtracted
    minimum (recorded as the PIDS global attribute ``origin_z``). NaN/inf
    cells take the grid minimum (i.e. 0 after shifting).
    """
    zt = np.asarray(dem_grid, dtype=np.float64).copy()
    finite = np.isfinite(zt)
    if not finite.any():
        return np.zeros(zt.shape, dtype=np.float32), 0.0
    min_val = float(zt[finite].min())
    zt[~finite] = min_val
    zt -= min_val
    return zt.astype(np.float32), min_val
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `& "C:\Users\kunih\miniconda3\Scripts\conda.exe" run -n voxcity python -m pytest tests/test_exporter_palm.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```powershell
git add src/voxcity/exporter/palm.py tests/test_exporter_palm.py
git commit -m "feat(palm): terrain (zt) builder with zero-min shift"
```

---

### Task 5: Building builders (2D, IDs, type, 3D)

**Files:**
- Modify: `src/voxcity/exporter/palm.py` (append)
- Modify: `tests/test_exporter_palm.py` (append)

- [ ] **Step 1: Write failing tests**

Append to `tests/test_exporter_palm.py`:

```python
from voxcity.exporter.palm import (
    _build_building_mask,
    _build_buildings,
    _build_buildings_3d,
    _has_elevated_segments,
)


def _empty_min_heights(ny, nx):
    mh = np.empty((ny, nx), dtype=object)
    for i in range(ny):
        for j in range(nx):
            mh[i, j] = []
    return mh


class TestBuildBuildings:
    def test_basic_fields(self):
        heights = np.array([[0.0, 10.0], [6.0, np.nan]])
        ids = np.array([[0, 7], [0, 0]])
        mask, segment_top_m = _build_building_mask(heights, None)
        b2d, bid, btype = _build_buildings(
            heights, ids, building_type=3, mask=mask, segment_top_m=segment_top_m
        )
        assert b2d.dtype == np.float32
        assert b2d[0, 0] == np.float32(FILL_FLOAT)
        assert b2d[1, 1] == np.float32(FILL_FLOAT)  # NaN height -> no building
        assert b2d[0, 1] == np.float32(10.0)
        assert bid.dtype == np.int32
        assert bid[0, 1] == 7
        # (1,0) has height but id 0 -> generated id > existing max
        assert bid[1, 0] > 7
        assert btype.dtype == np.int8
        assert btype[0, 1] == 3
        assert btype[0, 0] == FILL_BYTE

    def test_none_ids_all_generated(self):
        heights = np.array([[5.0, 0.0]])
        mask, segment_top_m = _build_building_mask(heights, None)
        b2d, bid, btype = _build_buildings(
            heights, None, building_type=2, mask=mask, segment_top_m=segment_top_m
        )
        assert bid[0, 0] >= 1
        assert bid[0, 1] == FILL_INT


class TestElevatedSegments:
    def test_no_segments(self):
        assert not _has_elevated_segments(None, 2.0)
        assert not _has_elevated_segments(_empty_min_heights(2, 2), 2.0)

    def test_ground_based_only(self):
        mh = _empty_min_heights(2, 2)
        mh[0, 0] = [[0.0, 10.0]]
        assert not _has_elevated_segments(mh, 2.0)

    def test_elevated_segment_detected(self):
        mh = _empty_min_heights(2, 2)
        mh[0, 0] = [[4.0, 10.0]]  # starts 2 voxels above ground
        assert _has_elevated_segments(mh, 2.0)

    def test_sub_voxel_min_rounds_to_ground(self):
        mh = _empty_min_heights(1, 1)
        mh[0, 0] = [[0.4, 10.0]]  # rounds to level 0 with meshsize 2.0
        assert not _has_elevated_segments(mh, 2.0)


class TestBuildBuildings3d:
    def test_overhang_column(self):
        heights = np.array([[10.0, 0.0]])
        mh = _empty_min_heights(1, 2)
        mh[0, 0] = [[4.0, 10.0]]
        mask, segment_top_m = _build_building_mask(heights, mh)
        b3d = _build_buildings_3d(
            heights, mh, meshsize=2.0, mask=mask, segment_top_m=segment_top_m
        )
        assert b3d.dtype == np.int8
        assert b3d.shape == (5, 1, 2)  # nz = round(10/2)
        col = b3d[:, 0, 0]
        assert list(col) == [0, 0, 1, 1, 1]  # levels 2..4 filled
        assert (b3d[:, 0, 1] == 0).all()

    def test_extrusion_fallback_when_no_segments(self):
        heights = np.array([[6.0]])
        mh = _empty_min_heights(1, 1)
        mask, segment_top_m = _build_building_mask(heights, mh)
        b3d = _build_buildings_3d(
            heights, mh, meshsize=2.0, mask=mask, segment_top_m=segment_top_m
        )
        assert list(b3d[:, 0, 0]) == [1, 1, 1]
```

Note: this Step-1 block uses the shared building-presence mask signature
(`mask`/`segment_top_m` from `_build_building_mask`) landed by
`6a94e9b fix(palm): unify the building-presence mask across LOD1 and LOD2
outputs`, after this task was originally implemented. Kept in sync with
the real test file (unlike other tasks' Step-1 blocks, which are
historical snapshots) because the pre-unification signature no longer
exists at all -- a snapshot using it would not run.

- [ ] **Step 2: Run tests to verify they fail**

Run: `& "C:\Users\kunih\miniconda3\Scripts\conda.exe" run -n voxcity python -m pytest tests/test_exporter_palm.py::TestBuildBuildings tests/test_exporter_palm.py::TestElevatedSegments tests/test_exporter_palm.py::TestBuildBuildings3d -v`
Expected: FAIL — ImportError

- [ ] **Step 3: Implement**

Append to `src/voxcity/exporter/palm.py`:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `& "C:\Users\kunih\miniconda3\Scripts\conda.exe" run -n voxcity python -m pytest tests/test_exporter_palm.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```powershell
git add src/voxcity/exporter/palm.py tests/test_exporter_palm.py
git commit -m "feat(palm): building builders (LOD1 fields, LOD2 mask, overhang detection)"
```

---

### Task 6: Surface classification builder

**Files:**
- Modify: `src/voxcity/exporter/palm.py` (append)
- Modify: `tests/test_exporter_palm.py` (append)

- [ ] **Step 1: Write failing tests**

Append to `tests/test_exporter_palm.py`:

```python
from voxcity.exporter.palm import _build_surface_types, IDX_VEGETATION, IDX_PAVEMENT, IDX_WATER


class TestBuildSurfaceTypes:
    def _run(self):
        # OSM raw indices: 0 Bareland, 8 Water, 11 Road, 12 Building
        lc = np.array([[0, 8], [11, 12], [0, 12]])
        building_mask = np.array([[False, False], [False, True], [False, False]])
        canopy_mask = np.array([[True, False], [False, False], [False, False]])
        # (1,1): building; (2,1): land cover says Building but no height
        return _build_surface_types(
            lc, 'OpenStreetMap', building_mask, canopy_mask,
            under_tree_vegetation_type=3, soil_type_code=3,
        )

    def test_categories(self):
        f = self._run()
        veg, pav, wat = f["vegetation_type"], f["pavement_type"], f["water_type"]
        assert veg[0, 0] == 3          # canopy overrides Bareland
        assert wat[0, 1] == 1          # Water -> lake
        assert pav[1, 0] == 1          # Road -> asphalt
        assert veg[2, 0] == 1          # Bareland -> bare soil
        assert pav[2, 1] == 2          # Building class w/o height -> concrete
        # building cell: everything fill
        assert veg[1, 1] == FILL_BYTE
        assert pav[1, 1] == FILL_BYTE
        assert wat[1, 1] == FILL_BYTE

    def test_exclusivity(self):
        f = self._run()
        set_count = sum(
            (f[k] != FILL_BYTE).astype(int)
            for k in ("vegetation_type", "pavement_type", "water_type")
        )
        building_mask = np.array([[False, False], [False, True], [False, False]])
        assert (set_count[~building_mask] == 1).all()
        assert (set_count[building_mask] == 0).all()

    def test_soil_and_fraction(self):
        f = self._run()
        soil, sf = f["soil_type"], f["surface_fraction"]
        assert soil[2, 0] == 3                      # vegetation -> soil
        assert soil[1, 0] == 3                      # pavement -> soil
        assert soil[0, 1] == FILL_BYTE              # water -> no soil
        assert soil[1, 1] == FILL_BYTE              # building -> no soil
        assert sf[IDX_VEGETATION, 2, 0] == 1.0
        assert sf[IDX_PAVEMENT, 2, 0] == 0.0
        assert sf[IDX_WATER, 0, 1] == 1.0
        assert sf[:, 1, 1].tolist() == [np.float32(FILL_FLOAT)] * 3
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `& "C:\Users\kunih\miniconda3\Scripts\conda.exe" run -n voxcity python -m pytest tests/test_exporter_palm.py::TestBuildSurfaceTypes -v`
Expected: FAIL — ImportError

- [ ] **Step 3: Implement**

Append to `src/voxcity/exporter/palm.py`:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `& "C:\Users\kunih\miniconda3\Scripts\conda.exe" run -n voxcity python -m pytest tests/test_exporter_palm.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```powershell
git add src/voxcity/exporter/palm.py tests/test_exporter_palm.py
git commit -m "feat(palm): mutually exclusive surface classification builder"
```

---

### Task 7: LAD builder

**Files:**
- Modify: `src/voxcity/exporter/palm.py` (append)
- Modify: `tests/test_exporter_palm.py` (append)

- [ ] **Step 1: Write failing tests**

Append to `tests/test_exporter_palm.py`:

```python
from voxcity.exporter.palm import _build_lad


class TestBuildLad:
    def test_crown_placement(self):
        top = np.array([[6.0, 0.0]])
        bottom = np.array([[1.8, 0.0]])
        building = np.zeros((1, 2), dtype=bool)
        lad, zlad = _build_lad(top, bottom, meshsize=2.0, lad_value=1.0,
                               building_mask=building)
        # zlad: surface 0, then centres 1, 3, 5 (max top 6, dz 2)
        assert zlad.tolist() == [0.0, 1.0, 3.0, 5.0]
        col = lad[:, 0, 0]
        assert col[0] == 0.0     # surface, below crown
        assert col[1] == 0.0     # z=1 < bottom 1.8
        assert col[2] == 1.0     # z=3 in crown
        assert col[3] == 1.0     # z=5 in crown
        # non-vegetated column: all fill
        assert (lad[:, 0, 1] == np.float32(FILL_FLOAT)).all()

    def test_building_clears_canopy(self):
        top = np.array([[6.0]])
        bottom = np.array([[1.8]])
        building = np.array([[True]])
        lad, zlad = _build_lad(top, bottom, meshsize=2.0, lad_value=1.0,
                               building_mask=building)
        assert lad is None and zlad is None  # nothing left to resolve

    def test_thin_crown_gets_one_layer(self):
        # bottom == top == 2.0 with dz 2: no zlad level in [2, 2] except... none
        # (levels are 0, 1) -> topmost level at/below top must be forced
        top = np.array([[2.0]])
        bottom = np.array([[2.0]])
        lad, zlad = _build_lad(top, bottom, meshsize=2.0, lad_value=0.5,
                               building_mask=np.array([[False]]))
        assert (lad[:, 0, 0] == 0.5).sum() == 1

    def test_no_canopy_returns_none(self):
        lad, zlad = _build_lad(np.zeros((2, 2)), np.zeros((2, 2)),
                               meshsize=1.0, lad_value=1.0,
                               building_mask=np.zeros((2, 2), dtype=bool))
        assert lad is None and zlad is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `& "C:\Users\kunih\miniconda3\Scripts\conda.exe" run -n voxcity python -m pytest tests/test_exporter_palm.py::TestBuildLad -v`
Expected: FAIL — ImportError

- [ ] **Step 3: Implement**

Append to `src/voxcity/exporter/palm.py`:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `& "C:\Users\kunih\miniconda3\Scripts\conda.exe" run -n voxcity python -m pytest tests/test_exporter_palm.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```powershell
git add src/voxcity/exporter/palm.py tests/test_exporter_palm.py
git commit -m "feat(palm): leaf area density (lad/zlad) builder"
```

---

### Task 8: Consistency validator

**Files:**
- Modify: `src/voxcity/exporter/palm.py` (append)
- Modify: `tests/test_exporter_palm.py` (append)

- [ ] **Step 1: Write failing tests**

Append to `tests/test_exporter_palm.py`. As shipped, `_validate_static_fields`
is added to the single consolidated `from voxcity.exporter.palm import (...)`
block at the top of the file rather than a standalone import line here — same
"all test imports at the top" convention already established by earlier
tasks (see Task 2's note):

```python
def _valid_fields():
    """Minimal internally consistent 1x2 field set: veg cell + building cell."""
    zt = np.array([[0.0, 1.0]], dtype=np.float32)
    b2d = np.array([[FILL_FLOAT, 8.0]], dtype=np.float32)
    bid = np.array([[FILL_INT, 5]], dtype=np.int32)
    btype = np.array([[FILL_BYTE, 3]], dtype=np.int8)
    veg = np.array([[3, FILL_BYTE]], dtype=np.int8)
    pav = np.full((1, 2), FILL_BYTE, dtype=np.int8)
    wat = np.full((1, 2), FILL_BYTE, dtype=np.int8)
    soil = np.array([[3, FILL_BYTE]], dtype=np.int8)
    sf = np.full((3, 1, 2), FILL_FLOAT, dtype=np.float32)
    sf[:, 0, 0] = [1.0, 0.0, 0.0]
    return {
        "zt": zt, "buildings_2d": b2d, "building_id": bid,
        "building_type": btype, "vegetation_type": veg,
        "pavement_type": pav, "water_type": wat, "soil_type": soil,
        "surface_fraction": sf, "buildings_3d": None, "lad": None,
    }


class TestValidator:
    def test_valid_fields_pass(self):
        _validate_static_fields(_valid_fields())  # no raise

    def test_zt_min_nonzero_fails(self):
        f = _valid_fields()
        f["zt"] = f["zt"] + 1.0
        with pytest.raises(RuntimeError, match="zt"):
            _validate_static_fields(f)

    def test_two_surface_types_fail(self):
        f = _valid_fields()
        f["pavement_type"][0, 0] = 1  # cell now veg AND pavement
        with pytest.raises(RuntimeError, match="exactly one"):
            _validate_static_fields(f)

    def test_surface_type_on_building_fails(self):
        f = _valid_fields()
        f["vegetation_type"][0, 1] = 3
        with pytest.raises(RuntimeError, match="building cells"):
            _validate_static_fields(f)

    def test_missing_soil_fails(self):
        f = _valid_fields()
        f["soil_type"][0, 0] = FILL_BYTE
        with pytest.raises(RuntimeError, match="soil_type"):
            _validate_static_fields(f)

    def test_bad_fraction_sum_fails(self):
        f = _valid_fields()
        f["surface_fraction"][0, 0, 0] = 0.5
        with pytest.raises(RuntimeError, match="surface_fraction"):
            _validate_static_fields(f)

    def test_building_without_id_fails(self):
        f = _valid_fields()
        f["building_id"][0, 1] = FILL_INT
        with pytest.raises(RuntimeError, match="building_id"):
            _validate_static_fields(f)

    def test_out_of_range_code_fails(self):
        f = _valid_fields()
        f["vegetation_type"][0, 0] = 99
        with pytest.raises(RuntimeError, match="vegetation_type"):
            _validate_static_fields(f)

    def test_buildings_3d_column_without_2d_fails(self):
        f = _valid_fields()
        b3d = np.zeros((2, 1, 2), dtype=np.int8)
        b3d[0, 0, 0] = 1  # column (0,0) has no buildings_2d
        f["buildings_3d"] = b3d
        with pytest.raises(RuntimeError, match="buildings_3d"):
            _validate_static_fields(f)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `& "C:\Users\kunih\miniconda3\Scripts\conda.exe" run -n voxcity python -m pytest tests/test_exporter_palm.py::TestValidator -v`
Expected: FAIL — ImportError

- [ ] **Step 3: Implement**

Superseded during implementation (see the "REQUIRED CHANGES" this task actually
shipped with): the LOD1/LOD2 presence check below is bidirectional across all
four of buildings_3d/buildings_2d/building_id/building_type (not the
one-directional `col & ~bld` check originally sketched here), and every
field's dtype is asserted explicitly. Append to `src/voxcity/exporter/palm.py`:

```python
# dtype the writer must produce for each field (see _write_static_driver):
# an int8<->int16 slip here is a silently non-PIDS-conformant file (`byte`
# vs `short`), so this is asserted explicitly rather than left to the
# writer's netCDF4 calls to fail loudly (they wouldn't -- netCDF4 happily
# casts on write).
_REQUIRED_DTYPES = {
    "zt": np.float32,
    "buildings_2d": np.float32,
    "surface_fraction": np.float32,
    "building_id": np.int32,
    "building_type": np.int8,
    "vegetation_type": np.int8,
    "pavement_type": np.int8,
    "water_type": np.int8,
    "soil_type": np.int8,
}
# Optional fields (may be None): checked only when present.
_OPTIONAL_DTYPES = {
    "buildings_3d": np.int8,
    "lad": np.float32,
}


def _validate_static_fields(fields):
    """Enforce PALM's documented runtime consistency rules before writing.

    A violation is an exporter bug (the builders should never produce one),
    so this raises RuntimeError rather than silently correcting.
    """
    problems = []

    for name, dtype in _REQUIRED_DTYPES.items():
        arr = fields[name]
        if arr.dtype != dtype:
            problems.append(
                f"{name} has dtype {arr.dtype}, expected {np.dtype(dtype)}"
            )
    for name, dtype in _OPTIONAL_DTYPES.items():
        arr = fields.get(name)
        if arr is not None and arr.dtype != dtype:
            problems.append(
                f"{name} has dtype {arr.dtype}, expected {np.dtype(dtype)}"
            )

    zt = fields["zt"]
    if not np.isfinite(zt).all():
        problems.append("zt contains non-finite values")
    elif abs(float(zt.min())) > 1e-4:
        problems.append(f"zt minimum is {float(zt.min())}, expected 0")

    veg = fields["vegetation_type"] != FILL_BYTE
    pav = fields["pavement_type"] != FILL_BYTE
    wat = fields["water_type"] != FILL_BYTE
    bld = fields["buildings_2d"] != np.float32(FILL_FLOAT)

    n_types = veg.astype(int) + pav.astype(int) + wat.astype(int)
    if (n_types[~bld] != 1).any():
        problems.append("non-building cells must have exactly one surface type")
    if (n_types[bld] != 0).any():
        problems.append("building cells must have no surface type")

    soil = fields["soil_type"] != FILL_BYTE
    if ((veg | pav) != soil).any():
        problems.append("soil_type must be set exactly where vegetation or pavement is")

    sf = fields["surface_fraction"]
    classified = veg | pav | wat
    if classified.any() and not np.allclose(sf.sum(axis=0)[classified], 1.0):
        problems.append("surface_fraction must sum to 1 on classified cells")

    if (bld & (fields["building_id"] == FILL_INT)).any():
        problems.append("building cells missing building_id")
    if (bld & (fields["building_type"] == FILL_BYTE)).any():
        problems.append("building cells missing building_type")

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
        has_id = fields["building_id"] != FILL_INT
        has_type = fields["building_type"] != FILL_BYTE
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
```

Note: the shipped test suite adds several classes beyond the Step-1 block
above (kept in sync because their pre-strengthening equivalents would not
run against the real dtype/presence checks): `TestValidatorDtypes`
(parametrized dtype-mismatch cases for every required and optional field),
`TestValidatorBuildings3dPresence` (one isolated test per one of the six
bidirectional presence directions, via a `_presence_fields(bld, col,
has_id, has_type)` helper), `TestValidatorAgainstRealBuilders` (builds
fields from the real `_build_*` functions for a small synthetic city and
asserts the validator accepts them, proving builders and validator agree),
and `TestValidatorLadRange` (the `lad` negative/non-finite rule had no
coverage in the original Step-1 block).

- [ ] **Step 4: Run tests to verify they pass**

Run: `& "C:\Users\kunih\miniconda3\Scripts\conda.exe" run -n voxcity python -m pytest tests/test_exporter_palm.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```powershell
git add src/voxcity/exporter/palm.py tests/test_exporter_palm.py
git commit -m "feat(palm): PIDS/runtime consistency validator"
```

---

### Task 9: NetCDF writer

**Files:**
- Modify: `src/voxcity/exporter/palm.py` (append)
- Modify: `tests/test_exporter_palm.py` (append)

- [ ] **Step 1: Write failing tests**

Append to `tests/test_exporter_palm.py`. As shipped, `_write_static_driver`
and `from netCDF4 import Dataset` are both added to the file's top-level
imports rather than imported locally here (same convention as Task 8's
Step 1; the plan's own per-test `from netCDF4 import Dataset` lines inside
`test_dims_dtypes_and_fills`/`test_optional_vars_absent` below are likewise
consolidated to the top of the file, not left as local imports):

```python
class TestWriter:
    def _write(self, tmp_path, with_3d=False, with_lad=True):
        fields = _valid_fields()
        if with_3d:
            b3d = np.zeros((2, 1, 2), dtype=np.int8)
            b3d[:, 0, 1] = 1
            fields["buildings_3d"] = b3d
        if with_lad:
            fields["lad"] = np.full((2, 1, 2), FILL_FLOAT, dtype=np.float32)
            fields["lad"][:, 0, 0] = [0.0, 1.0]
        coords = {
            "x": np.array([1.0, 3.0], dtype=np.float32),
            "y": np.array([1.0], dtype=np.float32),
        }
        if with_3d:
            coords["z"] = np.array([1.0, 3.0], dtype=np.float32)
        if with_lad:
            coords["zlad"] = np.array([0.0, 1.0], dtype=np.float32)
        attrs = {
            "Conventions": "CF-1.7",
            "origin_time": "2000-01-01 00:00:00 +00",
            "origin_lon": 139.7, "origin_lat": 35.68,
            "origin_x": 382000.0, "origin_y": 3949000.0,
            "origin_z": 0.0, "rotation_angle": 0.0,
        }
        path = tmp_path / "dom_static"
        _write_static_driver(path, fields, coords, attrs)
        return path

    def test_dims_dtypes_and_fills(self, tmp_path):
        path = self._write(tmp_path, with_3d=True)
        with Dataset(path) as nc:
            nc.set_auto_mask(False)
            assert nc.dimensions["x"].size == 2
            assert nc.dimensions["y"].size == 1
            assert nc.dimensions["z"].size == 2
            assert nc.dimensions["zlad"].size == 2
            assert nc.dimensions["nsurface_fraction"].size == 3
            zt = nc.variables["zt"]
            assert zt.dtype == np.float32
            assert zt._FillValue == np.float32(FILL_FLOAT)
            b2d = nc.variables["buildings_2d"]
            assert b2d.lod == 1
            assert b2d[0, 1] == np.float32(8.0)
            bid = nc.variables["building_id"]
            assert bid.dtype == np.int32
            assert bid._FillValue == FILL_INT
            for name in ("building_type", "vegetation_type", "pavement_type",
                         "water_type", "soil_type"):
                var = nc.variables[name]
                assert var.dtype == np.int8
                assert var._FillValue == FILL_BYTE
            b3d = nc.variables["buildings_3d"]
            assert b3d.lod == 2
            assert b3d.dimensions == ("z", "y", "x")
            lad = nc.variables["lad"]
            assert lad.dimensions == ("zlad", "y", "x")
            sf = nc.variables["surface_fraction"]
            assert sf.dimensions == ("nsurface_fraction", "y", "x")
            assert nc.rotation_angle == 0.0
            assert nc.origin_lon == pytest.approx(139.7)

    def test_optional_vars_absent(self, tmp_path):
        path = self._write(tmp_path, with_3d=False, with_lad=False)
        with Dataset(path) as nc:
            assert "buildings_3d" not in nc.variables
            assert "lad" not in nc.variables
            assert "z" not in nc.dimensions
            assert "zlad" not in nc.dimensions
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `& "C:\Users\kunih\miniconda3\Scripts\conda.exe" run -n voxcity python -m pytest tests/test_exporter_palm.py::TestWriter -v`
Expected: FAIL — ImportError

- [ ] **Step 3: Implement**

Superseded during implementation: `from netCDF4 import Dataset` is hoisted to
the module header (alongside the other third-party imports), not imported
locally inside this function as first sketched here. `netCDF4` is a hard
dependency (`pyproject.toml`: `netCDF4 = "*"`) with no optional-import guard
anywhere in this package, so there is no import-time cost to avoid and no
existing deferred-import pattern to match: `voxcity/exporter/netcdf.py` never
imports `netCDF4` directly at all (it defers the *optional* `xarray`
dependency behind a `try/except`, and only ever names `netcdf4` as a string
passed to `xr.Dataset.to_netcdf(engine=...)`), so it is not the precedent this
task originally assumed it was. Append to `src/voxcity/exporter/palm.py`:

```python
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
            var = nc.createVariable(name, dtype, dims, fill_value=fill)
            var.set_auto_mask(False)
            var[:] = data
            for k, v in var_attrs.items():
                setattr(var, k, v)

        yx = ("y", "x")
        write("zt", fields["zt"], yx, "f4", np.float32(FILL_FLOAT),
              units="m", long_name="terrain height")
        write("buildings_2d", fields["buildings_2d"], yx, "f4",
              np.float32(FILL_FLOAT), units="m", long_name="building height",
              lod=np.int32(1))
        write("building_id", fields["building_id"], yx, "i4", np.int32(FILL_INT),
              units="1", long_name="building id numbers")
        write("building_type", fields["building_type"], yx, "b",
              np.int8(FILL_BYTE), units="1", long_name="building type classification")
        write("vegetation_type", fields["vegetation_type"], yx, "b",
              np.int8(FILL_BYTE), units="1", long_name="vegetation type classification")
        write("pavement_type", fields["pavement_type"], yx, "b",
              np.int8(FILL_BYTE), units="1", long_name="pavement type classification")
        write("water_type", fields["water_type"], yx, "b",
              np.int8(FILL_BYTE), units="1", long_name="water type classification")
        write("soil_type", fields["soil_type"], yx, "b",
              np.int8(FILL_BYTE), units="1", long_name="soil type classification",
              lod=np.int32(1))
        write("surface_fraction", fields["surface_fraction"],
              ("nsurface_fraction", "y", "x"), "f4", np.float32(FILL_FLOAT),
              units="1", long_name="surface fraction")
        if fields.get("buildings_3d") is not None:
            write("buildings_3d", fields["buildings_3d"], ("z", "y", "x"), "b",
                  np.int8(FILL_BYTE), units="1",
                  long_name="building structure in 3d", lod=np.int32(2))
        if fields.get("lad") is not None:
            write("lad", fields["lad"], ("zlad", "y", "x"), "f4",
                  np.float32(FILL_FLOAT), units="m2 m-3",
                  long_name="leaf area density")
```

Note, corrected during implementation: `var.set_auto_mask(False)` at *write*
time turned out to have no effect on what is stored in the file -- verified
directly (write a fill value with and without the write-side call, in both
cases the raw file holds the literal fill float; only the *read*-side
`nc.set_auto_mask(False)` determines whether a fill value comes back as a
plain value or as a masked (`--`) entry of a `numpy.ma.MaskedArray`). The
call is kept (harmless, and protective if `data` were ever a masked array),
but it is not the load-bearing line the original text claimed; every test
in this module that inspects fill values disables masking on the *read*
side instead (`nc.set_auto_mask(False)` right after `Dataset(path)`), which
is what actually matters.

The shipped test suite also adds, beyond the Step-1 `TestWriter` block
above: `test_round_trip_values_and_fill_placement` (actual data values and
fill placement, not just structure/dtypes), `test_variable_attributes`
(`lod`/`units`/`long_name` on every variable that carries them, since PALM
reads `lod` to decide how to interpret buildings), `test_reopenable_and_
self_consistent` (`nsurface_fraction`'s own values match `IDX_*`, coordinate
values round-trip, and a second independent `Dataset` handle can open the
file), and a `TestWriterOverwrite` class pinning that re-writing the same
path truncates stale content (a leftover `lad` variable from an earlier
export does not survive a re-export that no longer has canopy) -- the
decision from the design's open question on overwrite behavior: truncate,
matching every other exporter in this package (`cityles.py`'s `open(filename,
'w')` writers).

- [ ] **Step 4: Run tests to verify they pass**

Run: `& "C:\Users\kunih\miniconda3\Scripts\conda.exe" run -n voxcity python -m pytest tests/test_exporter_palm.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```powershell
git add src/voxcity/exporter/palm.py tests/test_exporter_palm.py
git commit -m "feat(palm): PIDS netCDF4 static driver writer"
```

---

### Task 10: Orchestrator, adapter, registration, end-to-end tests

**Files:**
- Modify: `src/voxcity/exporter/palm.py` (append)
- Modify: `src/voxcity/exporter/__init__.py`
- Modify: `tests/test_exporter_palm.py` (append)

- [ ] **Step 1: Write failing end-to-end tests**

Append to `tests/test_exporter_palm.py`:

```python
from voxcity.exporter.palm import export_palm, PalmExporter
from voxcity.models import (
    VoxCity, VoxelGrid, BuildingGrid, LandCoverGrid, DemGrid, CanopyGrid,
    GridMetadata,
)


def make_city(ny=4, nx=4, meshsize=2.0, with_overhang=False, with_canopy=True,
              extras=None):
    """Small synthetic VoxCity in the internal south-up orientation."""
    meta = GridMetadata(crs="EPSG:4326",
                        bounds=(139.7000, 35.6800, 139.7011, 35.6809),
                        meshsize=meshsize)
    heights = np.zeros((ny, nx))
    heights[1, 1] = 10.0
    heights[2, 2] = 6.0
    ids = np.zeros((ny, nx), dtype=int)
    ids[1, 1] = 7
    ids[2, 2] = 8
    min_heights = np.empty((ny, nx), dtype=object)
    for i in range(ny):
        for j in range(nx):
            min_heights[i, j] = []
    min_heights[1, 1] = [[0.0, 10.0]]
    min_heights[2, 2] = [[4.0, 6.0]] if with_overhang else [[0.0, 6.0]]
    land_cover = np.zeros((ny, nx), dtype=int)  # 0 = Bareland
    land_cover[0, 0] = 8    # Water
    land_cover[0, 1] = 11   # Road
    land_cover[0, 2] = 5    # Tree
    land_cover[1, 1] = 12   # Building (has height)
    land_cover[2, 2] = 12
    dem = np.zeros((ny, nx))
    dem[3, :] = 1.5
    canopy_top = np.zeros((ny, nx))
    if with_canopy:
        canopy_top[0, 2] = 6.0
        canopy_top[1, 1] = 5.0  # over the building -> must be cleared
    voxels = VoxelGrid(classes=np.zeros((ny, nx, 8), dtype=np.int32), meta=meta)
    if extras is None:
        extras = {"rectangle_vertices": RECT,
                  "land_cover_source": "OpenStreetMap"}
    return VoxCity(
        voxels=voxels,
        buildings=BuildingGrid(heights=heights, min_heights=min_heights,
                               ids=ids, meta=meta),
        land_cover=LandCoverGrid(classes=land_cover, meta=meta),
        dem=DemGrid(elevation=dem, meta=meta),
        tree_canopy=CanopyGrid(top=canopy_top, bottom=None, meta=meta),
        extras=extras,
    )


class TestExportPalm:
    def test_file_contract(self, tmp_path):
        from netCDF4 import Dataset
        out = export_palm(make_city(), output_directory=str(tmp_path),
                          domain_name="testdom")
        path = Path(out)
        assert path.name == "testdom_static"
        with Dataset(path) as nc:
            nc.set_auto_mask(False)
            assert nc.dimensions["x"].size == 4
            assert nc.dimensions["y"].size == 4
            # coords are cell centres
            assert nc.variables["x"][:].tolist() == [1.0, 3.0, 5.0, 7.0]
            # georeference
            assert nc.origin_lon == pytest.approx(139.7000)
            assert nc.origin_lat == pytest.approx(35.6800)
            assert abs(nc.rotation_angle) < 0.5
            assert nc.origin_z == pytest.approx(0.0)
            assert nc.origin_time == "2000-01-01 00:00:00 +00"
            # terrain: dem row 3 = 1.5
            assert nc.variables["zt"][3, 0] == pytest.approx(1.5)
            # buildings
            assert nc.variables["buildings_2d"][1, 1] == np.float32(10.0)
            assert nc.variables["building_id"][1, 1] == 7
            assert nc.variables["building_type"][1, 1] == 3
            # no overhang -> LOD1 only
            assert "buildings_3d" not in nc.variables
            # surface classification
            assert nc.variables["water_type"][0, 0] == 1
            assert nc.variables["pavement_type"][0, 1] == 1
            assert nc.variables["vegetation_type"][0, 2] == 3   # under canopy
            assert nc.variables["vegetation_type"][1, 1] == FILL_BYTE
            # lad: canopy at (0,2) top 6, default ratio 0.3 -> bottom 1.8
            lad = nc.variables["lad"][:]
            zlad = nc.variables["zlad"][:].tolist()
            assert zlad == [0.0, 1.0, 3.0, 5.0]
            assert lad[2, 0, 2] == np.float32(1.0)
            assert lad[1, 0, 2] == np.float32(0.0)
            # canopy over building cleared
            assert (lad[:, 1, 1] == np.float32(FILL_FLOAT)).all()

    def test_exclusivity_invariant_whole_grid(self, tmp_path):
        from netCDF4 import Dataset
        out = export_palm(make_city(), output_directory=str(tmp_path))
        with Dataset(out) as nc:
            nc.set_auto_mask(False)
            veg = nc.variables["vegetation_type"][:] != FILL_BYTE
            pav = nc.variables["pavement_type"][:] != FILL_BYTE
            wat = nc.variables["water_type"][:] != FILL_BYTE
            bld = nc.variables["buildings_2d"][:] != np.float32(FILL_FLOAT)
            n = veg.astype(int) + pav.astype(int) + wat.astype(int)
            assert (n[~bld] == 1).all()
            assert (n[bld] == 0).all()

    def test_buildings_3d_auto(self, tmp_path):
        from netCDF4 import Dataset
        out = export_palm(make_city(with_overhang=True),
                          output_directory=str(tmp_path))
        with Dataset(out) as nc:
            nc.set_auto_mask(False)
            b3d = nc.variables["buildings_3d"]
            assert b3d.lod == 2
            col = b3d[:, 2, 2]
            assert col[0] == 0 and col[1] == 0  # below the overhang
            assert col[2] == 1                  # levels 2 (4-6 m)

    def test_buildings_3d_forced_off(self, tmp_path):
        from netCDF4 import Dataset
        out = export_palm(make_city(with_overhang=True), buildings_3d=False,
                          output_directory=str(tmp_path))
        with Dataset(out) as nc:
            assert "buildings_3d" not in nc.variables

    def test_trunk_ratio_recompute(self, tmp_path):
        from netCDF4 import Dataset
        out = export_palm(make_city(), trunk_height_ratio=0.5,
                          output_directory=str(tmp_path))
        with Dataset(out) as nc:
            nc.set_auto_mask(False)
            lad = nc.variables["lad"][:]
            # bottom = 3.0 now: z=1 below crown -> 0; z=3 in crown -> 1
            assert lad[1, 0, 2] == np.float32(0.0)
            assert lad[2, 0, 2] == np.float32(1.0)

    def test_missing_rectangle_vertices_raises(self, tmp_path):
        city = make_city(extras={"land_cover_source": "OpenStreetMap"})
        with pytest.raises(ValueError, match="rectangle_vertices"):
            export_palm(city, output_directory=str(tmp_path))

    def test_shape_mismatch_raises(self, tmp_path):
        city = make_city()
        city.dem.elevation = np.zeros((2, 2))
        with pytest.raises(ValueError, match="shape"):
            export_palm(city, output_directory=str(tmp_path))


class TestPalmExporterAdapter:
    def test_export_via_adapter(self, tmp_path):
        exporter = PalmExporter()
        out = exporter.export(make_city(), str(tmp_path), "mydom")
        assert Path(out).name == "mydom_static"
        assert Path(out).exists()

    def test_rejects_non_voxcity(self, tmp_path):
        with pytest.raises(TypeError):
            PalmExporter().export(object(), str(tmp_path), "x")

    def test_registered_in_package(self):
        import voxcity.exporter as ex
        assert "PalmExporter" in ex.__all__
        assert "export_palm" in ex.__all__
        assert ex.PalmExporter is PalmExporter
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `& "C:\Users\kunih\miniconda3\Scripts\conda.exe" run -n voxcity python -m pytest tests/test_exporter_palm.py::TestExportPalm -v`
Expected: FAIL — ImportError (`export_palm` not defined)

- [ ] **Step 3: Implement orchestrator and adapter**

Append to `src/voxcity/exporter/palm.py`:

```python
_TRUNK_RATIO_SENTINEL = object()  # internal sentinel — never pass this


def export_palm(city: VoxCity,
                output_directory: str = "output/palm",
                domain_name: str = "voxcity",
                origin_time: str = "2000-01-01 00:00:00 +00",
                lad: float = 1.0,
                trunk_height_ratio: float = _TRUNK_RATIO_SENTINEL,
                canopy_bottom_height_grid=None,
                building_type: int = 3,
                buildings_3d="auto",
                under_tree_vegetation_type: int = 3,
                soil_type: int = 3,
                land_cover_source: str | None = None,
                author: str | None = None,
                comment: str | None = None,
                **kwargs):
    """Export a VoxCity model to a PALM (PIDS) static driver NetCDF file.

    Parameters
    ----------
    city : VoxCity
        Model instance. ``city.extras['rectangle_vertices']`` is required.
    output_directory : str
        Directory for the output file.
    domain_name : str
        Output file is ``<domain_name>_static``.
    origin_time : str
        PIDS origin_time global attribute.
    lad : float
        Constant leaf area density (m2/m3) inside tree crowns.
    trunk_height_ratio : float, optional
        When explicitly provided, canopy bottoms are always recomputed as
        ``top * ratio`` (same semantics as export_cityles). When omitted,
        an explicit ``canopy_bottom_height_grid`` wins, then
        ``city.tree_canopy.bottom``, then the default ratio 0.3.
    canopy_bottom_height_grid : numpy.ndarray, optional
        Explicit canopy bottom heights; ignored when *trunk_height_ratio*
        is explicitly provided.
    building_type : int
        PALM building type class for all buildings (default 3).
    buildings_3d : bool or "auto"
        True always writes the LOD2 mask, False never, "auto" writes it only
        when the model contains segments starting above ground.
    under_tree_vegetation_type : int
        PALM vegetation type for ground beneath tree canopy (default 3).
    soil_type : int
        PALM soil type wherever vegetation or pavement is set (default 3).
    land_cover_source : str, optional
        Auto-detected from ``city.extras`` when omitted.
    author, comment : str, optional
        Extra global attributes.

    Returns
    -------
    str
        Path to the written ``<domain_name>_static`` file.
    """
    extras = city.extras or {}
    rect = extras.get("rectangle_vertices")
    if rect is None:
        raise ValueError(
            "city.extras['rectangle_vertices'] is required for PALM export "
            "(georeferencing); refusing to write an unreferenced static driver"
        )

    heights = city.buildings.heights
    shape = heights.shape
    named_grids = {
        "land_cover": city.land_cover.classes,
        "dem": city.dem.elevation,
    }
    canopy_top = city.tree_canopy.top if city.tree_canopy is not None else None
    if canopy_top is not None:
        named_grids["canopy_top"] = canopy_top
    for name, grid in named_grids.items():
        if grid.shape != shape:
            raise ValueError(
                f"{name} grid shape {grid.shape} does not match building "
                f"heights shape {shape}"
            )
    if city.voxels.classes.shape[:2] != shape:
        raise ValueError(
            f"voxel grid horizontal shape {city.voxels.classes.shape[:2]} "
            f"does not match building heights shape {shape}"
        )

    meshsize = float(city.voxels.meta.meshsize)
    land_cover_source = land_cover_source or extras.get("land_cover_source", "Standard")
    _logger.info(f"Exporting PALM static driver (source: {land_cover_source})")

    # ── canopy bottom resolution (export_cityles semantics) ──
    user_specified_ratio = trunk_height_ratio is not _TRUNK_RATIO_SENTINEL
    ratio = 0.3 if not user_specified_ratio else float(trunk_height_ratio)
    if canopy_top is None:
        canopy_top = np.zeros(shape, dtype=np.float64)
    if user_specified_ratio:
        canopy_bottom = canopy_top * ratio
    elif canopy_bottom_height_grid is not None:
        canopy_bottom = canopy_bottom_height_grid
    elif city.tree_canopy is not None and city.tree_canopy.bottom is not None:
        canopy_bottom = city.tree_canopy.bottom
    else:
        canopy_bottom = canopy_top * ratio

    # ── build all fields ──
    geo = _build_georeference(rect)
    zt, origin_z = _build_zt(city.dem.elevation)
    min_heights = city.buildings.min_heights
    # One presence predicate feeds both LOD1 and LOD2 so the two cannot
    # disagree (see Task 5): a cell is a building if it has a positive height
    # or LOD2 geometry rising above ground.
    building_mask, segment_top_m = _build_building_mask(heights, min_heights)
    buildings_2d, building_id, building_type_arr = _build_buildings(
        heights, city.buildings.ids, building_type, building_mask, segment_top_m
    )

    b3d = None
    want_3d = buildings_3d is True or (
        buildings_3d == "auto" and _has_elevated_segments(min_heights, meshsize)
    )
    if want_3d and building_mask.any():
        b3d = _build_buildings_3d(
            heights, min_heights, meshsize, building_mask, segment_top_m
        )

    lad_arr, zlad = _build_lad(canopy_top, canopy_bottom, meshsize, lad,
                               building_mask)
    canopy_mask = (
        np.nan_to_num(np.asarray(canopy_top, dtype=np.float64), nan=0.0) > 0.0
    ) & ~building_mask

    surfaces = _build_surface_types(
        city.land_cover.classes, land_cover_source, building_mask, canopy_mask,
        under_tree_vegetation_type, soil_type,
    )

    fields = {
        "zt": zt,
        "buildings_2d": buildings_2d,
        "building_id": building_id,
        "building_type": building_type_arr,
        "buildings_3d": b3d,
        "lad": lad_arr,
        **surfaces,
    }
    _validate_static_fields(fields)

    ny, nx = shape
    coords = {
        "x": ((np.arange(nx) + 0.5) * meshsize).astype(np.float32),
        "y": ((np.arange(ny) + 0.5) * meshsize).astype(np.float32),
    }
    if b3d is not None:
        coords["z"] = ((np.arange(b3d.shape[0]) + 0.5) * meshsize).astype(np.float32)
    if zlad is not None:
        coords["zlad"] = zlad

    attrs = {
        "Conventions": "CF-1.7",
        "title": f"VoxCity PALM static driver: {domain_name}",
        "source": "VoxCity",
        "origin_time": origin_time,
        "creation_time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S +00"),
        "origin_lon": geo["origin_lon"],
        "origin_lat": geo["origin_lat"],
        "origin_x": geo["origin_x"],
        "origin_y": geo["origin_y"],
        "origin_z": float(origin_z),
        "rotation_angle": geo["rotation_angle"],
        "comment": (
            f"origin_x/origin_y in EPSG:{geo['epsg']}"
            + (f"; {comment}" if comment else "")
        ),
        "author": author,
    }

    out_dir = Path(output_directory)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{domain_name}_static"
    _write_static_driver(out_path, fields, coords, attrs)

    _logger.info(
        f"PALM static driver written: {out_path} "
        f"({nx}x{ny} cells, dx={meshsize} m, "
        f"buildings_3d={'yes' if b3d is not None else 'no'}, "
        f"lad={'yes' if lad_arr is not None else 'no'})"
    )
    return str(out_path)


class PalmExporter:
    """Exporter adapter to write a VoxCity model to a PALM static driver."""

    def export(self, obj, output_directory: str, base_filename: str, **kwargs):
        if not isinstance(obj, VoxCity):
            raise TypeError("PalmExporter expects a VoxCity instance")
        return export_palm(
            obj,
            output_directory=output_directory,
            domain_name=base_filename,
            **kwargs,
        )
```

Now that `export_palm` and `PalmExporter` both exist, remove the forward-declaration `# noqa: F822` comment from the module's `__all__` line (added by code review during Unit 2, once `ruff check src` started flagging the forward-declared names as undefined) — it was only needed while those names were undefined, and leaving it in place would silently hide any future genuine `F822`/undefined-name error in this module:

```python
__all__ = ["PalmExporter", "export_palm"]
```

- [ ] **Step 4: Register in the exporter package**

In `src/voxcity/exporter/__init__.py`, add the import after the geotiff import:

```python
from .palm import *
```

and extend `__all__` (after the geotiff entries):

```python
    # palm
    "PalmExporter",
    "export_palm",
```

- [ ] **Step 5: Run the full test module**

Run: `& "C:\Users\kunih\miniconda3\Scripts\conda.exe" run -n voxcity python -m pytest tests/test_exporter_palm.py -v`
Expected: PASS (all tests)

- [ ] **Step 6: Run the full exporter test suite for regressions**

Run: `& "C:\Users\kunih\miniconda3\Scripts\conda.exe" run -n voxcity python -m pytest tests/ -k "exporter" -v`
Expected: PASS (no regressions in other exporters)

- [ ] **Step 7: Commit**

```powershell
git add src/voxcity/exporter/palm.py src/voxcity/exporter/__init__.py tests/test_exporter_palm.py
git commit -m "feat(palm): export_palm orchestrator, PalmExporter adapter, package registration"
```

---

### Task 11: Final verification

**Files:** none

- [ ] **Step 1: Run the whole fast test suite**

Run: `& "C:\Users\kunih\miniconda3\Scripts\conda.exe" run -n voxcity python -m pytest tests/ -m "not slow and not integration and not gee" -q`
Expected: PASS (no regressions anywhere)

- [ ] **Step 2: Verify the module imports cleanly from the package root**

Run: `& "C:\Users\kunih\miniconda3\Scripts\conda.exe" run -n voxcity python -c "from voxcity.exporter import export_palm, PalmExporter; print('ok')"`
Expected: `ok`

- [ ] **Step 3: Use superpowers:requesting-code-review, then superpowers:finishing-a-development-branch**

Follow those skills for review and merge/PR decisions. Remember: the follow-up
end-to-end validation (an actual PALM run on an exported file) is planned but
out of scope for this branch.
