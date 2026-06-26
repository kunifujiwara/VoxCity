# GSI Bare-Earth DEM Downloader Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a downloader that fetches bare-earth DEM tiles from the Geospatial Information Authority of Japan (GSI), mosaics them into an EPSG:3857 GeoTIFF, wires it into `get_dem_grid` as source `"GSI DEM Japan"`, and makes it the auto-selected DEM for Japan.

**Architecture:** New self-contained module `src/voxcity/downloader/gsi.py` mirrors the existing `oemj.py` tile→mosaic→GeoTIFF pattern. GSI elevation tiles (`.txt`, 256×256 CSV of meters, `e`=no-data) live on the XYZ Web-Mercator grid, so the mosaic is written natively in EPSG:3857 (zero resampling). The existing `create_dem_grid_from_geotiff_polygon` consumes the GeoTIFF unchanged (it reads `src.crs`).

**Tech Stack:** Python, `requests`, `numpy`, `osgeo.gdal`/`osr` (GDAL 3.12, matches `oemj.py`), `pytest` with `unittest.mock`. Tests run in the `voxcity` conda env.

**Test command (PowerShell):**
```
& "C:\Users\kunih\miniconda3\Scripts\conda.exe" run -n voxcity python -m pytest tests/test_downloader_gsi.py -v
```

**Reference facts (Web Mercator):**
- `_MERC_MAX = 20037508.342789244` (half the EPSG:3857 world extent, meters)
- World extent = `2 * _MERC_MAX`; tile size at zoom `z` = `world / 2**z`; pixel size = `tile / 256`
- GSI endpoints: `https://cyberjapandata.gsi.go.jp/xyz/{type}/{z}/{x}/{y}.txt`
- Resolution priority: `dem5a` (z15) → `dem5b` (z15) → `dem10b` (z14)

---

## Task 1: Tile math helpers + module skeleton

**Files:**
- Create: `src/voxcity/downloader/gsi.py`
- Test: `tests/test_downloader_gsi.py`

- [ ] **Step 1: Write the failing test**

```python
"""Tests for voxcity.downloader.gsi module."""
import math
import numpy as np
import pytest

from voxcity.downloader.gsi import (
    latlon_to_tile,
    tile_bounds_mercator,
    _bbox_from_rectangle_vertices,
    tile_range_for_bbox,
    _MERC_MAX,
)


class TestLatlonToTile:
    def test_known_tile_zoom15(self):
        # Tsukuba area center
        x, y = latlon_to_tile(36.225, 140.105, 15)
        assert isinstance(x, int) and isinstance(y, int)
        # Sanity: lon 140.105 at z15 -> x within world range
        assert 0 <= x < 2 ** 15
        assert 0 <= y < 2 ** 15

    def test_origin_corner(self):
        # lon -180, lat ~85.05 maps to tile (0, 0)
        x, y = latlon_to_tile(85.0511, -180.0, 0)
        assert (x, y) == (0, 0)


class TestTileBoundsMercator:
    def test_full_world_at_zoom0(self):
        minx, miny, maxx, maxy = tile_bounds_mercator(0, 0, 0)
        assert minx == pytest.approx(-_MERC_MAX)
        assert maxy == pytest.approx(_MERC_MAX)
        assert maxx == pytest.approx(_MERC_MAX)
        assert miny == pytest.approx(-_MERC_MAX)

    def test_pixel_extent_zoom15(self):
        minx, miny, maxx, maxy = tile_bounds_mercator(100, 200, 15)
        tile = (2 * _MERC_MAX) / (2 ** 15)
        assert (maxx - minx) == pytest.approx(tile)
        assert (maxy - miny) == pytest.approx(tile)


class TestBboxAndRange:
    def test_bbox(self):
        verts = [(140.09, 36.21), (140.12, 36.21), (140.12, 36.24), (140.09, 36.24)]
        assert _bbox_from_rectangle_vertices(verts) == (140.09, 36.21, 140.12, 36.24)

    def test_tile_range_orders_min_max(self):
        bbox = (140.09, 36.21, 140.12, 36.24)
        x_min, y_min, x_max, y_max = tile_range_for_bbox(bbox, 15)
        assert x_min <= x_max
        assert y_min <= y_max
```

- [ ] **Step 2: Run test to verify it fails**

Run: `& "C:\Users\kunih\miniconda3\Scripts\conda.exe" run -n voxcity python -m pytest tests/test_downloader_gsi.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'voxcity.downloader.gsi'`

- [ ] **Step 3: Write minimal implementation**

```python
"""
Module for downloading bare-earth DEM (Digital Terrain Model) tiles from the
Geospatial Information Authority of Japan (GSI / 国土地理院) and converting them
into a VoxCity-compatible georeferenced GeoTIFF.

GSI publishes elevation tiles on the standard XYZ Web-Mercator tile grid at
``https://cyberjapandata.gsi.go.jp/xyz/{type}/{z}/{x}/{y}.txt``. Each tile is a
256x256 grid of elevation values (meters) as CSV text, with ``e`` marking
no-data. Resolutions, finest first: dem5a (5 m laser, z15), dem5b (5 m photo,
z15), dem10b (10 m nationwide, z14).

The mosaic is written natively in EPSG:3857 (no resampling); the downstream
``create_dem_grid_from_geotiff_polygon`` reprojects from the file's own CRS.

Example:
    >>> verts = [(140.09, 36.21), (140.12, 36.21), (140.12, 36.24), (140.09, 36.24)]
    >>> save_gsi_dem_as_geotiff(verts, "dem.tif")
"""

import os
import math
import time

import numpy as np
import requests
from osgeo import gdal, osr

__all__ = ["save_gsi_dem_as_geotiff"]

# Half the EPSG:3857 (Web Mercator) world extent, in meters.
_MERC_MAX = 20037508.342789244

GSI_TILE_SIZE = 256
GSI_NODATA = -9999.0

# DEM product types in priority order (finest resolution first).
GSI_DEM_TYPES = [
    {"type": "dem5a", "zoom": 15},   # 5 m mesh, airborne laser survey
    {"type": "dem5b", "zoom": 15},   # 5 m mesh, photogrammetry
    {"type": "dem10b", "zoom": 14},  # 10 m mesh, nationwide
]
_ZOOM_BY_TYPE = {item["type"]: item["zoom"] for item in GSI_DEM_TYPES}


def latlon_to_tile(lat, lon, zoom):
    """Convert lat/lon to integer XYZ Web-Mercator tile indices (x, y)."""
    lat_rad = math.radians(lat)
    n = 2.0 ** zoom
    xtile = int((lon + 180.0) / 360.0 * n)
    ytile = int(
        (1.0 - math.log(math.tan(lat_rad) + (1 / math.cos(lat_rad))) / math.pi)
        / 2.0
        * n
    )
    return xtile, ytile


def tile_bounds_mercator(x, y, zoom):
    """Return (minx, miny, maxx, maxy) of a tile in EPSG:3857 meters."""
    world = 2 * _MERC_MAX
    tile = world / (2.0 ** zoom)
    minx = -_MERC_MAX + x * tile
    maxx = minx + tile
    maxy = _MERC_MAX - y * tile
    miny = maxy - tile
    return (minx, miny, maxx, maxy)


def _bbox_from_rectangle_vertices(rectangle_vertices):
    """Return (min_lon, min_lat, max_lon, max_lat) from (lon, lat) vertices."""
    if not rectangle_vertices:
        raise ValueError("rectangle_vertices is empty")
    lons = [p[0] for p in rectangle_vertices]
    lats = [p[1] for p in rectangle_vertices]
    return (min(lons), min(lats), max(lons), max(lats))


def tile_range_for_bbox(bbox, zoom):
    """Return (x_min, y_min, x_max, y_max) tile indices covering bbox."""
    min_lon, min_lat, max_lon, max_lat = bbox
    x0, y0 = latlon_to_tile(max_lat, min_lon, zoom)  # top-left
    x1, y1 = latlon_to_tile(min_lat, max_lon, zoom)  # bottom-right
    return (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `& "C:\Users\kunih\miniconda3\Scripts\conda.exe" run -n voxcity python -m pytest tests/test_downloader_gsi.py -v`
Expected: PASS (all tests in TestLatlonToTile, TestTileBoundsMercator, TestBboxAndRange)

- [ ] **Step 5: Commit**

```bash
git add src/voxcity/downloader/gsi.py tests/test_downloader_gsi.py
git commit -m "feat(downloader): GSI DEM tile math helpers"
```

---

## Task 2: DEM tile text parser

**Files:**
- Modify: `src/voxcity/downloader/gsi.py`
- Test: `tests/test_downloader_gsi.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_downloader_gsi.py`:

```python
from voxcity.downloader.gsi import parse_dem_tile_text, GSI_NODATA


class TestParseDemTileText:
    def test_full_grid(self):
        # 256 rows x 256 cols of "1.5"
        line = ",".join(["1.5"] * 256)
        text = "\n".join([line] * 256)
        arr = parse_dem_tile_text(text)
        assert arr.shape == (256, 256)
        assert arr.dtype == np.float32
        assert np.allclose(arr, 1.5)

    def test_nodata_token(self):
        line = ",".join(["e"] * 256)
        text = "\n".join([line] * 256)
        arr = parse_dem_tile_text(text, nodata=-9999.0)
        assert np.allclose(arr, -9999.0)

    def test_mixed_and_ragged(self):
        # First cell real, rest no-data; short rows; missing rows -> nodata
        text = "12.25,e,e\ne,3.0"
        arr = parse_dem_tile_text(text, nodata=-1.0)
        assert arr.shape == (256, 256)
        assert arr[0, 0] == pytest.approx(12.25)
        assert arr[0, 1] == pytest.approx(-1.0)
        assert arr[1, 0] == pytest.approx(-1.0)
        assert arr[1, 1] == pytest.approx(3.0)
        # Untouched cell stays nodata
        assert arr[5, 5] == pytest.approx(-1.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `& "C:\Users\kunih\miniconda3\Scripts\conda.exe" run -n voxcity python -m pytest tests/test_downloader_gsi.py::TestParseDemTileText -v`
Expected: FAIL with `ImportError: cannot import name 'parse_dem_tile_text'`

- [ ] **Step 3: Write minimal implementation**

Add to `src/voxcity/downloader/gsi.py` (after `tile_range_for_bbox`):

```python
def parse_dem_tile_text(text, nodata=GSI_NODATA, size=GSI_TILE_SIZE):
    """Parse a GSI DEM ``.txt`` tile (CSV of meters, ``e`` = no-data).

    Returns a ``(size, size)`` float32 array. Ragged/short rows and missing
    rows are filled with ``nodata``.
    """
    arr = np.full((size, size), nodata, dtype=np.float32)
    lines = [ln for ln in text.strip().splitlines() if ln.strip()]
    for i, line in enumerate(lines[:size]):
        cells = line.split(",")
        for j, cell in enumerate(cells[:size]):
            c = cell.strip()
            if c == "" or c == "e":
                continue
            arr[i, j] = float(c)
    return arr
```

- [ ] **Step 4: Run test to verify it passes**

Run: `& "C:\Users\kunih\miniconda3\Scripts\conda.exe" run -n voxcity python -m pytest tests/test_downloader_gsi.py::TestParseDemTileText -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/voxcity/downloader/gsi.py tests/test_downloader_gsi.py
git commit -m "feat(downloader): parse GSI DEM tile text"
```

---

## Task 3: Resolution auto-detect (`check_dem_availability`)

**Files:**
- Modify: `src/voxcity/downloader/gsi.py`
- Test: `tests/test_downloader_gsi.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_downloader_gsi.py`:

```python
from unittest.mock import patch, MagicMock
from voxcity.downloader.gsi import check_dem_availability


def _resp(status):
    m = MagicMock()
    m.status_code = status
    return m


class TestCheckDemAvailability:
    def test_picks_finest_available(self):
        # dem5a (first probe) returns 200 -> chosen immediately
        with patch("voxcity.downloader.gsi.requests.get", return_value=_resp(200)) as g:
            dem_type, zoom = check_dem_availability(36.225, 140.105, sleep=0)
        assert (dem_type, zoom) == ("dem5a", 15)
        assert g.call_count == 1

    def test_falls_through_to_dem10b(self):
        # dem5a, dem5b -> 404; dem10b -> 200
        responses = [_resp(404), _resp(404), _resp(200)]
        with patch("voxcity.downloader.gsi.requests.get", side_effect=responses):
            dem_type, zoom = check_dem_availability(36.225, 140.105, sleep=0)
        assert (dem_type, zoom) == ("dem10b", 14)

    def test_all_fail_defaults_to_dem10b(self):
        with patch("voxcity.downloader.gsi.requests.get", return_value=_resp(404)):
            dem_type, zoom = check_dem_availability(36.225, 140.105, sleep=0)
        assert (dem_type, zoom) == ("dem10b", 14)

    def test_network_error_is_skipped(self):
        import requests as _rq
        with patch("voxcity.downloader.gsi.requests.get",
                   side_effect=_rq.exceptions.ConnectTimeout()):
            dem_type, zoom = check_dem_availability(36.225, 140.105, sleep=0)
        assert (dem_type, zoom) == ("dem10b", 14)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `& "C:\Users\kunih\miniconda3\Scripts\conda.exe" run -n voxcity python -m pytest tests/test_downloader_gsi.py::TestCheckDemAvailability -v`
Expected: FAIL with `ImportError: cannot import name 'check_dem_availability'`

- [ ] **Step 3: Write minimal implementation**

Add to `src/voxcity/downloader/gsi.py`:

```python
_GSI_XYZ_URL = "https://cyberjapandata.gsi.go.jp/xyz/{dem_type}/{zoom}/{x}/{y}.txt"


def check_dem_availability(lat, lon, *, timeout_s=5, sleep=0.2):
    """Probe the center point and return (dem_type, zoom) for the finest
    available GSI DEM product. Falls back to ('dem10b', 14)."""
    for item in GSI_DEM_TYPES:
        x, y = latlon_to_tile(lat, lon, item["zoom"])
        url = _GSI_XYZ_URL.format(dem_type=item["type"], zoom=item["zoom"], x=x, y=y)
        try:
            if sleep:
                time.sleep(sleep)
            resp = requests.get(url, timeout=timeout_s)
            if resp.status_code == 200:
                return item["type"], item["zoom"]
        except requests.exceptions.RequestException:
            continue
    return "dem10b", 14
```

- [ ] **Step 4: Run test to verify it passes**

Run: `& "C:\Users\kunih\miniconda3\Scripts\conda.exe" run -n voxcity python -m pytest tests/test_downloader_gsi.py::TestCheckDemAvailability -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/voxcity/downloader/gsi.py tests/test_downloader_gsi.py
git commit -m "feat(downloader): GSI DEM resolution auto-detect"
```

---

## Task 4: Download + compose mosaic

**Files:**
- Modify: `src/voxcity/downloader/gsi.py`
- Test: `tests/test_downloader_gsi.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_downloader_gsi.py`:

```python
from voxcity.downloader.gsi import download_dem_tiles, compose_dem_array


def _txt_resp(value):
    m = MagicMock()
    m.status_code = 200
    line = ",".join([str(value)] * 256)
    m.text = "\n".join([line] * 256)
    return m


class TestDownloadAndCompose:
    def test_download_fills_missing_with_nodata(self):
        # 1x2 tile range; first tile 200, second 404
        tile_range = (10, 20, 10, 21)  # x_min,y_min,x_max,y_max
        responses = [_txt_resp(5.0), _resp(404)]
        with patch("voxcity.downloader.gsi.requests.get", side_effect=responses):
            tiles = download_dem_tiles(tile_range, "dem5a", 15, sleep=0, nodata=-9999.0)
        assert set(tiles.keys()) == {(10, 20), (10, 21)}
        assert np.allclose(tiles[(10, 20)], 5.0)
        assert np.allclose(tiles[(10, 21)], -9999.0)

    def test_download_all_fail_raises(self):
        tile_range = (10, 20, 10, 20)
        with patch("voxcity.downloader.gsi.requests.get", return_value=_resp(404)):
            with pytest.raises(ValueError):
                download_dem_tiles(tile_range, "dem5a", 15, sleep=0)

    def test_compose_places_blocks(self):
        tile_range = (10, 20, 11, 20)  # 2 wide, 1 tall
        tiles = {
            (10, 20): np.full((256, 256), 1.0, dtype=np.float32),
            (11, 20): np.full((256, 256), 2.0, dtype=np.float32),
        }
        mosaic = compose_dem_array(tiles, tile_range, nodata=-9999.0)
        assert mosaic.shape == (256, 512)
        assert np.allclose(mosaic[:, :256], 1.0)
        assert np.allclose(mosaic[:, 256:], 2.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `& "C:\Users\kunih\miniconda3\Scripts\conda.exe" run -n voxcity python -m pytest tests/test_downloader_gsi.py::TestDownloadAndCompose -v`
Expected: FAIL with `ImportError: cannot import name 'download_dem_tiles'`

- [ ] **Step 3: Write minimal implementation**

Add to `src/voxcity/downloader/gsi.py`:

```python
def download_dem_tiles(tile_range, dem_type, zoom, *, nodata=GSI_NODATA,
                       sleep=0.4, timeout_s=10):
    """Download every tile in ``tile_range`` (x_min, y_min, x_max, y_max).

    Returns ``{(x, y): (256, 256) float32}``. Missing/failed tiles become
    nodata blocks. Raises ValueError if no tile was retrieved.
    """
    x_min, y_min, x_max, y_max = tile_range
    tiles = {}
    any_ok = False
    for x in range(x_min, x_max + 1):
        for y in range(y_min, y_max + 1):
            url = _GSI_XYZ_URL.format(dem_type=dem_type, zoom=zoom, x=x, y=y)
            block = np.full((GSI_TILE_SIZE, GSI_TILE_SIZE), nodata, dtype=np.float32)
            try:
                if sleep:
                    time.sleep(sleep)
                resp = requests.get(url, timeout=timeout_s)
                if resp.status_code == 200:
                    block = parse_dem_tile_text(resp.text, nodata=nodata)
                    any_ok = True
            except requests.exceptions.RequestException:
                pass
            tiles[(x, y)] = block
    if not any_ok:
        raise ValueError(
            "No GSI DEM tiles available for the requested area "
            "(is it outside Japan coverage?)."
        )
    return tiles


def compose_dem_array(tiles, tile_range, nodata=GSI_NODATA):
    """Assemble per-tile 256x256 blocks into one float32 mosaic."""
    x_min, y_min, x_max, y_max = tile_range
    cols = (x_max - x_min + 1) * GSI_TILE_SIZE
    rows = (y_max - y_min + 1) * GSI_TILE_SIZE
    mosaic = np.full((rows, cols), nodata, dtype=np.float32)
    for (x, y), block in tiles.items():
        r0 = (y - y_min) * GSI_TILE_SIZE
        c0 = (x - x_min) * GSI_TILE_SIZE
        mosaic[r0:r0 + GSI_TILE_SIZE, c0:c0 + GSI_TILE_SIZE] = block
    return mosaic
```

- [ ] **Step 4: Run test to verify it passes**

Run: `& "C:\Users\kunih\miniconda3\Scripts\conda.exe" run -n voxcity python -m pytest tests/test_downloader_gsi.py::TestDownloadAndCompose -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/voxcity/downloader/gsi.py tests/test_downloader_gsi.py
git commit -m "feat(downloader): download and compose GSI DEM mosaic"
```

---

## Task 5: Write EPSG:3857 GeoTIFF

**Files:**
- Modify: `src/voxcity/downloader/gsi.py`
- Test: `tests/test_downloader_gsi.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_downloader_gsi.py`:

```python
from voxcity.downloader.gsi import save_dem_as_geotiff, tile_bounds_mercator


class TestSaveDemAsGeotiff:
    def test_roundtrip_crs_transform_nodata(self, tmp_path):
        import rasterio
        tile_range = (29000, 12900, 29000, 12900)  # single tile
        zoom = 15
        array = np.arange(256 * 256, dtype=np.float32).reshape(256, 256)
        out = tmp_path / "dem.tif"
        save_dem_as_geotiff(array, tile_range, zoom, str(out), nodata=-9999.0)
        assert out.exists()
        with rasterio.open(str(out)) as src:
            assert src.crs.to_epsg() == 3857
            assert src.nodata == -9999.0
            data = src.read(1)
            assert np.allclose(data, array)
            # Origin matches top-left tile bounds
            minx, miny, maxx, maxy = tile_bounds_mercator(29000, 12900, zoom)
            assert src.transform.c == pytest.approx(minx, rel=1e-9)
            assert src.transform.f == pytest.approx(maxy, rel=1e-9)
            # Pixel size = tile / 256
            tile = (2 * 20037508.342789244) / (2 ** zoom)
            assert src.transform.a == pytest.approx(tile / 256, rel=1e-9)
            assert src.transform.e == pytest.approx(-tile / 256, rel=1e-9)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `& "C:\Users\kunih\miniconda3\Scripts\conda.exe" run -n voxcity python -m pytest tests/test_downloader_gsi.py::TestSaveDemAsGeotiff -v`
Expected: FAIL with `ImportError: cannot import name 'save_dem_as_geotiff'`

- [ ] **Step 3: Write minimal implementation**

Add to `src/voxcity/downloader/gsi.py`:

```python
def save_dem_as_geotiff(array, tile_range, zoom, filepath, nodata=GSI_NODATA):
    """Write the mosaic as a single-band float32 GeoTIFF in EPSG:3857."""
    x_min, y_min, x_max, y_max = tile_range
    origin_minx, _, _, origin_maxy = tile_bounds_mercator(x_min, y_min, zoom)
    pixel = (2 * _MERC_MAX) / (2.0 ** zoom) / GSI_TILE_SIZE

    out_dir = os.path.dirname(filepath)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    height, width = array.shape
    driver = gdal.GetDriverByName("GTiff")
    dataset = driver.Create(filepath, width, height, 1, gdal.GDT_Float32)
    dataset.SetGeoTransform((origin_minx, pixel, 0, origin_maxy, 0, -pixel))

    srs = osr.SpatialReference()
    srs.ImportFromEPSG(3857)
    dataset.SetProjection(srs.ExportToWkt())

    band = dataset.GetRasterBand(1)
    band.WriteArray(np.asarray(array, dtype=np.float32))
    band.SetNoDataValue(float(nodata))
    dataset = None
    return filepath
```

- [ ] **Step 4: Run test to verify it passes**

Run: `& "C:\Users\kunih\miniconda3\Scripts\conda.exe" run -n voxcity python -m pytest tests/test_downloader_gsi.py::TestSaveDemAsGeotiff -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/voxcity/downloader/gsi.py tests/test_downloader_gsi.py
git commit -m "feat(downloader): write GSI DEM as EPSG:3857 GeoTIFF"
```

---

## Task 6: Orchestrator `save_gsi_dem_as_geotiff`

**Files:**
- Modify: `src/voxcity/downloader/gsi.py`
- Test: `tests/test_downloader_gsi.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_downloader_gsi.py`:

```python
from voxcity.downloader.gsi import save_gsi_dem_as_geotiff


class TestSaveGsiDemAsGeotiff:
    def _verts(self):
        return [(140.09, 36.21), (140.12, 36.21), (140.12, 36.24), (140.09, 36.24)]

    def test_auto_detect_then_write(self, tmp_path):
        out = tmp_path / "dem.tif"
        with patch("voxcity.downloader.gsi.check_dem_availability",
                   return_value=("dem5a", 15)) as chk, \
             patch("voxcity.downloader.gsi.requests.get", side_effect=_txt_resp_factory()):
            path = save_gsi_dem_as_geotiff(self._verts(), str(out), sleep=0)
        assert path == str(out)
        assert out.exists()
        chk.assert_called_once()

    def test_forced_type_skips_probe(self, tmp_path):
        out = tmp_path / "dem.tif"
        with patch("voxcity.downloader.gsi.check_dem_availability") as chk, \
             patch("voxcity.downloader.gsi.requests.get", side_effect=_txt_resp_factory()):
            save_gsi_dem_as_geotiff(self._verts(), str(out), dem_type="dem10b", sleep=0)
        chk.assert_not_called()

    def test_invalid_type_raises(self, tmp_path):
        out = tmp_path / "dem.tif"
        with pytest.raises(ValueError):
            save_gsi_dem_as_geotiff(self._verts(), str(out), dem_type="bogus", sleep=0)
```

Also add this helper near the top of the test file (after `_txt_resp`):

```python
def _txt_resp_factory():
    """Infinite generator of valid 200 tile responses (value 3.0)."""
    def gen():
        while True:
            yield _txt_resp(3.0)
    return gen()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `& "C:\Users\kunih\miniconda3\Scripts\conda.exe" run -n voxcity python -m pytest tests/test_downloader_gsi.py::TestSaveGsiDemAsGeotiff -v`
Expected: FAIL with `ImportError: cannot import name 'save_gsi_dem_as_geotiff'`

- [ ] **Step 3: Write minimal implementation**

Add to `src/voxcity/downloader/gsi.py`:

```python
def save_gsi_dem_as_geotiff(rectangle_vertices, filepath, dem_type=None,
                            nodata=GSI_NODATA, sleep=0.4, timeout_s=10):
    """Download GSI bare-earth DEM for an ROI and save an EPSG:3857 GeoTIFF.

    Args:
        rectangle_vertices: list of (lon, lat) tuples defining the ROI.
        filepath: output GeoTIFF path.
        dem_type: None to auto-detect the finest available product, or one of
                  'dem5a' / 'dem5b' / 'dem10b' to force it.
        nodata: no-data fill value.
        sleep: seconds between requests (politeness; set 0 in tests).
        timeout_s: per-request timeout.

    Returns:
        The written filepath.

    Raises:
        ValueError: if dem_type is invalid or no tiles cover the area.
    """
    bbox = _bbox_from_rectangle_vertices(rectangle_vertices)
    min_lon, min_lat, max_lon, max_lat = bbox
    mid_lat = (min_lat + max_lat) / 2.0
    mid_lon = (min_lon + max_lon) / 2.0

    if dem_type is None:
        dem_type, zoom = check_dem_availability(
            mid_lat, mid_lon, timeout_s=timeout_s, sleep=sleep
        )
    else:
        if dem_type not in _ZOOM_BY_TYPE:
            raise ValueError(
                f"Unknown dem_type {dem_type!r}; expected one of "
                f"{sorted(_ZOOM_BY_TYPE)}"
            )
        zoom = _ZOOM_BY_TYPE[dem_type]

    tile_range = tile_range_for_bbox(bbox, zoom)
    tiles = download_dem_tiles(
        tile_range, dem_type, zoom, nodata=nodata, sleep=sleep, timeout_s=timeout_s
    )
    mosaic = compose_dem_array(tiles, tile_range, nodata=nodata)
    save_dem_as_geotiff(mosaic, tile_range, zoom, filepath, nodata=nodata)
    return filepath
```

- [ ] **Step 4: Run test to verify it passes**

Run: `& "C:\Users\kunih\miniconda3\Scripts\conda.exe" run -n voxcity python -m pytest tests/test_downloader_gsi.py -v`
Expected: PASS (entire file)

- [ ] **Step 5: Commit**

```bash
git add src/voxcity/downloader/gsi.py tests/test_downloader_gsi.py
git commit -m "feat(downloader): orchestrate GSI DEM download to GeoTIFF"
```

---

## Task 7: Export from downloader package

**Files:**
- Modify: `src/voxcity/downloader/__init__.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_downloader_gsi.py`:

```python
class TestPackageExport:
    def test_exported_from_downloader(self):
        import voxcity.downloader as dl
        assert hasattr(dl, "save_gsi_dem_as_geotiff")
        assert "save_gsi_dem_as_geotiff" in dl.__all__
```

- [ ] **Step 2: Run test to verify it fails**

Run: `& "C:\Users\kunih\miniconda3\Scripts\conda.exe" run -n voxcity python -m pytest tests/test_downloader_gsi.py::TestPackageExport -v`
Expected: FAIL with `AssertionError` (attribute / __all__ missing)

- [ ] **Step 3: Write minimal implementation**

In `src/voxcity/downloader/__init__.py`, add the import after the existing `from .oemj import *` line (line 5):

```python
from .gsi import *
```

And add to the `__all__` list, immediately after the `# oemj` block entry `"save_oemj_as_geotiff",`:

```python
    # gsi
    "save_gsi_dem_as_geotiff",
```

- [ ] **Step 4: Run test to verify it passes**

Run: `& "C:\Users\kunih\miniconda3\Scripts\conda.exe" run -n voxcity python -m pytest tests/test_downloader_gsi.py::TestPackageExport -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/voxcity/downloader/__init__.py tests/test_downloader_gsi.py
git commit -m "feat(downloader): export save_gsi_dem_as_geotiff"
```

---

## Task 8: Wire into `get_dem_grid`

**Files:**
- Modify: `src/voxcity/generator/grids.py:357-400`
- Test: `tests/test_generator_gsi_dem.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_generator_gsi_dem.py`:

```python
"""Tests for the GSI DEM source wired into get_dem_grid."""
import os
from unittest.mock import patch

import numpy as np

from voxcity.generator.grids import get_dem_grid


VERTS = [(140.09, 36.21), (140.12, 36.21), (140.12, 36.24), (140.09, 36.24)]


def test_gsi_source_calls_downloader_and_builds_grid(tmp_path):
    """source='GSI DEM Japan' must route to the GSI downloader (no Earth
    Engine) and return a grid from the written GeoTIFF."""
    called = {}

    def fake_save(rectangle_vertices, filepath, dem_type=None, **kwargs):
        called["dem_type"] = dem_type
        called["filepath"] = filepath
        # Write a tiny valid EPSG:3857 GeoTIFF so the grid builder can read it.
        from voxcity.downloader.gsi import save_dem_as_geotiff
        arr = np.full((256, 256), 12.0, dtype=np.float32)
        save_dem_as_geotiff(arr, (29000, 12900, 29000, 12900), 15, filepath)
        return filepath

    with patch("voxcity.downloader.gsi.save_gsi_dem_as_geotiff", side_effect=fake_save), \
         patch("voxcity.generator.grids.initialize_earth_engine") as init_ee:
        grid = get_dem_grid(
            VERTS, meshsize=10, source="GSI DEM Japan",
            output_dir=str(tmp_path), gsi_dem_type="dem10b", gridvis=False,
        )

    init_ee.assert_not_called()
    assert called["dem_type"] == "dem10b"
    assert os.path.basename(called["filepath"]) == "dem.tif"
    assert isinstance(grid, np.ndarray)
    assert grid.ndim == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `& "C:\Users\kunih\miniconda3\Scripts\conda.exe" run -n voxcity python -m pytest tests/test_generator_gsi_dem.py -v`
Expected: FAIL — current code has no `"GSI DEM Japan"` branch, so it falls into the Earth Engine path and `initialize_earth_engine` is called (assert fails) or errors.

- [ ] **Step 3: Write minimal implementation**

In `src/voxcity/generator/grids.py`, locate the branch at line 363:

```python
    if source == "Local file":
        geotiff_path = kwargs["dem_path"]
    else:
```

Insert a new `elif` between the `if` and the `else` so it reads:

```python
    if source == "Local file":
        geotiff_path = kwargs["dem_path"]
    elif source == "GSI DEM Japan":
        from ..downloader.gsi import save_gsi_dem_as_geotiff
        geotiff_path = os.path.join(output_dir, "dem.tif")
        save_gsi_dem_as_geotiff(
            rectangle_vertices, geotiff_path, dem_type=kwargs.get("gsi_dem_type")
        )
    else:
```

(The existing Earth Engine block stays as the `else` body unchanged. The shared
`create_dem_grid_from_geotiff_polygon` call below the branch handles the
resulting GeoTIFF for all three cases.)

- [ ] **Step 4: Run test to verify it passes**

Run: `& "C:\Users\kunih\miniconda3\Scripts\conda.exe" run -n voxcity python -m pytest tests/test_generator_gsi_dem.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/voxcity/generator/grids.py tests/test_generator_gsi_dem.py
git commit -m "feat(generator): add 'GSI DEM Japan' source to get_dem_grid"
```

---

## Task 9: Auto-select GSI DEM for Japan

**Files:**
- Modify: `src/voxcity/generator/api.py:229-236` (coverage map), `api.py:412-424` (DEM selection), `api.py:292` (docstring)
- Test: `tests/test_generator_gsi_dem.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_generator_gsi_dem.py`:

```python
from voxcity.generator.api import auto_select_data_sources, _DEM_COVERAGE


# Tokyo ROI (Japan)
JAPAN_VERTS = [(139.76, 35.67), (139.77, 35.67), (139.77, 35.68), (139.76, 35.68)]
# Manhattan ROI (USA) — must stay USGS, proving Japan branch is scoped
USA_VERTS = [(-74.01, 40.70), (-74.00, 40.70), (-74.00, 40.71), (-74.01, 40.71)]


def test_japan_auto_selects_gsi():
    sources = auto_select_data_sources(JAPAN_VERTS)
    assert sources["dem_source"] == "GSI DEM Japan"


def test_usa_still_usgs():
    sources = auto_select_data_sources(USA_VERTS)
    assert sources["dem_source"] == "USGS 3DEP 1m"


def test_gsi_in_dem_coverage_map():
    assert "GSI DEM Japan" in _DEM_COVERAGE
```

- [ ] **Step 2: Run test to verify it fails**

Run: `& "C:\Users\kunih\miniconda3\Scripts\conda.exe" run -n voxcity python -m pytest tests/test_generator_gsi_dem.py -k "japan or coverage or usgs" -v`
Expected: FAIL — `test_japan_auto_selects_gsi` returns `'FABDEM'`; `test_gsi_in_dem_coverage_map` fails (key missing).

- [ ] **Step 3: Write minimal implementation**

(a) In `src/voxcity/generator/api.py`, add to the `_DEM_COVERAGE` dict (currently lines 229-236), after the `'Netherlands 0.5m DTM'` entry:

```python
    'GSI DEM Japan': lambda f: f['is_japan'],
```

(b) In the DEM-source selection block (currently lines 413-424), add an `is_japan`
branch before the final `else`:

```python
    # DEM source
    if is_usa:
        dem_source = 'USGS 3DEP 1m'
    elif is_england:
        dem_source = 'England 1m DTM'
    elif is_australia:
        dem_source = 'AUSTRALIA 5M DEM'
    elif is_france:
        dem_source = 'DEM France 1m'
    elif is_netherlands:
        dem_source = 'Netherlands 0.5m DTM'
    elif is_japan:
        dem_source = 'GSI DEM Japan'
    else:
        dem_source = 'FABDEM'
```

(c) Update the auto-select docstring DEM rule (currently line 292) from:

```python
    - DEM: High-resolution where available (USA, England, Australia, France, Netherlands), else 'FABDEM'.
```

to:

```python
    - DEM: High-resolution where available (USA, England, Australia, France, Netherlands, Japan), else 'FABDEM'.
      Japan -> 'GSI DEM Japan' (bare-earth GSI DEM, auto-detected 5 m/10 m).
```

- [ ] **Step 4: Run test to verify it passes**

Run: `& "C:\Users\kunih\miniconda3\Scripts\conda.exe" run -n voxcity python -m pytest tests/test_generator_gsi_dem.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/voxcity/generator/api.py tests/test_generator_gsi_dem.py
git commit -m "feat(api): auto-select GSI DEM for Japan"
```

---

## Task 10: Optional live integration test (skippable)

**Files:**
- Create: `tests/test_downloader_gsi_integration.py`

- [ ] **Step 1: Write the test (network-gated)**

Create `tests/test_downloader_gsi_integration.py`:

```python
"""Live network integration test for GSI DEM download. Skipped by default.

Enable with:  VOXCITY_LIVE_GSI=1 pytest tests/test_downloader_gsi_integration.py
"""
import os

import numpy as np
import pytest

LIVE = os.environ.get("VOXCITY_LIVE_GSI") == "1"
pytestmark = pytest.mark.skipif(not LIVE, reason="set VOXCITY_LIVE_GSI=1 to run")


def test_tsukuba_download(tmp_path):
    import rasterio
    from voxcity.downloader.gsi import save_gsi_dem_as_geotiff

    verts = [(140.09, 36.21), (140.12, 36.21), (140.12, 36.24), (140.09, 36.24)]
    out = tmp_path / "tsukuba_dem.tif"
    save_gsi_dem_as_geotiff(verts, str(out))
    assert out.exists()
    with rasterio.open(str(out)) as src:
        assert src.crs.to_epsg() == 3857
        data = src.read(1)
    # At least some real (non-nodata) elevation present.
    assert np.any(data > -1000)
```

- [ ] **Step 2: Run to verify it is collected and skipped**

Run: `& "C:\Users\kunih\miniconda3\Scripts\conda.exe" run -n voxcity python -m pytest tests/test_downloader_gsi_integration.py -v`
Expected: 1 skipped (reason: set VOXCITY_LIVE_GSI=1 to run)

- [ ] **Step 3: (Manual, optional) Run live**

Run (PowerShell): `$env:VOXCITY_LIVE_GSI=1; & "C:\Users\kunih\miniconda3\Scripts\conda.exe" run -n voxcity python -m pytest tests/test_downloader_gsi_integration.py -v; $env:VOXCITY_LIVE_GSI=$null`
Expected: PASS (requires internet + GSI availability)

- [ ] **Step 4: Commit**

```bash
git add tests/test_downloader_gsi_integration.py
git commit -m "test(downloader): optional live GSI DEM integration test"
```

---

## Final verification

- [ ] **Run the full new test set**

Run: `& "C:\Users\kunih\miniconda3\Scripts\conda.exe" run -n voxcity python -m pytest tests/test_downloader_gsi.py tests/test_generator_gsi_dem.py tests/test_downloader_gsi_integration.py -v`
Expected: all pass (integration test skipped unless `VOXCITY_LIVE_GSI=1`).

- [ ] **Sanity-check no regressions in adjacent modules**

Run: `& "C:\Users\kunih\miniconda3\Scripts\conda.exe" run -n voxcity python -m pytest tests/test_downloader_gee.py tests/test_downloader_utils.py -q`
Expected: pass (unchanged behavior).
```
