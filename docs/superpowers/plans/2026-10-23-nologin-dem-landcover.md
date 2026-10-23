# Login-free DEM and land cover sources Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Read `'COPERNICUS'`, `'ESA WorldCover'`, `'FABDEM'` and a new `'USGS 3DEP 10m'` source without Earth Engine or any account, and use them as the no-Earth-Engine DEM defaults.

**Architecture:** The generic windowed COG reader moves from `downloader/aws_opendata.py` to a new `downloader/cog.py` and gains tile helpers, a reusable HEAD probe, `pad_pixels` and `fill_nodata`. `aws_opendata.py` gets Copernicus, WorldCover and USGS adapters; a new `downloader/fabdem.py` reads FABDEM tiles out of the University of Bristol zips through `/vsizip//vsicurl/`. `grids.py` dispatches these names before Earth Engine is touched; `api.py` changes the no-Earth-Engine DEM auto-selection.

**Tech Stack:** Python, rasterio 1.5 / GDAL 3.12 (`/vsicurl/`, `/vsizip/`), requests, numpy, scipy.ndimage, pytest.

**Spec:** `docs/superpowers/specs/2026-10-23-nologin-dem-landcover-design.md`

**Branch:** `feat/aws-opendata-canopy` (already checked out). No worktrees.

---

## Conventions for every task

- **Running tests.** `conda` is not on PATH. From the repo root in the Bash tool:

  ```bash
  "C:/Users/kunih/miniconda3/Scripts/conda.exe" run -n voxcity python -m pytest <paths> -q -p no:cacheprovider
  ```

  Multi-line `python -c` fails under `conda run`; write throwaway scripts to `C:\Users\kunih\AppData\Local\Temp\claude\c--Users-kunih-OneDrive-00-Codes-python-VoxCity\850214bc-5850-454d-a346-5626c4f25c64\scratchpad`.
- **Commits** use conventional subjects and end with:

  ```
  Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
  ```

  Never `--no-verify`, never amend, never stage `node_modules/`.
- **No network in unit tests.** Patch `requests.head`, the adapter fetch functions, or read local GeoTIFFs written to `tmp_path`. Live tests are opt-in with `VOXCITY_LIVE_AWS=1`.
- **Logging in tests.** The voxcity logger does not propagate; request the `propagate_voxcity_logs` fixture (tests/conftest.py) and use `caplog.at_level(logging.WARNING, logger="voxcity")`.
- **Verified facts** (live, 2026-10-23):
  - Copernicus: `https://copernicus-dem-30m.s3.amazonaws.com/Copernicus_DSM_COG_10_N52_00_E013_00_DEM/Copernicus_DSM_COG_10_N52_00_E013_00_DEM.tif` (EPSG:4326 float32, no nodata).
  - WorldCover: `https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map/ESA_WorldCover_10m_2021_v200_N51E012_Map.tif` (EPSG:4326 uint8 codes, nodata 0).
  - USGS: `https://prd-tnm.s3.amazonaws.com/StagedProducts/Elevation/13/TIFF/current/n41w074/USGS_13_n41w074.tif` (EPSG:4269 float32, nodata −999999; tag is the tile's **NW** corner).
  - FABDEM: `/vsizip//vsicurl/https://data.bris.ac.uk/datasets/s5hqmjcdj8yo2ibzi9b4ew3sn/N50E010-N60E020_FABDEM_V1-2.zip/N52E013_FABDEM_V1-2.tif` and `…/N40W080-N50W070_FABDEM_V1-2.zip/N40W074_FABDEM_V1-2.tif` both read.
  - `rasterio.merge.merge(sources, bounds=…, nodata=np.nan, dtype="float32")` turns source nodata **and** uncovered area into NaN while keeping real 0 values.

---

## File structure

| File | Responsibility |
|---|---|
| `src/voxcity/downloader/cog.py` (new) | Generic: bbox helper, GDAL env, `fetch_cog_crop`, `_resample`, `_fill_nearest`, HEAD probe (`_remote_exists`, `probe_existing`), tile helpers (`_lat_tag`, `_lon_tag`, `_tile_origins`, `tile_origins_for`). |
| `src/voxcity/downloader/aws_opendata.py` (modify) | Dataset adapters: CHMv2 (existing), Copernicus, WorldCover, USGS 3DEP 10m. |
| `src/voxcity/downloader/fabdem.py` (new) | FABDEM zip/member naming and fetch. |
| `src/voxcity/generator/grids.py` (modify) | DEM and land-cover dispatch for the four names. |
| `src/voxcity/generator/api.py` (modify) | Source URL, coverage, no-EE DEM rule, docstrings. |
| `tests/test_downloader_cog.py` (new) | Offline tests for the new `cog.py` features. |
| `tests/test_downloader_aws_dem_lc.py` (new) | Offline tests for the three AWS adapters. |
| `tests/test_downloader_fabdem.py` (new) | Offline tests for FABDEM naming and fetch wiring. |
| `tests/test_generator_nologin_sources.py` (new) | Offline tests for `grids.py` dispatch and `api.py` auto-selection. |
| `tests/test_downloader_nologin_integration.py` (new) | Live, opt-in. |
| `tests/test_downloader_aws_opendata.py`, `tests/test_issue9_land_cover_missing_file.py`, `tests/test_generator_gsi_dem.py` (modify) | Patch targets / changed expectations. |
| `pyproject.toml`, `README.md`, `CHANGELOG.md`, `docs/reference/data_sources.md` (modify) | Coverage omit and docs. |

---

### Task 1: Move the generic reader into `downloader/cog.py` (pure refactor)

**Files:**
- Create: `src/voxcity/downloader/cog.py`
- Modify: `src/voxcity/downloader/aws_opendata.py`
- Modify: `tests/test_downloader_aws_opendata.py` (patch targets only)

- [ ] **Step 1: Baseline**

Run: `"C:/Users/kunih/miniconda3/Scripts/conda.exe" run -n voxcity python -m pytest tests/test_downloader_aws_opendata.py tests/test_generator_canopy_chmv2.py tests/test_import_hygiene.py -q -p no:cacheprovider`
Expected: all pass. Note the count.

- [ ] **Step 2: Create `src/voxcity/downloader/cog.py`**

Move these definitions **verbatim** out of `aws_opendata.py` into the new file: `_bbox_from_rectangle_vertices`, `_remote_exists`, `_GDAL_HTTP_ENV` (with its comment), `_RESAMPLING`, `_gdal_path`, `fetch_cog_crop`, `_resample`. File header and imports:

```python
"""
Generic reader for Cloud-Optimized GeoTIFFs over anonymous HTTP(S).

Everything here goes over plain anonymous HTTPS. There is no boto3, no AWS
CLI and no credential lookup, so it works the same on a laptop, in CI and in
Google Colab without signing in to anything.

``fetch_cog_crop`` reads only the window covering a rectangle from each
source through GDAL's ``/vsicurl/`` (HTTP range requests), mosaics the
pieces, optionally resamples, and writes a small local GeoTIFF. Dataset
adapters live in ``aws_opendata.py`` and ``fabdem.py``.
"""

from __future__ import annotations

import logging
import math
import os
from typing import List, Optional, Sequence, Tuple

import numpy as np
import rasterio
import requests
from rasterio.enums import Resampling
from rasterio.errors import RasterioIOError
from rasterio.merge import merge
from rasterio.transform import from_origin
from rasterio.warp import reproject, transform_bounds

from ..utils.logging import get_logger

__all__ = ["fetch_cog_crop"]

_logger = get_logger(__name__)
```

In the moved `_remote_exists`, change the two log messages from `"CHMv2 tile probe failed for %s: %s"` / `"CHMv2 tile probe for %s returned HTTP %s"` to `"Tile probe failed for %s: %s"` / `"Tile probe for %s returned HTTP %s"` (it now serves every dataset; no test asserts the "CHMv2" wording).

- [ ] **Step 3: Trim `aws_opendata.py`**

Delete the moved definitions from `aws_opendata.py`. Replace its import block with:

```python
from __future__ import annotations

import math
import os
from typing import List, Optional, Sequence, Tuple

from ..utils.logging import get_logger
from .cog import _bbox_from_rectangle_vertices, _remote_exists, fetch_cog_crop
from .gsi import latlon_to_tile
```

Keep `__all__ = ["fetch_cog_crop", "chmv2_tile_urls", "fetch_chmv2_canopy_height", "CHMV2_BASE_URL"]` unchanged so `from voxcity.downloader.aws_opendata import fetch_cog_crop` keeps working. Update the module docstring's "Two layers" paragraph to say the generic reader now lives in `cog.py` and is re-exported here.

- [ ] **Step 4: Move test patch targets**

Patches on names that now live in `cog.py` must target `cog`. In `tests/test_downloader_aws_opendata.py` replace every occurrence:

```bash
sed -i \
  -e 's/voxcity\.downloader\.aws_opendata\.requests\.head/voxcity.downloader.cog.requests.head/g' \
  -e 's/voxcity\.downloader\.aws_opendata\.rasterio\.open/voxcity.downloader.cog.rasterio.open/g' \
  -e 's/voxcity\.downloader\.aws_opendata\.merge/voxcity.downloader.cog.merge/g' \
  tests/test_downloader_aws_opendata.py
```

Patches of `voxcity.downloader.aws_opendata.fetch_cog_crop` and `voxcity.downloader.aws_opendata.chmv2_tile_urls` stay as they are: `aws_opendata` calls those names through its own module globals.

- [ ] **Step 5: Run the same tests**

Run the Step 1 command. Expected: the same count, all passing. If a test fails, the cause is a missed patch target: grep the failing test for `aws_opendata.` and decide whether that name now lives in `cog`.

- [ ] **Step 6: Commit**

```bash
git add src/voxcity/downloader/cog.py src/voxcity/downloader/aws_opendata.py tests/test_downloader_aws_opendata.py
git commit -m "refactor(downloader): move the generic COG reader into downloader/cog.py

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 2: `cog.py` features — tile helpers, `probe_existing`, `pad_pixels`, `fill_nodata`, zip support

**Files:**
- Modify: `src/voxcity/downloader/cog.py`
- Modify: `src/voxcity/downloader/aws_opendata.py` (`chmv2_tile_urls` uses `probe_existing`)
- Test: `tests/test_downloader_cog.py` (new)

- [ ] **Step 1: Write the failing tests**

Create `tests/test_downloader_cog.py`:

```python
"""Offline tests for voxcity.downloader.cog."""
import logging
import os
import zipfile
from unittest.mock import patch

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from voxcity.downloader.cog import (
    _GDAL_HTTP_ENV,
    _MAX_TILES,
    _gdal_path,
    _lat_tag,
    _lon_tag,
    _tile_origins,
    fetch_cog_crop,
    probe_existing,
    tile_origins_for,
)


class _Resp:
    def __init__(self, status_code):
        self.status_code = status_code


def _write(path, values, left, top, res, nodata=None, dtype="float32", crs="EPSG:4326"):
    values = np.asarray(values, dtype=dtype)
    with rasterio.open(
        path, "w", driver="GTiff", width=values.shape[1], height=values.shape[0], count=1,
        dtype=dtype, crs=crs, transform=from_origin(left, top, res, res), nodata=nodata,
    ) as ds:
        ds.write(values, 1)
    return str(path)


def _rect(w, s, e, n):
    return [(w, s), (e, s), (e, n), (w, n)]


class TestTags:
    def test_lat_tags(self):
        assert _lat_tag(52) == "N52"
        assert _lat_tag(0) == "N00"
        assert _lat_tag(-5) == "S05"

    def test_lon_tags(self):
        assert _lon_tag(13) == "E013"
        assert _lon_tag(0) == "E000"
        assert _lon_tag(-74) == "W074"


class TestTileOrigins:
    def test_berlin_one_degree(self):
        assert _tile_origins(13.366, 52.513, 13.374, 52.518, 1) == [(52, 13)]

    def test_straddles_longitude_line(self):
        assert _tile_origins(-74.5, 40.2, -73.5, 40.8, 1) == [(40, -75), (40, -74)]

    def test_three_degree_tiles(self):
        assert _tile_origins(13.366, 52.513, 13.374, 52.518, 3) == [(51, 12)]

    def test_around_origin(self):
        assert _tile_origins(-0.5, -0.5, 0.5, 0.5, 1) == [(-1, -1), (-1, 0), (0, -1), (0, 0)]

    def test_guard_rejects_antimeridian_span(self, propagate_voxcity_logs, caplog):
        with caplog.at_level(logging.WARNING, logger="voxcity"):
            assert tile_origins_for(_rect(-179.9, 10.0, 179.9, 10.5), 1, "Test DEM") == []
        assert any(r.levelno == logging.WARNING for r in caplog.records)

    def test_guard_rejects_too_many_tiles(self):
        # 5 x 5 one-degree tiles = 25 > _MAX_TILES
        assert _MAX_TILES == 16
        assert tile_origins_for(_rect(10.5, 40.5, 14.5, 44.5), 1, "Test DEM") == []

    def test_guard_passes_normal_area(self):
        assert tile_origins_for(_rect(13.366, 52.513, 13.374, 52.518), 1, "Test DEM") == [(52, 13)]


class TestProbeExisting:
    def test_keeps_200_only_and_warns_on_unexpected_status(self, propagate_voxcity_logs, caplog):
        status = {"https://x/a.tif": 200, "https://x/b.tif": 404, "https://x/c.tif": 503}
        with caplog.at_level(logging.WARNING, logger="voxcity"), patch(
            "voxcity.downloader.cog.requests.head",
            side_effect=lambda url, timeout: _Resp(status[url]),
        ):
            assert probe_existing(list(status)) == ["https://x/a.tif"]
        warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warnings) == 1 and "c.tif" in warnings[0]


class TestZipSupport:
    def test_zip_extension_allowed_for_vsicurl(self):
        assert ".zip" in _GDAL_HTTP_ENV["CPL_VSIL_CURL_ALLOWED_EXTENSIONS"].split(",")
        assert ".tif" in _GDAL_HTTP_ENV["CPL_VSIL_CURL_ALLOWED_EXTENSIONS"].split(",")

    def test_vsi_paths_pass_through(self):
        p = "/vsizip//vsicurl/https://example.org/a.zip/b.tif"
        assert _gdal_path(p) == p

    def test_reads_tif_inside_local_zip(self, tmp_path):
        tif = _write(tmp_path / "t.tif", np.full((100, 100), 7.0), 13.0, 53.0, 0.001)
        zpath = tmp_path / "pack.zip"
        with zipfile.ZipFile(zpath, "w") as z:
            z.write(tif, arcname="N52E013_test.tif")
        out = fetch_cog_crop(
            [f"/vsizip/{zpath.as_posix()}/N52E013_test.tif"],
            _rect(13.02, 52.92, 13.05, 52.95), str(tmp_path / "out.tif"),
        )
        with rasterio.open(out) as ds:
            assert np.all(ds.read(1) == 7.0)


class TestPadPixels:
    def test_pad_widens_crop_by_expected_pixels(self, tmp_path):
        src = _write(tmp_path / "s.tif", np.ones((200, 200)), 13.0, 53.0, 0.001)
        rect = _rect(13.05, 52.85, 13.10, 52.90)
        with rasterio.open(fetch_cog_crop([src], rect, str(tmp_path / "p1.tif"))) as ds:
            w1, h1 = ds.width, ds.height
        with rasterio.open(fetch_cog_crop([src], rect, str(tmp_path / "p5.tif"), pad_pixels=5)) as ds:
            w5, h5 = ds.width, ds.height
        assert (w5 - w1, h5 - h1) == (8, 8)


class TestFillNodata:
    def test_fills_hole_and_uncovered_strip_but_keeps_real_zero(self, tmp_path):
        vals = np.full((100, 100), 30.0)
        vals[40:60, 40:60] = -999999.0   # nodata hole
        vals[0, 0] = 0.0                 # a real sea-level value
        src = _write(tmp_path / "s.tif", vals, 13.0, 53.0, 0.001, nodata=-999999.0)
        # Box runs 0.02 deg west of the raster (uncovered strip).
        rect = _rect(12.98, 52.92, 13.08, 52.999)
        out = fetch_cog_crop([src], rect, str(tmp_path / "o.tif"), fill_nodata=True)
        with rasterio.open(out) as ds:
            a = ds.read(1)
            assert ds.dtypes[0] == "float32"
            r0, c0 = ds.index(13.0005, 52.9995)      # the real-zero pixel
        assert np.isfinite(a).all()
        assert a.min() >= 0.0 and a.max() <= 30.0
        assert a[r0, c0] == 0.0
        assert not np.any(a == -999999.0)

    def test_all_nodata_returns_none_and_writes_nothing(self, tmp_path):
        src = _write(tmp_path / "s.tif", np.full((50, 50), -999999.0), 13.0, 53.0, 0.001, nodata=-999999.0)
        out_path = tmp_path / "o.tif"
        assert fetch_cog_crop([src], _rect(13.01, 52.96, 13.02, 52.97), str(out_path), fill_nodata=True) is None
        assert not out_path.exists()

    def test_default_keeps_old_behaviour(self, tmp_path):
        src = _write(tmp_path / "s.tif", np.full((50, 50), 5, dtype="uint8"), 13.0, 53.0, 0.001, dtype="uint8")
        out = fetch_cog_crop([src], _rect(13.01, 52.96, 13.02, 52.97), str(tmp_path / "o.tif"))
        with rasterio.open(out) as ds:
            assert ds.dtypes[0] == "uint8"
```

- [ ] **Step 2: Run to verify they fail**

Run: `"C:/Users/kunih/miniconda3/Scripts/conda.exe" run -n voxcity python -m pytest tests/test_downloader_cog.py -q -p no:cacheprovider`
Expected: `ImportError: cannot import name '_MAX_TILES'` (and the other new names).

- [ ] **Step 3: Implement in `cog.py`**

(a) In `_GDAL_HTTP_ENV`, change `"CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".tif"` to `".tif,.zip"` and update the comment above the dict to say ".tif rasters and .zip archives (FABDEM)".

(b) Add after `_bbox_from_rectangle_vertices`:

```python
# Tiles are 1 or 3 degrees; more than this many means an area far beyond any
# city-scale model (or vertices straddling the antimeridian).
_MAX_TILES = 16


def _lat_tag(lat: int) -> str:
    """'N52' / 'S05' for an integer latitude."""
    return f"{'N' if lat >= 0 else 'S'}{abs(lat):02d}"


def _lon_tag(lon: int) -> str:
    """'E013' / 'W074' for an integer longitude."""
    return f"{'E' if lon >= 0 else 'W'}{abs(lon):03d}"


def _tile_origins(
    min_lon: float, min_lat: float, max_lon: float, max_lat: float, step: int
) -> List[Tuple[int, int]]:
    """Sorted (lat0, lon0) south-west corners of every ``step``-degree tile touching the bbox."""
    lat_lo = math.floor(min_lat / step) * step
    lat_hi = math.floor(max_lat / step) * step
    lon_lo = math.floor(min_lon / step) * step
    lon_hi = math.floor(max_lon / step) * step
    return sorted(
        (lat, lon)
        for lat in range(lat_lo, lat_hi + 1, step)
        for lon in range(lon_lo, lon_hi + 1, step)
    )


def tile_origins_for(
    rectangle_vertices: Sequence[Tuple[float, float]], step: int, label: str
) -> List[Tuple[int, int]]:
    """
    Tile origins for ``rectangle_vertices``, or [] (with a warning) when the
    area spans more than 180 degrees of longitude or needs more than
    ``_MAX_TILES`` tiles.
    """
    min_lon, min_lat, max_lon, max_lat = _bbox_from_rectangle_vertices(rectangle_vertices)
    origins = _tile_origins(min_lon, min_lat, max_lon, max_lat, step)
    if max_lon - min_lon > 180 or len(origins) > _MAX_TILES:
        _logger.warning(
            "%s: area is too large or crosses the antimeridian (%d tiles, longitude "
            "span %.1f deg; limit %d tiles); skipping this source",
            label, len(origins), max_lon - min_lon, _MAX_TILES,
        )
        return []
    return origins
```

(c) Add after `_remote_exists`:

```python
def probe_existing(urls: Sequence[str], timeout: int = 10) -> List[str]:
    """The subset of ``urls`` that answer an anonymous HEAD with 200, in order."""
    return [url for url in urls if _remote_exists(url, timeout)]
```

(d) In `fetch_cog_crop`, add two keyword-only parameters after `timeout: int = 60,`:

```python
    pad_pixels: int = 1,
    fill_nodata: bool = False,
```

Document them in the docstring Args:

```
        pad_pixels: Source pixels of padding added on every side of the
             rectangle (default 1, so cell-centre sampling at the edge never
             falls outside the crop; DEM adapters use more for interpolation).
        fill_nodata: Merge as float32 with NaN for source nodata and for area
             no source covers, then fill those cells from the nearest valid
             pixel. Returns None if no pixel is valid. Real zero values are
             kept.
```

Replace the padding comment and `bounds = …` line with:

```python
            # Padding so cell-centre sampling (and DEM interpolation) at the
            # edge never falls just outside the crop.
            bounds = (
                left - pad_pixels * px, bottom - pad_pixels * py,
                right + pad_pixels * px, top + pad_pixels * py,
            )
            merge_kwargs = {"nodata": np.nan, "dtype": "float32"} if fill_nodata else {}
```

and change the merge call to `data, transform = merge(sources, bounds=bounds, **merge_kwargs)`.

Right after the `with rasterio.Env(**env):` block ends (before `if res is not None:`), insert:

```python
    if fill_nodata:
        filled = _fill_nearest(data[0])
        if filled is None:
            _logger.warning("No valid pixels in the requested window; no raster written")
            return None
        data = filled[np.newaxis, ...]
```

Add the helper next to `_resample`:

```python
def _fill_nearest(a: np.ndarray) -> Optional[np.ndarray]:
    """Replace non-finite cells with the nearest finite cell; None if none are finite."""
    from scipy import ndimage

    invalid = ~np.isfinite(a)
    if invalid.all():
        return None
    if not invalid.any():
        return a
    idx = ndimage.distance_transform_edt(invalid, return_distances=False, return_indices=True)
    return a[tuple(idx)]
```

Update `__all__` in `cog.py` to `["fetch_cog_crop", "probe_existing", "tile_origins_for"]`.

(e) In `aws_opendata.py`, change the import to `from .cog import _bbox_from_rectangle_vertices, fetch_cog_crop, probe_existing` (drop `_remote_exists`; if `tests/test_downloader_aws_opendata.py` imports `_remote_exists` from `aws_opendata`, change that test import to `from voxcity.downloader.cog import _remote_exists`). In `chmv2_tile_urls`, replace the probing loop with:

```python
    base = base_url.rstrip("/")
    return probe_existing([f"{base}/chm/{qk}.tif" for qk in quadkeys], timeout)
```

- [ ] **Step 4: Run the tests**

Run: `"C:/Users/kunih/miniconda3/Scripts/conda.exe" run -n voxcity python -m pytest tests/test_downloader_cog.py tests/test_downloader_aws_opendata.py tests/test_generator_canopy_chmv2.py tests/test_import_hygiene.py -q -p no:cacheprovider`
Expected: all pass.

If `test_fills_hole_and_uncovered_strip_but_keeps_real_zero` fails only on `a[r0, c0] == 0.0`, print `a[r0-1:r0+2, c0-1:c0+2]` and check that `ds.index` hit the intended pixel (the raster's top-left pixel centre is 13.0005, 52.9995). Do not weaken the assertion.

- [ ] **Step 5: Commit**

```bash
git add src/voxcity/downloader/cog.py src/voxcity/downloader/aws_opendata.py tests/test_downloader_cog.py tests/test_downloader_aws_opendata.py
git commit -m "feat(downloader): cog tile helpers, probe_existing, pad_pixels, fill_nodata and zip support

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 3: Copernicus DEM and USGS 3DEP 10m adapters

**Files:**
- Modify: `src/voxcity/downloader/aws_opendata.py`
- Test: `tests/test_downloader_aws_dem_lc.py` (new)

- [ ] **Step 1: Write the failing tests**

Create `tests/test_downloader_aws_dem_lc.py`:

```python
"""Offline tests for the Copernicus, WorldCover and USGS 3DEP adapters."""
import os
from unittest.mock import patch

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from voxcity.downloader import aws_opendata as aod

BERLIN = [(13.366, 52.513), (13.374, 52.513), (13.374, 52.518), (13.366, 52.518)]
MANHATTAN = [(-73.99, 40.75), (-73.97, 40.75), (-73.97, 40.765), (-73.99, 40.765)]
HOUSTON = [(-95.56, 29.735), (-95.54, 29.735), (-95.54, 29.75), (-95.56, 29.75)]


def _echo(urls, timeout=10):
    return list(urls)


class TestCopernicusNames:
    def test_berlin(self):
        assert aod._copernicus_tile_name(52, 13) == "Copernicus_DSM_COG_10_N52_00_E013_00_DEM"

    def test_manhattan(self):
        assert aod._copernicus_tile_name(40, -74) == "Copernicus_DSM_COG_10_N40_00_W074_00_DEM"

    def test_southern_hemisphere(self):
        assert aod._copernicus_tile_name(-34, 151) == "Copernicus_DSM_COG_10_S34_00_E151_00_DEM"

    def test_urls_are_probed(self):
        with patch("voxcity.downloader.aws_opendata.probe_existing", side_effect=_echo) as probe:
            urls = aod.copernicus_tile_urls(BERLIN)
        assert urls == [
            "https://copernicus-dem-30m.s3.amazonaws.com/"
            "Copernicus_DSM_COG_10_N52_00_E013_00_DEM/Copernicus_DSM_COG_10_N52_00_E013_00_DEM.tif"
        ]
        probe.assert_called_once()


class TestUsgsNames:
    def test_manhattan_uses_north_west_corner(self):
        assert aod._usgs_3dep_tile_tag(40, -74) == "n41w074"

    def test_houston(self):
        assert aod._usgs_3dep_tile_tag(29, -96) == "n30w096"

    def test_eastern_hemisphere_format(self):
        assert aod._usgs_3dep_tile_tag(13, 144) == "n14e144"

    def test_urls(self):
        with patch("voxcity.downloader.aws_opendata.probe_existing", side_effect=_echo):
            urls = aod.usgs_3dep_10m_tile_urls(MANHATTAN)
        assert urls == [
            "https://prd-tnm.s3.amazonaws.com/StagedProducts/Elevation/13/TIFF/current/"
            "n41w074/USGS_13_n41w074.tif"
        ]


class TestDemFetchWiring:
    @pytest.mark.parametrize(
        "fetch_name, urls_name, expected",
        [
            ("fetch_copernicus_dem", "copernicus_tile_urls", {"pad_pixels": 4}),
            ("fetch_usgs_3dep_10m", "usgs_3dep_10m_tile_urls", {"pad_pixels": 10, "fill_nodata": True}),
        ],
    )
    def test_writes_dem_tif_with_dataset_options(self, tmp_path, fetch_name, urls_name, expected):
        captured = {}

        def fake_crop(urls, verts, out_path, **kwargs):
            captured.update(kwargs, urls=urls, out_path=out_path)
            open(out_path, "wb").close()
            return out_path

        with patch(f"voxcity.downloader.aws_opendata.{urls_name}", return_value=["u1"]), \
             patch("voxcity.downloader.aws_opendata.fetch_cog_crop", side_effect=fake_crop):
            out = getattr(aod, fetch_name)(BERLIN, str(tmp_path / "o"))

        assert out == str(tmp_path / "o" / "dem.tif")
        assert captured["urls"] == ["u1"]
        for k, v in expected.items():
            assert captured[k] == v

    @pytest.mark.parametrize(
        "fetch_name, urls_name",
        [("fetch_copernicus_dem", "copernicus_tile_urls"), ("fetch_usgs_3dep_10m", "usgs_3dep_10m_tile_urls")],
    )
    def test_no_tiles_returns_none(self, tmp_path, fetch_name, urls_name):
        with patch(f"voxcity.downloader.aws_opendata.{urls_name}", return_value=[]), \
             patch("voxcity.downloader.aws_opendata.fetch_cog_crop") as crop:
            assert getattr(aod, fetch_name)(BERLIN, str(tmp_path)) is None
        crop.assert_not_called()
```

- [ ] **Step 2: Run to verify they fail**

Run: `"C:/Users/kunih/miniconda3/Scripts/conda.exe" run -n voxcity python -m pytest tests/test_downloader_aws_dem_lc.py -q -p no:cacheprovider`
Expected: `AttributeError: module 'voxcity.downloader.aws_opendata' has no attribute '_copernicus_tile_name'`.

- [ ] **Step 3: Implement**

In `aws_opendata.py`, extend the `cog` import to:

```python
from .cog import (
    _bbox_from_rectangle_vertices,
    _lat_tag,
    _lon_tag,
    fetch_cog_crop,
    probe_existing,
    tile_origins_for,
)
```

Append to the module:

```python
# --- Copernicus DEM GLO-30 (surface model; buildings and trees included) ---

COPERNICUS_BASE_URL = "https://copernicus-dem-30m.s3.amazonaws.com"


def _copernicus_tile_name(lat0: int, lon0: int) -> str:
    """Tile (and folder) name for the 1-degree tile with south-west corner (lat0, lon0)."""
    return f"Copernicus_DSM_COG_10_{_lat_tag(lat0)}_00_{_lon_tag(lon0)}_00_DEM"


def copernicus_tile_urls(
    rectangle_vertices: Sequence[Tuple[float, float]],
    base_url: str = COPERNICUS_BASE_URL,
    timeout: int = 10,
) -> List[str]:
    """URLs of the Copernicus GLO-30 tiles that cover the area and exist (open ocean has none)."""
    base = base_url.rstrip("/")
    names = [_copernicus_tile_name(lat, lon) for lat, lon in tile_origins_for(rectangle_vertices, 1, "Copernicus DEM")]
    return probe_existing([f"{base}/{n}/{n}.tif" for n in names], timeout)


def fetch_copernicus_dem(
    rectangle_vertices: Sequence[Tuple[float, float]],
    output_dir: str,
    *,
    base_url: str = COPERNICUS_BASE_URL,
) -> Optional[str]:
    """
    Write ``output_dir/dem.tif`` from Copernicus DEM GLO-30 on AWS (no login).

    Missing tiles (open ocean) are zero-filled by the mosaic, i.e. sea level.
    Returns None when no tile exists or none could be read.
    """
    urls = copernicus_tile_urls(rectangle_vertices, base_url=base_url)
    if not urls:
        _logger.info("Copernicus DEM: no tiles cover this area (or the bucket is unreachable)")
        return None
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "dem.tif")
    # ~120 m of padding at 1 arc-second, matching the 100 m buffer of the
    # Earth Engine DEM path, so interpolation at the edge has neighbours.
    return fetch_cog_crop(urls, rectangle_vertices, out_path, pad_pixels=4)


# --- USGS 3DEP 1/3 arc-second seamless DEM (bare earth, US) ---

USGS_3DEP_10M_BASE_URL = "https://prd-tnm.s3.amazonaws.com/StagedProducts/Elevation/13/TIFF/current"


def _usgs_3dep_tile_tag(lat0: int, lon0: int) -> str:
    """USGS tiles are named by their NORTH-west corner, e.g. 'n41w074' for lat 40..41, lon -74..-73."""
    return f"{_lat_tag(lat0 + 1)}{_lon_tag(lon0)}".lower()


def usgs_3dep_10m_tile_urls(
    rectangle_vertices: Sequence[Tuple[float, float]],
    base_url: str = USGS_3DEP_10M_BASE_URL,
    timeout: int = 10,
) -> List[str]:
    """URLs of the USGS 3DEP 1/3 arc-second tiles that cover the area and exist."""
    base = base_url.rstrip("/")
    tags = [_usgs_3dep_tile_tag(lat, lon) for lat, lon in tile_origins_for(rectangle_vertices, 1, "USGS 3DEP 10m")]
    return probe_existing([f"{base}/{t}/USGS_13_{t}.tif" for t in tags], timeout)


def fetch_usgs_3dep_10m(
    rectangle_vertices: Sequence[Tuple[float, float]],
    output_dir: str,
    *,
    base_url: str = USGS_3DEP_10M_BASE_URL,
) -> Optional[str]:
    """
    Write ``output_dir/dem.tif`` from the USGS 3DEP 1/3 arc-second DEM on AWS (no login).

    Cells with the dataset's nodata (-999999) are filled from the nearest valid
    pixel. Returns None when no tile exists or nothing valid could be read.
    """
    urls = usgs_3dep_10m_tile_urls(rectangle_vertices, base_url=base_url)
    if not urls:
        _logger.info("USGS 3DEP 10m: no tiles cover this area (or the bucket is unreachable)")
        return None
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "dem.tif")
    # ~100 m of padding at 1/3 arc-second.
    return fetch_cog_crop(urls, rectangle_vertices, out_path, pad_pixels=10, fill_nodata=True)
```

Add the four public names to `__all__`: `"copernicus_tile_urls", "fetch_copernicus_dem", "usgs_3dep_10m_tile_urls", "fetch_usgs_3dep_10m"`. Update the module docstring's dataset list.

- [ ] **Step 4: Run the tests**

Run: `"C:/Users/kunih/miniconda3/Scripts/conda.exe" run -n voxcity python -m pytest tests/test_downloader_aws_dem_lc.py tests/test_downloader_aws_opendata.py -q -p no:cacheprovider`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/voxcity/downloader/aws_opendata.py tests/test_downloader_aws_dem_lc.py
git commit -m "feat(downloader): Copernicus DEM and USGS 3DEP 10m from AWS Open Data

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 4: ESA WorldCover adapter

**Files:**
- Modify: `src/voxcity/downloader/aws_opendata.py`
- Test: `tests/test_downloader_aws_dem_lc.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_downloader_aws_dem_lc.py`:

```python
from voxcity.geoprocessor.raster import create_land_cover_grid_from_geotiff_polygon
from voxcity.utils.lc import get_land_cover_classes

_ESA_CLASSES = get_land_cover_classes("ESA WorldCover")
_ESA_NAMES = list(_ESA_CLASSES.values())
_CODES = [10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 100]
_CODE_TO_NAME = dict(zip(_CODES, _ESA_NAMES))
_CODE_TO_NAME[0] = "Open water"


class TestWorldCoverNames:
    def test_berlin(self):
        assert aod._worldcover_tile_name(51, 12) == "ESA_WorldCover_10m_2021_v200_N51E012_Map.tif"

    def test_southern_western(self):
        assert aod._worldcover_tile_name(-36, -60) == "ESA_WorldCover_10m_2021_v200_S36W060_Map.tif"

    def test_urls_use_three_degree_tiles(self):
        with patch("voxcity.downloader.aws_opendata.probe_existing", side_effect=_echo):
            urls = aod.worldcover_tile_urls(BERLIN)
        assert urls == [
            "https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map/"
            "ESA_WorldCover_10m_2021_v200_N51E012_Map.tif"
        ]


class TestWorldCoverRoundTrip:
    """Codes -> RGB land_cover.tif -> VoxCity's own colour lookup -> class names."""

    @pytest.mark.parametrize("code", _CODES + [0])
    def test_every_code_maps_to_its_class(self, tmp_path, code):
        codes_tif = tmp_path / "codes.tif"
        with rasterio.open(
            codes_tif, "w", driver="GTiff", width=200, height=200, count=1, dtype="uint8",
            crs="EPSG:4326", transform=from_origin(13.36, 52.52, 0.0001, 0.0001), nodata=0,
        ) as ds:
            ds.write(np.full((200, 200), code, dtype="uint8"), 1)

        with patch("voxcity.downloader.aws_opendata.worldcover_tile_urls", return_value=[str(codes_tif)]):
            out = aod.fetch_esa_worldcover(BERLIN, str(tmp_path / "o"))

        assert out == str(tmp_path / "o" / "land_cover.tif")
        assert not os.path.exists(tmp_path / "o" / "_esa_worldcover_codes.tif")
        with rasterio.open(out) as ds:
            assert ds.count == 3 and ds.dtypes[0] == "uint8"
        grid = create_land_cover_grid_from_geotiff_polygon(out, 5.0, _ESA_CLASSES, BERLIN)
        assert np.all(grid == _CODE_TO_NAME[code])

    def test_no_tiles_returns_none(self, tmp_path):
        with patch("voxcity.downloader.aws_opendata.worldcover_tile_urls", return_value=[]):
            assert aod.fetch_esa_worldcover(BERLIN, str(tmp_path)) is None
```

- [ ] **Step 2: Run to verify they fail**

Run: `"C:/Users/kunih/miniconda3/Scripts/conda.exe" run -n voxcity python -m pytest tests/test_downloader_aws_dem_lc.py -q -p no:cacheprovider`
Expected: `AttributeError: … '_worldcover_tile_name'`.

- [ ] **Step 3: Implement**

Add `import numpy as np` and `import rasterio` to `aws_opendata.py` imports. Append:

```python
# --- ESA WorldCover 10 m 2021 v200 ---

WORLDCOVER_BASE_URL = "https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map"
# Class codes in the order of get_land_cover_classes('ESA WorldCover').
_WORLDCOVER_CODES = (10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 100)


def _worldcover_tile_name(lat0: int, lon0: int) -> str:
    """File name of the 3-degree tile with south-west corner (lat0, lon0)."""
    return f"ESA_WorldCover_10m_2021_v200_{_lat_tag(lat0)}{_lon_tag(lon0)}_Map.tif"


def worldcover_tile_urls(
    rectangle_vertices: Sequence[Tuple[float, float]],
    base_url: str = WORLDCOVER_BASE_URL,
    timeout: int = 10,
) -> List[str]:
    """URLs of the WorldCover tiles that cover the area and exist."""
    base = base_url.rstrip("/")
    names = [_worldcover_tile_name(lat, lon) for lat, lon in tile_origins_for(rectangle_vertices, 3, "ESA WorldCover")]
    return probe_existing([f"{base}/{n}" for n in names], timeout)


def _worldcover_rgb_lut() -> np.ndarray:
    """(256, 3) uint8 lookup from class code to the RGB VoxCity's ESA class table expects.

    Uses the colours of get_land_cover_classes('ESA WorldCover') so the
    existing colour matching maps every pixel exactly. Code 0 (outside the
    tile set, i.e. ocean) and any unknown code map to Open water.
    """
    from ..utils.lc import get_land_cover_classes

    colors = list(get_land_cover_classes("ESA WorldCover").keys())
    lut = np.empty((256, 3), dtype=np.uint8)
    lut[:] = colors[_WORLDCOVER_CODES.index(80)]
    for code, rgb in zip(_WORLDCOVER_CODES, colors):
        lut[code] = rgb
    return lut


def fetch_esa_worldcover(
    rectangle_vertices: Sequence[Tuple[float, float]],
    output_dir: str,
    *,
    base_url: str = WORLDCOVER_BASE_URL,
) -> Optional[str]:
    """
    Write ``output_dir/land_cover.tif`` (3-band RGB) from ESA WorldCover on AWS (no login).

    The RGB matches the 'ESA WorldCover' class table, so the rest of the
    land-cover pipeline is unchanged. Returns None when no tile exists or
    none could be read.
    """
    urls = worldcover_tile_urls(rectangle_vertices, base_url=base_url)
    if not urls:
        _logger.info("ESA WorldCover: no tiles cover this area (or the bucket is unreachable)")
        return None
    os.makedirs(output_dir, exist_ok=True)
    codes_path = os.path.join(output_dir, "_esa_worldcover_codes.tif")
    if fetch_cog_crop(urls, rectangle_vertices, codes_path) is None:
        return None
    with rasterio.open(codes_path) as src:
        codes = src.read(1)
        crs, transform = src.crs, src.transform
    os.remove(codes_path)
    rgb = _worldcover_rgb_lut()[codes]
    out_path = os.path.join(output_dir, "land_cover.tif")
    with rasterio.open(
        out_path, "w", driver="GTiff", height=codes.shape[0], width=codes.shape[1], count=3,
        dtype="uint8", crs=crs, transform=transform, compress="deflate",
    ) as dst:
        dst.write(np.moveaxis(rgb, -1, 0))
    return out_path
```

Add `"worldcover_tile_urls", "fetch_esa_worldcover"` to `__all__`.

- [ ] **Step 4: Run the tests**

Run: `"C:/Users/kunih/miniconda3/Scripts/conda.exe" run -n voxcity python -m pytest tests/test_downloader_aws_dem_lc.py tests/test_import_hygiene.py -q -p no:cacheprovider`
Expected: all pass. If a round-trip case fails, print the unique values of `grid` and the RGB written for that code; the fix belongs in `_worldcover_rgb_lut`, never in the expected names.

- [ ] **Step 5: Commit**

```bash
git add src/voxcity/downloader/aws_opendata.py tests/test_downloader_aws_dem_lc.py
git commit -m "feat(downloader): ESA WorldCover from AWS Open Data, written in VoxCity's class colours

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 5: FABDEM from the University of Bristol

**Files:**
- Create: `src/voxcity/downloader/fabdem.py`
- Test: `tests/test_downloader_fabdem.py` (new)

- [ ] **Step 1: Write the failing tests**

Create `tests/test_downloader_fabdem.py`:

```python
"""Offline tests for voxcity.downloader.fabdem."""
from unittest.mock import patch

from voxcity.downloader import fabdem

BERLIN = [(13.366, 52.513), (13.374, 52.513), (13.374, 52.518), (13.366, 52.518)]
MANHATTAN = [(-73.99, 40.75), (-73.97, 40.75), (-73.97, 40.765), (-73.99, 40.765)]


class TestNames:
    def test_zip_berlin(self):
        assert fabdem._fabdem_zip_name(52, 13) == "N50E010-N60E020_FABDEM_V1-2.zip"

    def test_zip_manhattan(self):
        assert fabdem._fabdem_zip_name(40, -74) == "N40W080-N50W070_FABDEM_V1-2.zip"

    def test_zip_southern(self):
        assert fabdem._fabdem_zip_name(-5, 20) == "S10E020-N00E030_FABDEM_V1-2.zip"

    def test_members(self):
        assert fabdem._fabdem_member_name(52, 13) == "N52E013_FABDEM_V1-2.tif"
        assert fabdem._fabdem_member_name(40, -74) == "N40W074_FABDEM_V1-2.tif"


class TestPaths:
    def test_berlin_path(self):
        assert fabdem.fabdem_paths(BERLIN) == [
            "/vsizip//vsicurl/https://data.bris.ac.uk/datasets/s5hqmjcdj8yo2ibzi9b4ew3sn/"
            "N50E010-N60E020_FABDEM_V1-2.zip/N52E013_FABDEM_V1-2.tif"
        ]

    def test_manhattan_path(self):
        assert fabdem.fabdem_paths(MANHATTAN) == [
            "/vsizip//vsicurl/https://data.bris.ac.uk/datasets/s5hqmjcdj8yo2ibzi9b4ew3sn/"
            "N40W080-N50W070_FABDEM_V1-2.zip/N40W074_FABDEM_V1-2.tif"
        ]


class TestFetch:
    def test_writes_dem_tif_with_fill_and_padding(self, tmp_path):
        captured = {}

        def fake_crop(paths, verts, out_path, **kwargs):
            captured.update(kwargs, paths=paths, out_path=out_path)
            open(out_path, "wb").close()
            return out_path

        with patch("voxcity.downloader.fabdem.fetch_cog_crop", side_effect=fake_crop):
            out = fabdem.fetch_fabdem(BERLIN, str(tmp_path / "o"))

        assert out == str(tmp_path / "o" / "dem.tif")
        assert captured["paths"] == fabdem.fabdem_paths(BERLIN)
        assert captured["pad_pixels"] == 4 and captured["fill_nodata"] is True

    def test_guard_rejection_returns_none(self, tmp_path):
        huge = [(-179.9, 10.0), (179.9, 10.0), (179.9, 10.5), (-179.9, 10.5)]
        with patch("voxcity.downloader.fabdem.fetch_cog_crop") as crop:
            assert fabdem.fetch_fabdem(huge, str(tmp_path)) is None
        crop.assert_not_called()
```

- [ ] **Step 2: Run to verify they fail**

Run: `"C:/Users/kunih/miniconda3/Scripts/conda.exe" run -n voxcity python -m pytest tests/test_downloader_fabdem.py -q -p no:cacheprovider`
Expected: `ImportError: cannot import name 'fabdem'`.

- [ ] **Step 3: Implement**

Create `src/voxcity/downloader/fabdem.py`:

```python
"""
FABDEM v1-2 read without any account.

FABDEM (Hawker et al., 2022) is Copernicus GLO-30 with forests and buildings
removed. The University of Bristol serves it as 10 x 10 degree zip archives
of 1 x 1 degree GeoTIFFs. GDAL reads a single tile out of a remote zip with
HTTP range requests (``/vsizip//vsicurl/``), so only the window around the
area is transferred, not the 1-2 GB archive.

Licence: CC BY-NC-SA 4.0 (non-commercial). Cite Hawker, L. et al. (2022),
"A 30 m global map of elevation with forests and buildings removed",
Environmental Research Letters 17 024016.
"""

from __future__ import annotations

import math
import os
from typing import List, Optional, Sequence, Tuple

from ..utils.logging import get_logger
from .cog import _lat_tag, _lon_tag, fetch_cog_crop, tile_origins_for

__all__ = ["fabdem_paths", "fetch_fabdem", "FABDEM_BASE_URL"]

_logger = get_logger(__name__)

FABDEM_BASE_URL = "https://data.bris.ac.uk/datasets/s5hqmjcdj8yo2ibzi9b4ew3sn"


def _fabdem_zip_name(lat0: int, lon0: int) -> str:
    """Archive holding the 1-degree tile at (lat0, lon0): named by its 10-degree corners."""
    zlat = math.floor(lat0 / 10) * 10
    zlon = math.floor(lon0 / 10) * 10
    return f"{_lat_tag(zlat)}{_lon_tag(zlon)}-{_lat_tag(zlat + 10)}{_lon_tag(zlon + 10)}_FABDEM_V1-2.zip"


def _fabdem_member_name(lat0: int, lon0: int) -> str:
    """GeoTIFF name of the 1-degree tile with south-west corner (lat0, lon0)."""
    return f"{_lat_tag(lat0)}{_lon_tag(lon0)}_FABDEM_V1-2.tif"


def fabdem_paths(
    rectangle_vertices: Sequence[Tuple[float, float]],
    base_url: str = FABDEM_BASE_URL,
) -> List[str]:
    """GDAL paths of the FABDEM tiles covering the area.

    Not probed: a tile missing from its archive (open ocean) simply fails to
    open and is skipped by fetch_cog_crop.
    """
    base = base_url.rstrip("/")
    return [
        f"/vsizip//vsicurl/{base}/{_fabdem_zip_name(lat, lon)}/{_fabdem_member_name(lat, lon)}"
        for lat, lon in tile_origins_for(rectangle_vertices, 1, "FABDEM")
    ]


def fetch_fabdem(
    rectangle_vertices: Sequence[Tuple[float, float]],
    output_dir: str,
    *,
    base_url: str = FABDEM_BASE_URL,
) -> Optional[str]:
    """
    Write ``output_dir/dem.tif`` from FABDEM (no login). Nodata cells are
    filled from the nearest valid pixel. Returns None when no tile could be
    read.
    """
    paths = fabdem_paths(rectangle_vertices, base_url=base_url)
    if not paths:
        return None
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "dem.tif")
    # ~120 m of padding at 1 arc-second, as for Copernicus.
    return fetch_cog_crop(paths, rectangle_vertices, out_path, pad_pixels=4, fill_nodata=True)
```

- [ ] **Step 4: Run the tests**

Run: `"C:/Users/kunih/miniconda3/Scripts/conda.exe" run -n voxcity python -m pytest tests/test_downloader_fabdem.py tests/test_import_hygiene.py -q -p no:cacheprovider`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/voxcity/downloader/fabdem.py tests/test_downloader_fabdem.py
git commit -m "feat(downloader): FABDEM read from the Bristol archives without a login

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 6: DEM dispatch in `get_dem_grid`

**Files:**
- Modify: `src/voxcity/generator/grids.py` (`get_dem_grid`)
- Test: `tests/test_generator_nologin_sources.py` (new)

- [ ] **Step 1: Write the failing tests**

Create `tests/test_generator_nologin_sources.py`:

```python
"""Offline tests: login-free DEM / land cover sources in grids.py and api.py."""
import os
from unittest.mock import patch

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from voxcity.generator.grids import get_dem_grid, get_land_cover_grid
from voxcity.geoprocessor.raster.core import compute_grid_shape

VERTS = [(13.3660, 52.5130), (13.3690, 52.5130), (13.3690, 52.5150), (13.3660, 52.5150)]
MESH = 10.0
EE = "voxcity.downloader.gee.initialize_earth_engine"

DEM_FETCHERS = {
    "COPERNICUS": "voxcity.downloader.aws_opendata.fetch_copernicus_dem",
    "USGS 3DEP 10m": "voxcity.downloader.aws_opendata.fetch_usgs_3dep_10m",
    "FABDEM": "voxcity.downloader.fabdem.fetch_fabdem",
}


def _fake_dem(value):
    def _fetch(rectangle_vertices, output_dir, **kwargs):
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, "dem.tif")
        with rasterio.open(
            path, "w", driver="GTiff", width=60, height=40, count=1, dtype="float32",
            crs="EPSG:4326", transform=from_origin(13.365, 52.516, 0.0001, 0.0001),
        ) as ds:
            ds.write(np.full((40, 60), value, dtype="float32"), 1)
        return path
    return _fetch


@pytest.mark.parametrize("source", list(DEM_FETCHERS))
def test_dem_source_reads_without_earth_engine(tmp_path, source):
    with patch(DEM_FETCHERS[source], side_effect=_fake_dem(42.0)) as fetch, \
         patch(EE) as init_ee:
        grid = get_dem_grid(VERTS, MESH, source, str(tmp_path), gridvis=False)
    fetch.assert_called_once()
    init_ee.assert_not_called()
    assert grid.shape == compute_grid_shape(VERTS, MESH)
    np.testing.assert_allclose(grid, 42.0)


@pytest.mark.parametrize("source", list(DEM_FETCHERS))
def test_dem_source_failure_falls_back_to_flat(tmp_path, source):
    with patch(DEM_FETCHERS[source], return_value=None), patch(EE) as init_ee:
        grid = get_dem_grid(VERTS, MESH, source, str(tmp_path), gridvis=False)
    init_ee.assert_not_called()
    assert grid.shape == compute_grid_shape(VERTS, MESH)
    assert np.all(grid == 0)
```

- [ ] **Step 2: Run to verify they fail**

Run: `"C:/Users/kunih/miniconda3/Scripts/conda.exe" run -n voxcity python -m pytest tests/test_generator_nologin_sources.py -q -p no:cacheprovider`
Expected: 6 failures (the fetchers are never called; Earth Engine init is called).

- [ ] **Step 3: Implement**

In `src/voxcity/generator/grids.py`, add above `def get_dem_grid(`:

```python
# DEM sources read without Earth Engine or any account.
_NOLOGIN_DEM_SOURCES = ("COPERNICUS", "FABDEM", "USGS 3DEP 10m")


def _fetch_nologin_dem(source, rectangle_vertices, output_dir):
    """Write output_dir/dem.tif for a login-free DEM source; the path or None."""
    if source == "FABDEM":
        from ..downloader.fabdem import fetch_fabdem
        return fetch_fabdem(rectangle_vertices, output_dir)
    from ..downloader import aws_opendata
    if source == "COPERNICUS":
        return aws_opendata.fetch_copernicus_dem(rectangle_vertices, output_dir)
    return aws_opendata.fetch_usgs_3dep_10m(rectangle_vertices, output_dir)
```

In `get_dem_grid`, add a branch between the `"GSI DEM Japan"` branch and the `else:` that initialises Earth Engine:

```python
    elif source in _NOLOGIN_DEM_SOURCES:
        geotiff_path = _fetch_nologin_dem(source, rectangle_vertices, output_dir)
        if geotiff_path is None:
            _logger.warning(
                "%s could not be read for this area — falling back to flat DEM.", source
            )
            from ..geoprocessor.raster.core import compute_grid_shape
            return np.zeros(compute_grid_shape(rectangle_vertices, meshsize), dtype=float)
```

Leave the Earth Engine `else:` branch and `gee.get_dem_image` untouched.

- [ ] **Step 4: Run the tests**

Run: `"C:/Users/kunih/miniconda3/Scripts/conda.exe" run -n voxcity python -m pytest tests/test_generator_nologin_sources.py tests/test_generator_gsi_dem.py tests/test_generator_citygml_dem_source.py tests/test_import_hygiene.py -q -p no:cacheprovider`
Expected: the 6 new tests pass. `tests/test_generator_gsi_dem.py::test_usa_downgrades_to_flat_when_earth_engine_unavailable` still passes (the api.py change comes in Task 8). Everything else passes.

- [ ] **Step 5: Commit**

```bash
git add src/voxcity/generator/grids.py tests/test_generator_nologin_sources.py
git commit -m "feat(generator): COPERNICUS, FABDEM and USGS 3DEP 10m DEMs without Earth Engine

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 7: ESA WorldCover dispatch in `get_land_cover_grid`

**Files:**
- Modify: `src/voxcity/generator/grids.py` (`get_land_cover_grid`)
- Modify: `tests/test_issue9_land_cover_missing_file.py`
- Test: `tests/test_generator_nologin_sources.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_generator_nologin_sources.py`:

```python
from voxcity.utils.lc import get_land_cover_classes

WC_FETCH = "voxcity.downloader.aws_opendata.fetch_esa_worldcover"


def _fake_worldcover(rgb):
    def _fetch(rectangle_vertices, output_dir, **kwargs):
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, "land_cover.tif")
        with rasterio.open(
            path, "w", driver="GTiff", width=60, height=40, count=3, dtype="uint8",
            crs="EPSG:4326", transform=from_origin(13.365, 52.516, 0.0001, 0.0001),
        ) as ds:
            for band, value in enumerate(rgb, start=1):
                ds.write(np.full((40, 60), value, dtype="uint8"), band)
        return path
    return _fetch


def test_worldcover_reads_without_earth_engine(tmp_path):
    trees_rgb = list(get_land_cover_classes("ESA WorldCover"))[0]   # (0, 112, 0) = 'Trees'
    with patch(WC_FETCH, side_effect=_fake_worldcover(trees_rgb)) as fetch, patch(EE) as init_ee:
        grid = get_land_cover_grid(VERTS, MESH, "ESA WorldCover", str(tmp_path), gridvis=False, quiet=True)
    fetch.assert_called_once()
    init_ee.assert_not_called()
    assert grid.shape == compute_grid_shape(VERTS, MESH)
    assert len(np.unique(grid)) == 1


class _ReachedOSM(Exception):
    pass


def test_worldcover_failure_falls_back_to_openstreetmap(tmp_path):
    with patch(WC_FETCH, return_value=None), patch(EE) as init_ee, \
         patch("voxcity.generator.grids.load_land_cover_gdf_from_osm", side_effect=_ReachedOSM):
        with pytest.raises(_ReachedOSM):
            get_land_cover_grid(VERTS, MESH, "ESA WorldCover", str(tmp_path), gridvis=False, quiet=True)
    init_ee.assert_not_called()
```

Rewrite the first test of `tests/test_issue9_land_cover_missing_file.py` so the issue #9 contract (a Earth Engine exporter that writes nothing raises `ProcessingError`) is still covered by a source that still uses Earth Engine. Replace `test_esa_export_writes_no_file_raises_processing_error` with:

```python
def test_gee_export_writes_no_file_raises_processing_error(monkeypatch, tmp_path):
    rect = [(0.0, 0.0), (0.001, 0.0), (0.001, 0.001), (0.0, 0.001)]
    monkeypatch.setattr("voxcity.downloader.gee.initialize_earth_engine", lambda *a, **k: None)
    monkeypatch.setattr("voxcity.downloader.gee.get_roi", lambda *a, **k: object())
    monkeypatch.setattr("voxcity.downloader.gee.save_geotiff_esri_landcover", lambda *a, **k: None)  # writes nothing
    with pytest.raises(ProcessingError):
        grids_mod.get_land_cover_grid(rect, 5.0, "ESRI 10m Annual Land Cover", str(tmp_path), gridvis=False, quiet=True)
```

and in the module docstring change "save_geotiff_esa_land_cover" to "save_geotiff_esri_landcover" and add one sentence: "'ESA WorldCover' is read from AWS without Earth Engine, so the exporter case uses 'ESRI 10m Annual Land Cover'."

- [ ] **Step 2: Run to verify the new tests fail**

Run: `"C:/Users/kunih/miniconda3/Scripts/conda.exe" run -n voxcity python -m pytest tests/test_generator_nologin_sources.py tests/test_issue9_land_cover_missing_file.py -q -p no:cacheprovider`
Expected: the two WorldCover tests fail (Earth Engine init is called / no fallback); the rewritten issue-9 test passes already.

- [ ] **Step 3: Implement**

In `get_land_cover_grid`:

(a) Change `if source not in ["OpenStreetMap", "OpenEarthMapJapan"]:` to
`if source not in ["OpenStreetMap", "OpenEarthMapJapan", "ESA WorldCover"]:`.

(b) Replace the whole `elif source == 'ESA WorldCover':` branch with:

```python
    elif source == 'ESA WorldCover':
        # Read from AWS Open Data without Earth Engine or any account.
        from ..downloader.aws_opendata import fetch_esa_worldcover
        if fetch_esa_worldcover(rectangle_vertices, output_dir) is None:
            _logger.warning(
                "ESA WorldCover could not be read for this area — falling back to OpenStreetMap."
            )
            effective_source = 'OpenStreetMap'
            land_cover_gdf = load_land_cover_gdf_from_osm(rectangle_vertices, **_cache_kwargs)
```

`fetch_esa_worldcover` writes `output_dir/land_cover.tif`, which is the `geotiff_path` the rest of the function reads. `save_geotiff_esa_land_cover` stays imported at the top of the function only if another branch uses it; if nothing else references it, remove it from that local import list.

- [ ] **Step 4: Run the tests**

Run: `"C:/Users/kunih/miniconda3/Scripts/conda.exe" run -n voxcity python -m pytest tests/test_generator_nologin_sources.py tests/test_issue9_land_cover_missing_file.py tests/test_import_hygiene.py -q -p no:cacheprovider`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/voxcity/generator/grids.py tests/test_generator_nologin_sources.py tests/test_issue9_land_cover_missing_file.py
git commit -m "feat(generator): ESA WorldCover land cover without Earth Engine

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 8: `api.py` — no-Earth-Engine DEM rule, coverage, URL, docs

**Files:**
- Modify: `src/voxcity/generator/api.py`
- Modify: `tests/test_generator_gsi_dem.py`
- Test: `tests/test_generator_nologin_sources.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_generator_nologin_sources.py`:

```python
class _StopBeforeBuild(Exception):
    """Sentinel to abort get_voxcity right after source resolution."""


def _resolved_sources_without_ee(rectangle_vertices):
    """get_voxcity's auto-selected sources with Earth Engine unavailable (same
    pattern as tests/test_generator_gsi_dem.py)."""
    from voxcity.generator import api

    captured = {}

    def _capture(_verts, _b, _bc, land_cover, _canopy, dem):
        captured.update(land_cover=land_cover, dem=dem)
        raise _StopBeforeBuild

    with patch(EE, side_effect=Exception("EE unavailable")), \
         patch.object(api, "_warn_source_coverage", side_effect=_capture):
        try:
            api.get_voxcity(rectangle_vertices, meshsize=10)
        except _StopBeforeBuild:
            pass
    return captured


USA_VERTS = [(-122.42, 37.77), (-122.41, 37.77), (-122.41, 37.78), (-122.42, 37.78)]
GERMANY_VERTS = VERTS
JAPAN_VERTS = [(139.75, 35.68), (139.76, 35.68), (139.76, 35.69), (139.75, 35.69)]


def test_no_ee_dem_usa_is_usgs_10m():
    assert _resolved_sources_without_ee(USA_VERTS)["dem"] == "USGS 3DEP 10m"


def test_no_ee_dem_germany_is_fabdem():
    assert _resolved_sources_without_ee(GERMANY_VERTS)["dem"] == "FABDEM"


def test_no_ee_dem_japan_stays_gsi():
    assert _resolved_sources_without_ee(JAPAN_VERTS)["dem"] == "GSI DEM Japan"


def test_usgs_10m_registered():
    from voxcity.generator.api import _DEM_COVERAGE, _SOURCE_URLS
    assert _SOURCE_URLS["USGS 3DEP 10m"].startswith("https://www.usgs.gov/")
    assert "USGS 3DEP 10m" in _DEM_COVERAGE
```

In `tests/test_generator_gsi_dem.py`, rename `test_usa_downgrades_to_flat_when_earth_engine_unavailable` to `test_usa_uses_usgs_10m_when_earth_engine_unavailable`, change its docstring to "Outside Japan the US DEM is read from AWS (USGS 3DEP 10m) when EE is unavailable.", and its assertion to `== "USGS 3DEP 10m"`.

- [ ] **Step 2: Run to verify they fail**

Run: `"C:/Users/kunih/miniconda3/Scripts/conda.exe" run -n voxcity python -m pytest tests/test_generator_nologin_sources.py tests/test_generator_gsi_dem.py -q -p no:cacheprovider`
Expected: `test_no_ee_dem_usa_is_usgs_10m`, `test_no_ee_dem_germany_is_fabdem`, `test_usgs_10m_registered` and the renamed GSI test fail ('Flat' / KeyError); the Japan test passes.

- [ ] **Step 3: Implement**

(a) `_SOURCE_URLS`: after the `'USGS 3DEP 1m'` entry add
`'USGS 3DEP 10m': 'https://www.usgs.gov/3d-elevation-program/about-3dep-products-services',`

(b) `_DEM_COVERAGE`: after `'USGS 3DEP 1m'` add `'USGS 3DEP 10m': lambda f: f['is_usa'],`

(c) No-Earth-Engine block, land cover: change
`if auto_sources['land_cover_source'] not in ('OpenStreetMap', 'OpenEarthMapJapan'):` to
`if auto_sources['land_cover_source'] not in ('OpenStreetMap', 'OpenEarthMapJapan', 'ESA WorldCover'):`

(d) No-Earth-Engine block, DEM: replace

```python
            # DEM: fallback to Flat (no elevation data without GEE), except
            # for sources that don't require Earth Engine. 'GSI DEM Japan'
            # downloads bare-earth DEM tiles directly over HTTP from GSI, so
            # it stays usable even when GEE is unavailable.
            if auto_sources['dem_source'] != 'GSI DEM Japan':
                auto_sources['dem_source'] = 'Flat'
```

with

```python
            # DEM: sources read without Earth Engine or any account.
            # Japan keeps 'GSI DEM Japan' (GSI tiles over HTTP); the US gets
            # the USGS 3DEP 1/3 arc-second bare-earth DEM from AWS; everywhere
            # else gets FABDEM (Bristol archives) - the same DEM Earth Engine
            # users get by default there.
            if auto_sources['dem_source'] == 'GSI DEM Japan':
                pass
            elif auto_sources['dem_source'] == 'USGS 3DEP 1m':
                auto_sources['dem_source'] = 'USGS 3DEP 10m'
            else:
                auto_sources['dem_source'] = 'FABDEM'
```

(e) Docstrings. In `auto_select_data_sources`, after the DEM bullet add:
`  When Earth Engine is unavailable: Japan -> 'GSI DEM Japan', USA -> 'USGS 3DEP 10m', elsewhere -> 'FABDEM' (all read without an account).`
In `get_voxcity`, after the `dem_source:` arg line add:

```
            'COPERNICUS', 'FABDEM' and 'USGS 3DEP 10m' are read without Earth Engine or
            any account (AWS Open Data / University of Bristol); FABDEM is CC BY-NC-SA 4.0.
```

and after the `land_cover_source:` arg line add:

```
            'ESA WorldCover' is read from AWS Open Data without Earth Engine or any account.
```

- [ ] **Step 4: Run the tests**

Run: `"C:/Users/kunih/miniconda3/Scripts/conda.exe" run -n voxcity python -m pytest tests/test_generator_nologin_sources.py tests/test_generator_gsi_dem.py tests/test_generator_canopy_chmv2.py tests/test_import_hygiene.py -q -p no:cacheprovider`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/voxcity/generator/api.py tests/test_generator_nologin_sources.py tests/test_generator_gsi_dem.py
git commit -m "feat(api): no-Earth-Engine DEM defaults to USGS 3DEP 10m (US) and FABDEM elsewhere

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 9: Live tests, coverage omit, docs

**Files:**
- Create: `tests/test_downloader_nologin_integration.py`
- Modify: `pyproject.toml`, `README.md`, `CHANGELOG.md`, `docs/reference/data_sources.md`

- [ ] **Step 1: Write the live tests**

Create `tests/test_downloader_nologin_integration.py`:

```python
"""Live tests for the login-free DEM and land cover sources. Skipped by default.

Enable with:  VOXCITY_LIVE_AWS=1 pytest tests/test_downloader_nologin_integration.py
No account or credentials are needed.
"""
import os

import numpy as np
import pytest

LIVE = os.environ.get("VOXCITY_LIVE_AWS") == "1"
pytestmark = [
    pytest.mark.slow,
    pytest.mark.integration,
    pytest.mark.skipif(not LIVE, reason="set VOXCITY_LIVE_AWS=1 to run"),
]

BERLIN = [(13.366, 52.513), (13.374, 52.513), (13.374, 52.518), (13.366, 52.518)]
MANHATTAN = [(-73.99, 40.75), (-73.97, 40.75), (-73.97, 40.765), (-73.99, 40.765)]


def _read(path):
    import rasterio
    with rasterio.open(path) as ds:
        return ds.read(), ds.crs.to_epsg(), ds.dtypes[0]


def test_copernicus_berlin(tmp_path):
    from voxcity.downloader.aws_opendata import fetch_copernicus_dem
    a, epsg, dtype = _read(fetch_copernicus_dem(BERLIN, str(tmp_path)))
    assert epsg == 4326 and dtype == "float32"
    assert 20 < np.median(a) < 80


def test_usgs_manhattan(tmp_path):
    from voxcity.downloader.aws_opendata import fetch_usgs_3dep_10m
    a, epsg, _ = _read(fetch_usgs_3dep_10m(MANHATTAN, str(tmp_path)))
    assert epsg == 4269
    assert np.isfinite(a).all() and a.min() > -50 and a.max() < 60


def test_fabdem_berlin(tmp_path):
    from voxcity.downloader.fabdem import fetch_fabdem
    a, epsg, _ = _read(fetch_fabdem(BERLIN, str(tmp_path)))
    assert epsg == 4326
    assert 25 < np.median(a) < 50


def test_worldcover_berlin(tmp_path):
    from voxcity.downloader.aws_opendata import fetch_esa_worldcover
    a, epsg, dtype = _read(fetch_esa_worldcover(BERLIN, str(tmp_path)))
    assert epsg == 4326 and dtype == "uint8" and a.shape[0] == 3
    trees = (a[0] == 0) & (a[1] == 112) & (a[2] == 0)
    assert trees.mean() > 0.3   # Tiergarten is mostly woodland
```

- [ ] **Step 2: Run live once, then confirm skip**

Run: `export VOXCITY_LIVE_AWS=1; "C:/Users/kunih/miniconda3/Scripts/conda.exe" run -n voxcity python -m pytest tests/test_downloader_nologin_integration.py -q -p no:cacheprovider`
Expected: `4 passed`. Then without the variable (`unset VOXCITY_LIVE_AWS`): `4 skipped`.

If a live assertion fails, print the actual value range and report it; do not widen ranges without stating the measured values in the report.

- [ ] **Step 3: Coverage omit**

In `pyproject.toml` `[tool.coverage.run] omit`, after the `aws_opendata.py` entry add `"*/voxcity/downloader/fabdem.py",`. (`cog.py` is covered by offline tests and is not omitted.)

- [ ] **Step 4: README and data sources page**

In `README.md`, read the "Flexible Input Sources" list (~line 165) and the "References of Data Sources" DEM and land cover tables (~line 690). Make these edits:
- Feature list: after the DEM line add `  - No account needed: CHMv2 canopy, ESA WorldCover, Copernicus DEM, FABDEM and USGS 3DEP 10m are read directly without Earth Engine.`
- DEM table: add a row `| [USGS 3DEP 1/3 arc-second](https://www.usgs.gov/3d-elevation-program/about-3dep-products-services) | United States | ~10 m | Bare-earth DEM from lidar and other sources, read from AWS Open Data without an account |`. If the table already has FABDEM and Copernicus rows, append " · read without an account" to their last column; FABDEM's row also gets " · CC BY-NC-SA 4.0".
- Land cover table: append " · read from AWS Open Data without an account" to the ESA WorldCover row.

In `docs/reference/data_sources.md`, read the file and mirror the same additions in its format.

- [ ] **Step 5: CHANGELOG**

Under the existing `## Unreleased` → `### Added` section, add:

```markdown
- **Terrain and land cover without Earth Engine or any account.**
  `'COPERNICUS'` (Copernicus GLO-30 on AWS), `'ESA WorldCover'` (2021 v200 on
  AWS) and `'FABDEM'` (University of Bristol archives) are now read directly,
  with only the window around the area transferred. New source
  `'USGS 3DEP 10m'` reads the US 1/3 arc-second bare-earth DEM from AWS.
  When Earth Engine is unavailable, the auto-selected DEM is now
  `'USGS 3DEP 10m'` in the US and `'FABDEM'` elsewhere (Japan keeps
  `'GSI DEM Japan'`) instead of `'Flat'`. If a DEM cannot be read the grid
  falls back to flat; if WorldCover cannot be read land cover falls back to
  OpenStreetMap.
```

and under a new `### Changed` heading in the same Unreleased section:

```markdown
- `'ESA WorldCover'`, `'COPERNICUS'` and `'FABDEM'` no longer use Earth
  Engine even when it is available. The data is the same; it is read at
  native resolution instead of through an Earth Engine export, so values can
  differ slightly at cell edges. FABDEM remains CC BY-NC-SA 4.0
  (non-commercial). No global DEM removes buildings in high-rise cores; in
  such areas prefer national lidar or `'Flat'`.
```

- [ ] **Step 6: Full offline suite**

Run: `"C:/Users/kunih/miniconda3/Scripts/conda.exe" run -n voxcity python -m pytest tests -q -m "not slow and not integration and not gee" -p no:cacheprovider` (timeout 600000 ms).
Expected: only the 4 known pre-existing failures (`tests/importer/test_voxelize_meshlib.py::test_meshlib_matches_trimesh_on_cube`, three in `tests/test_readme_demo_gif.py`). Report any other failure with its output.

- [ ] **Step 7: Commit**

```bash
git add tests/test_downloader_nologin_integration.py pyproject.toml README.md CHANGELOG.md docs/reference/data_sources.md
git commit -m "docs: login-free DEM and land cover sources; live tests

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

## After the plan (controller, not an implementer task)

Validation run, published as a page: the five-site lidar benchmark through `get_dem_grid` for `'FABDEM'`, `'COPERNICUS'` and `'USGS 3DEP 10m'`, and full no-Earth-Engine `get_voxcity` builds for the Berlin box (FABDEM terrain, CHMv2 canopy) and a US box (USGS 3DEP 10m terrain).
