# AWS Open Data canopy source (Meta/WRI CHMv2) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `'High Resolution 1m Global Canopy Height Maps v2'` as a canopy source that reads Meta/WRI CHMv2 tiles from the public AWS bucket over anonymous HTTPS — no Earth Engine, no AWS account — so it works from Google Colab.

**Architecture:** A new module `src/voxcity/downloader/aws_opendata.py` holds a generic `fetch_cog_crop` (opens Cloud-Optimized GeoTIFFs via GDAL `/vsicurl/`, mosaics only the bbox window with `rasterio.merge.merge`, optionally warps to a coarser resolution with `rasterio.warp.reproject`, writes a small local GeoTIFF) and a thin CHMv2 adapter (zoom-10 quadkey math → tile URLs → `requests.head` existence probe → `fetch_cog_crop`). `get_canopy_height_grid` in `grids.py` gains a branch that runs *before* Earth Engine initialisation and reuses the existing GeoTIFF sampler; `api.py` registers the source and uses it as the no-Earth-Engine auto-selection.

**Tech Stack:** Python, `rasterio` 1.5 / GDAL 3.12 (already required), `requests` (already required), `numpy`, `pytest` with `unittest.mock.patch` / `monkeypatch`.

**Spec:** `docs/superpowers/specs/2026-10-23-aws-opendata-canopy-design.md`

**Branch:** `feat/aws-opendata-canopy` (already created off `main`; the spec commit is on it). Work on this branch directly — no worktree.

---

## Conventions for every task

- **Running tests.** `conda` is not on PATH. From the repo root, in the Bash tool:

  ```bash
  "C:/Users/kunih/miniconda3/Scripts/conda.exe" run -n voxcity python -m pytest tests/test_downloader_aws_opendata.py -q
  ```

  (PowerShell equivalent: `& "C:\Users\kunih\miniconda3\Scripts\conda.exe" run -n voxcity python -m pytest ... -q`.) `-q` overrides the `-v` in `pyproject.toml`'s `addopts`; keep it so output stays short.

- **Commits** use the repo's conventional style (`feat(scope): …`, `test(scope): …`, `docs: …`) and end with the attribution line:

  ```
  Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
  ```

- **No network in unit tests.** Every unit test either works on GeoTIFFs written into `tmp_path` or patches `requests.head` / the downloader functions. Only `tests/test_downloader_aws_opendata_integration.py` touches the network, and it is skipped unless `VOXCITY_LIVE_AWS=1`.

- **Verified facts you can rely on** (checked against the live bucket on 2026-10-23):
  - Tile URL pattern: `https://dataforgood-fb-data.s3.amazonaws.com/forests/v2/global/dinov3_global_chm_v2_ml3/chm/<10-digit quadkey>.tif`
  - Tiles are EPSG:3857 COGs, uint8, ~1.1943 m/px, 32768×32768 px, 512×512 blocks, overviews, no nodata tag; 255 is the "no canopy" sentinel.
  - Quadkeys for a point at lon 13.3, lat 52.1 (Berlin): `['1202120103']`. For bbox lon 13.35–13.37, lat 52.1: `['1202120103', '1202120112']`. For bbox lon 13.35–13.37, lat 52.26–52.27: `['1202120101', '1202120103', '1202120110', '1202120112']`. All four of those tiles exist (HEAD → 200).
  - `rasterio.merge.merge(sources, bounds=…)` pads area outside every source with 0 (it does **not** clip to the sources).
  - `rasterio` rejects `Resampling.max` for reads/merge ("can be used for warp operations but not for reads and writes"); `rasterio.warp.reproject` accepts it.

---

## File structure

| File | Responsibility |
|---|---|
| `src/voxcity/downloader/aws_opendata.py` (new) | Generic `fetch_cog_crop` + CHMv2 adapter (`_quadkey`, `_quadkeys_for_bbox`, `chmv2_tile_urls`, `fetch_chmv2_canopy_height`). Module-level `requests`/`rasterio` imports, same as `gsi.py` and `gba.py` (neither is on the import-hygiene forbidden list; `grids.py` already imports `rasterio` at module top). |
| `src/voxcity/generator/grids.py` (modify) | New `_finish_canopy_grid` helper; new CHMv2 branch in `get_canopy_height_grid` before Earth Engine init. |
| `src/voxcity/generator/api.py` (modify) | `_SOURCE_URLS` entry; no-EE auto-selection uses CHMv2; docstring mentions the source and the two new kwargs. |
| `tests/test_downloader_aws_opendata.py` (new) | Unit tests for the module, offline. |
| `tests/test_generator_canopy_chmv2.py` (new) | Unit tests for the `grids.py` branch and `api.py` auto-selection, offline. |
| `tests/test_downloader_aws_opendata_integration.py` (new) | Live test, opt-in via `VOXCITY_LIVE_AWS=1`. |
| `pyproject.toml` (modify) | Coverage `omit` entry. |
| `README.md`, `CHANGELOG.md` (modify) | Source lists and changelog entry. |
| `docs/superpowers/specs/2026-10-23-aws-opendata-canopy-design.md` (modify) | Three small corrections discovered while planning (see Task 9). |

---

### Task 1: Module skeleton and quadkey math

**Files:**
- Create: `src/voxcity/downloader/aws_opendata.py`
- Test: `tests/test_downloader_aws_opendata.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_downloader_aws_opendata.py`:

```python
"""Tests for voxcity.downloader.aws_opendata (offline)."""
import os

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from voxcity.downloader.aws_opendata import (
    CHMV2_BASE_URL,
    _quadkey,
    _quadkeys_for_bbox,
)


class TestQuadkey:
    def test_zoom1_corners(self):
        # Bing quadkey digits: 0=NW, 1=NE, 2=SW, 3=SE
        assert _quadkey(0, 0, 1) == "0"
        assert _quadkey(1, 0, 1) == "1"
        assert _quadkey(0, 1, 1) == "2"
        assert _quadkey(1, 1, 1) == "3"

    def test_length_equals_zoom(self):
        assert len(_quadkey(523, 341, 10)) == 10


class TestQuadkeysForBbox:
    def test_berlin_point_matches_live_tile_key(self):
        # Verified against the bucket's tiles.geojson on 2026-10-23.
        assert _quadkeys_for_bbox(13.3, 52.1, 13.3, 52.1) == ["1202120103"]

    def test_bbox_straddling_tile_edge_yields_two_keys(self):
        assert _quadkeys_for_bbox(13.35, 52.1, 13.37, 52.1) == [
            "1202120103",
            "1202120112",
        ]

    def test_bbox_straddling_both_axes_yields_four_sorted_unique_keys(self):
        keys = _quadkeys_for_bbox(13.35, 52.26, 13.37, 52.27)
        assert keys == ["1202120101", "1202120103", "1202120110", "1202120112"]
        assert keys == sorted(set(keys))
        assert all(len(k) == 10 for k in keys)


def test_base_url_has_no_trailing_slash():
    assert not CHMV2_BASE_URL.endswith("/")
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `"C:/Users/kunih/miniconda3/Scripts/conda.exe" run -n voxcity python -m pytest tests/test_downloader_aws_opendata.py -q`
Expected: collection error, `ModuleNotFoundError: No module named 'voxcity.downloader.aws_opendata'`.

- [ ] **Step 3: Create the module with the quadkey helpers**

Create `src/voxcity/downloader/aws_opendata.py`:

```python
"""
Downloader for rasters published on the AWS Open Data Registry.

Everything here goes over plain anonymous HTTPS. There is no boto3, no AWS
CLI and no credential lookup, so it works the same on a laptop, in CI and in
Google Colab without signing in to anything.

Two layers:

- ``fetch_cog_crop``: generic. Given URLs (or local paths) of Cloud-Optimized
  GeoTIFFs, read only the window covering a rectangle through GDAL's
  ``/vsicurl/`` (HTTP range requests), mosaic the pieces, optionally resample
  to a coarser resolution, and write a small local GeoTIFF.
- CHMv2 adapter: Meta / WRI High Resolution Canopy Height Maps v2
  (https://registry.opendata.aws/dataforgood-fb-forestsv2/, CC BY 4.0).
  Tiles are zoom-10 Web Mercator tiles named by their 10-digit Bing quadkey,
  EPSG:3857, ~1.19 m/px, uint8 with 255 as the "no canopy" sentinel.

Usage:
    path = fetch_chmv2_canopy_height(rectangle_vertices, output_dir)
"""

from __future__ import annotations

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
from .gsi import latlon_to_tile

__all__ = ["fetch_cog_crop", "chmv2_tile_urls", "fetch_chmv2_canopy_height", "CHMV2_BASE_URL"]

_logger = get_logger(__name__)

CHMV2_BASE_URL = (
    "https://dataforgood-fb-data.s3.amazonaws.com/forests/v2/global/dinov3_global_chm_v2_ml3"
)
CHMV2_ZOOM = 10


def _bbox_from_rectangle_vertices(vertices: Sequence[Tuple[float, float]]) -> Tuple[float, float, float, float]:
    """Return (min_lon, min_lat, max_lon, max_lat) from (lon, lat) vertices."""
    if not vertices:
        raise ValueError("rectangle_vertices must be a non-empty sequence of (lon, lat)")
    lons = [v[0] for v in vertices]
    lats = [v[1] for v in vertices]
    return (min(lons), min(lats), max(lons), max(lats))


def _quadkey(x: int, y: int, zoom: int) -> str:
    """Bing Maps quadkey for XYZ tile (x, y) at ``zoom``."""
    digits = []
    for i in range(zoom, 0, -1):
        digit = 0
        mask = 1 << (i - 1)
        if x & mask:
            digit += 1
        if y & mask:
            digit += 2
        digits.append(str(digit))
    return "".join(digits)


def _quadkeys_for_bbox(
    min_lon: float, min_lat: float, max_lon: float, max_lat: float, zoom: int = CHMV2_ZOOM
) -> List[str]:
    """Sorted, de-duplicated quadkeys of every ``zoom`` tile touching the bbox."""
    x0, y0 = latlon_to_tile(max_lat, min_lon, zoom)  # top-left
    x1, y1 = latlon_to_tile(min_lat, max_lon, zoom)  # bottom-right
    keys = {
        _quadkey(x, y, zoom)
        for x in range(min(x0, x1), max(x0, x1) + 1)
        for y in range(min(y0, y1), max(y0, y1) + 1)
    }
    return sorted(keys)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `"C:/Users/kunih/miniconda3/Scripts/conda.exe" run -n voxcity python -m pytest tests/test_downloader_aws_opendata.py -q`
Expected: `7 passed`.

- [ ] **Step 5: Commit**

```bash
git add src/voxcity/downloader/aws_opendata.py tests/test_downloader_aws_opendata.py
git commit -m "feat(downloader): aws_opendata module with CHMv2 quadkey math

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 2: `chmv2_tile_urls` with HEAD probing

**Files:**
- Modify: `src/voxcity/downloader/aws_opendata.py`
- Test: `tests/test_downloader_aws_opendata.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_downloader_aws_opendata.py` (and add `chmv2_tile_urls` to the import list at the top):

```python
from unittest.mock import patch

import requests as _requests

from voxcity.downloader.aws_opendata import chmv2_tile_urls


class _Resp:
    def __init__(self, status_code):
        self.status_code = status_code


class TestChmv2TileUrls:
    BBOX_VERTS = [(13.35, 52.1), (13.37, 52.1), (13.37, 52.1), (13.35, 52.1)]

    def test_keeps_only_tiles_that_exist(self):
        # Two candidate keys: 1202120103 (exists) and 1202120112 (pretend missing).
        def fake_head(url, timeout):
            return _Resp(200 if url.endswith("/chm/1202120103.tif") else 404)

        with patch("voxcity.downloader.aws_opendata.requests.head", side_effect=fake_head) as head:
            urls = chmv2_tile_urls(self.BBOX_VERTS)

        assert urls == [f"{CHMV2_BASE_URL}/chm/1202120103.tif"]
        assert head.call_count == 2

    def test_request_exception_counts_as_missing(self):
        with patch(
            "voxcity.downloader.aws_opendata.requests.head",
            side_effect=_requests.ConnectionError("offline"),
        ):
            assert chmv2_tile_urls(self.BBOX_VERTS) == []

    def test_custom_base_url_trailing_slash_is_normalised(self):
        with patch("voxcity.downloader.aws_opendata.requests.head", return_value=_Resp(200)):
            urls = chmv2_tile_urls([(13.3, 52.1)], base_url="https://example.org/mirror/")
        assert urls == ["https://example.org/mirror/chm/1202120103.tif"]
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `"C:/Users/kunih/miniconda3/Scripts/conda.exe" run -n voxcity python -m pytest tests/test_downloader_aws_opendata.py -q`
Expected: `ImportError: cannot import name 'chmv2_tile_urls'`.

- [ ] **Step 3: Implement**

Append to `src/voxcity/downloader/aws_opendata.py`:

```python
def _remote_exists(url: str, timeout: int) -> bool:
    """True if an anonymous HEAD returns 200. Network errors count as missing."""
    try:
        return requests.head(url, timeout=timeout).status_code == 200
    except requests.RequestException:
        return False


def chmv2_tile_urls(
    rectangle_vertices: Sequence[Tuple[float, float]],
    base_url: str = CHMV2_BASE_URL,
    timeout: int = 30,
) -> List[str]:
    """
    URLs of the CHMv2 tiles that cover ``rectangle_vertices`` and exist.

    The dataset covers land only, so a missing tile key simply means no
    coverage; it is dropped rather than treated as an error.
    """
    min_lon, min_lat, max_lon, max_lat = _bbox_from_rectangle_vertices(rectangle_vertices)
    base = base_url.rstrip("/")
    urls = []
    for qk in _quadkeys_for_bbox(min_lon, min_lat, max_lon, max_lat):
        url = f"{base}/chm/{qk}.tif"
        if _remote_exists(url, timeout):
            urls.append(url)
        else:
            _logger.debug("CHMv2 tile %s not available", qk)
    return urls
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `"C:/Users/kunih/miniconda3/Scripts/conda.exe" run -n voxcity python -m pytest tests/test_downloader_aws_opendata.py -q`
Expected: `10 passed`.

- [ ] **Step 5: Commit**

```bash
git add src/voxcity/downloader/aws_opendata.py tests/test_downloader_aws_opendata.py
git commit -m "feat(downloader): chmv2_tile_urls probes tile existence over anonymous HTTPS

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 3: `fetch_cog_crop` — native-resolution windowed mosaic

**Files:**
- Modify: `src/voxcity/downloader/aws_opendata.py`
- Test: `tests/test_downloader_aws_opendata.py`

Test fixtures write tiny EPSG:3857 GeoTIFFs. The rectangle used by the tests is chosen in Web Mercator metres and converted back to lon/lat with `pyproj`, so the WGS84→3857 round trip inside `fetch_cog_crop` lands on known pixels.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_downloader_aws_opendata.py` (add `fetch_cog_crop` to the import list):

```python
from pyproj import Transformer

from voxcity.downloader.aws_opendata import fetch_cog_crop

_TO_LONLAT = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)


def _merc_rect(x0, y0, x1, y1):
    """Rectangle vertices (lon, lat) for an EPSG:3857 box."""
    corners = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
    return [tuple(_TO_LONLAT.transform(x, y)) for x, y in corners]


def _write_tile(path, left, top, values, res=1.0):
    """Write a uint8 EPSG:3857 GeoTIFF with its top-left corner at (left, top)."""
    values = np.asarray(values, dtype="uint8")
    transform = from_origin(left, top, res, res)
    with rasterio.open(
        path, "w", driver="GTiff", width=values.shape[1], height=values.shape[0],
        count=1, dtype="uint8", crs="EPSG:3857", transform=transform,
    ) as ds:
        ds.write(values, 1)
    return str(path)


# Tiles live near Berlin in EPSG:3857 so lat/lon <-> metres stays well-conditioned.
_X0, _Y0 = 1_480_000.0, 6_860_000.0   # top-left of tile A (metres)


@pytest.fixture
def tile_a(tmp_path):
    # 20x20 px, value = row index (0..19), 1 m/px, top-left (_X0, _Y0)
    vals = np.repeat(np.arange(20, dtype="uint8")[:, None], 20, axis=1)
    return _write_tile(tmp_path / "a.tif", _X0, _Y0, vals)


@pytest.fixture
def tile_b(tmp_path):
    # Same rows, immediately east of tile A, constant 200
    vals = np.full((20, 20), 200, dtype="uint8")
    return _write_tile(tmp_path / "b.tif", _X0 + 20, _Y0, vals)


class TestFetchCogCrop:
    def test_single_tile_crop_has_expected_extent_and_values(self, tile_a, tmp_path):
        # Box 5..15 m east, 5..15 m south of the tile origin -> rows 5..14
        rect = _merc_rect(_X0 + 5, _Y0 - 15, _X0 + 15, _Y0 - 5)
        out = fetch_cog_crop([tile_a], rect, str(tmp_path / "out.tif"))
        assert out == str(tmp_path / "out.tif")
        with rasterio.open(out) as ds:
            assert ds.crs.to_epsg() == 3857
            assert ds.dtypes[0] == "uint8"
            assert ds.res == (1.0, 1.0)
            # One-pixel padding on every side: rows 4..15 -> 12 px tall
            assert ds.height == 12 and ds.width == 12
            arr = ds.read(1)
        assert arr[0, 0] == 4 and arr[-1, 0] == 15

    def test_two_tiles_mosaic_seamlessly(self, tile_a, tile_b, tmp_path):
        # Box straddling the A|B boundary at x = _X0 + 20
        rect = _merc_rect(_X0 + 15, _Y0 - 10, _X0 + 25, _Y0 - 5)
        out = fetch_cog_crop([tile_a, tile_b], rect, str(tmp_path / "out.tif"))
        with rasterio.open(out) as ds:
            arr = ds.read(1)
            left = ds.transform.c
        col_of_boundary = int(round((_X0 + 20) - left))
        assert arr[0, col_of_boundary - 1] < 200   # from A (row value)
        assert arr[0, col_of_boundary] == 200       # from B

    def test_bbox_outside_tile_is_zero_filled(self, tile_a, tmp_path):
        # Box extends 10 m north of tile A (nothing there)
        rect = _merc_rect(_X0 + 2, _Y0 - 5, _X0 + 8, _Y0 + 10)
        out = fetch_cog_crop([tile_a], rect, str(tmp_path / "out.tif"))
        with rasterio.open(out) as ds:
            arr = ds.read(1)
        assert arr[0, :].max() == 0          # north rim: padding
        assert arr[-2, :].max() > 0          # inside tile A (row ~4)

    def test_unopenable_source_is_skipped(self, tile_a, tmp_path):
        rect = _merc_rect(_X0 + 5, _Y0 - 15, _X0 + 15, _Y0 - 5)
        out = fetch_cog_crop(
            [str(tmp_path / "missing.tif"), tile_a], rect, str(tmp_path / "out.tif")
        )
        assert out is not None and os.path.exists(out)

    def test_all_sources_unopenable_returns_none(self, tmp_path):
        rect = _merc_rect(_X0 + 5, _Y0 - 15, _X0 + 15, _Y0 - 5)
        out = fetch_cog_crop([str(tmp_path / "missing.tif")], rect, str(tmp_path / "out.tif"))
        assert out is None
        assert not os.path.exists(tmp_path / "out.tif")

    def test_creates_output_directory(self, tile_a, tmp_path):
        rect = _merc_rect(_X0 + 5, _Y0 - 15, _X0 + 15, _Y0 - 5)
        out = fetch_cog_crop([tile_a], rect, str(tmp_path / "nested" / "dir" / "out.tif"))
        assert os.path.exists(out)

    def test_http_url_gets_vsicurl_prefix(self, tile_a, tmp_path, monkeypatch):
        opened = []
        real_open = rasterio.open

        def spy_open(path, *a, **k):
            opened.append(path)
            if isinstance(path, str) and path.startswith("/vsicurl/"):
                raise RasterioIOError("no network in tests")
            return real_open(path, *a, **k)

        monkeypatch.setattr("voxcity.downloader.aws_opendata.rasterio.open", spy_open)
        rect = _merc_rect(_X0 + 5, _Y0 - 15, _X0 + 15, _Y0 - 5)
        fetch_cog_crop(["https://example.org/t.tif", tile_a], rect, str(tmp_path / "out.tif"))
        assert opened[0] == "/vsicurl/https://example.org/t.tif"
```

Also add `from rasterio.errors import RasterioIOError` to the test imports.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `"C:/Users/kunih/miniconda3/Scripts/conda.exe" run -n voxcity python -m pytest tests/test_downloader_aws_opendata.py -q`
Expected: `ImportError: cannot import name 'fetch_cog_crop'`.

- [ ] **Step 3: Implement the native-resolution path**

Append to `src/voxcity/downloader/aws_opendata.py`:

```python
# GDAL options for reading COGs over HTTP: skip directory listings (one HEAD
# instead of a bucket LIST), only treat .tif as rasters, retry transient errors.
_GDAL_HTTP_ENV = {
    "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
    "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".tif",
    "GDAL_HTTP_MAX_RETRY": "3",
    "GDAL_HTTP_RETRY_DELAY": "1",
}

_RESAMPLING = {
    "nearest": Resampling.nearest,
    "average": Resampling.average,
    "max": Resampling.max,
}


def _gdal_path(url: str) -> str:
    """Prefix http(s) URLs for GDAL's virtual curl filesystem; leave local paths alone."""
    if url.startswith(("http://", "https://")):
        return f"/vsicurl/{url}"
    return url


def fetch_cog_crop(
    urls: Sequence[str],
    rectangle_vertices: Sequence[Tuple[float, float]],
    out_path: str,
    *,
    res: Optional[float] = None,
    resampling: str = "nearest",
    timeout: int = 60,
) -> Optional[str]:
    """
    Crop one or more Cloud-Optimized GeoTIFFs to a rectangle and write a local GeoTIFF.

    Only the window covering the rectangle is read from each source (HTTP
    range requests for remote COGs), then the pieces are mosaicked. The output
    is in the CRS of the first openable source. Area not covered by any source
    is filled with 0.

    Args:
        urls: http(s) URLs or local paths of GeoTIFFs. Sources that fail to
              open are logged and skipped.
        rectangle_vertices: (lon, lat) vertices of the area of interest (WGS84).
        out_path: Where to write the cropped GeoTIFF.
        res: If given, resample the crop to this pixel size (in the source CRS
             units, metres for EPSG:3857) using ``resampling``.
        resampling: 'nearest' (default), 'average' or 'max'. Only used with ``res``.
        timeout: Per-request HTTP timeout in seconds (GDAL_HTTP_TIMEOUT).

    Returns:
        ``out_path``, or None if no source could be opened.
    """
    if resampling not in _RESAMPLING:
        raise ValueError(f"resampling must be one of {sorted(_RESAMPLING)}, got {resampling!r}")

    min_lon, min_lat, max_lon, max_lat = _bbox_from_rectangle_vertices(rectangle_vertices)
    env = dict(_GDAL_HTTP_ENV, GDAL_HTTP_TIMEOUT=str(timeout))

    with rasterio.Env(**env):
        sources = []
        try:
            for url in urls:
                try:
                    sources.append(rasterio.open(_gdal_path(url)))
                except RasterioIOError as exc:
                    _logger.info("Skipping %s: %s", url, exc)
            if not sources:
                return None

            crs = sources[0].crs
            px, py = sources[0].res
            left, bottom, right, top = transform_bounds(
                "EPSG:4326", crs, min_lon, min_lat, max_lon, max_lat
            )
            # One source pixel of padding so cell-centre sampling at the edge
            # never falls just outside the crop.
            bounds = (left - px, bottom - py, right + px, top + py)
            data, transform = merge(sources, bounds=bounds)
        finally:
            for src in sources:
                src.close()

    if res is not None:
        data, transform = _resample(data, transform, crs, res, _RESAMPLING[resampling])

    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    profile = {
        "driver": "GTiff",
        "height": data.shape[1],
        "width": data.shape[2],
        "count": 1,
        "dtype": data.dtype,
        "crs": crs,
        "transform": transform,
        "compress": "deflate",
    }
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(data[0], 1)
    return out_path


def _resample(data, transform, crs, res: float, resampling: Resampling):
    """Warp a (1, H, W) array to pixel size ``res`` in the same CRS."""
    src = data[0]
    width_m = src.shape[1] * transform.a
    height_m = src.shape[0] * (-transform.e)
    width = max(1, int(math.ceil(width_m / res)))
    height = max(1, int(math.ceil(height_m / res)))
    dst_transform = from_origin(transform.c, transform.f, res, res)
    dst = np.zeros((height, width), dtype=src.dtype)
    reproject(
        src, dst,
        src_transform=transform, src_crs=crs,
        dst_transform=dst_transform, dst_crs=crs,
        resampling=resampling,
    )
    return dst[np.newaxis, ...], dst_transform
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `"C:/Users/kunih/miniconda3/Scripts/conda.exe" run -n voxcity python -m pytest tests/test_downloader_aws_opendata.py -q`
Expected: `17 passed`.

If `test_single_tile_crop_has_expected_extent_and_values` fails on the exact height/width, the WGS84 round trip put the box edge within floating-point noise of a pixel boundary; shift the box by +0.5 m on each axis in that one test (rows then read 4/15 → still 4 and 15 after padding). Do not loosen the assertion to a range.

- [ ] **Step 5: Commit**

```bash
git add src/voxcity/downloader/aws_opendata.py tests/test_downloader_aws_opendata.py
git commit -m "feat(downloader): fetch_cog_crop mosaics the bbox window of remote or local COGs

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 4: `fetch_cog_crop` — `res` / `resampling` path

**Files:**
- Modify: `src/voxcity/downloader/aws_opendata.py` (already contains `_resample` from Task 3 — this task only adds tests, then fixes anything they reveal)
- Test: `tests/test_downloader_aws_opendata.py`

- [ ] **Step 1: Write the tests**

Append to `tests/test_downloader_aws_opendata.py`:

```python
class TestFetchCogCropResample:
    def test_res_max_coarsens_and_takes_block_maximum(self, tile_a, tmp_path):
        # Rows 0..19 have value = row. Crop rows 0..9 (+1 px pad -> 0..10 with
        # row -1 zero-filled), then 5 m blocks with 'max'.
        rect = _merc_rect(_X0 + 0, _Y0 - 10, _X0 + 10, _Y0 - 0)
        out = fetch_cog_crop([tile_a], rect, str(tmp_path / "out.tif"), res=5.0, resampling="max")
        with rasterio.open(out) as ds:
            assert ds.res == (5.0, 5.0)
            arr = ds.read(1)
        # First 5 m block covers padded row -1 (0) and rows 0..3 -> max 3
        assert arr[0, 0] == 3
        # Second block covers rows 4..8 -> max 8
        assert arr[1, 0] == 8

    def test_res_average_is_between_min_and_max(self, tile_a, tmp_path):
        rect = _merc_rect(_X0 + 0, _Y0 - 10, _X0 + 10, _Y0 - 0)
        out = fetch_cog_crop([tile_a], rect, str(tmp_path / "out.tif"), res=5.0, resampling="average")
        with rasterio.open(out) as ds:
            arr = ds.read(1)
        assert 4 <= arr[1, 0] <= 8

    def test_res_nearest_keeps_integer_values(self, tile_a, tmp_path):
        rect = _merc_rect(_X0 + 0, _Y0 - 10, _X0 + 10, _Y0 - 0)
        out = fetch_cog_crop([tile_a], rect, str(tmp_path / "out.tif"), res=2.0, resampling="nearest")
        with rasterio.open(out) as ds:
            arr = ds.read(1)
        assert set(np.unique(arr)) <= set(range(0, 20))

    def test_invalid_resampling_name_raises(self, tile_a, tmp_path):
        rect = _merc_rect(_X0 + 0, _Y0 - 10, _X0 + 10, _Y0 - 0)
        with pytest.raises(ValueError, match="resampling"):
            fetch_cog_crop([tile_a], rect, str(tmp_path / "out.tif"), res=5.0, resampling="cubic")
```

- [ ] **Step 2: Run the tests**

Run: `"C:/Users/kunih/miniconda3/Scripts/conda.exe" run -n voxcity python -m pytest tests/test_downloader_aws_opendata.py -q`
Expected: `21 passed`. (The implementation landed in Task 3; these tests confirm the `reproject` path works with all three resampling modes and that `max` — which `merge` would reject — is handled.)

If `test_res_max_coarsens_and_takes_block_maximum` fails on the exact values, print `arr` and check the padding: the crop's top edge is at `_Y0 + 1` (one pixel of padding), so the first 5 m block spans source rows −1..3. If the first block instead spans rows 0..4 the padding is missing — fix `fetch_cog_crop`, not the test.

- [ ] **Step 3: Commit**

```bash
git add tests/test_downloader_aws_opendata.py
git commit -m "test(downloader): cover fetch_cog_crop resampling via reproject

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 5: `fetch_chmv2_canopy_height`

**Files:**
- Modify: `src/voxcity/downloader/aws_opendata.py`
- Test: `tests/test_downloader_aws_opendata.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_downloader_aws_opendata.py` (add `fetch_chmv2_canopy_height` to the imports):

```python
from voxcity.downloader.aws_opendata import fetch_chmv2_canopy_height


class TestFetchChmv2CanopyHeight:
    VERTS = [(13.3, 52.1), (13.31, 52.1), (13.31, 52.11), (13.3, 52.11)]

    def test_writes_canopy_height_tif_in_output_dir(self, tmp_path):
        calls = {}

        def fake_tile_urls(verts, base_url=CHMV2_BASE_URL, timeout=30):
            return ["https://example.org/chm/1202120103.tif"]

        def fake_fetch(urls, verts, out_path, *, res=None, resampling="nearest", timeout=60):
            calls.update(urls=urls, out_path=out_path, res=res, resampling=resampling)
            open(out_path, "wb").close()
            return out_path

        with patch("voxcity.downloader.aws_opendata.chmv2_tile_urls", side_effect=fake_tile_urls), \
             patch("voxcity.downloader.aws_opendata.fetch_cog_crop", side_effect=fake_fetch):
            out = fetch_chmv2_canopy_height(
                self.VERTS, str(tmp_path / "out"), res=5.0, resampling="max"
            )

        assert out == str(tmp_path / "out" / "canopy_height.tif")
        assert os.path.exists(out)
        assert calls["urls"] == ["https://example.org/chm/1202120103.tif"]
        assert calls["res"] == 5.0 and calls["resampling"] == "max"

    def test_no_tiles_returns_none_without_fetching(self, tmp_path):
        with patch("voxcity.downloader.aws_opendata.chmv2_tile_urls", return_value=[]), \
             patch("voxcity.downloader.aws_opendata.fetch_cog_crop") as fetch:
            assert fetch_chmv2_canopy_height(self.VERTS, str(tmp_path)) is None
        fetch.assert_not_called()

    def test_fetch_failure_propagates_none(self, tmp_path):
        with patch("voxcity.downloader.aws_opendata.chmv2_tile_urls", return_value=["u"]), \
             patch("voxcity.downloader.aws_opendata.fetch_cog_crop", return_value=None):
            assert fetch_chmv2_canopy_height(self.VERTS, str(tmp_path)) is None
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `"C:/Users/kunih/miniconda3/Scripts/conda.exe" run -n voxcity python -m pytest tests/test_downloader_aws_opendata.py -q`
Expected: `ImportError: cannot import name 'fetch_chmv2_canopy_height'`.

- [ ] **Step 3: Implement**

Append to `src/voxcity/downloader/aws_opendata.py`:

```python
def fetch_chmv2_canopy_height(
    rectangle_vertices: Sequence[Tuple[float, float]],
    output_dir: str,
    *,
    res: Optional[float] = None,
    resampling: str = "nearest",
    base_url: str = CHMV2_BASE_URL,
) -> Optional[str]:
    """
    Write ``output_dir/canopy_height.tif`` cropped from Meta/WRI CHMv2 tiles.

    Anonymous HTTPS only; nothing to sign in to. Returns the GeoTIFF path, or
    None when no tile covers the area or none could be read (offline).
    """
    urls = chmv2_tile_urls(rectangle_vertices, base_url=base_url)
    if not urls:
        _logger.info("CHMv2: no tiles cover this area (or the bucket is unreachable)")
        return None
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "canopy_height.tif")
    return fetch_cog_crop(urls, rectangle_vertices, out_path, res=res, resampling=resampling)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `"C:/Users/kunih/miniconda3/Scripts/conda.exe" run -n voxcity python -m pytest tests/test_downloader_aws_opendata.py -q`
Expected: `24 passed`.

- [ ] **Step 5: Commit**

```bash
git add src/voxcity/downloader/aws_opendata.py tests/test_downloader_aws_opendata.py
git commit -m "feat(downloader): fetch_chmv2_canopy_height writes canopy_height.tif from AWS tiles

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 6: Extract `_finish_canopy_grid` in `grids.py` (pure refactor)

**Files:**
- Modify: `src/voxcity/generator/grids.py:437-453`
- Test: existing canopy tests (`tests/test_pipeline_osm_canopy_coverage.py`, `tests/test_parallel_canopy_repair.py`, `tests/test_pipeline_sequential_coverage.py`, `tests/test_pipeline_parallel_coverage.py`)

- [ ] **Step 1: Run the existing canopy tests to get a baseline**

Run: `"C:/Users/kunih/miniconda3/Scripts/conda.exe" run -n voxcity python -m pytest tests/test_pipeline_osm_canopy_coverage.py tests/test_parallel_canopy_repair.py tests/test_pipeline_sequential_coverage.py tests/test_pipeline_parallel_coverage.py -q`
Expected: all pass. Note the count.

- [ ] **Step 2: Add the helper and use it in the Earth Engine path**

In `src/voxcity/generator/grids.py`, insert this function immediately **above** `def get_canopy_height_grid(` (currently line 326):

```python
def _finish_canopy_grid(canopy_height_grid, meshsize, **kwargs):
    """Shared tail for raster canopy sources: sentinel cleanup, trunk ratio, viz.

    The Meta and ETH datasets store canopy height as uint8 and use 255 as
    their nodata / "no canopy" sentinel.  Replace with 0 so downstream code
    treats those cells as bare ground instead of 255-metre-tall trees.
    """
    canopy_height_grid = np.where(canopy_height_grid >= 255, 0, canopy_height_grid)

    trunk_height_ratio = kwargs.get("trunk_height_ratio")
    if trunk_height_ratio is None:
        trunk_height_ratio = 11.76 / 19.98
    canopy_bottom_grid = canopy_height_grid * float(trunk_height_ratio)

    grid_vis = kwargs.get("gridvis", True)
    if grid_vis:
        canopy_height_grid_nan = canopy_height_grid.copy()
        canopy_height_grid_nan[canopy_height_grid_nan == 0] = np.nan
        from ..visualizer.grids import visualize_numerical_grid
        visualize_numerical_grid(canopy_height_grid_nan, meshsize, "Tree canopy height", cmap='Greens', label='Tree canopy height (m)')
    return canopy_height_grid, canopy_bottom_grid
```

Then replace the tail of `get_canopy_height_grid` — everything from the comment `# The ETH dataset stores canopy height as uint8 and uses 255 as its` down to and including `return canopy_height_grid, canopy_bottom_grid` (currently lines 437–453) — with:

```python
    return _finish_canopy_grid(canopy_height_grid, meshsize, **kwargs)
```

The function now ends:

```python
    save_geotiff(image, geotiff_path, scale=meshsize, region=roi, crs='EPSG:4326')
    canopy_height_grid = create_height_grid_from_geotiff_polygon(geotiff_path, meshsize, rectangle_vertices)
    return _finish_canopy_grid(canopy_height_grid, meshsize, **kwargs)
```

- [ ] **Step 3: Run the same tests again**

Run: `"C:/Users/kunih/miniconda3/Scripts/conda.exe" run -n voxcity python -m pytest tests/test_pipeline_osm_canopy_coverage.py tests/test_parallel_canopy_repair.py tests/test_pipeline_sequential_coverage.py tests/test_pipeline_parallel_coverage.py tests/test_import_hygiene.py -q`
Expected: same count passing as Step 1 plus the import-hygiene tests, no failures.

- [ ] **Step 4: Commit**

```bash
git add src/voxcity/generator/grids.py
git commit -m "refactor(generator): extract _finish_canopy_grid from get_canopy_height_grid

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 7: CHMv2 branch in `get_canopy_height_grid`

**Files:**
- Modify: `src/voxcity/generator/grids.py` (insert before the `try: initialize_earth_engine()` block)
- Test: `tests/test_generator_canopy_chmv2.py` (new)

- [ ] **Step 1: Write the failing tests**

Create `tests/test_generator_canopy_chmv2.py`:

```python
"""Tests for the CHMv2 (AWS Open Data) canopy source wired into get_canopy_height_grid."""
import os
from unittest.mock import patch

import numpy as np
import rasterio
from pyproj import Transformer
from rasterio.transform import from_origin

from voxcity.generator.grids import get_canopy_height_grid
from voxcity.geoprocessor.raster.core import compute_grid_shape

CHMV2 = "High Resolution 1m Global Canopy Height Maps v2"

# ~100 m x 100 m box in Berlin
VERTS = [(13.3000, 52.1000), (13.3015, 52.1000), (13.3015, 52.1009), (13.3000, 52.1009)]
MESH = 10.0

_TO_MERC = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)


def _write_constant_tif(path, value):
    """Write a 1 m EPSG:3857 raster of ``value`` generously covering VERTS."""
    xs, ys = zip(*[_TO_MERC.transform(lon, lat) for lon, lat in VERTS])
    left, top = min(xs) - 20, max(ys) + 20
    width = int(max(xs) - min(xs)) + 40
    height = int(max(ys) - min(ys)) + 40
    with rasterio.open(
        path, "w", driver="GTiff", width=width, height=height, count=1, dtype="uint8",
        crs="EPSG:3857", transform=from_origin(left, top, 1.0, 1.0),
    ) as ds:
        ds.write(np.full((height, width), value, dtype="uint8"), 1)
    return str(path)


def _fake_fetch_returning(value):
    def _fetch(rectangle_vertices, output_dir, *, res=None, resampling="nearest", base_url=None):
        os.makedirs(output_dir, exist_ok=True)
        return _write_constant_tif(os.path.join(output_dir, "canopy_height.tif"), value)
    return _fetch


def test_chmv2_builds_grid_without_earth_engine(tmp_path):
    with patch("voxcity.downloader.aws_opendata.fetch_chmv2_canopy_height",
               side_effect=_fake_fetch_returning(12)) as fetch, \
         patch("voxcity.downloader.gee.initialize_earth_engine",
               side_effect=AssertionError("Earth Engine must not be touched")):
        top, bottom = get_canopy_height_grid(VERTS, MESH, CHMV2, str(tmp_path), gridvis=False)

    assert top.shape == compute_grid_shape(VERTS, MESH)
    assert np.all(top == 12)
    np.testing.assert_allclose(bottom, top * (11.76 / 19.98))
    # kwargs defaults forwarded
    _, kwargs = fetch.call_args
    assert kwargs == {"res": None, "resampling": "nearest"}


def test_chmv2_forwards_resampling_kwargs(tmp_path):
    with patch("voxcity.downloader.aws_opendata.fetch_chmv2_canopy_height",
               side_effect=_fake_fetch_returning(12)) as fetch:
        get_canopy_height_grid(
            VERTS, MESH, CHMV2, str(tmp_path), gridvis=False,
            canopy_resample_res=MESH, canopy_resampling="max",
        )
    _, kwargs = fetch.call_args
    assert kwargs == {"res": MESH, "resampling": "max"}


def test_chmv2_255_sentinel_becomes_zero(tmp_path):
    with patch("voxcity.downloader.aws_opendata.fetch_chmv2_canopy_height",
               side_effect=_fake_fetch_returning(255)):
        top, _ = get_canopy_height_grid(VERTS, MESH, CHMV2, str(tmp_path), gridvis=False)
    assert np.all(top == 0)


def test_chmv2_no_tiles_falls_back_to_static(tmp_path):
    shape = compute_grid_shape(VERTS, MESH)
    land_cover = np.zeros(shape, dtype=int)   # no tree class anywhere
    with patch("voxcity.downloader.aws_opendata.fetch_chmv2_canopy_height", return_value=None), \
         patch("voxcity.downloader.gee.initialize_earth_engine",
               side_effect=AssertionError("Earth Engine must not be touched")):
        top, bottom = get_canopy_height_grid(
            VERTS, MESH, CHMV2, str(tmp_path), gridvis=False,
            land_cover_like=land_cover, land_cover_source="OpenStreetMap",
        )
    assert top.shape == shape
    assert np.all(top == 0) and np.all(bottom == 0)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `"C:/Users/kunih/miniconda3/Scripts/conda.exe" run -n voxcity python -m pytest tests/test_generator_canopy_chmv2.py -q`
Expected: 4 failures. The first three fail with `AssertionError: Earth Engine must not be touched` or `ValueError: Unsupported canopy source` (the source is unknown and the code reaches the EE block); the fourth fails the same way.

- [ ] **Step 3: Add the branch**

In `src/voxcity/generator/grids.py`, inside `get_canopy_height_grid`, insert this block immediately **before** the line `    try:` that precedes `initialize_earth_engine()` (after the `GeoDataFrame` block's `return canopy_top, canopy_bottom`):

```python
    # AWS Open Data path (no Earth Engine, no account): Meta/WRI CHMv2 tiles
    # are read over anonymous HTTPS. Must run before initialize_earth_engine()
    # so an EE failure never masks this source.
    if source == 'High Resolution 1m Global Canopy Height Maps v2':
        from ..downloader.aws_opendata import fetch_chmv2_canopy_height
        geotiff_path = fetch_chmv2_canopy_height(
            rectangle_vertices, output_dir,
            res=kwargs.get('canopy_resample_res'),
            resampling=kwargs.get('canopy_resampling', 'nearest'),
        )
        if geotiff_path is None:
            _logger.warning(
                "CHMv2 (AWS) has no tiles for this area or could not be reached "
                "— falling back to Static canopy heights."
            )
            return get_canopy_height_grid(rectangle_vertices, meshsize, 'Static', output_dir, **kwargs)
        canopy_height_grid = create_height_grid_from_geotiff_polygon(geotiff_path, meshsize, rectangle_vertices)
        return _finish_canopy_grid(canopy_height_grid, meshsize, **kwargs)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `"C:/Users/kunih/miniconda3/Scripts/conda.exe" run -n voxcity python -m pytest tests/test_generator_canopy_chmv2.py tests/test_import_hygiene.py -q`
Expected: all pass (4 new + hygiene).

- [ ] **Step 5: Commit**

```bash
git add src/voxcity/generator/grids.py tests/test_generator_canopy_chmv2.py
git commit -m "feat(generator): 'High Resolution 1m Global Canopy Height Maps v2' canopy source via AWS Open Data

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 8: Register the source in `api.py` and use it when Earth Engine is unavailable

**Files:**
- Modify: `src/voxcity/generator/api.py` (`_SOURCE_URLS` ~line 59; no-EE block lines 506–513; `get_voxcity` docstring ~line 422)
- Test: `tests/test_generator_canopy_chmv2.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_generator_canopy_chmv2.py`:

```python
class _StopBeforeBuild(Exception):
    """Sentinel to abort get_voxcity right after source resolution."""


def _resolved_canopy_source_when_ee_unavailable(rectangle_vertices):
    """Run get_voxcity's auto-selection with Earth Engine unavailable and
    capture the canopy source it resolves to, short-circuiting before the
    heavy model build via a patched _warn_source_coverage (same pattern as
    tests/test_generator_gsi_dem.py)."""
    from voxcity.generator import api

    captured = {}

    def _capture(_verts, _b, _bc, _lc, canopy_height_source, _dem):
        captured["canopy"] = canopy_height_source
        raise _StopBeforeBuild

    with patch("voxcity.downloader.gee.initialize_earth_engine",
               side_effect=Exception("EE unavailable")), \
         patch.object(api, "_warn_source_coverage", side_effect=_capture):
        try:
            api.get_voxcity(rectangle_vertices, meshsize=10)
        except _StopBeforeBuild:
            pass
    return captured.get("canopy")


JAPAN_VERTS = [(139.75, 35.68), (139.76, 35.68), (139.76, 35.69), (139.75, 35.69)]
USA_VERTS = [(-122.42, 37.77), (-122.41, 37.77), (-122.41, 37.78), (-122.42, 37.78)]


def test_no_ee_auto_selects_chmv2_outside_japan():
    assert _resolved_canopy_source_when_ee_unavailable(USA_VERTS) == CHMV2


def test_no_ee_auto_selects_chmv2_in_japan():
    assert _resolved_canopy_source_when_ee_unavailable(JAPAN_VERTS) == CHMV2


def test_chmv2_has_a_source_url():
    from voxcity.generator.api import _SOURCE_URLS
    assert _SOURCE_URLS[CHMV2] == "https://registry.opendata.aws/dataforgood-fb-forestsv2/"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `"C:/Users/kunih/miniconda3/Scripts/conda.exe" run -n voxcity python -m pytest tests/test_generator_canopy_chmv2.py -q`
Expected: the three new tests fail (`'OpenStreetMap' != CHMV2`, `'Static' != CHMV2`, `KeyError`).

- [ ] **Step 3: Edit `api.py`**

(a) In `_SOURCE_URLS`, after the line
`    'High Resolution 1m Global Canopy Height Maps': 'https://gee-community-catalog.org/projects/meta_trees/',`
add:

```python
    'High Resolution 1m Global Canopy Height Maps v2': 'https://registry.opendata.aws/dataforgood-fb-forestsv2/',
```

(b) Replace the canopy part of the no-EE block (currently):

```python
            # Canopy height: region-dependent fallback
            # - Japan: 'Static' (OpenEarthMapJapan land cover provides good tree coverage)
            # - Other regions: 'OpenStreetMap' (provides actual tree locations and forest polygons)
            is_japan_area = (auto_sources['land_cover_source'] == 'OpenEarthMapJapan')
            if is_japan_area:
                auto_sources['canopy_height_source'] = 'Static'
            else:
                auto_sources['canopy_height_source'] = 'OpenStreetMap'
```

with:

```python
            # Canopy height: Meta/WRI CHMv2 is read from the AWS Open Data
            # bucket over anonymous HTTPS, so it needs neither Earth Engine
            # nor an account. get_canopy_height_grid falls back to 'Static'
            # by itself if the area has no tiles or the bucket is unreachable.
            auto_sources['canopy_height_source'] = 'High Resolution 1m Global Canopy Height Maps v2'
```

(c) In the `get_voxcity` docstring, after the line
`        canopy_height_source: Canopy height source (default: auto-selected based on location)`
insert:

```python
            Use 'High Resolution 1m Global Canopy Height Maps v2' for Meta/WRI CHMv2 read
            from AWS Open Data over anonymous HTTPS (no Earth Engine or account needed;
            this is also the automatic choice when Earth Engine is unavailable).
```

and in the `**kwargs` list, after the `force_refresh` bullet, add:

```python
                  - canopy_resample_res: float, for the CHMv2 source only: resample the ~1.2 m
                    tiles to this pixel size (m) before sampling at cell centres (default: None,
                    i.e. nearest native pixel, matching the Earth Engine sources).
                  - canopy_resampling: 'nearest' (default) | 'average' | 'max'; aggregation
                    used when canopy_resample_res is set.
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `"C:/Users/kunih/miniconda3/Scripts/conda.exe" run -n voxcity python -m pytest tests/test_generator_canopy_chmv2.py tests/test_generator_gsi_dem.py tests/test_import_hygiene.py -q`
Expected: all pass. (`test_generator_gsi_dem.py` exercises the same no-EE block for DEM and must stay green.)

- [ ] **Step 5: Commit**

```bash
git add src/voxcity/generator/api.py tests/test_generator_canopy_chmv2.py
git commit -m "feat(api): auto-select CHMv2 (AWS) canopy when Earth Engine is unavailable

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 9: Live integration test, coverage omit, docs, spec corrections

**Files:**
- Create: `tests/test_downloader_aws_opendata_integration.py`
- Modify: `pyproject.toml` (coverage `omit`, ~line 128), `README.md` (lines 169 and 692), `CHANGELOG.md` (top), `docs/superpowers/specs/2026-10-23-aws-opendata-canopy-design.md`

- [ ] **Step 1: Write the integration test**

Create `tests/test_downloader_aws_opendata_integration.py`:

```python
"""Live network integration test for the CHMv2 AWS Open Data download. Skipped by default.

Enable with:  VOXCITY_LIVE_AWS=1 pytest tests/test_downloader_aws_opendata_integration.py
No AWS account or credentials are needed; the bucket is public.
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


def test_berlin_window_download(tmp_path):
    import rasterio
    from voxcity.downloader.aws_opendata import fetch_chmv2_canopy_height

    # ~300 m x 300 m in Tiergarten, Berlin (tile 1202120103)
    verts = [(13.350, 52.512), (13.3545, 52.512), (13.3545, 52.5147), (13.350, 52.5147)]
    out = fetch_chmv2_canopy_height(verts, str(tmp_path))
    assert out == str(tmp_path / "canopy_height.tif")
    with rasterio.open(out) as ds:
        assert ds.crs.to_epsg() == 3857
        assert ds.dtypes[0] == "uint8"
        arr = ds.read(1)
    assert arr.shape[0] > 200 and arr.shape[1] > 200   # ~1.19 m/px over ~300 m
    assert arr.max() <= 255
    assert np.any((arr >= 1) & (arr <= 100))            # some trees in the park
```

- [ ] **Step 2: Run it live once**

Run: `VOXCITY_LIVE_AWS=1 "C:/Users/kunih/miniconda3/Scripts/conda.exe" run -n voxcity python -m pytest tests/test_downloader_aws_opendata_integration.py -q`
Expected: `1 passed` in a few seconds. Then run it without the env var and confirm `1 skipped`.

- [ ] **Step 3: Coverage omit**

In `pyproject.toml`, in the `[tool.coverage.run] omit` list, after `  "*/voxcity/downloader/ocean.py",` add:

```toml
  "*/voxcity/downloader/aws_opendata.py",
```

- [ ] **Step 4: README**

Line 169 — change

```
  - Canopy Height: High Resolution 1m Global Canopy Height Maps, ETH Global Sentinel-2 10m
```

to

```
  - Canopy Height: High Resolution 1m Global Canopy Height Maps (v1 via Earth Engine; v2 via AWS Open Data, no account needed), ETH Global Sentinel-2 10m
```

In the "Tree Canopy Height" table (~line 692), after the v1 row add:

```
| [High Resolution 1m Global Canopy Height Maps v2](https://registry.opendata.aws/dataforgood-fb-forestsv2/) | Worldwide (land) | ~1.2 m | Meta/WRI CHMv2, read from AWS Open Data over anonymous HTTPS — works in Google Colab without Earth Engine |
```

- [ ] **Step 5: CHANGELOG**

Insert at the top of `CHANGELOG.md`, directly under `# Changelog`:

```markdown
## Unreleased

### Added

- **Canopy height from AWS Open Data, no account required.** New source
  `'High Resolution 1m Global Canopy Height Maps v2'` reads Meta/WRI CHMv2
  Cloud-Optimized GeoTIFFs from the public `dataforgood-fb-data` bucket over
  anonymous HTTPS, fetching only the window covering the area (no 185 MB tile
  downloads, no Earth Engine, no AWS credentials; works in Google Colab).
  When Earth Engine is unavailable, `get_voxcity` now auto-selects this
  source instead of `'Static'`/`'OpenStreetMap'`; it still falls back to
  `'Static'` if the area has no tiles or the bucket cannot be reached.
  `voxcity.downloader.aws_opendata.fetch_cog_crop` is the reusable core for
  adding further AWS-hosted rasters. Optional kwargs `canopy_resample_res`
  and `canopy_resampling` ('nearest' | 'average' | 'max') control
  aggregation for coarse meshes.
```

- [ ] **Step 6: Spec corrections**

Three facts changed while planning; update `docs/superpowers/specs/2026-10-23-aws-opendata-canopy-design.md` so it matches the code:

(a) In **Constraints** item 5, replace the sentence with:
`5. **Import hygiene:** \`rasterio\` and \`requests\` are imported at module top like \`gsi.py\`/\`gba.py\` (neither is on the forbidden list in \`tests/test_import_hygiene.py\`); the module itself is imported lazily from \`grids.py\`.`

(b) In **Components → 1. Generic core**, replace the sentence beginning "Mosaics the windowed reads with" with:
`Mosaics the windowed reads with \`rasterio.merge.merge(datasets, bounds=…)\` at native resolution (area outside every source is zero-filled). If \`res\` is given, the mosaic is then warped to that pixel size with \`rasterio.warp.reproject\` using the named \`Resampling\` (\`nearest\` | \`average\` | \`max\`) — \`merge\` itself rejects \`max\`.`

(c) In **Testing**, replace "bbox partly outside the tile yields a crop clipped to the tile" with "bbox partly outside the tile yields a crop zero-filled outside the tile".

- [ ] **Step 7: Run the whole unit suite**

Run: `"C:/Users/kunih/miniconda3/Scripts/conda.exe" run -n voxcity python -m pytest tests -q -m "not slow and not integration and not gee" -x`
Expected: all pass, no failures. Paste the summary line into the commit message body if anything about the count is noteworthy.

- [ ] **Step 8: Commit**

```bash
git add tests/test_downloader_aws_opendata_integration.py pyproject.toml README.md CHANGELOG.md
git add -f docs/superpowers/specs/2026-10-23-aws-opendata-canopy-design.md
git commit -m "docs: document the CHMv2 AWS canopy source; add live integration test

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

(`docs/superpowers/` is gitignored; earlier specs were force-added, hence `-f`.)

---

## Done when

- `pytest tests -m "not slow and not integration and not gee"` is green.
- `VOXCITY_LIVE_AWS=1 pytest tests/test_downloader_aws_opendata_integration.py` passes.
- `get_voxcity(verts, 5, canopy_height_source='High Resolution 1m Global Canopy Height Maps v2', building_source='OpenStreetMap', land_cover_source='OpenStreetMap', dem_source='Flat')` runs end-to-end on a machine with no Earth Engine login (manual check — do it once in a fresh Colab or with `EARTHENGINE_TOKEN` unset).
