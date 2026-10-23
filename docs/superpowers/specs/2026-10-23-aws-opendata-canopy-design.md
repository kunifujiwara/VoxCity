# AWS Open Data canopy source (Meta/WRI CHMv2) — design

Date: 2026-10-23
Branch: feat/aws-opendata-canopy (off main)

## Goal

Let `get_voxcity` build a canopy-height grid from the Meta/WRI High Resolution
Canopy Height Maps v2 (CHMv2) hosted on the AWS Open Data Registry, with **no
account or sign-in to any service**. It must work identically in Google Colab,
on a laptop, and in CI. The implementation is split into a small reusable
"remote COG → cropped local GeoTIFF" core and a thin CHMv2 adapter, so later
AWS datasets (ESA WorldCover, Copernicus DEM, GHSL built height) can be added
as adapters of roughly thirty lines each.

## Dataset facts (verified 2026-10-23)

| Item | Value |
|---|---|
| Registry | https://registry.opendata.aws/dataforgood-fb-forestsv2/ (CC BY 4.0) |
| Bucket / prefix | `dataforgood-fb-data` / `forests/v2/global/dinov3_global_chm_v2_ml3/` (us-east-1) |
| HTTPS base | `https://dataforgood-fb-data.s3.amazonaws.com/forests/v2/global/dinov3_global_chm_v2_ml3/` |
| Tiles | `chm/<quadkey>.tif`, 10-digit Bing quadkeys = zoom-10 Web Mercator tiles |
| Per-tile metadata | `metadata/<quadkey>.geojson` (acquisition-date polygons; not used here) |
| Index | `tiles.geojson` (58 MB; not used — quadkeys are computed instead) |
| Raster | Cloud-Optimized GeoTIFF, EPSG:3857, 32768×32768 px, ~1.194 m/px, uint8, deflate, 512×512 blocks, overviews 2–64, no nodata tag (255 is the "no canopy" sentinel, as in the existing ETH path), ~185 MB per tile |
| Coverage | Land only; a missing tile key means no coverage, not an error |
| Anonymous access | Confirmed: bucket listing, `HEAD`, and `rasterio` `/vsicurl/` windowed reads all work with no credentials, no `~/.aws`, no env keys |

Measured: opening a tile ≈ 1.7 s, reading a 500 m × 500 m window ≈ 1.1 s.

## Constraints

1. **No login.** Plain anonymous HTTPS only. No `boto3`, no AWS CLI, no
   `s3://` or `/vsis3/` (either would consult a credential chain).
2. **No full-tile downloads.** Read only the bbox window via HTTP range
   requests through GDAL's `/vsicurl/`.
3. **No index download.** Compute candidate quadkeys from the bbox; probe
   existence with `requests.head`.
4. **Dependencies already present:** `rasterio>=1.3.11`, `requests`, `pyproj`,
   `numpy`. Nothing new in `pyproject.toml`.
5. **Import hygiene:** heavy imports (`rasterio`, `requests`) happen inside
   functions so `tests/test_import_hygiene.py` stays green.

## Components

All new code lives in `src/voxcity/downloader/aws_opendata.py`.

### 1. Generic core — `fetch_cog_crop`

```python
def fetch_cog_crop(
    urls: Sequence[str],
    rectangle_vertices: Sequence[Tuple[float, float]],
    out_path: str,
    *,
    res: Optional[float] = None,
    resampling: str = "nearest",
    timeout: int = 60,
) -> Optional[str]:
```

- `urls` may be `http(s)://…` (prefixed with `/vsicurl/` on open) or local
  paths (used as-is; this is how unit tests exercise the core offline).
- Opens each source inside `rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
  CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif", GDAL_HTTP_MAX_RETRY=3,
  GDAL_HTTP_RETRY_DELAY=1, GDAL_HTTP_TIMEOUT=timeout)`.
- Takes the CRS of the first openable source as the working CRS; transforms the
  WGS84 bbox of `rectangle_vertices` into it (padded by one source pixel).
- Mosaics the windowed reads with `rasterio.merge.merge(datasets, bounds=…)`
  (merge reads only the needed window per dataset, which GDAL serves as range
  requests). If `res` is given, passes `res=(res, res)` and the `Resampling`
  enum named by `resampling` (`nearest` | `average` | `max`); otherwise native
  resolution.
- Writes `out_path` as a deflate-compressed GeoTIFF in the working CRS and
  returns `out_path`. Returns `None` if no source could be opened.
- A source that fails to open is logged at INFO and skipped; it does not abort
  the mosaic.

### 2. CHMv2 adapter

```python
CHMV2_BASE_URL = "https://dataforgood-fb-data.s3.amazonaws.com/forests/v2/global/dinov3_global_chm_v2_ml3"
CHMV2_ZOOM = 10

def _quadkeys_for_bbox(min_lon, min_lat, max_lon, max_lat, zoom=CHMV2_ZOOM) -> List[str]
def chmv2_tile_urls(rectangle_vertices, base_url=CHMV2_BASE_URL, timeout=30) -> List[str]
def fetch_chmv2_canopy_height(
    rectangle_vertices, output_dir, *, res=None, resampling="nearest",
    base_url=CHMV2_BASE_URL,
) -> Optional[str]
```

- `_quadkeys_for_bbox` is pure math (Web Mercator tile x/y at `zoom` for the
  four bbox corners, then the Bing quadkey string for every tile in the x/y
  range). Known value for tests: Berlin (13.3, 52.1) → `1202120103`.
- `chmv2_tile_urls` builds `f"{base_url}/chm/{qk}.tif"` for each quadkey and
  keeps those for which `requests.head(url, timeout=timeout).status_code == 200`.
  A `requests.RequestException` on a probe is treated as "missing".
- `fetch_chmv2_canopy_height` calls the two above and `fetch_cog_crop` with
  `out_path = os.path.join(output_dir, "canopy_height.tif")`. Returns `None`
  when no tiles exist or none could be read.

### 3. Dispatch in `get_canopy_height_grid` (`src/voxcity/generator/grids.py`)

Add a branch **before** the `initialize_earth_engine()` call (currently line
414), so this source never touches Earth Engine:

```python
if source == 'High Resolution 1m Global Canopy Height Maps v2':
    from ..downloader.aws_opendata import fetch_chmv2_canopy_height
    geotiff_path = fetch_chmv2_canopy_height(
        rectangle_vertices, output_dir,
        res=kwargs.get('canopy_resample_res'),
        resampling=kwargs.get('canopy_resampling', 'nearest'),
    )
    if geotiff_path is None:
        _logger.warning("CHMv2 (AWS) has no tiles for this area or could not be reached — falling back to Static canopy heights.")
        return get_canopy_height_grid(rectangle_vertices, meshsize, 'Static', output_dir, **kwargs)
    canopy_height_grid = create_height_grid_from_geotiff_polygon(geotiff_path, meshsize, rectangle_vertices)
    # then the shared tail: >=255 → 0, trunk ratio, optional visualisation
```

The shared tail (lines 437–453 today) is extracted into a private helper
`_finish_canopy_grid(canopy_height_grid, meshsize, **kwargs)` used by both the
EE path and the new branch, so the sentinel handling and trunk-ratio logic are
not duplicated.

### 4. Registration in `src/voxcity/generator/api.py`

- `_SOURCE_URLS['High Resolution 1m Global Canopy Height Maps v2'] =
  'https://registry.opendata.aws/dataforgood-fb-forestsv2/'`.
- In the "Earth Engine not available" branch of `get_voxcity` (lines 506–513),
  auto-selected canopy becomes `'High Resolution 1m Global Canopy Height Maps v2'`
  for every region (it falls back to Static by itself if the area has no
  tiles). The Earth-Engine-available default stays
  `'High Resolution 1m Global Canopy Height Maps'` (v1); changing the global
  default is a separate decision because it changes outputs for existing users.
- Docstring of `get_voxcity` lists the new source string.

## Data flow

```
rectangle_vertices ─► bbox (WGS84) ─► zoom-10 quadkeys ─► HEAD-filtered URLs
   ─► /vsicurl/ windowed reads ─► merge ─► output_dir/canopy_height.tif
   (EPSG:3857, native ~1.19 m unless res given)
   ─► create_height_grid_from_geotiff_polygon (nearest pixel at cell centres)
   ─► >=255 → 0 ─► canopy_bottom = top × trunk_height_ratio
```

Point-sampling the native raster at mesh cell centres is the same behaviour
the GEE path has today (its export at `scale=meshsize` is nearest-neighbour),
so results are comparable. Users with coarse meshes can pass
`canopy_resample_res=meshsize, canopy_resampling='max'` (or `'average'`)
through `get_voxcity(**kwargs)`; this is not the default.

## Error handling

| Situation | Behaviour |
|---|---|
| Tile HEAD returns non-200 or raises | Tile skipped silently (expected over ocean / outside coverage) |
| Tile opens but read fails | Logged at INFO, tile skipped |
| Zero usable tiles (no coverage, offline) | `WARNING` + fall back to `'Static'`, matching the existing EE-unavailable behaviour |
| Transient HTTP errors during reads | GDAL retry via `GDAL_HTTP_MAX_RETRY=3` |
| Bucket becomes private in future | Surfaces as the zero-tile fallback; the live integration test is how we notice |

## Testing

Unit tests, no network, in `tests/test_downloader_aws_opendata.py`:

- `_quadkeys_for_bbox`: Berlin point → `['1202120103']`; a bbox straddling a
  tile edge → two keys; keys are 10 characters and sorted/deduplicated.
- `chmv2_tile_urls`: `requests.head` patched to return 200/404/raise; only the
  200 URL survives.
- `fetch_cog_crop` on COGs written to `tmp_path` with `rasterio` (EPSG:3857,
  uint8): single tile crop has expected bounds and values; two adjacent tiles
  mosaic seamlessly; bbox partly outside the tile yields a crop clipped to the
  tile; `res=5, resampling='max'` yields the coarser grid with max values;
  a non-existent path plus a valid one still succeeds; all-invalid returns
  `None`.
- `get_canopy_height_grid` with `fetch_chmv2_canopy_height` monkeypatched:
  (a) returns a path → grid shape matches `compute_grid_shape`, `>=255` cells
  become 0, no EE call happens (patch `initialize_earth_engine` to raise);
  (b) returns `None` → Static fallback path is taken.

Integration test `tests/test_downloader_aws_opendata_integration.py`, marked
`slow` and `integration`, skipped unless `VOXCITY_LIVE_AWS=1`: fetches a
300 m × 300 m Berlin window and asserts the GeoTIFF exists, CRS is EPSG:3857,
values are within 0–255, and at least one value is in 1–100.

Housekeeping: add `"*/voxcity/downloader/aws_opendata.py"` to the coverage
`omit` list beside the other network downloaders; `test_import_hygiene.py`
must pass unchanged.

## Out of scope

- `boto3`, `/vsis3/`, `tiles.geojson`, per-tile acquisition-date metadata.
- Download caching for rasters (no raster source caches today; `@cached_download`
  is vector-only).
- Other AWS datasets (ESA WorldCover, Copernicus DEM, GHSL). They become
  adapters over `fetch_cog_crop` later.
- Changing the Earth-Engine-available default canopy source to v2.
