# Login-free DEM and land cover sources — design

Date: 2026-10-23
Branch: feat/aws-opendata-canopy (continues the CHMv2 work; one PR for all login-free sources)
Builds on: `2026-10-23-aws-opendata-canopy-design.md`

## Goal

Make the terrain and land-cover layers of `get_voxcity` work without any
account or sign-in, as the CHMv2 canopy source already does. Four datasets:

| Source name | Data | Host | Status |
|---|---|---|---|
| `'COPERNICUS'` | Copernicus DEM GLO-30 (surface model) | AWS `copernicus-dem-30m` | existing name, now read from AWS |
| `'ESA WorldCover'` | ESA WorldCover 10 m 2021 v200 | AWS `esa-worldcover` (eu-central-1) | existing name, now read from AWS |
| `'USGS 3DEP 10m'` | USGS 3DEP 1/3 arc-second seamless DEM (bare earth, US) | AWS `prd-tnm` | new name |
| `'FABDEM'` | FABDEM v1-2 (Copernicus with forests and buildings removed) | University of Bristol data server | existing name, now read from Bristol |

All four are read through GDAL range requests (`/vsicurl/`, plus `/vsizip/`
for FABDEM), so only the window around the area is transferred.

## Decisions (made with the user)

1. **Login-free first for same-data names.** `'ESA WorldCover'`,
   `'COPERNICUS'` and `'FABDEM'` no longer touch Earth Engine. This also
   changes behaviour for users who have Earth Engine: they get the same data,
   read at native resolution instead of through an Earth Engine export.
2. **No-Earth-Engine DEM auto-selection.** Japan keeps `'GSI DEM Japan'`; an
   area whose auto-selected DEM is `'USGS 3DEP 1m'` (the US) gets
   `'USGS 3DEP 10m'`; every other area gets `'FABDEM'` instead of `'Flat'`.
   Outside the US and Japan this is the same DEM Earth Engine users already
   get by default.
3. **Land cover auto-selection is unchanged.** WorldCover is used when named.
4. **GEDTM30 is not added.** A five-site lidar benchmark (below) showed FABDEM
   more accurate everywhere.

## Evidence

Per-pixel comparison against USGS 3DEP lidar bare earth (RMSE m, bias m),
resampled bilinearly onto the USGS grid, 2026-10-23:

| Site | FABDEM | GEDTM30 | Copernicus |
|---|---|---|---|
| Manhattan midtown (high-rise) | 61.1 (+53.6) | 66.5 (+57.8) | 67.8 (+59.2) |
| Central Park (urban park) | 11.4 (+10.3) | 12.5 (+8.1) | 17.1 (+15.7) |
| Park Slope (low-rise) | 3.7 (+3.3) | 7.1 (+6.6) | 8.2 (+7.8) |
| West Houston (suburb) | 2.2 (+1.8) | 2.3 (+1.5) | 5.1 (+4.1) |
| Shenandoah NP (forest) | 5.8 (−3.0) | 7.1 (−5.0) | 9.9 (+7.8) |

No global model handles high-rise cores; that limitation is documented, not
solved, here.

## Dataset facts (verified 2026-10-23)

| | Copernicus | WorldCover | USGS 3DEP 10m | FABDEM |
|---|---|---|---|---|
| Base URL | `https://copernicus-dem-30m.s3.amazonaws.com` | `https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map` | `https://prd-tnm.s3.amazonaws.com/StagedProducts/Elevation/13/TIFF/current` | `https://data.bris.ac.uk/datasets/s5hqmjcdj8yo2ibzi9b4ew3sn` |
| Tile | 1° | 3° | 1° | 1° inside 10° zips |
| Named by | SW corner | SW corner (multiple of 3) | **NW** corner | SW corner; zip by 10° SW corner |
| Example | `Copernicus_DSM_COG_10_N52_00_E013_00_DEM/Copernicus_DSM_COG_10_N52_00_E013_00_DEM.tif` | `ESA_WorldCover_10m_2021_v200_N51E012_Map.tif` | `n41w074/USGS_13_n41w074.tif` | `N50E010-N60E020_FABDEM_V1-2.zip/N52E013_FABDEM_V1-2.tif` |
| CRS | EPSG:4326 | EPSG:4326 | EPSG:4269 | EPSG:4326 |
| Resolution | 1″ lat (lon varies with latitude) | 1/12000° (~10 m) | 1/3″ | 1″ |
| Type / nodata | float32 / none (missing tiles over ocean) | uint8 class codes / 0 | float32 / −999999 | float32 / per file |
| COG | yes (1024 blocks, overviews) | yes | yes (512 blocks, overviews) | GeoTIFF inside zip; window reads measured at 1–3 s |
| License | Copernicus DEM licence (free, attribution) | CC BY 4.0 | US public domain | **CC BY-NC-SA 4.0 (non-commercial)** |

Copernicus is a surface model: buildings and trees are included, as with the
existing Earth Engine source. FABDEM's non-commercial licence is the same one
Earth Engine users already accept for `'FABDEM'`; it is stated in the
docstring, README and CHANGELOG.

## Components

### 1. `src/voxcity/downloader/cog.py` (new — generic, moved out of `aws_opendata.py`)

Moves verbatim: `_GDAL_HTTP_ENV`, `_RESAMPLING`, `_gdal_path`, `_resample`,
`fetch_cog_crop`, `_remote_exists`, `_bbox_from_rectangle_vertices`.
`aws_opendata.py` imports them and keeps `fetch_cog_crop` in its `__all__` so
`from voxcity.downloader.aws_opendata import fetch_cog_crop` still works.
Tests that monkeypatch internals (`merge`, `rasterio.open`, `requests.head`)
move their patch targets to `voxcity.downloader.cog`.

Additions:

- `CPL_VSIL_CURL_ALLOWED_EXTENSIONS` becomes `".tif,.zip"` (FABDEM zips).
- `probe_existing(urls, timeout=10) -> list[str]`: the HEAD loop now inside
  `chmv2_tile_urls`, made reusable.
- `fetch_cog_crop(..., pad_pixels: int = 1, fill_nodata: bool = False)`:
  - `pad_pixels` replaces the hard-coded one-pixel padding.
  - `fill_nodata=True`: cells equal to the first source's nodata value, NaN,
    or outside every source are filled from the nearest valid pixel
    (`scipy.ndimage.distance_transform_edt` with `return_indices=True`). If
    no cell is valid, return `None` and write nothing. Uncovered area is
    tracked by merging with an explicit fill value rather than inferring it
    from zeros, so a real 0 m elevation is never "filled".
  - Default behaviour (`pad_pixels=1`, `fill_nodata=False`) is unchanged;
    every existing CHMv2 test must still pass.

### 2. `src/voxcity/downloader/aws_opendata.py` (adapters)

Each dataset is a pure tile-naming function (unit-tested against live keys)
plus a fetch function returning a path or `None`. A shared helper
`_tile_origins(bbox, step)` yields SW corners (`floor(v/step)*step`) of every
`step`-degree tile touching the bbox, capped at 16 tiles (a warning and `[]`
above that, or for a longitude span over 180°).

- `copernicus_tile_urls(rectangle_vertices)` → HEAD-probed URLs.
  `fetch_copernicus_dem(rectangle_vertices, output_dir)` → `output_dir/dem.tif`
  with `pad_pixels=4`, `fill_nodata=False` (missing ocean tiles are zero-filled
  by the merge, which is sea level).
- `worldcover_tile_urls(rectangle_vertices)` (3° tiles).
  `fetch_esa_worldcover(rectangle_vertices, output_dir)` → reads class codes
  (nearest, `pad_pixels=1`) into a temporary file, maps code → RGB, writes a
  3-band uint8 `output_dir/land_cover.tif`. The RGB for each code is taken
  from `get_land_cover_classes('ESA WorldCover')` in code order
  10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 100, so the existing colour lookup
  maps every pixel exactly. Code 0 (outside the tile set: ocean) maps to the
  Open water colour.
- `usgs3dep10m_tile_urls(rectangle_vertices)`: tag is `n{lat0+1:02d}w{abs(lon0):03d}`
  for SW corner (lat0, lon0) in the western hemisphere; the general form uses
  `n`/`s` and `e`/`w` with the same NW-corner rule.
  `fetch_usgs_3dep_10m(rectangle_vertices, output_dir)` → `output_dir/dem.tif`
  with `pad_pixels=10`, `fill_nodata=True` (−999999 cells).

### 3. `src/voxcity/downloader/fabdem.py` (new)

- `fabdem_paths(rectangle_vertices, base_url=FABDEM_BASE_URL)` →
  `/vsizip//vsicurl/{base}/{zip}/{member}` paths. Member:
  `{N|S}{lat0:02d}{E|W}{lon0:03d}_FABDEM_V1-2.tif`. Zip: 10° SW corner
  `zlat=floor(lat0/10)*10`, `zlon=floor(lon0/10)*10`, named
  `{zlat}{zlon}-{zlat+10}{zlon+10}_FABDEM_V1-2.zip` in the same N/S, E/W
  format (verified: `N50E010-N60E020`, `N40W080-N50W070`).
- No HEAD probe: a missing member (ocean) fails to open and is skipped at
  INFO level by `fetch_cog_crop`.
- `fetch_fabdem(rectangle_vertices, output_dir)` → `output_dir/dem.tif` with
  `pad_pixels=4`, `fill_nodata=True`.

### 4. `src/voxcity/generator/grids.py`

- `get_land_cover_grid`: add `'ESA WorldCover'` to the sources that skip
  `initialize_earth_engine()`. Its branch calls `fetch_esa_worldcover`; on
  `None` it logs a warning and continues as `'OpenStreetMap'`
  (`effective_source`), matching today's Earth Engine failure behaviour.
- `get_dem_grid`: before the Earth Engine `try`, a branch for
  `'COPERNICUS'`, `'FABDEM'` and `'USGS 3DEP 10m'` calls the matching fetch
  function. On `None`: a warning and the flat grid (`compute_grid_shape`
  zeros), exactly as the current Earth Engine failure path returns. Otherwise
  the shared `create_dem_grid_from_geotiff_polygon` tail runs unchanged.
- `src/voxcity/downloader/gee.py` `get_dem_image` keeps its `'COPERNICUS'`
  and `'FABDEM'` branches (unused by the generator, kept for direct callers).

### 5. `src/voxcity/generator/api.py`

- `_SOURCE_URLS['USGS 3DEP 10m'] =
  'https://www.usgs.gov/3d-elevation-program/about-3dep-products-services'`.
- `_DEM_COVERAGE['USGS 3DEP 10m'] = lambda f: f['is_usa']`.
- No-Earth-Engine block, DEM part becomes:
  `GSI DEM Japan` → unchanged; `USGS 3DEP 1m` → `USGS 3DEP 10m`; anything
  else → `FABDEM`.
- No-Earth-Engine land-cover downgrade keeps `'ESA WorldCover'` (add it to
  the allowed tuple) in case a caller's auto-selection ever picks it.
- Docstrings: `get_voxcity` and `auto_select_data_sources` describe the new
  behaviour.

## Error handling

| Situation | Behaviour |
|---|---|
| AWS tile missing (HEAD 404) | Dropped silently |
| HEAD non-200/404 or network error | Dropped with a WARNING |
| FABDEM member missing (ocean) | Open fails; skipped at INFO |
| Area needs > 16 tiles or crosses the antimeridian | WARNING, `[]`, then the fallback |
| All reads fail / nothing valid | `None` → DEM: WARNING + flat; land cover: WARNING + OpenStreetMap |
| USGS −999999 inside the window | Filled from the nearest valid pixel |

## Testing

Offline (`tests/test_downloader_cog.py`, `tests/test_downloader_aws_dem_lc.py`,
`tests/test_downloader_fabdem.py`, `tests/test_generator_nologin_sources.py`):

- Tile naming against verified keys: Copernicus `N52_00_E013_00` for Berlin
  and `N40_00_W074_00` for Manhattan; WorldCover `N51E012` for Berlin; USGS
  `n41w074` for Manhattan and `n30w096` for West Houston; FABDEM
  `N50E010-N60E020…/N52E013…` and `N40W080-N50W070…/N40W074…`. Southern and
  eastern-hemisphere formatting is covered by one synthetic case each.
- `fetch_cog_crop`: `pad_pixels` widens the crop by the expected pixels;
  `fill_nodata` fills a nodata hole and an uncovered strip from neighbours,
  leaves a real 0 untouched, and returns `None` when nothing is valid; a
  `/vsizip/` path is passed to GDAL unchanged; existing CHMv2 tests still pass.
- WorldCover round trip: a synthetic code raster (all 11 codes plus 0) →
  `fetch_esa_worldcover` (with `fetch_cog_crop` reading a local file) →
  `create_land_cover_grid_from_geotiff_polygon` with the ESA class table →
  the expected class names, cell for cell.
- `grids.py` dispatch: each of the four names calls its fetch function and
  never `initialize_earth_engine` (plain `patch` + `assert_not_called`); a
  `None` return produces the documented fallback.
- `api.py` auto-selection with Earth Engine unavailable: a US box →
  `'USGS 3DEP 10m'`, a German box → `'FABDEM'`, a Japanese box →
  `'GSI DEM Japan'`.

Live, opt-in (`VOXCITY_LIVE_AWS=1`, same flag as CHMv2): one small window
per dataset, asserting CRS, dtype and a plausible value range.

Validation (reported as a published page, not a committed test): rerun the
five-site lidar benchmark through `get_dem_grid` for `'FABDEM'`,
`'COPERNICUS'` and `'USGS 3DEP 10m'`, and a full no-Earth-Engine
`get_voxcity` build for the Berlin box (FABDEM terrain) and a US box
(USGS 3DEP 10m terrain).

## Out of scope

- GEDTM30, Mapzen terrain tiles, GHSL built height.
- USGS 3DEP 1 m from AWS (project-based; needs an index service).
- Land-cover auto-selection changes.
- The pre-existing OSM tree canopy transpose bug (separate branch).
