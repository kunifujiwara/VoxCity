# GSI Bare-Earth DEM Downloader — Design

**Date:** 2026-06-26
**Status:** Approved (pending spec review)

## Goal

Add a downloader module that fetches bare-earth DEM (Digital Terrain Model)
data from the Geospatial Information Authority of Japan (GSI / 国土地理院) and
converts it into VoxCity's standard DEM format (a georeferenced GeoTIFF
consumed by the existing grid pipeline). Make GSI the auto-selected DEM source
for areas in Japan.

## Background

VoxCity DEM data flows through `get_dem_grid`
([`src/voxcity/generator/grids.py:357`](../../../src/voxcity/generator/grids.py)).
That function either:

- pulls a DEM image from Google Earth Engine (NASA SRTM, COPERNICUS, FABDEM,
  USGS 3DEP, etc.) and saves a GeoTIFF, **or**
- reads a user-supplied local GeoTIFF (`source="Local file"`).

In both cases the final step is
`create_dem_grid_from_geotiff_polygon(geotiff_path, meshsize, rectangle_vertices, ...)`
([`src/voxcity/geoprocessor/raster/raster.py:51`](../../../src/voxcity/geoprocessor/raster/raster.py)),
which opens the GeoTIFF, reads its **own** CRS (`src.crs`), and reprojects pixel
centres to a metric UTM zone for interpolation. **It therefore consumes any CRS
faithfully** — the chosen output CRS does not need to match other sources.

GSI publishes elevation tiles on the standard XYZ Web-Mercator tile grid at
`https://cyberjapandata.gsi.go.jp/xyz/{type}/{z}/{x}/{y}.txt`. Each tile is a
256×256 grid of elevation values as CSV text, with `e` marking no-data. Tile
resolutions, in priority order:

| type     | zoom | description                              |
|----------|------|------------------------------------------|
| `dem5a`  | 15   | 5 m mesh, airborne laser survey (finest) |
| `dem5b`  | 15   | 5 m mesh, photogrammetry                 |
| `dem10b` | 14   | 10 m mesh, nationwide coverage           |

The existing module
[`src/voxcity/downloader/oemj.py`](../../../src/voxcity/downloader/oemj.py)
already implements the same tile-download → mosaic → GeoTIFF pattern for GSI-style
RGB imagery, and serves as the structural template.

## Design Decisions

1. **Integration:** Add a new DEM source `"GSI DEM Japan"` handled directly in
   `get_dem_grid` (same UX as NASA/COPERNICUS/etc.). The downloader module fetches
   tiles and writes the GeoTIFF; the existing
   `create_dem_grid_from_geotiff_polygon` call is reused unchanged.
2. **Resolution selection:** Auto-detect the best available type by probing the
   center tile in order `dem5a → dem5b → dem10b`, falling back to `dem10b`.
   Caller may force a type via the `gsi_dem_type` kwarg.
3. **Output CRS:** Native **EPSG:3857** (Web Mercator). Tiles arrive on the
   mercator grid; writing them natively means *zero resampling* of elevation
   values. The grid consumer reprojects from `src.crs`, so the end result is
   identical to a reprojected raster but lossless and simpler. (A linear 4326
   geotransform would be slightly wrong since mercator pixels are not
   equal-latitude-spaced; a `gdal.Warp` to 4326 would resample. Both rejected.)
4. **Caching:** None on disk. Tiles are held in memory for one call; the written
   `dem.tif` is the durable artifact. (Skip-existing tile caching from the
   reference script is intentionally omitted — YAGNI.)
5. **Auto-selection:** GSI becomes the default DEM for Japan, taking priority
   over the global `FABDEM` fallback.

## Components

### New module: `src/voxcity/downloader/gsi.py`

**Public API**

```python
save_gsi_dem_as_geotiff(rectangle_vertices, filepath, dem_type=None,
                        nodata=-9999.0, sleep=0.4, timeout_s=10) -> str
```

- `rectangle_vertices`: list of `(lon, lat)` tuples (VoxCity ROI convention).
- `dem_type=None` → auto-detect; or `'dem5a'`/`'dem5b'`/`'dem10b'` to force.
- Returns the path written. Raises `ValueError` if no tiles cover the area.

**Internal helpers**

- `latlon_to_tile(lat, lon, zoom) -> (xtile, ytile)` — mercator tile index.
- `tile_bounds_mercator(x, y, zoom) -> (minx, miny, maxx, maxy)` — tile extent in
  EPSG:3857 meters (for the geotransform).
- `check_dem_availability(lat, lon, *, timeout_s, sleep) -> (dem_type, zoom)` —
  probe center tile in priority order; fall back to `('dem10b', 14)`.
- `download_dem_tiles(bounds, dem_type, zoom, *, nodata, sleep, timeout_s)` —
  fetch each `.txt`, parse 256×256 CSV of floats (`e` → `nodata`), 404 / missing
  → block of `nodata`. Returns `{(x, y): np.ndarray(256, 256)}`.
- `compose_dem_array(tiles, bounds) -> np.ndarray` — assemble into one
  `float32` mosaic of shape `(rows*256, cols*256)`.
- `save_dem_as_geotiff(array, bounds, zoom, filepath, nodata)` — write a
  single-band `GDT_Float32` GeoTIFF in EPSG:3857 with the mercator geotransform
  and nodata flag set. Uses `osgeo.gdal`/`osr` (consistent with `oemj.py`).

### Pipeline wiring: `get_dem_grid` (`grids.py:357`)

Add a branch handled before the Earth Engine path:

```python
elif source == "GSI DEM Japan":
    from ..downloader.gsi import save_gsi_dem_as_geotiff
    geotiff_path = os.path.join(output_dir, "dem.tif")
    save_gsi_dem_as_geotiff(rectangle_vertices, geotiff_path,
                            dem_type=kwargs.get("gsi_dem_type"))
```

No Earth Engine initialization for this source. The subsequent
`create_dem_grid_from_geotiff_polygon` call is unchanged.

### Exports: `downloader/__init__.py`

- Add `from .gsi import *`.
- Add `"save_gsi_dem_as_geotiff"` to `__all__`.

### Auto-selection: `api.py`

- In the DEM-source block (`api.py:413`), add `elif is_japan: dem_source = 'GSI DEM Japan'`
  before the `else: dem_source = 'FABDEM'`.
- Add to `_DEM_COVERAGE` (`api.py:229`): `'GSI DEM Japan': lambda f: f['is_japan']`.
- Update the auto-select docstring DEM rule (`api.py:292`) to note
  `Japan -> 'GSI DEM Japan'`.

## Data Flow

```
rectangle_vertices
  → bbox (min/max lon/lat)
  → check_dem_availability(center) → (dem_type, zoom)
  → tile range covering bbox
  → download_dem_tiles → {(x,y): 256x256 float32, e→nodata}
  → compose_dem_array → float32 mosaic
  → save_dem_as_geotiff → EPSG:3857 single-band GeoTIFF (dem.tif)
  → create_dem_grid_from_geotiff_polygon → dem_grid (np.ndarray)
```

## Error Handling

- **No coverage / all tiles 404:** raise `ValueError` with a clear message
  (area likely outside Japan).
- **Individual tile failure (404 / network):** fill that block with `nodata`.
  Downstream `np.where(dem < -1000, 0, dem)` in the grid function neutralizes it.
- **Network / timeout:** mirror `oemj.py` — bounded `timeout_s`, polite `sleep`
  between requests, clear error messages. No insecure SSL / HTTP fallback unless
  later requested.

## Testing

**Unit (network mocked):**
- `latlon_to_tile` against known reference values.
- CSV tile parser: normal rows, `e` no-data tokens, short/ragged rows.
- `compose_dem_array`: correct block placement and mosaic dimensions.
- `save_dem_as_geotiff`: geotransform corners match `tile_bounds_mercator`;
  CRS is EPSG:3857; nodata flag set; round-trip read returns the written array.
- `check_dem_availability`: priority order and `dem10b` fallback (mock HTTP).

**Integration (optional, live network, skippable):**
- Small Tsukuba bbox (`36.21,140.09 → 36.24,140.12`) → produces a valid GeoTIFF;
  `get_dem_grid(..., source="GSI DEM Japan")` returns a finite grid.

## Out of Scope (YAGNI)

- On-disk tile caching / resume.
- Refactoring shared mercator tile math into `utils.py` (no second consumer yet;
  `gsi.py` stays self-contained).
- DSM / surface (non-bare-earth) products.
- Insecure SSL / HTTP fallback paths.
