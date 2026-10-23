# Changelog

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
  Previously, when Earth Engine was unavailable, canopy came from
  OpenStreetMap trees (or 'Static' in Japan), so the no-Earth-Engine
  behavior changes.
  `voxcity.downloader.aws_opendata.fetch_cog_crop` is the reusable core for
  adding further AWS-hosted rasters. Optional kwargs `canopy_resample_res`
  and `canopy_resampling` ('nearest' | 'average' | 'max') control
  aggregation for coarse meshes. `canopy_resample_res` is in ground metres
  (converted to Web Mercator units at the area's centre latitude).
  The no-Earth-Engine fallback when CHMv2 has no tiles is the land-cover-based
  `'Static'` source, not OpenStreetMap trees; pass
  `canopy_height_source='OpenStreetMap'` explicitly to get OSM point trees
  without Earth Engine. If only some tiles are reachable, the unreachable
  ones are logged and treated as no canopy.
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

### Changed

- `'ESA WorldCover'`, `'COPERNICUS'` and `'FABDEM'` no longer use Earth
  Engine even when it is available. The data is the same; it is read at
  native resolution instead of through an Earth Engine export, so values can
  differ slightly at cell edges. FABDEM remains CC BY-NC-SA 4.0
  (non-commercial). No global DEM removes buildings in high-rise cores; in
  such areas prefer national lidar or `'Flat'`.

## 1.7.0 (2026-10-05)

### Breaking

- **`make_surface_face_key` no longer takes `face_index`.** The fourth,
  required positional parameter is gone, so any external caller passing it
  raises `TypeError`; the emitted key also loses its trailing `:i<index>`
  segment. The index was the face's position in whatever enumeration minted
  the key, which made the key depend on enumeration *order* --
  `create_voxel_mesh` emits faces direction-major while a voxel walk goes
  voxel-major, so the two minted different keys for the same face on any
  building larger than one voxel, and a selector resolved against one
  producer matched nothing when drawn by the other. Building id, centroid
  and normal already identify a face uniquely.

  Migration: drop the argument. Keys already stored by an application do
  not need rewriting -- `surface_zone_mask` normalizes both sides of every
  face-key comparison, so an old-format key still resolves to its face.
  Applications doing their own raw key comparisons should route them
  through `normalize_surface_face_key`.

  Versioned as a minor bump rather than a major one only because the 2.0.0
  slot below belongs to an abandoned bump that this line was reverted from
  (see `947bac2`); by semver this break warrants a major, and the
  maintainer may want to renumber before publishing.

### Added

- `voxcity.geoprocessor.normalize_surface_face_key(key)` — reduces a face
  key to its order-independent form by stripping a legacy trailing
  `:i<index>`. Applied to both sides of every face-key comparison in
  `surface_zone_mask`, so keys minted before the change above keep
  resolving. `surface_face_meta_version` deliberately stays at 1: a cached
  mesh restored from a saved session can carry old-format keys, and
  normalization reconciles that on read without invalidating the cache.

### Fixed

- **GSI DEM: the 10 m fallback now actually runs, and partial 5 m coverage
  yields a uniform 10 m raster.** The downloader requested the tile set
  `dem10b`, but GSI serves DEM10B at `/xyz/dem/`, so every 10 m request 404'd
  and an area without full 5 m coverage came back mostly no-data — which
  `create_dem_grid_from_geotiff_polygon` turns into 0 m, rendering as a
  sea-level plain with the 5 m region standing on it as a cliff-edged plateau.
  Separately, patching 10 m pixels into the 5 m grid is no longer done at all:
  where the 5 m products leave a gap the 10 m product can fill, the whole area
  is written at 10 m instead. Mixing them put steps of up to 22 m into terrain
  whose real pixel-to-pixel gradient peaked at 8 m. Gaps the 10 m product also
  lacks (sea, outside Japan) do not trigger the switch, so coastal areas keep
  their 5 m surface.

## 1.6.3 (2026-08-22)

### Changed

- Python support widened to `>=3.10,<3.14`. Google Colab now ships Python
  3.13, which the previous `<3.13` cap silently excluded — `pip install
  voxcity` there failed with "No matching distribution found". The full
  dependency set (including `numba`/`llvmlite` and the optional `taichi`
  GPU extra) resolves and the test suite passes on 3.13.

## 2.0.0 (2026-07-21)

### Breaking

- **HDF5 format v3 (`voxcity_results.v3`) is required by the loader.**
  Files written by VoxCity 1.x no longer load; convert them once with
  `voxcity.io.migrate_h5(src, dst)` or `python -m voxcity.migrate FILE ...`.
  v3 files are self-describing: root/group/dataset `axes` attributes,
  `rotation_angle` (derived from geometry at save time), and a structured
  `rectangle_vertices` dataset.
- Saving now errors if `extras['rotation_angle']` disagrees with the angle
  derived from `rectangle_vertices` by more than 0.1 degrees.

### Added

- `voxcity.direction_to_axis_vector(azimuth_deg, elevation_deg, rotation_angle_deg)`
  — the single public azimuth-to-axis-vector mapping (compass azimuth,
  component 0 = north). Broadcasts over arrays.
- `voxcity.check_axes(file_or_attrs)` — assert a file declares the
  `north,east,up` contract.
- `voxcity.GridProjector.from_city(city)` / `.from_h5(path)` — lon/lat <-> cell
  without hand-assembling a `GridGeom`.
- `VoxCity.to_xarray()` — zero-copy named-dimension view
  (dims `("north", "east", "up")`, cell-centre metre coordinates).
- `python -m voxcity.migrate` — batch converter with provenance attrs
  (`migrated_from`, `geometry_source`).
- Empirical orientation guard test (NE-corner building -> high-i/high-j),
  the permanent version of the check that caught the `voxcity_vwind`
  axis-swap incident.

### Changed

- Solar simulators (`radiation.py`, `sky.py`, `common/geometry.py`) build
  direction vectors via `direction_to_axis_vector`. Outputs are
  bit-identical, except the SVF ray fan may differ by <=1 ulp (azimuth
  generation moved from radians to degrees).
