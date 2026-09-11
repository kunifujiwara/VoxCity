# Changelog

## 1.7.0 (unreleased)

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
