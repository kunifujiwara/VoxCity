# Changelog

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
