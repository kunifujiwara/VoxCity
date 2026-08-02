# App / library boundary

Record of the seam between `app/` (FastAPI backend + Vite frontend) and the
`voxcity` library, prepared so that splitting `app/` into its own repository is
mechanical. Current state: the app consumes voxcity **only through public
names** (guarded, see section 3), and `app/backend/requirements.txt` declares
`voxcity>=1.6.0` — the first version carrying every name the app imports.
This document records the surface, the guards, and what a split still
requires. **The split itself has not been performed.**

Regenerate section 1 from the tree, never from memory:

```bash
git grep -h "^\s*from voxcity\|^\s*import voxcity" -- 'app/backend/*.py' 'app/preprocessing/*.py' | sed 's/^\s*//' | sort -u
```

## 1. Public surface the app consumes

| Module | Names | Used for |
|---|---|---|
| `voxcity.generator` | `Voxelizer`, `auto_select_data_sources`, `get_voxcity`, `get_voxcity_CityGML`, `regenerate_voxels` | model generation endpoints; nDSM/LOD2 re-voxelisation |
| `voxcity.models` | `BuildingGrid`, `CanopyGrid`, `DemGrid`, `GridMetadata`, `LandCoverGrid`, `VoxCity`, `VoxelGrid` | typed grid containers held in backend state (`state.py`, `main.py`) |
| `voxcity.io` | `save_voxcity`, `load_voxcity` | session save/load (`session_io.py`) |
| `voxcity.simulator_gpu.solar` | `get_building_global_solar_irradiance_using_epw`, `get_global_solar_irradiance_using_epw`, `clear_all_caches` | solar simulation endpoints; cache invalidation on model change |
| `voxcity.simulator_gpu.visibility` | `get_landmark_visibility_map`, `get_surface_view_factor`, `get_view_index`, `mark_building_by_id`, `clear_visibility_cache` | visibility simulation endpoints; cache invalidation |
| `voxcity.exporter` | `export_cityles`, `export_obj`, `save_voxel_netcdf` | export endpoints |
| `voxcity.visualizer` | `get_voxel_color_map`, `visualize_voxcity_plotly` | scene colours (`scene_geometry.py`); plotly preview HTML |
| `voxcity.utils` | `GridProjector`, `LAND_COVER_CLASSES`, `get_land_cover_classes` | grid↔geo projection (`zoning.py`, overlays); land-cover metadata |
| `voxcity.geoprocessor.geojson` | `build_building_geojson`, `build_canopy_geojson`, `build_lc_geojson`, `get_lc_source_colors` | 2-D map overlay endpoints and editor palette (promoted out of the ipyleaflet-heavy `draw` package: ~0.14 s import vs ~1.5 s) |
| `voxcity.geoprocessor.raster` | `compute_grid_geometry`, `compute_cell_center_coords` | grid frame for overlays, zoning, nDSM refine, OBJ/DXF placement |
| `voxcity.importer` | `add_buildings_from_obj` | OBJ building import endpoint |
| `voxcity.downloader` | `initialize_earth_engine` | GEE initialisation at startup |

`app/preprocessing/precompute_citygml_cache.py` additionally imports
`voxcity.downloader.citygml` (see section 2).

## 2. Accepted deep surface

Imports deliberately left deeper than package level (source: commit `17e4015`;
every row re-verified against the tree on 2026-08-02 — all still accurate):

| Deep import | Names | Why accepted |
|---|---|---|
| `voxcity.importer.loader` | `load_obj_groups`, `classify_roles`, `group_material_name` | not re-exported from `importer/__init__.py` |
| `voxcity.importer.transform` | `build_placement_transform` | not re-exported from `importer/__init__.py` |
| `voxcity.simulator_gpu.init_taichi` | `ensure_initialized`, `reset` | `reset` not re-exported from `simulator_gpu` |
| `voxcity.utils.orientation` | `check_axes` | not in `utils/__init__.py` |
| `voxcity.exporter.geotiff` | `export_geotiffs` | `test_export_geotiff.py` monkeypatches the module; the package re-export (`from .geotiff import *`, `exporter/__init__.py:8`) binds eagerly and would dodge the patch |
| `voxcity.downloader.citygml` | `load_buid_dem_veg_from_citygml` | not re-exported from `downloader/__init__.py` |
| `voxcity.geoprocessor.mesh` | `create_voxel_mesh`, `create_sim_surface_mesh`, `BUILDING_SURFACE_CLASSES` | not re-exported from `geoprocessor/__init__.py` |

If any of these names gains a package-level re-export later, move the app to
it and shrink this table.

## 3. Guards

**Boundary guard** — `app/backend/test_api_boundary.py`: grep-based; fails on
any `voxcity...._private` module import or `_name` from-import anywhere under
`app/` (`.py` files and `.ipynb` code cells; `archive/` and `node_modules/`
excluded). This is what keeps section 1 public and section 2 the *only* deep
surface.

**Library-side twins** (commit `41484e9`) — library claims previously pinned
only through app tests, now runnable from a library-only checkout:

| Test file | Pins |
|---|---|
| `tests/test_grid_geometry_south_up_contract.py` | south-up cell-centre contract of `compute_grid_geometry`/`compute_cell_center_coords`: row 0 at the `vertex[0]` end of side_1, lat increasing with row for the app's [SW, NW, NE, SE] convention, corner-vertex pairing on a rotated rectangle, projector consistency |
| `tests/test_geojson_builders_public.py` | orientation of the promoted `voxcity.geoprocessor.geojson` builders on a non-square grid with an asymmetric southern band; `build_lc_geojson`'s cls-index and style-colour property contract |
| `tests/test_save_voxcity_frame_roundtrip.py` | `save_voxcity` → `check_axes` → `GridProjector.from_h5` content round trip, keyed by an off-centre marker's lon/lat (attribute-only checks passed while false earlier in this project) |
| `tests/test_generator_update.py` | `regenerate_voxels` is a function of the 2.5-D component grids alone — the claim behind the app's LOD2 routing around it |

## 4. Library-claim inventory (app test → library claim → library-side coverage)

| App test | Library claim | Library-side coverage |
|---|---|---|
| `app/backend/test_frame_consumers.py`, `test_model_geo_overlays.py`, `test_ndsm_refine.py` | `compute_grid_geometry` south-up frame and rotated-rectangle corner pairing | `tests/test_grid_geometry_south_up_contract.py` |
| `app/backend/test_model_geo_overlays.py` | GeoJSON builder orientation and `build_lc_geojson` property contract | `tests/test_geojson_builders_public.py` |
| `app/backend/test_session_io.py`, `test_session_save_load.py` | H5 save/load preserves the grid frame (content, not just attributes) | `tests/test_save_voxcity_frame_roundtrip.py` |
| `app/backend/test_ndsm_pipeline.py`, `test_ndsm_lod2_geometry.py`, `test_generate_lod2.py` | `regenerate_voxels` depends only on the component grids (safe to route LOD2 around it) | `tests/test_generator_update.py` |
| `app/backend/test_ndsm_pipeline.py` (LOD2 canopy) | `reapply_canopy` adds trees without touching other classes / preserves mesh-vegetation columns / is idempotent and path-independent / honours the z-datum and terrain | `VoxCityGML/tests/test_reapply_canopy.py:366,393,434,543` (delegated to voxcitygml's own suite) |
| `app/backend/test_ndsm_lod2_geometry.py` (frame alignment) | land cover and voxels share a frame (tree land cover aligns with TREE voxels, not flipped) | `VoxCityGML/tests/test_frame_contract.py:127-173` |

## 5. What a split still requires — the checklist

1. **Publish voxcity ≥ 1.6.0** (with `voxcity.geoprocessor.geojson` and the
   `raster.__all__` additions) to PyPI or an index the app can install from.
   The floor in `app/backend/requirements.txt` is meaningless until an
   installable release exists at or above it.
2. **Switch the app to the released install** — replace the
   `PYTHONPATH=<checkout>` launch with `pip install voxcity>=1.6.0`, and add
   app CI that runs `app/backend` against the *released* voxcity, not the
   sibling checkout.
3. **Resolve the two reverse-direction entanglements** (library → app, found
   in Task 2; re-verified 2026-08-02):
   - `tests/app/` — **10 test files, every one of them imports
     `app.backend.*`** (`test_building_surfaces_endpoint`,
     `test_edit_building_height`, `test_ground_roofs_flag`,
     `test_rectangle_from_dimensions`, `test_scene_geometry`,
     `test_scene_geometry_highlight`, `test_surface_zone_edges`,
     `test_surface_zone_edges_endpoint`, `test_surface_zones`, `test_zones`;
     plus an empty `__init__.py`). The **library's** suite breaks the moment
     `app/` leaves. They must move to the app repo or be dropped.
   - `app/backend/test_model_geo_overlays.py:27` imports
     `tests.importer.conftest` (`make_flat_voxcity`) — an app test reaching
     into the library's test tree. Duplicate the fixture into `app/backend`
     or promote it to a shipped helper before the split.
4. **Move `app/` and its three suites**: the backend pytest suite
   (`app/backend/test_*.py`), the `tests/app/` suite (currently in the
   library's test tree — see item 3), and the frontend vitest suite
   (`app/frontend/src/**/*.test.*`).
5. **Version-skew hazard on this machine**: conda envs here have resolved
   stale site-packages copies over the intended checkout before (the
   voxcitygml 1.1.8 incident — an installed old build shadowed the checkout
   and silently mirrored the model; voxcitygml's own `voxcity>=1.3.2` floor in
   its `pyproject.toml` exists for the same reason). The `voxcity>=1.6.0`
   floor converts that failure mode from a silent wrong-behaviour run into a
   loud resolver/import error. After the split, an app env must never contain
   an unversioned editable voxcity plus a PyPI voxcity simultaneously.

## 6. Known issues logged here, not fixed

- `build_lc_geojson(..., land_cover_source=None)` raises `UnboundLocalError`:
  it calls `get_land_cover_classes(None)` (`src/voxcity/geoprocessor/geojson.py:263`),
  whose if/elif chain in `src/voxcity/utils/lc.py` (function at line 66) never
  binds `land_cover_classes` for `None`, crashing at the `return` on line 175.
  Pre-existing, now reachable through the public name. Note
  `get_lc_source_colors` *does* implement the None fallback its docstring
  claims (`geojson.py:91-92`) — but the lc build path crashes at line 263,
  before ever reaching it (line 265).
