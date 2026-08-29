"""Water-surface DEM flattening on the CityGML/PLATEAU path.

The raster pipeline flattens every connected water body's DEM cells to that
body's minimum before voxelizing.  The CityGML path resolves its DEM from a
terrain TIN that mixes bank and water-surface elevations, so without the same
rule a river quantizes to two ground levels and leaves a voxel-high cliff
inside the water.  These tests pin the shared rule onto that path.
"""
import numpy as np
import pytest

import voxcity.generator.api as api

RECT = [(139.770, 35.646), (139.770, 35.650),
        (139.775, 35.650), (139.775, 35.646)]

# OpenStreetMap land-cover indices (0-based, standard = index + 1).
OSM_WATER = 8       # -> standard class 9 (Water)
OSM_DEVELOPED = 10  # -> standard class 11 (Developed space)


# --------------------------------------------------------------------------
# The extracted helper: same rule, same kwargs semantics as the pipeline.
# --------------------------------------------------------------------------

def _one_body_grids():
    """A single 4-connected water body straddling two DEM levels."""
    land_cover = np.full((4, 4), OSM_DEVELOPED)
    land_cover[1:3, :] = OSM_WATER
    dem = np.full((4, 4), 20.0)
    dem[1:3, :2] = 12.0
    dem[1:3, 2:] = 15.0
    return dem, land_cover


def test_helper_flattens_one_body_to_its_minimum():
    dem, land_cover = _one_body_grids()

    flattened, extras = api._flatten_citygml_water_dem(
        dem, land_cover, "OpenStreetMap", {})

    assert np.all(flattened[1:3, :] == 12.0), "water must sit at one level"
    assert np.all(flattened[0, :] == 20.0), "dry land must be untouched"
    assert np.all(flattened[3, :] == 20.0)


def test_helper_keeps_separate_bodies_at_their_own_levels():
    """Two disconnected water bodies each keep their own minimum — flattening
    is per-component, not global."""
    land_cover = np.full((3, 5), OSM_DEVELOPED)
    land_cover[1, :2] = OSM_WATER     # body A
    land_cover[1, 3:] = OSM_WATER     # body B
    dem = np.full((3, 5), 30.0)
    dem[1, 0], dem[1, 1] = 10.0, 11.0
    dem[1, 3], dem[1, 4] = 50.0, 51.0

    flattened, extras = api._flatten_citygml_water_dem(
        dem, land_cover, "OpenStreetMap", {})

    assert flattened[1, 0] == 10.0 and flattened[1, 1] == 10.0
    assert flattened[1, 3] == 50.0 and flattened[1, 4] == 50.0
    assert extras["water_dem_flattening"]["water_body_count"] == 2


def test_helper_opt_out_is_identity():
    dem, land_cover = _one_body_grids()

    flattened, extras = api._flatten_citygml_water_dem(
        dem, land_cover, "OpenStreetMap", {"flatten_water_dem": False})

    assert np.array_equal(flattened, dem)
    assert extras["flatten_water_dem"] is False
    assert extras["water_dem_flattening"]["applied"] is False
    assert extras["water_dem_flattening"]["reason"] == "disabled"


def test_helper_extras_keys_match_the_pipeline():
    dem, land_cover = _one_body_grids()

    _, extras = api._flatten_citygml_water_dem(
        dem, land_cover, "OpenStreetMap", {})

    assert set(extras) == {"flatten_water_dem", "water_dem_connectivity",
                           "water_dem_flattening"}
    assert extras["flatten_water_dem"] is True
    assert extras["water_dem_connectivity"] == 4  # pipeline default
    info = extras["water_dem_flattening"]
    assert set(info) == {"applied", "reason", "water_body_count",
                         "water_cell_count", "water_dem_min_values"}
    assert info["applied"] is True
    assert info["water_cell_count"] == 8


def test_helper_forwards_connectivity_kwarg():
    """connectivity=8 joins diagonally touching water; the default 4 does not."""
    land_cover = np.array([[OSM_WATER, OSM_DEVELOPED],
                           [OSM_DEVELOPED, OSM_WATER]])
    dem = np.array([[4.0, 100.0], [100.0, 8.0]])

    four, extras_4 = api._flatten_citygml_water_dem(
        dem, land_cover, "OpenStreetMap", {})
    eight, extras_8 = api._flatten_citygml_water_dem(
        dem, land_cover, "OpenStreetMap", {"water_dem_connectivity": 8})

    assert four[1, 1] == 8.0
    assert extras_4["water_dem_connectivity"] == 4
    assert eight[1, 1] == 4.0
    assert extras_8["water_dem_connectivity"] == 8


def test_helper_rejects_bad_connectivity():
    dem, land_cover = _one_body_grids()
    with pytest.raises(ValueError):
        api._flatten_citygml_water_dem(
            dem, land_cover, "OpenStreetMap", {"water_dem_connectivity": 5})


# --------------------------------------------------------------------------
# Wiring: the real ``get_voxcity_CityGML`` must apply the rule.
# --------------------------------------------------------------------------

def _empty_min_height_grid(shape):
    grid = np.empty(shape, dtype=object)
    for index in np.ndindex(shape):
        grid[index] = []
    return grid


def _stub_citygml_run(monkeypatch, dem, land_cover):
    """Stub every heavy step of ``get_voxcity_CityGML`` (CityGML parsing,
    land cover, building footprints, terrain interpolation) so the real
    function can be driven end to end on tiny synthetic grids."""
    shape = land_cover.shape
    monkeypatch.setattr(api, "load_buid_dem_veg_from_citygml",
                        lambda **kwargs: (None, None, None))
    monkeypatch.setattr(api, "get_land_cover_grid",
                        lambda *a, **k: land_cover.copy())
    monkeypatch.setattr(
        api, "create_building_height_grid_from_gdf_polygon",
        lambda *a, **k: (np.zeros(shape), _empty_min_height_grid(shape),
                         np.zeros(shape, dtype=int), None))
    monkeypatch.setattr(api, "create_dem_grid_from_gdf_polygon",
                        lambda *a, **k: dem.copy())


def test_citygml_path_flattens_water_dem(monkeypatch):
    dem, land_cover = _one_body_grids()
    _stub_citygml_run(monkeypatch, dem, land_cover)

    city = api.get_voxcity_CityGML(
        RECT, "OpenStreetMap", "Static", 5.0, citygml_path="unused.gml",
        gridvis=False, save_voxcity_data=False)

    elevation = np.asarray(city.dem.elevation)
    assert np.all(elevation[1:3, :] == 12.0), (
        "the CityGML path must flatten each water body to its minimum")
    assert np.all(elevation[0, :] == 20.0)
    assert city.extras["flatten_water_dem"] is True
    assert city.extras["water_dem_connectivity"] == 4
    assert city.extras["water_dem_flattening"]["applied"] is True
    assert city.extras["water_dem_flattening"]["water_body_count"] == 1

    # The user-facing symptom: without flattening the river quantizes to two
    # ground levels and PALM renders (and thermally models) the step face as a
    # wall. Every water column must end up at the same ground level.
    voxels = np.asarray(city.voxels.classes)
    water_ground_levels = {int(np.max(np.nonzero(voxels[i, j, :])[0]))
                           for i in (1, 2) for j in range(4)}
    assert len(water_ground_levels) == 1, (
        f"water columns must share one ground level, got {water_ground_levels}")


def test_citygml_path_flattens_against_the_post_perimeter_water_mask(monkeypatch):
    """Perimeter removal overwrites edge land cover with Developed space, which
    shrinks water bodies that reach the AOI edge. Flattening must run AFTER
    that, or a perimeter cell's elevation leaks in as the body's minimum.

    This pins the call's *position* in ``get_voxcity_CityGML``: moving it up to
    the DEM-resolution line (a plausible "tidy it closer to where the DEM is
    made" refactor) is a silent numerical regression that no other test sees.
    """
    land_cover = np.full((10, 10), OSM_DEVELOPED)
    land_cover[5, 0:6] = OSM_WATER          # river reaching the west edge
    dem = np.full((10, 10), 50.0)
    dem[5, 0] = 1.0                          # the minimum sits IN the perimeter ring
    dem[5, 1:6] = 20.0
    _stub_citygml_run(monkeypatch, dem, land_cover)

    city = api.get_voxcity_CityGML(
        RECT, "OpenStreetMap", "Static", 5.0, citygml_path="unused.gml",
        gridvis=False, save_voxcity_data=False, remove_perimeter_object=0.1)

    elevation = np.asarray(city.dem.elevation)
    assert np.all(elevation[5, 1:6] == 20.0), (
        "flattening must use the post-perimeter mask; the perimeter cell's "
        "1.0 m must not become the body minimum")
    assert city.extras["water_dem_flattening"]["water_cell_count"] == 5


def test_citygml_path_honours_flatten_water_dem_opt_out(monkeypatch):
    dem, land_cover = _one_body_grids()
    _stub_citygml_run(monkeypatch, dem, land_cover)

    city = api.get_voxcity_CityGML(
        RECT, "OpenStreetMap", "Static", 5.0, citygml_path="unused.gml",
        gridvis=False, save_voxcity_data=False, flatten_water_dem=False)

    elevation = np.asarray(city.dem.elevation)
    assert set(np.unique(elevation[1:3, :])) == {12.0, 15.0}, (
        "flatten_water_dem=False must preserve the raw DEM")
    assert city.extras["flatten_water_dem"] is False
    assert city.extras["water_dem_flattening"]["applied"] is False
