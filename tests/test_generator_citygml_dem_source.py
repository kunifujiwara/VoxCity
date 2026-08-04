"""DEM source routing for get_voxcity_CityGML (helper-level tests)."""
import numpy as np
import pytest

import voxcity.generator.api as api

RECT = [(139.770, 35.646), (139.770, 35.650),
        (139.775, 35.650), (139.775, 35.646)]


def test_default_uses_terrain_gdf(monkeypatch):
    sentinel = np.full((3, 4), 7.0)
    calls = {}

    def fake_from_gdf(terrain_gdf, meshsize, rectangle_vertices):
        calls["terrain_gdf"] = terrain_gdf
        return sentinel

    monkeypatch.setattr(api, "create_dem_grid_from_gdf_polygon",
                        fake_from_gdf)
    grid = api._resolve_citygml_dem_grid(
        dem_source=None, terrain_gdf="TERRAIN",
        land_cover_grid=np.zeros((3, 4)), rectangle_vertices=RECT,
        meshsize=5.0, output_dir="out", kwargs={"gridvis": False})
    assert grid is sentinel
    assert calls["terrain_gdf"] == "TERRAIN"


@pytest.mark.parametrize("choice", ["CityGML Terrain", "GeoDataFrame"])
def test_citygml_terrain_aliases_use_terrain_gdf(monkeypatch, choice):
    sentinel = np.full((3, 4), 7.0)
    monkeypatch.setattr(api, "create_dem_grid_from_gdf_polygon",
                        lambda *a, **k: sentinel)
    grid = api._resolve_citygml_dem_grid(
        dem_source=choice, terrain_gdf="TERRAIN",
        land_cover_grid=np.zeros((3, 4)), rectangle_vertices=RECT,
        meshsize=5.0, output_dir="out", kwargs={"gridvis": False})
    assert grid is sentinel


def test_flat_returns_zeros_matching_land_cover_shape():
    grid = api._resolve_citygml_dem_grid(
        dem_source="Flat", terrain_gdf=None,
        land_cover_grid=np.ones((3, 4)), rectangle_vertices=RECT,
        meshsize=5.0, output_dir="out", kwargs={})
    assert grid.shape == (3, 4)
    assert (grid == 0).all()


def test_named_source_routes_to_get_dem_grid(monkeypatch):
    sentinel = np.full((3, 4), 2.0)
    calls = {}

    def fake_get_dem_grid(rectangle_vertices, meshsize, source, output_dir,
                          **kwargs):
        calls["source"] = source
        calls["dem_interpolation"] = kwargs.get("dem_interpolation")
        return sentinel

    import voxcity.generator.grids as grids_mod
    monkeypatch.setattr(grids_mod, "get_dem_grid", fake_get_dem_grid)

    grid = api._resolve_citygml_dem_grid(
        dem_source="FABDEM", terrain_gdf=None,
        land_cover_grid=np.zeros((3, 4)), rectangle_vertices=RECT,
        meshsize=5.0, output_dir="out",
        kwargs={"dem_interpolation": True, "gridvis": False})
    assert grid is sentinel
    assert calls["source"] == "FABDEM"
    assert calls["dem_interpolation"] is True


def test_get_voxcity_citygml_signature_carries_dem_source():
    """The app probes this signature to decide whether it can forward an
    explicit DEM choice on the LOD1 non-cached path."""
    import inspect
    assert "dem_source" in inspect.signature(api.get_voxcity_CityGML).parameters


class _DemSourceCaptured(Exception):
    """Raised by the fake ``_resolve_citygml_dem_grid`` below to short-circuit
    ``get_voxcity_CityGML`` the instant it reaches the DEM step, once the
    shim has already decided what ``dem_source`` to pass down."""


def _stub_pipeline_up_to_dem_step(monkeypatch):
    """Stub every step of ``get_voxcity_CityGML`` that runs before the DEM
    line, so a test can drive the real function (not just the extracted
    helper) down to the ``flat_dem`` shim without running CityGML parsing,
    building-footprint processing, or canopy-height downloads.
    """
    monkeypatch.setattr(api, "load_buid_dem_veg_from_citygml",
                        lambda **kwargs: (None, None, None))
    monkeypatch.setattr(api, "get_land_cover_grid",
                        lambda *a, **k: np.zeros((3, 4)))
    monkeypatch.setattr(
        api, "create_building_height_grid_from_gdf_polygon",
        lambda *a, **k: (np.zeros((3, 4)), np.zeros((3, 4)),
                         np.zeros((3, 4), dtype=int), None))


def test_flat_dem_shim_resolves_flat_through_get_voxcity_citygml(monkeypatch):
    """``flat_dem=True`` is the pre-``dem_source`` spelling of "no terrain".
    Driven through the real entry point (not the helper directly), it must
    still resolve to ``dem_source="Flat"`` when ``dem_source`` is omitted."""
    _stub_pipeline_up_to_dem_step(monkeypatch)
    captured = {}

    def fake_resolve(dem_source, *args, **kwargs):
        captured["dem_source"] = dem_source
        raise _DemSourceCaptured

    monkeypatch.setattr(api, "_resolve_citygml_dem_grid", fake_resolve)

    with pytest.raises(_DemSourceCaptured):
        api.get_voxcity_CityGML(
            RECT, "OpenStreetMap", "Static", 5.0,
            citygml_path="unused.gml", flat_dem=True, gridvis=False)

    assert captured["dem_source"] == "Flat"


def test_flat_dem_shim_yields_to_explicit_dem_source(monkeypatch):
    """When a caller passes both the legacy ``flat_dem=True`` and an explicit
    ``dem_source``, the explicit value must win (the shim's
    ``and dem_source is None`` guard)."""
    _stub_pipeline_up_to_dem_step(monkeypatch)
    captured = {}

    def fake_resolve(dem_source, *args, **kwargs):
        captured["dem_source"] = dem_source
        raise _DemSourceCaptured

    monkeypatch.setattr(api, "_resolve_citygml_dem_grid", fake_resolve)

    with pytest.raises(_DemSourceCaptured):
        api.get_voxcity_CityGML(
            RECT, "OpenStreetMap", "Static", 5.0,
            citygml_path="unused.gml", flat_dem=True,
            dem_source="FABDEM", gridvis=False)

    assert captured["dem_source"] == "FABDEM"
