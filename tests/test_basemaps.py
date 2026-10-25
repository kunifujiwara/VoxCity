"""Tests for voxcity.utils.basemaps and the basemap defaults that use it."""

import inspect
import warnings
from unittest.mock import patch

import pytest

from voxcity.utils.basemaps import (
    DEFAULT_STATIC_BASEMAP,
    ESRI_BASEMAPS,
    add_static_basemap,
    clamp_zoom,
    esri_tile_url,
    resolve_static_basemap,
)


class TestResolveStaticBasemap:
    def test_default_is_esri_light_gray(self):
        source = resolve_static_basemap(None)
        assert source["name"] == DEFAULT_STATIC_BASEMAP == "Esri light gray"
        assert source["url"] == esri_tile_url("Esri light gray")
        assert source["max_zoom"] == 16

    @pytest.mark.parametrize("name", list(ESRI_BASEMAPS))
    def test_every_name_builds_a_keyless_esri_url(self, name):
        source = resolve_static_basemap(name)
        assert source["url"].startswith("https://server.arcgisonline.com/")
        assert source["url"].endswith("/MapServer/tile/{z}/{y}/{x}")
        assert source.build_url(x=1, y=2, z=3).endswith("/tile/3/2/1")

    @pytest.mark.parametrize("legacy, expected", [
        ("CartoDB light", "Esri light gray"),
        ("CartoDB light no labels", "Esri light gray"),
        ("CartoDB dark", "Esri dark gray"),
        ("CartoDB dark no labels", "Esri dark gray"),
        ("CartoDB voyager", "Esri street"),
    ])
    def test_carto_names_map_to_esri_with_warning(self, legacy, expected):
        with pytest.warns(FutureWarning, match="API key"):
            source = resolve_static_basemap(legacy)
        assert source["name"] == expected

    def test_provider_and_url_template_pass_through(self):
        import xyzservices.providers as xyz

        provider = xyz.Esri.WorldImagery
        assert resolve_static_basemap(provider) is provider
        url = "https://example.com/tiles/{z}/{x}/{y}.png?key=abc"
        assert resolve_static_basemap(url) == url

    def test_unknown_name_raises(self):
        with pytest.raises(ValueError, match="Unknown basemap 'Stamen toner'"):
            resolve_static_basemap("Stamen toner")


class TestClampZoom:
    def test_explicit_zoom_capped_at_provider_max(self):
        source = resolve_static_basemap("Esri light gray")
        assert clamp_zoom(source, 18) == 16
        assert clamp_zoom(source, 12) == 12

    def test_auto_and_unbounded_sources_unchanged(self):
        assert clamp_zoom(resolve_static_basemap("Esri light gray"), "auto") == "auto"
        assert clamp_zoom("https://example.com/{z}/{x}/{y}.png", 21) == 21


class TestAddStaticBasemap:
    @patch("contextily.add_basemap")
    def test_passes_resolved_source_and_clamped_zoom(self, mock_add):
        add_static_basemap("ax", "Esri dark gray", zoom=19, crs="EPSG:3857")
        _, kwargs = mock_add.call_args
        assert kwargs["source"]["name"] == "Esri dark gray"
        assert kwargs["zoom"] == 16
        assert kwargs["crs"] == "EPSG:3857"

    def test_silences_inferred_zoom_clip_warning(self):
        def fake_add_basemap(ax, **kwargs):
            warnings.warn(
                "The inferred zoom level of 18 is not valid for the current "
                "tile provider (valid zooms: 0 - 16).")

        with patch("contextily.add_basemap", side_effect=fake_add_basemap):
            with warnings.catch_warnings():
                warnings.simplefilter("error")
                add_static_basemap("ax")


class TestDefaultsNoLongerUseCarto:
    @pytest.mark.parametrize("module, func", [
        ("voxcity.visualizer.maps", "plot_grid"),
        ("voxcity.visualizer.maps", "visualize_land_cover_grid_on_map"),
        ("voxcity.visualizer.maps", "visualize_building_height_grid_on_map"),
        ("voxcity.visualizer.maps", "visualize_numerical_grid_on_map"),
        ("voxcity.visualizer.grids", "visualize_landcover_grid_on_basemap"),
        ("voxcity.visualizer.grids", "visualize_numerical_grid_on_basemap"),
        ("voxcity.visualizer.grids", "visualize_numerical_gdf_on_basemap"),
    ])
    def test_static_plot_default_basemap(self, module, func):
        import importlib

        fn = getattr(importlib.import_module(module), func)
        assert inspect.signature(fn).parameters["basemap"].default == DEFAULT_STATIC_BASEMAP

    def test_leaflet_default_layer_is_esri_street(self):
        pytest.importorskip("ipyleaflet")
        from voxcity.geoprocessor.draw._common import (
            create_basemap_tiles,
            default_basemap_layer,
        )

        layer = default_basemap_layer()
        assert layer.url == esri_tile_url("Esri street")
        for tile in create_basemap_tiles().values():
            assert "carto" not in tile.url and "openstreetmap.org" not in tile.url

    def test_interactive_maps_do_not_fall_back_to_osm(self):
        pytest.importorskip("ipyleaflet")
        from voxcity.geoprocessor.draw import create_rectangle_map, draw_rectangle_map

        for ui, _ in (create_rectangle_map(), draw_rectangle_map()):
            maps = [ui] if type(ui).__name__ == "Map" else [
                w for w in ui.children if type(w).__name__ == "Map"]
            urls = [layer.url for m in maps for layer in m.layers if hasattr(layer, "url")]
            assert urls and not any("openstreetmap.org" in u for u in urls)
