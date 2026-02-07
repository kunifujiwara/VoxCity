"""Round 6 – cover heights.py extract_building_heights_from_geotiff (lines 161-197) and complement_building_heights_from_gdf additional stats printing."""
from __future__ import annotations

import numpy as np
import pytest
from unittest.mock import patch, MagicMock
import geopandas as gpd
from shapely.geometry import Polygon, box


# ===========================================================================
# Tests for extract_building_heights_from_geotiff
# ===========================================================================

class TestExtractBuildingHeightsFromGeotiff:
    """Cover heights.py lines 161-197."""

    @patch("voxcity.geoprocessor.heights.rasterio")
    def test_assigns_heights(self, mock_rasterio):
        from voxcity.geoprocessor.heights import extract_building_heights_from_geotiff

        # Build a simple GDF with one building polygon that has no height
        poly = box(0.0, 0.0, 0.001, 0.001)
        gdf = gpd.GeoDataFrame({"geometry": [poly], "height": [0.0]}, crs="EPSG:4326")

        # Mock rasterio context
        mock_src = MagicMock()
        mock_src.crs = "EPSG:4326"
        mock_src.nodata = -9999
        # Mock mask returns array with valid height data
        masked_data = np.array([[[10.0, 12.0, 11.0]]])
        mock_rasterio.mask.mask.return_value = (masked_data, MagicMock())
        mock_rasterio.open.return_value.__enter__ = MagicMock(return_value=mock_src)
        mock_rasterio.open.return_value.__exit__ = MagicMock(return_value=False)

        result = extract_building_heights_from_geotiff("fake.tif", gdf)
        # Should have assigned a height value
        assert result["height"].iloc[0] == pytest.approx(11.0)

    @patch("voxcity.geoprocessor.heights.rasterio")
    def test_no_data_in_raster(self, mock_rasterio):
        from voxcity.geoprocessor.heights import extract_building_heights_from_geotiff

        poly = box(0.0, 0.0, 0.001, 0.001)
        gdf = gpd.GeoDataFrame({"geometry": [poly], "height": [np.nan]}, crs="EPSG:4326")

        mock_src = MagicMock()
        mock_src.crs = "EPSG:4326"
        mock_src.nodata = -9999
        # All nodata → empty heights
        masked_data = np.array([[[-9999, -9999]]])
        mock_rasterio.mask.mask.return_value = (masked_data, MagicMock())
        mock_rasterio.open.return_value.__enter__ = MagicMock(return_value=mock_src)
        mock_rasterio.open.return_value.__exit__ = MagicMock(return_value=False)

        result = extract_building_heights_from_geotiff("fake.tif", gdf)
        assert np.isnan(result["height"].iloc[0])

    @patch("voxcity.geoprocessor.heights.rasterio")
    def test_skip_existing_heights(self, mock_rasterio):
        from voxcity.geoprocessor.heights import extract_building_heights_from_geotiff

        poly = box(0.0, 0.0, 0.001, 0.001)
        gdf = gpd.GeoDataFrame({"geometry": [poly], "height": [25.0]}, crs="EPSG:4326")

        mock_src = MagicMock()
        mock_src.crs = "EPSG:4326"
        mock_rasterio.open.return_value.__enter__ = MagicMock(return_value=mock_src)
        mock_rasterio.open.return_value.__exit__ = MagicMock(return_value=False)

        result = extract_building_heights_from_geotiff("fake.tif", gdf)
        # Should keep original height
        assert result["height"].iloc[0] == pytest.approx(25.0)


# ===========================================================================
# Tests for complement_building_heights_from_gdf – stats branch
# ===========================================================================

class TestComplementBuildingStatsOutput:
    """Cover heights.py lines 128-156 (statistics printing branch)."""

    def test_stats_printed_when_missing_heights(self, capsys):
        from voxcity.geoprocessor.heights import complement_building_heights_from_gdf

        # Primary GDF with 2 buildings: one with height, one without
        primary_gdf = gpd.GeoDataFrame({
            "geometry": [box(0, 0, 1, 1), box(2, 2, 3, 3)],
            "height": [10.0, 0.0],  # second has no height
            "id": [1, 2],
        }, crs="EPSG:4326")

        # Reference GDF with one building overlapping the second primary, with height
        ref_gdf = gpd.GeoDataFrame({
            "geometry": [box(2.1, 2.1, 2.9, 2.9), box(5, 5, 6, 6)],
            "height": [15.0, 20.0],
            "id": [101, 102],
        }, crs="EPSG:4326")

        result = complement_building_heights_from_gdf(primary_gdf, ref_gdf)
        captured = capsys.readouterr()
        # Stats should mention the number of buildings without height
        assert "did not have height data" in captured.out
        assert len(result) >= 2  # At least original + non-overlapping ref buildings
