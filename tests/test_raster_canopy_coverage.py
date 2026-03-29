"""Tests for geoprocessor/raster/canopy.py to improve coverage."""

import numpy as np
import pytest
import geopandas as gpd
from shapely.geometry import Polygon, Point, MultiPolygon


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def small_polygon():
    """~200m x 200m rectangle in Tokyo."""
    return [
        (139.756, 35.671),
        (139.756, 35.673),
        (139.758, 35.673),
        (139.758, 35.671),
    ]


@pytest.fixture
def veg_gdf_epsg4326():
    """Small vegetation GeoDataFrame with a polygon inside the test area."""
    poly = Polygon([
        (139.7565, 35.6715),
        (139.7575, 35.6715),
        (139.7575, 35.6725),
        (139.7565, 35.6725),
    ])
    return gpd.GeoDataFrame(
        {"height": [10.0]},
        geometry=[poly],
        crs="EPSG:4326",
    )


@pytest.fixture
def terrain_gdf():
    """Simple terrain point cloud."""
    pts = [Point(139.756 + i * 0.0005, 35.671 + j * 0.0005)
           for i in range(5) for j in range(5)]
    elevations = [float(i) for i in range(25)]
    return gpd.GeoDataFrame(
        {"elevation": elevations},
        geometry=pts,
        crs="EPSG:4326",
    )


# ---------------------------------------------------------------------------
# create_vegetation_height_grid_from_gdf_polygon
# ---------------------------------------------------------------------------

class TestCreateVegetationHeightGrid:
    def test_basic(self, veg_gdf_epsg4326, small_polygon):
        from voxcity.geoprocessor.raster.canopy import create_vegetation_height_grid_from_gdf_polygon
        grid = create_vegetation_height_grid_from_gdf_polygon(veg_gdf_epsg4326, 50, small_polygon)
        assert grid.ndim == 2
        assert np.any(grid > 0)

    def test_no_crs(self, veg_gdf_epsg4326, small_polygon):
        from voxcity.geoprocessor.raster.canopy import create_vegetation_height_grid_from_gdf_polygon
        gdf = veg_gdf_epsg4326.copy()
        gdf.crs = None
        with pytest.warns(UserWarning, match="no CRS"):
            grid = create_vegetation_height_grid_from_gdf_polygon(gdf, 50, small_polygon)
        assert grid.ndim == 2

    def test_non_4326_crs(self, veg_gdf_epsg4326, small_polygon):
        from voxcity.geoprocessor.raster.canopy import create_vegetation_height_grid_from_gdf_polygon
        gdf = veg_gdf_epsg4326.to_crs(epsg=32654)  # UTM 54N
        grid = create_vegetation_height_grid_from_gdf_polygon(gdf, 50, small_polygon)
        assert grid.ndim == 2

    def test_missing_height_column(self, small_polygon):
        from voxcity.geoprocessor.raster.canopy import create_vegetation_height_grid_from_gdf_polygon
        gdf = gpd.GeoDataFrame(geometry=[Point(0, 0)], crs="EPSG:4326")
        with pytest.raises(ValueError, match="height"):
            create_vegetation_height_grid_from_gdf_polygon(gdf, 50, small_polygon)

    def test_shapely_polygon_input(self, veg_gdf_epsg4326, small_polygon):
        from voxcity.geoprocessor.raster.canopy import create_vegetation_height_grid_from_gdf_polygon
        poly = Polygon(small_polygon)
        grid = create_vegetation_height_grid_from_gdf_polygon(veg_gdf_epsg4326, 50, poly)
        assert grid.ndim == 2

    def test_invalid_polygon_type(self, veg_gdf_epsg4326):
        from voxcity.geoprocessor.raster.canopy import create_vegetation_height_grid_from_gdf_polygon
        with pytest.raises(ValueError, match="polygon"):
            create_vegetation_height_grid_from_gdf_polygon(veg_gdf_epsg4326, 50, "bad")

    def test_tiny_polygon_returns_empty(self, veg_gdf_epsg4326):
        from voxcity.geoprocessor.raster.canopy import create_vegetation_height_grid_from_gdf_polygon
        tiny = [(139.756, 35.671), (139.756, 35.671001), (139.756001, 35.671001), (139.756001, 35.671)]
        with pytest.warns(UserWarning, match="smaller than mesh_size"):
            grid = create_vegetation_height_grid_from_gdf_polygon(veg_gdf_epsg4326, 9999, tiny)
        assert grid.size == 0


# ---------------------------------------------------------------------------
# create_dem_grid_from_gdf_polygon
# ---------------------------------------------------------------------------

class TestCreateDemGridFromGdfPolygon:
    def test_basic(self, terrain_gdf, small_polygon):
        from voxcity.geoprocessor.raster.canopy import create_dem_grid_from_gdf_polygon
        grid = create_dem_grid_from_gdf_polygon(terrain_gdf, 50, small_polygon)
        assert grid.ndim == 2
        assert not np.all(np.isnan(grid))

    def test_no_crs(self, terrain_gdf, small_polygon):
        from voxcity.geoprocessor.raster.canopy import create_dem_grid_from_gdf_polygon
        gdf = terrain_gdf.copy()
        gdf.crs = None
        with pytest.warns(UserWarning, match="no CRS"):
            grid = create_dem_grid_from_gdf_polygon(gdf, 50, small_polygon)
        assert grid.ndim == 2

    def test_non_4326_crs(self, terrain_gdf, small_polygon):
        from voxcity.geoprocessor.raster.canopy import create_dem_grid_from_gdf_polygon
        gdf = terrain_gdf.to_crs(epsg=32654)
        grid = create_dem_grid_from_gdf_polygon(gdf, 50, small_polygon)
        assert grid.ndim == 2

    def test_missing_elevation_column(self, small_polygon):
        from voxcity.geoprocessor.raster.canopy import create_dem_grid_from_gdf_polygon
        gdf = gpd.GeoDataFrame(
            {"val": [1.0]},
            geometry=[Point(139.757, 35.672)],
            crs="EPSG:4326",
        )
        with pytest.raises(ValueError, match="elevation"):
            create_dem_grid_from_gdf_polygon(gdf, 50, small_polygon)

    def test_shapely_polygon_input(self, terrain_gdf, small_polygon):
        from voxcity.geoprocessor.raster.canopy import create_dem_grid_from_gdf_polygon
        poly = Polygon(small_polygon)
        grid = create_dem_grid_from_gdf_polygon(terrain_gdf, 50, poly)
        assert grid.ndim == 2

    def test_invalid_polygon_type(self, terrain_gdf):
        from voxcity.geoprocessor.raster.canopy import create_dem_grid_from_gdf_polygon
        with pytest.raises(ValueError, match="polygon"):
            create_dem_grid_from_gdf_polygon(terrain_gdf, 50, 42)

    def test_tiny_polygon_returns_empty(self, terrain_gdf):
        from voxcity.geoprocessor.raster.canopy import create_dem_grid_from_gdf_polygon
        tiny = [(139.756, 35.671), (139.756, 35.671001), (139.756001, 35.671001), (139.756001, 35.671)]
        with pytest.warns(UserWarning, match="smaller than mesh_size"):
            grid = create_dem_grid_from_gdf_polygon(terrain_gdf, 9999, tiny)
        assert grid.size == 0


# ---------------------------------------------------------------------------
# create_canopy_grids_from_tree_gdf
# ---------------------------------------------------------------------------

class TestCreateCanopyGridsFromTreeGdf:
    @pytest.fixture
    def rect_verts(self):
        return [
            (139.756, 35.671),
            (139.756, 35.673),
            (139.758, 35.673),
            (139.758, 35.671),
        ]

    def _make_point_tree_gdf(self, lon, lat, top_h=10.0, bot_h=2.0, dia=25.0):
        return gpd.GeoDataFrame(
            {
                "top_height": [top_h],
                "bottom_height": [bot_h],
                "crown_diameter": [dia],
                "geometry_type": ["point"],
            },
            geometry=[Point(lon, lat)],
            crs="EPSG:4326",
        )

    def _make_polygon_tree_gdf(self, polygon, top_h=8.0, bot_h=1.0):
        return gpd.GeoDataFrame(
            {
                "top_height": [top_h],
                "bottom_height": [bot_h],
                "crown_diameter": [0.0],
                "geometry_type": ["polygon"],
            },
            geometry=[polygon],
            crs="EPSG:4326",
        )

    def test_none_input(self, rect_verts):
        from voxcity.geoprocessor.raster.canopy import create_canopy_grids_from_tree_gdf
        top, bot = create_canopy_grids_from_tree_gdf(None, 10, rect_verts)
        assert top.size == 0

    def test_empty_gdf(self, rect_verts):
        from voxcity.geoprocessor.raster.canopy import create_canopy_grids_from_tree_gdf
        gdf = gpd.GeoDataFrame(
            {"top_height": [], "bottom_height": [], "crown_diameter": [], "geometry_type": []},
            geometry=[],
            crs="EPSG:4326",
        )
        top, bot = create_canopy_grids_from_tree_gdf(gdf, 10, rect_verts)
        assert top.size == 0

    def test_missing_column(self, rect_verts):
        from voxcity.geoprocessor.raster.canopy import create_canopy_grids_from_tree_gdf
        gdf = gpd.GeoDataFrame(geometry=[Point(0, 0)], crs="EPSG:4326")
        with pytest.raises(ValueError, match="top_height"):
            create_canopy_grids_from_tree_gdf(gdf, 10, rect_verts)

    def test_point_trees(self, rect_verts):
        from voxcity.geoprocessor.raster.canopy import create_canopy_grids_from_tree_gdf
        gdf = self._make_point_tree_gdf(139.757, 35.672)
        top, bot = create_canopy_grids_from_tree_gdf(gdf, 10, rect_verts)
        assert top.shape == bot.shape
        assert np.any(top > 0)
        assert np.all(bot <= top)

    def test_polygon_trees(self, rect_verts):
        from voxcity.geoprocessor.raster.canopy import create_canopy_grids_from_tree_gdf
        poly = Polygon([
            (139.7565, 35.6715),
            (139.7575, 35.6715),
            (139.7575, 35.6725),
            (139.7565, 35.6725),
        ])
        gdf = self._make_polygon_tree_gdf(poly)
        top, bot = create_canopy_grids_from_tree_gdf(gdf, 10, rect_verts)
        assert np.any(top > 0)

    def test_multipolygon_trees(self, rect_verts):
        from voxcity.geoprocessor.raster.canopy import create_canopy_grids_from_tree_gdf
        p1 = Polygon([(139.7565, 35.6715), (139.757, 35.6715), (139.757, 35.672), (139.7565, 35.672)])
        p2 = Polygon([(139.757, 35.672), (139.7575, 35.672), (139.7575, 35.6725), (139.757, 35.6725)])
        mp = MultiPolygon([p1, p2])
        gdf = gpd.GeoDataFrame(
            {"top_height": [12.0], "bottom_height": [3.0], "crown_diameter": [0.0], "geometry_type": ["polygon"]},
            geometry=[mp],
            crs="EPSG:4326",
        )
        top, bot = create_canopy_grids_from_tree_gdf(gdf, 10, rect_verts)
        assert np.any(top > 0)

    def test_no_crs(self, rect_verts):
        from voxcity.geoprocessor.raster.canopy import create_canopy_grids_from_tree_gdf
        gdf = self._make_point_tree_gdf(139.757, 35.672)
        gdf.crs = None
        with pytest.warns(UserWarning, match="no CRS"):
            top, bot = create_canopy_grids_from_tree_gdf(gdf, 10, rect_verts)
        assert top.shape == bot.shape

    def test_non_4326_crs(self, rect_verts):
        from voxcity.geoprocessor.raster.canopy import create_canopy_grids_from_tree_gdf
        gdf = self._make_point_tree_gdf(139.757, 35.672).to_crs(epsg=32654)
        top, bot = create_canopy_grids_from_tree_gdf(gdf, 10, rect_verts)
        assert top.shape == bot.shape

    def test_auto_geom_type_detection(self, rect_verts):
        """Test without 'geometry_type' column — should auto-detect from geometry."""
        from voxcity.geoprocessor.raster.canopy import create_canopy_grids_from_tree_gdf
        gdf = self._make_point_tree_gdf(139.757, 35.672)
        gdf = gdf.drop(columns=["geometry_type"])
        top, bot = create_canopy_grids_from_tree_gdf(gdf, 10, rect_verts)
        assert np.any(top > 0)

    def test_swapped_heights(self, rect_verts):
        """bottom > top should be auto-corrected."""
        from voxcity.geoprocessor.raster.canopy import create_canopy_grids_from_tree_gdf
        gdf = self._make_point_tree_gdf(139.757, 35.672, top_h=2.0, bot_h=10.0)
        top, bot = create_canopy_grids_from_tree_gdf(gdf, 10, rect_verts)
        assert np.all(bot <= top)

    def test_zero_height_tree_ignored(self, rect_verts):
        from voxcity.geoprocessor.raster.canopy import create_canopy_grids_from_tree_gdf
        gdf = self._make_point_tree_gdf(139.757, 35.672, top_h=0.0, bot_h=0.0, dia=5.0)
        top, _ = create_canopy_grids_from_tree_gdf(gdf, 10, rect_verts)
        assert np.all(top == 0)

    def test_zero_diameter_tree_ignored(self, rect_verts):
        from voxcity.geoprocessor.raster.canopy import create_canopy_grids_from_tree_gdf
        gdf = self._make_point_tree_gdf(139.757, 35.672, top_h=10.0, bot_h=2.0, dia=0.0)
        top, _ = create_canopy_grids_from_tree_gdf(gdf, 10, rect_verts)
        assert np.all(top == 0)

    def test_polygon_zero_height_ignored(self, rect_verts):
        from voxcity.geoprocessor.raster.canopy import create_canopy_grids_from_tree_gdf
        poly = Polygon([(139.7565, 35.6715), (139.7575, 35.6715),
                         (139.7575, 35.6725), (139.7565, 35.6725)])
        gdf = self._make_polygon_tree_gdf(poly, top_h=0.0, bot_h=0.0)
        top, _ = create_canopy_grids_from_tree_gdf(gdf, 10, rect_verts)
        assert np.all(top == 0)
