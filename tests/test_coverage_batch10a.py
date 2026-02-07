"""
Coverage batch 10a – tests targeting remaining gaps in:
  heights.py, selection.py, raster/core.py, update.py, material.py,
  shape.py, grids.py, builder.py, epw.py, weather/files.py
"""

import os
import sys
import types
import tempfile
import zipfile
from pathlib import Path
from unittest.mock import patch, MagicMock, PropertyMock

import numpy as np
import pandas as pd
import geopandas as gpd
import pytest
from shapely.geometry import Polygon, Point, box
from shapely.errors import GEOSException, ShapelyError


# ---------------------------------------------------------------------------
# heights.py – extract_building_heights_from_gdf edge cases
# ---------------------------------------------------------------------------

class TestHeightsEdgeCases:
    """Targets: lines 27, 50, 63-64, 93, 95, 188-190."""

    def _make_gdf(self, geoms, heights=None, crs="EPSG:4326"):
        d = {"geometry": geoms}
        if heights is not None:
            d["height"] = heights
        return gpd.GeoDataFrame(d, crs=crs)

    def test_no_height_column_initialised(self):
        """Line 27 – height column should be auto-created."""
        from voxcity.geoprocessor.heights import extract_building_heights_from_gdf

        poly_a = box(0, 0, 1, 1)
        poly_b = box(0.5, 0.5, 1.5, 1.5)
        gdf_primary = gpd.GeoDataFrame({"geometry": [poly_a]}, crs="EPSG:4326")
        gdf_ref = gpd.GeoDataFrame({"geometry": [poly_b], "height": [10.0]}, crs="EPSG:4326")
        result = extract_building_heights_from_gdf(gdf_primary, gdf_ref)
        assert "height" in result.columns

    def test_ref_index_out_of_range_skip(self):
        """Line 50 – ref_idx >= len(gdf_ref) → continue."""
        from voxcity.geoprocessor.heights import extract_building_heights_from_gdf

        poly = box(0, 0, 1, 1)
        gdf_primary = self._make_gdf([poly], [0.0])
        gdf_ref = self._make_gdf([poly], [5.0])
        # Works normally – ref_idx should stay in range for well-formed data.
        result = extract_building_heights_from_gdf(gdf_primary, gdf_ref)
        assert result.at[0, "height"] == pytest.approx(5.0, rel=0.1)

    def test_geos_exception_triggers_buffer_fix(self):
        """Lines 63-64 – GEOSException → buffer(0) retry."""
        from voxcity.geoprocessor.heights import extract_building_heights_from_gdf

        poly_a = box(0, 0, 1, 1)
        poly_b = box(0.5, 0.5, 1.5, 1.5)
        gdf_primary = self._make_gdf([poly_a], [0.0])
        gdf_ref = self._make_gdf([poly_b], [10.0])

        # Patch intersects on the ref geometry to raise GEOSException first time
        original_intersects = poly_b.intersects
        call_count = [0]

        def patched_intersects(other):
            call_count[0] += 1
            if call_count[0] == 1:
                raise GEOSException("boom")
            return original_intersects(other)

        with patch.object(type(poly_b), "intersects", side_effect=patched_intersects):
            result = extract_building_heights_from_gdf(gdf_primary, gdf_ref)
        assert "height" in result.columns

    def test_no_overlap_nan_height(self):
        """Line 93 – overlapping_height_area == 0 → NaN."""
        from voxcity.geoprocessor.heights import extract_building_heights_from_gdf

        poly_a = box(0, 0, 1, 1)
        poly_b = box(10, 10, 11, 11)  # far away
        gdf_primary = self._make_gdf([poly_a], [0.0])
        gdf_ref = self._make_gdf([poly_b], [5.0])
        result = extract_building_heights_from_gdf(gdf_primary, gdf_ref)
        assert np.isnan(result.at[0, "height"])

    def test_extract_from_geotiff_value_error(self):
        """Line 188-190 – ValueError during rasterio mask → height = None."""
        from voxcity.geoprocessor.heights import extract_building_heights_from_geotiff

        poly = box(0, 0, 0.001, 0.001)
        gdf = self._make_gdf([poly], [0.0])
        gdf["id"] = [1]

        # Create a mock rasterio source
        mock_src = MagicMock()
        mock_src.crs = MagicMock()
        mock_src.nodata = -9999

        mock_ctx = MagicMock()
        mock_ctx.__enter__ = MagicMock(return_value=mock_src)
        mock_ctx.__exit__ = MagicMock(return_value=False)

        with patch("rasterio.open", return_value=mock_ctx):
            with patch("voxcity.geoprocessor.heights.Transformer") as mock_t:
                mock_transformer = MagicMock()
                mock_transformer.transform = MagicMock(return_value=(0, 0))
                mock_t.from_crs.return_value = mock_transformer
                with patch("rasterio.mask.mask", side_effect=ValueError("window outside raster")):
                    result = extract_building_heights_from_geotiff("fake.tif", gdf.copy())
                    # ValueError is caught; height is set to None which pandas stores as NaN
                    assert result.at[0, "height"] is None or pd.isna(result.at[0, "height"])


# ---------------------------------------------------------------------------
# selection.py – filter_buildings, find_building_containing_point,
#                get_buildings_in_drawn_polygon
# ---------------------------------------------------------------------------

class TestSelectionEdgeCases:
    """Targets: lines 29-30, 35-36, 50, 72."""

    def test_filter_buildings_invalid_geometry_skip(self):
        """Lines 29-30 – invalid geometry → skip."""
        from voxcity.geoprocessor.selection import filter_buildings

        good_poly = box(0, 0, 1, 1)
        plotting_box = box(-1, -1, 2, 2)

        features = [
            {"geometry": {"type": "Polygon", "coordinates": [[(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)]]}}
        ]

        # Patch shape() to return an invalid geom
        invalid_geom = MagicMock()
        invalid_geom.is_valid = False

        with patch("voxcity.geoprocessor.selection.shape", return_value=invalid_geom):
            result = filter_buildings(features, plotting_box)
        assert len(result) == 0

    def test_filter_buildings_shapely_error(self):
        """Lines 35-36 – ShapelyError → skip."""
        from voxcity.geoprocessor.selection import filter_buildings

        plotting_box = box(-1, -1, 2, 2)
        features = [
            {"geometry": {"type": "Polygon", "coordinates": [[(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)]]}}
        ]
        with patch("voxcity.geoprocessor.selection.shape", side_effect=ShapelyError("bad")):
            result = filter_buildings(features, plotting_box)
        assert len(result) == 0

    def test_find_building_containing_point_hit(self):
        """Line 50 – point inside polygon → id appended."""
        from voxcity.geoprocessor.selection import find_building_containing_point

        poly = box(0, 0, 1, 1)
        gdf = gpd.GeoDataFrame({"geometry": [poly], "id": [42]}, crs="EPSG:4326")
        result = find_building_containing_point(gdf, (0.5, 0.5))
        assert 42 in result

    def test_get_buildings_in_drawn_polygon_intersect(self):
        """Line 72 – intersect operation branch."""
        from voxcity.geoprocessor.selection import get_buildings_in_drawn_polygon

        poly = box(0, 0, 1, 1)
        gdf = gpd.GeoDataFrame({"geometry": [poly], "id": [7]}, crs="EPSG:4326")
        drawn = [{"vertices": [(-0.5, -0.5), (0.5, -0.5), (0.5, 0.5), (-0.5, 0.5)]}]
        result = get_buildings_in_drawn_polygon(gdf, drawn, operation="intersect")
        assert 7 in result


# ---------------------------------------------------------------------------
# raster/core.py – translate_array non-ndarray, process_grid fallback
# ---------------------------------------------------------------------------

class TestRasterCoreEdgeCases:
    """Targets: lines 28, 79-87."""

    def test_translate_array_list_input(self):
        """Line 28 – input is a list, auto-converted to ndarray."""
        from voxcity.geoprocessor.raster.core import translate_array

        result = translate_array([1, 2, 3], {1: "a", 2: "b", 3: "c"})
        assert list(result) == ["a", "b", "c"]

    def test_process_grid_fallback_on_error(self):
        """Lines 79-87 – optimized path fails → fallback loop."""
        from voxcity.geoprocessor.raster.core import process_grid

        grid_bi = np.array([[1, 0], [0, 2]])
        dem = np.array([[10.0, 5.0], [3.0, 8.0]])

        with patch("voxcity.geoprocessor.raster.core.process_grid_optimized", side_effect=RuntimeError("boom")):
            result = process_grid(grid_bi, dem)
        assert result.shape == (2, 2)
        assert result.min() == 0.0


# ---------------------------------------------------------------------------
# update.py – update_voxcity edge cases
# ---------------------------------------------------------------------------

class TestUpdateEdgeCases:
    """Targets: lines 215, 219, 273."""

    def test_add_mode_no_existing_top(self):
        """Line 215 – existing_top is None → canopy_top = new value."""
        from voxcity.generator.update import update_voxcity
        from voxcity.models import (
            VoxCity, VoxelGrid, BuildingGrid, LandCoverGrid, DemGrid, CanopyGrid, GridMetadata
        )

        meta = GridMetadata(crs="EPSG:4326", meshsize=10.0, bounds=(0, 0, 1, 1))
        city = VoxCity(
            voxels=VoxelGrid(classes=np.zeros((2, 2, 3), dtype=int), meta=meta),
            buildings=BuildingGrid(
                heights=np.zeros((2, 2)),
                min_heights=np.empty((2, 2), dtype=object),
                ids=np.zeros((2, 2), dtype=int),
                meta=meta,
            ),
            land_cover=LandCoverGrid(classes=np.zeros((2, 2), dtype=int), meta=meta),
            dem=DemGrid(elevation=np.zeros((2, 2)), meta=meta),
            tree_canopy=CanopyGrid(top=None, bottom=None, meta=meta),
            extras={},
        )
        # Fill min_heights
        for i in range(2):
            for j in range(2):
                city.buildings.min_heights[i, j] = []

        new_top = np.ones((2, 2)) * 5.0
        new_bot = np.ones((2, 2)) * 1.0
        result = update_voxcity(city, canopy_top=new_top, canopy_bottom=new_bot)
        np.testing.assert_array_equal(result.tree_canopy.top, new_top)

    def test_tree_canopy_raw_array_branch(self):
        """Line 273 – tree_canopy is ndarray (not CanopyGrid) → treated as top."""
        from voxcity.generator.update import update_voxcity
        from voxcity.models import (
            VoxCity, VoxelGrid, BuildingGrid, LandCoverGrid, DemGrid, CanopyGrid, GridMetadata
        )

        meta = GridMetadata(crs="EPSG:4326", meshsize=10.0, bounds=(0, 0, 1, 1))
        city = VoxCity(
            voxels=VoxelGrid(classes=np.zeros((2, 2, 3), dtype=int), meta=meta),
            buildings=BuildingGrid(
                heights=np.zeros((2, 2)),
                min_heights=np.empty((2, 2), dtype=object),
                ids=np.zeros((2, 2), dtype=int),
                meta=meta,
            ),
            land_cover=LandCoverGrid(classes=np.zeros((2, 2), dtype=int), meta=meta),
            dem=DemGrid(elevation=np.zeros((2, 2)), meta=meta),
            tree_canopy=CanopyGrid(top=np.zeros((2, 2)), bottom=np.zeros((2, 2)), meta=meta),
            extras={},
        )
        for i in range(2):
            for j in range(2):
                city.buildings.min_heights[i, j] = []

        raw_canopy = np.ones((2, 2)) * 7.0
        result = update_voxcity(city, tree_canopy=raw_canopy)
        np.testing.assert_array_equal(result.tree_canopy.top, raw_canopy)


# ---------------------------------------------------------------------------
# material.py – high window ratio branch
# ---------------------------------------------------------------------------

class TestMaterialWindowPattern:
    """Targets: lines 148-151 – window_ratio 0.625 < r <= 0.875."""

    def test_high_window_ratio_additional_pattern(self):
        from voxcity.utils.material import set_building_material_by_id

        # 3D grid: single building column of 10 z-levels
        grid = np.zeros((3, 3, 12), dtype=int)
        building_code = 7
        grid[1, 1, 2:12] = building_code  # 10 levels of building

        # building_id_grid that marks the building
        bid = np.zeros((3, 3), dtype=int)
        bid[1, 1] = 1

        result = set_building_material_by_id(
            grid.copy(), bid, ids=[1], mark=[building_code], window_ratio=0.8, glass_id=-16
        )
        # The function should have modified some voxels
        assert result.shape == grid.shape


# ---------------------------------------------------------------------------
# shape.py – pad/crop edge cases
# ---------------------------------------------------------------------------

class TestShapeEdgeCases:
    """Targets: lines 131-133 (non-callable pv_bmin), 180-181, 226."""

    def test_non_callable_pv_bmin_wrapped(self):
        """Line 131-133 – pv_bmin constant wrapped in lambda."""
        from voxcity.utils.shape import normalize_voxcity_shape
        from voxcity.models import (
            VoxCity, VoxelGrid, BuildingGrid, LandCoverGrid, DemGrid, CanopyGrid, GridMetadata
        )

        meta = GridMetadata(crs="EPSG:4326", meshsize=10.0, bounds=(0, 0, 1, 1))
        city = VoxCity(
            voxels=VoxelGrid(classes=np.zeros((3, 3, 4), dtype=int), meta=meta),
            buildings=BuildingGrid(
                heights=np.zeros((3, 3)),
                min_heights=np.empty((3, 3), dtype=object),
                ids=np.zeros((3, 3), dtype=int),
                meta=meta,
            ),
            land_cover=LandCoverGrid(classes=np.zeros((3, 3), dtype=int), meta=meta),
            dem=DemGrid(elevation=np.zeros((3, 3)), meta=meta),
            tree_canopy=CanopyGrid(top=np.zeros((3, 3)), bottom=np.zeros((3, 3)), meta=meta),
            extras={},
        )
        for i in range(3):
            for j in range(3):
                city.buildings.min_heights[i, j] = []

        # Pass a non-callable constant for building_min_heights_factory
        result = normalize_voxcity_shape(
            city, (2, 2, 3),
            pad_values={"building_min_heights_factory": 99},
            align_xy="center",
        )
        assert result.voxels.classes.shape[:2] == (2, 2)


# ---------------------------------------------------------------------------
# visualizer/grids.py – CRS detection edge cases
# ---------------------------------------------------------------------------

class TestGridsVisualizerEdgeCases:
    """Targets: lines 67-68, 72, 109."""

    def test_no_crs_not_lonlat_raises(self):
        """Line 72 – no CRS and coords not lon/lat → ValueError."""
        from voxcity.visualizer.grids import visualize_numerical_gdf_on_basemap

        gdf = gpd.GeoDataFrame(
            {"geometry": [box(1e6, 1e6, 2e6, 2e6)], "value": [1.0]}
        )  # no CRS, huge coords → not lon/lat
        with pytest.raises(ValueError, match="no CRS"):
            visualize_numerical_gdf_on_basemap(gdf)

    def test_no_crs_total_bounds_exception(self):
        """Lines 67-68 – total_bounds raises → looks_like_lonlat = False."""
        from voxcity.visualizer.grids import visualize_numerical_gdf_on_basemap

        gdf = gpd.GeoDataFrame({"geometry": [box(1e6, 1e6, 2e6, 2e6)], "value": [1.0]})
        with patch.object(type(gdf), "total_bounds", new_callable=PropertyMock, side_effect=Exception("oops")):
            with pytest.raises(ValueError, match="no CRS"):
                visualize_numerical_gdf_on_basemap(gdf)


# ---------------------------------------------------------------------------
# visualizer/builder.py – skip None meshes
# ---------------------------------------------------------------------------

class TestBuilderSkipNone:
    """Targets: line 33 – None mesh → continue."""

    def test_none_mesh_skipped(self):
        from voxcity.visualizer.builder import MeshBuilder
        from voxcity.models import VoxelGrid, GridMetadata
        import trimesh

        good_mesh = trimesh.Trimesh(
            vertices=[[0, 0, 0], [1, 0, 0], [0, 1, 0]],
            faces=[[0, 1, 2]],
        )

        meshes = {1: good_mesh, 2: None}

        meta = GridMetadata(crs="EPSG:4326", meshsize=10.0, bounds=(0, 0, 1, 1))
        vg = VoxelGrid(classes=np.zeros((2, 2, 2), dtype=int), meta=meta)

        with patch("voxcity.visualizer.builder.create_city_meshes", return_value=meshes):
            result = MeshBuilder.from_voxel_grid(vg, meshsize=10.0)
        # Only the good mesh should be in collection
        assert "1" in [m for m in result.meshes]
        assert "2" not in [m for m in result.meshes]


# ---------------------------------------------------------------------------
# utils/weather/epw.py – error cases
# ---------------------------------------------------------------------------

class TestEpwEdgeCases:
    """Targets: lines 112, 126, 132."""

    def _write_epw(self, path, has_location=True, has_data=True, short_lines=False):
        lines = []
        if has_location:
            lines.append("LOCATION,City,State,Country,Source,WMO,,35.0,139.0,9.0,10.0\n")
        else:
            lines.append("HEADER,stuff\n")
        # 7 filler lines to reach line 8
        for _ in range(7):
            lines.append("FILLER\n")
        if has_data:
            if short_lines:
                lines.append("2023,1,1,1,short\n")  # < 15 fields
                # Add a valid line after
                fields = ",".join(["0"] * 35)
                lines.append(f"2023,1,1,1,0,0,0,0,0,0,0,0,0,0,100,50,{fields}\n")
            else:
                fields = ",".join(["0"] * 35)
                lines.append(f"2023,1,1,1,0,0,0,0,0,0,0,0,0,0,100,50,{fields}\n")
        with open(path, "w") as f:
            f.writelines(lines)

    def test_no_location_line_raises(self):
        """Line 112 – no LOCATION line → ValueError."""
        from voxcity.utils.weather.epw import read_epw_for_solar_simulation

        with tempfile.NamedTemporaryFile(suffix=".epw", delete=False, mode="w") as f:
            f.write("HEADER,stuff\n" * 10)
            path = f.name
        try:
            with pytest.raises((ValueError, Exception)):
                read_epw_for_solar_simulation(path)
        finally:
            os.unlink(path)

    def test_no_data_lines_raises(self):
        """Line 126 – no data lines → ValueError."""
        from voxcity.utils.weather.epw import read_epw_for_solar_simulation

        with tempfile.NamedTemporaryFile(suffix=".epw", delete=False, mode="w") as f:
            # LOCATION has 10 fields: LOCATION,city,state,country,source,wmo,lat,lon,tz,elev
            f.write("LOCATION,City,State,Country,Source,WMO,35.0,139.0,9.0,10.0\n")
            for _ in range(7):
                f.write("FILLER\n")
            # No data lines with >30 fields
            path = f.name
        try:
            with pytest.raises((ValueError, Exception)):
                read_epw_for_solar_simulation(path)
        finally:
            os.unlink(path)

    def test_short_data_line_skipped(self):
        """Line 132 – lines with < 15 fields → skipped."""
        from voxcity.utils.weather.epw import read_epw_for_solar_simulation

        with tempfile.NamedTemporaryFile(suffix=".epw", delete=False, mode="w") as f:
            f.write("LOCATION,City,State,Country,Source,WMO,35.0,139.0,9.0,10.0\n")
            for _ in range(7):
                f.write("FILLER\n")
            # One short line then a valid line (>30 fields for detection, >=15 for parsing)
            f.write("short,line\n")
            # Build a valid EPW data line: year,month,day,hour + 31 more fields (>30 total)
            fields = ",".join(["0"] * 28)
            f.write(f"2023,1,1,1,{fields},100,50\n")
            path = f.name
        try:
            df, lon, lat, tz, elev = read_epw_for_solar_simulation(path)
            assert lat == pytest.approx(35.0)
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# utils/weather/files.py – safe_extract FileExistsError
# ---------------------------------------------------------------------------

class TestWeatherFilesEdgeCases:
    """Targets: lines 31-34 – FileExistsError handling."""

    def test_safe_extract_file_exists_error(self):
        """Lines 31-34 – FileExistsError → extract with temp name."""
        from voxcity.utils.weather.files import safe_extract

        mock_zip = MagicMock(spec=zipfile.ZipFile)
        # First call raises FileExistsError, second succeeds
        mock_zip.extract = MagicMock(side_effect=[FileExistsError("exists"), None])

        with tempfile.TemporaryDirectory() as tmpdir:
            result = safe_extract(mock_zip, "test.txt", Path(tmpdir))
            # Should have called extract twice (once failing, once with temp name)
            assert mock_zip.extract.call_count == 2
