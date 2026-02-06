"""
Additional tests for voxcity.geoprocessor.utils module.
Tests for raster operations and coordinate utilities.
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.crs import CRS
from rasterio.transform import from_bounds
from shapely.geometry import Polygon, box


class TestRasterIntersectsPolygon:
    """Tests for raster_intersects_polygon function."""

    @pytest.fixture
    def sample_geotiff(self, tmp_path):
        """Create a sample GeoTIFF for testing."""
        path = tmp_path / "test.tif"
        
        # Create a small raster at 139-140E, 35-36N
        data = np.ones((10, 10), dtype=np.uint8)
        transform = from_bounds(139.0, 35.0, 140.0, 36.0, 10, 10)
        
        with rasterio.open(
            path, 'w',
            driver='GTiff',
            height=10, width=10,
            count=1,
            dtype=data.dtype,
            crs='EPSG:4326',
            transform=transform
        ) as dst:
            dst.write(data, 1)
        
        return str(path)

    def test_intersects_with_overlapping_polygon(self, sample_geotiff):
        """Test that overlapping polygon returns True."""
        from voxcity.geoprocessor.utils import raster_intersects_polygon
        
        polygon = box(139.5, 35.5, 140.5, 36.5)  # Overlaps with raster
        assert raster_intersects_polygon(sample_geotiff, polygon)

    def test_no_intersection_with_distant_polygon(self, sample_geotiff):
        """Test that non-overlapping polygon returns False."""
        from voxcity.geoprocessor.utils import raster_intersects_polygon
        
        polygon = box(150.0, 40.0, 151.0, 41.0)  # Far from raster
        assert not raster_intersects_polygon(sample_geotiff, polygon)

    def test_contained_polygon(self, sample_geotiff):
        """Test polygon fully contained within raster."""
        from voxcity.geoprocessor.utils import raster_intersects_polygon
        
        polygon = box(139.3, 35.3, 139.7, 35.7)  # Fully inside raster
        assert raster_intersects_polygon(sample_geotiff, polygon)


class TestSaveRaster:
    """Tests for save_raster function."""

    def test_copies_raster_file(self, tmp_path):
        """Test that raster is correctly copied."""
        from voxcity.geoprocessor.utils import save_raster
        
        # Create source file
        src_path = tmp_path / "source.tif"
        dst_path = tmp_path / "dest.tif"
        
        data = np.ones((5, 5), dtype=np.uint8)
        transform = from_bounds(0, 0, 1, 1, 5, 5)
        
        with rasterio.open(
            src_path, 'w',
            driver='GTiff',
            height=5, width=5,
            count=1,
            dtype=data.dtype,
            crs='EPSG:4326',
            transform=transform
        ) as dst:
            dst.write(data, 1)
        
        save_raster(str(src_path), str(dst_path))
        
        assert dst_path.exists()
        
        with rasterio.open(dst_path) as src:
            copied_data = src.read(1)
        
        np.testing.assert_array_equal(copied_data, data)


class TestGetRasterBbox:
    """Tests for get_raster_bbox function."""

    def test_returns_box_geometry(self, tmp_path):
        """Test that function returns a shapely box."""
        from voxcity.geoprocessor.utils import get_raster_bbox
        
        path = tmp_path / "test.tif"
        data = np.ones((10, 10), dtype=np.uint8)
        transform = from_bounds(100.0, 10.0, 101.0, 11.0, 10, 10)
        
        with rasterio.open(
            path, 'w',
            driver='GTiff',
            height=10, width=10,
            count=1,
            dtype=data.dtype,
            crs='EPSG:4326',
            transform=transform
        ) as dst:
            dst.write(data, 1)
        
        bbox = get_raster_bbox(str(path))
        
        assert bbox.bounds == pytest.approx((100.0, 10.0, 101.0, 11.0))


class TestConvertFormatLatLon:
    """Tests for convert_format_lat_lon function."""

    def test_closes_polygon(self):
        """Test that polygon is closed by repeating first point."""
        from voxcity.geoprocessor.utils import convert_format_lat_lon
        
        coords = [[0, 0], [1, 0], [1, 1], [0, 1]]
        result = convert_format_lat_lon(coords)
        
        assert len(result) == 5
        assert result[0] == result[-1]
        assert result[-1] == [0, 0]

    def test_preserves_original_order(self):
        """Test that coordinate order is preserved."""
        from voxcity.geoprocessor.utils import convert_format_lat_lon
        
        coords = [[139.0, 35.0], [139.1, 35.0], [139.1, 35.1]]
        result = convert_format_lat_lon(coords)
        
        assert result[0] == [139.0, 35.0]
        assert result[1] == [139.1, 35.0]
        assert result[2] == [139.1, 35.1]


class TestValidatePolygonCoordinates:
    """Tests for validate_polygon_coordinates function."""

    def test_valid_closed_polygon(self):
        """Test that valid closed polygon returns True."""
        from voxcity.geoprocessor.utils import validate_polygon_coordinates
        
        geometry = {
            'type': 'Polygon',
            'coordinates': [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]]
        }
        assert validate_polygon_coordinates(geometry)

    def test_auto_closes_open_polygon(self):
        """Test that open polygon is automatically closed."""
        from voxcity.geoprocessor.utils import validate_polygon_coordinates
        
        geometry = {
            'type': 'Polygon',
            'coordinates': [[[0, 0], [1, 0], [1, 1], [0, 1]]]
        }
        result = validate_polygon_coordinates(geometry)
        
        assert result
        assert geometry['coordinates'][0][0] == geometry['coordinates'][0][-1]

    def test_invalid_polygon_too_few_points(self):
        """Test that polygon with < 4 points returns False."""
        from voxcity.geoprocessor.utils import validate_polygon_coordinates
        
        geometry = {
            'type': 'Polygon',
            'coordinates': [[[0, 0], [1, 0]]]  # Only 2 points
        }
        assert not validate_polygon_coordinates(geometry)

    def test_valid_multipolygon(self):
        """Test that valid MultiPolygon returns True."""
        from voxcity.geoprocessor.utils import validate_polygon_coordinates
        
        geometry = {
            'type': 'MultiPolygon',
            'coordinates': [[[[0, 0], [1, 0], [1, 1], [0, 0]]]]
        }
        assert validate_polygon_coordinates(geometry)

    def test_invalid_geometry_type(self):
        """Test that non-polygon type returns False."""
        from voxcity.geoprocessor.utils import validate_polygon_coordinates
        
        geometry = {
            'type': 'Point',
            'coordinates': [0, 0]
        }
        assert not validate_polygon_coordinates(geometry)


class TestNormalizeToOneMeter:
    """Tests for normalize_to_one_meter function."""

    def test_normalizes_vector(self):
        """Test that vector is normalized to 1 meter."""
        from voxcity.geoprocessor.utils import normalize_to_one_meter
        
        vector = np.array([3.0, 4.0])  # Length 5
        result = normalize_to_one_meter(vector, 5.0)
        
        # Should be 1/5 of original
        np.testing.assert_array_almost_equal(result, np.array([0.6, 0.8]))

    def test_preserves_direction(self):
        """Test that direction is preserved."""
        from voxcity.geoprocessor.utils import normalize_to_one_meter
        
        vector = np.array([10.0, 0.0])
        result = normalize_to_one_meter(vector, 10.0)
        
        # Direction should be preserved (pointing in +X)
        assert result[0] > 0
        assert result[1] == 0


class TestHaversineDistance:
    """Tests for haversine_distance function."""

    def test_zero_distance_same_point(self):
        """Test distance between same point is zero."""
        from voxcity.geoprocessor.utils import haversine_distance
        
        dist = haversine_distance(139.0, 35.0, 139.0, 35.0)
        assert dist == pytest.approx(0.0, abs=0.001)

    def test_known_distance(self):
        """Test a known distance (approx Tokyo to Osaka ~400km)."""
        from voxcity.geoprocessor.utils import haversine_distance
        
        # Tokyo: 139.69, 35.69
        # Osaka: 135.50, 34.69
        dist = haversine_distance(139.69, 35.69, 135.50, 34.69)
        
        # Should be approximately 400 km
        assert 350 < dist < 450

    def test_symmetric(self):
        """Test that distance is symmetric."""
        from voxcity.geoprocessor.utils import haversine_distance
        
        d1 = haversine_distance(0, 0, 10, 10)
        d2 = haversine_distance(10, 10, 0, 0)
        
        assert d1 == pytest.approx(d2, rel=1e-10)
