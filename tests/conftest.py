import pytest
import numpy as np
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

# Add the src directory to the Python path
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

@pytest.fixture
def sample_rectangle_vertices():
    """Sample rectangle vertices for testing"""
    return [
        (139.7564216011559, 35.671290792464255),
        (139.7564216011559, 35.67579720669077),
        (139.76194439884412, 35.67579720669077),
        (139.76194439884412, 35.671290792464255)
    ]

@pytest.fixture
def sample_polygon():
    """Sample polygon for testing"""
    from shapely.geometry import Polygon
    return Polygon([
        (139.7564, 35.6713),
        (139.7564, 35.6758),
        (139.7619, 35.6758),
        (139.7619, 35.6713)
    ])

@pytest.fixture
def sample_voxel_grid():
    """Sample voxel grid for testing"""
    return np.random.randint(0, 5, (10, 10, 10), dtype=int)

@pytest.fixture
def sample_building_height_grid():
    """Sample building height grid for testing"""
    return np.random.uniform(0, 50, (10, 10))

@pytest.fixture
def sample_building_id_grid():
    """Sample building ID grid for testing"""
    return np.random.randint(0, 10, (10, 10), dtype=int)

@pytest.fixture
def sample_land_cover_grid():
    """Sample land cover grid for testing"""
    return np.random.randint(1, 5, (10, 10), dtype=int)

@pytest.fixture
def sample_canopy_height_grid():
    """Sample canopy height grid for testing"""
    return np.random.uniform(0, 20, (10, 10))

@pytest.fixture
def sample_dem_grid():
    """Sample DEM grid for testing"""
    return np.random.uniform(0, 100, (10, 10))

@pytest.fixture
def sample_land_cover_array():
    """Sample land cover array for testing"""
    return np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

@pytest.fixture
def sample_numerical_grid():
    """Sample numerical grid for testing"""
    return np.random.uniform(0, 100, (10, 10))

@pytest.fixture
def sample_land_cover_classes():
    """Sample land cover classes for testing"""
    return {
        1: 'Building',
        2: 'Road',
        3: 'Vegetation',
        4: 'Water'
    }

@pytest.fixture
def sample_building_gdf():
    """Sample building GeoDataFrame for testing"""
    try:
        import geopandas as gpd
        from shapely.geometry import Polygon
        
        buildings = [
            {
                'geometry': Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
                'height': 25.0,
                'min_height': 0.0,
                'id': 1
            }
        ]
        return gpd.GeoDataFrame(buildings, crs='EPSG:4326')
    except ImportError:
        pytest.skip("geopandas not available")

@pytest.fixture
def sample_osm_data():
    """Sample OSM data for testing"""
    return {
        "elements": [
            {
                "type": "way",
                "id": 123,
                "nodes": [1, 2, 3, 4],
                "tags": {"building": "yes", "height": "25"}
            },
            {
                "type": "node",
                "id": 1,
                "lat": 35.6713,
                "lon": 139.7564
            }
        ]
    }

@pytest.fixture
def temp_output_dir():
    """Temporary directory for test outputs"""
    with tempfile.TemporaryDirectory() as tmpdirname:
        yield Path(tmpdirname)

@pytest.fixture
def sample_sun_direction():
    """Sample sun direction vector"""
    return np.array([0.0, 0.0, 1.0])

@pytest.fixture
def sample_view_point():
    """Sample view point coordinates"""
    return np.array([5.0, 5.0, 5.0]) 