import pytest
import os
import sys
from pathlib import Path

# Add the src directory to the Python path
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
def test_data_dir():
    """Path to test data directory"""
    return Path(__file__).parent / "test_data"

@pytest.fixture
def sample_geojson():
    """Sample GeoJSON feature for testing"""
    return {
        "type": "Feature",
        "geometry": {
            "type": "Polygon",
            "coordinates": [[[139.7564, 35.6713], [139.7564, 35.6758], 
                           [139.7619, 35.6758], [139.7619, 35.6713], 
                           [139.7564, 35.6713]]]
        },
        "properties": {
            "height": 25.0,
            "min_height": 0.0,
            "id": 1
        }
    }

@pytest.fixture(scope="session")
def gee_authenticated():
    """Check if Google Earth Engine is authenticated and available."""
    return os.getenv('GEE_AUTHENTICATED', 'false').lower() == 'true' 