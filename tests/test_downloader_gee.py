import os
import pytest

# Skip all tests in this module if Earth Engine is not authenticated
pytest_plugins = []
gee_available = os.getenv('GEE_AUTHENTICATED', 'false').lower() == 'true'

pytestmark = pytest.mark.skipif(
    not gee_available,
    reason="Google Earth Engine authentication not available"
)


@pytest.fixture(scope="module")
def ee_initialized():
    """Initialize Earth Engine for tests."""
    if not gee_available:
        pytest.skip("GEE not authenticated")
    
    import ee
    try:
        ee.Initialize()
        return True
    except Exception as e:
        pytest.skip(f"Failed to initialize Earth Engine: {e}")


def test_gee_basic_functionality(ee_initialized):
    """Test basic Earth Engine functionality."""
    import ee
    
    # Test basic collection access
    collection = ee.ImageCollection('MODIS/006/MCD12Q1')
    assert collection is not None
    
    # Test simple filtering
    filtered = collection.filterDate('2020-01-01', '2020-12-31')
    size = filtered.size()
    assert size.getInfo() > 0


def test_gee_landcover_data_access(ee_initialized):
    """Test access to landcover data used by voxcity."""
    import ee
    
    # Test accessing land cover data that voxcity might use
    landcover = ee.Image('MODIS/006/MCD12Q1/2020_01_01')
    assert landcover is not None
    
    # Test getting basic properties
    projection = landcover.select('LC_Type1').projection()
    assert projection is not None


def test_gee_image_download_mock(ee_initialized):
    """Test Earth Engine image download preparation (without actual download)."""
    import ee
    
    # Create a small test region
    geometry = ee.Geometry.Rectangle([-122.3, 37.8, -122.2, 37.9])  # Small SF area
    
    # Get a simple image
    image = ee.Image('MODIS/006/MCD12Q1/2020_01_01').select('LC_Type1')
    
    # Test clipping to region
    clipped = image.clip(geometry)
    assert clipped is not None
    
    # Test getting download URL (but don't actually download)
    url = clipped.getDownloadURL({
        'scale': 500,
        'region': geometry,
        'format': 'GeoTIFF'
    })
    assert url.startswith('https://')
