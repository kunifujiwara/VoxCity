import pytest
import os
import sys
import json
import math
import logging
from pathlib import Path

import numpy as np

# Disable Numba JIT so coverage.py can trace @njit function bodies.
# Must be set before numba is first imported by any test module.
os.environ["NUMBA_DISABLE_JIT"] = "1"

# Configure matplotlib to use non-interactive backend BEFORE importing pyplot
# This prevents visualization windows from appearing during tests
import matplotlib
matplotlib.use('Agg')

# Add the src directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))


# ---------------------------------------------------------------------------
# Shared helpers for v3 HDF5 / projector tests (test_io_v3.py,
# test_projector_factories.py). Kept as plain module-level functions/
# constants (not fixtures) so they can be imported directly, mirroring how
# they were originally defined in test_io_v3.py.
# ---------------------------------------------------------------------------

RECT = [(0.0, 0.0), (0.0, 0.01), (0.01, 0.01), (0.01, 0.0)]  # axis-aligned, ~1.1 km


def make_city(shape=(4, 5, 6), meshsize=2.0, extras=None):
    from voxcity.models import (
        GridMetadata,
        VoxelGrid,
        BuildingGrid,
        LandCoverGrid,
        DemGrid,
        CanopyGrid,
        VoxCity,
    )

    ny, nx, nz = shape
    meta = GridMetadata(crs="EPSG:4326", bounds=(0.0, 0.0, 0.01, 0.01), meshsize=meshsize)
    min_heights = np.empty((ny, nx), dtype=object)
    for idx in np.ndindex((ny, nx)):
        min_heights[idx] = []
    return VoxCity(
        voxels=VoxelGrid(classes=np.zeros(shape, dtype=np.int8), meta=meta),
        buildings=BuildingGrid(
            heights=np.zeros((ny, nx)),
            min_heights=min_heights,
            ids=np.zeros((ny, nx)),
            meta=meta,
        ),
        land_cover=LandCoverGrid(classes=np.ones((ny, nx), dtype=int), meta=meta),
        dem=DemGrid(elevation=np.zeros((ny, nx)), meta=meta),
        tree_canopy=CanopyGrid(top=np.zeros((ny, nx)), meta=meta),
        extras=dict(extras) if extras is not None else {"rectangle_vertices": RECT},
    )


def rotated_rect(angle_deg, size_deg=0.01):
    a = math.radians(angle_deg)
    d1 = (size_deg * math.sin(a), size_deg * math.cos(a))
    d2 = (size_deg * math.cos(a), -size_deg * math.sin(a))
    return [
        (0.0, 0.0),
        (d1[0], d1[1]),
        (d1[0] + d2[0], d1[1] + d2[1]),
        (d2[0], d2[1]),
    ]


def write_v2_file(path, with_vertices=True):
    """Hand-write a minimal pre-v3 (v2) file, as 1.x versions produced."""
    import h5py

    ny, nx, nz = 4, 5, 6
    extras = {"source": "test"}
    if with_vertices:
        extras["rectangle_vertices"] = RECT
    with h5py.File(path, "w") as f:
        f.attrs["__format__"] = "voxcity_results.v2"
        f.attrs["crs"] = "EPSG:4326"
        f.attrs["meshsize"] = 2.0
        f.attrs["bounds"] = [0.0, 0.0, 0.01, 0.01]
        vc = f.create_group("voxcity")
        vc.create_dataset("voxel_grid", data=np.zeros((ny, nx, nz), dtype=np.int8))
        vc.create_dataset("building_height", data=np.zeros((ny, nx)))
        vc.create_dataset("building_id", data=np.zeros((ny, nx)))
        vc.create_dataset("dem", data=np.zeros((ny, nx)))
        vc.create_dataset("land_cover", data=np.ones((ny, nx), dtype=int))
        vc.attrs["extras_json"] = json.dumps(extras)
    return str(path)


@pytest.fixture
def propagate_voxcity_logs():
    """Temporarily enable propagation on the voxcity logger so caplog can capture records."""
    voxcity_logger = logging.getLogger("voxcity")
    old_propagate = voxcity_logger.propagate
    voxcity_logger.propagate = True
    yield
    voxcity_logger.propagate = old_propagate


def pytest_collection_modifyitems(config, items):
    """Reorder tests to run integration tests last.
    
    This helps avoid Taichi/GPU state interference between tests.
    Integration tests that use GPU rendering should run after all other tests.
    """
    integration_tests = []
    other_tests = []
    
    for item in items:
        if 'integration' in item.keywords or 'test_integration' in item.nodeid:
            integration_tests.append(item)
        else:
            other_tests.append(item)
    
    # Run other tests first, then integration tests
    items[:] = other_tests + integration_tests

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


@pytest.fixture(autouse=True)
def cleanup_matplotlib():
    """Automatically close all matplotlib figures after each test.
    
    This prevents memory warnings about too many open figures and
    ensures tests don't interfere with each other.
    """
    yield
    import matplotlib.pyplot as plt
    plt.close('all')