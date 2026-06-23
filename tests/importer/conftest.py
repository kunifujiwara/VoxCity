"""Shared fixtures for importer tests."""
import numpy as np
import pytest
import trimesh

from voxcity.models import (
    VoxCity, VoxelGrid, BuildingGrid, LandCoverGrid, DemGrid, CanopyGrid, GridMetadata,
)

GROUND_CODE = -1
BUILDING_CODE = -3


def make_flat_voxcity(nx=20, ny=20, nz=10, meshsize=1.0):
    """Build a minimal flat-DEM VoxCity object for import tests.

    Axis-aligned 1-degree-ish rectangle is unnecessary; we use a small real
    rectangle near the equator so distances are well-behaved. Ground voxel at
    k=0 (land cover), everything above is air (0).
    """
    # Small axis-aligned rectangle (lon, lat): SW, NW, NE, SE
    lon0, lat0 = 0.0, 0.0
    # ~meshsize*nx meters wide. 1 deg lat ~= 111320 m.
    dlat = (meshsize * nx) / 111320.0
    dlon = (meshsize * ny) / 111320.0
    rectangle_vertices = [
        (lon0, lat0),
        (lon0, lat0 + dlat),
        (lon0 + dlon, lat0 + dlat),
        (lon0 + dlon, lat0),
    ]
    meta = GridMetadata(crs="EPSG:4326", bounds=(lon0, lat0, lon0 + dlon, lat0 + dlat), meshsize=meshsize)

    classes = np.zeros((nx, ny, nz), dtype=np.int8)
    classes[:, :, 0] = GROUND_CODE  # ground/landcover layer

    heights = np.zeros((nx, ny), dtype=float)
    ids = np.zeros((nx, ny), dtype=np.int32)
    min_heights = np.empty((nx, ny), dtype=object)
    for i in range(nx):
        for j in range(ny):
            min_heights[i, j] = []
    dem = np.zeros((nx, ny), dtype=float)
    lc = np.zeros((nx, ny), dtype=np.int32)
    canopy = np.zeros((nx, ny), dtype=float)

    return VoxCity(
        voxels=VoxelGrid(classes=classes, meta=meta),
        buildings=BuildingGrid(heights=heights, min_heights=min_heights, ids=ids, meta=meta),
        land_cover=LandCoverGrid(classes=lc, meta=meta),
        dem=DemGrid(elevation=dem, meta=meta),
        tree_canopy=CanopyGrid(top=canopy, bottom=None, meta=meta),
        extras={"rectangle_vertices": rectangle_vertices},
    )


@pytest.fixture
def flat_voxcity():
    return make_flat_voxcity()


@pytest.fixture
def box_obj_factory(tmp_path):
    """Return a factory writing a single-box OBJ and returning its path.

    box(origin=(x,y,z), size=(sx,sy,sz), name=...) -> Path to .obj
    """
    def _factory(origin=(0.0, 0.0, 0.0), size=(2.0, 2.0, 3.0), name="building1", filename="model.obj"):
        mesh = trimesh.creation.box(extents=size)
        # trimesh box is centered at origin; move so min corner = origin
        mesh.apply_translation(np.array(size) / 2.0 + np.array(origin))
        path = tmp_path / filename
        # Export to OBJ (extension drives the format)
        mesh.export(str(path))
        return path

    return _factory
