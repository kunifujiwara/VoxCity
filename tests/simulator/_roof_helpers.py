"""Shared fixtures for the 'include building rooftops' feature tests."""
import numpy as np

from voxcity.models import (
    VoxCity, VoxelGrid, BuildingGrid, LandCoverGrid, DemGrid, CanopyGrid, GridMetadata,
)

GROUND_LC = 1     # positive land-cover code -> exposed walkable ground
BUILDING = -3
AIR = 0


def make_voxcity_with_building(nx=6, ny=6, nz=12, meshsize=1.0, bh=4):
    """Flat ground (positive LC top) with one building column at (2, 2).

    Column layout for a *street* cell: k0 = GROUND_LC, k1.. = AIR.
    Column layout for the *building* cell (2,2): k0 = GROUND_LC,
    k1..bh = BUILDING, above = AIR. Roof top is at k = bh.
    """
    lon0, lat0 = 0.0, 0.0
    dlat = (meshsize * nx) / 111320.0
    dlon = (meshsize * ny) / 111320.0
    rect = [(lon0, lat0), (lon0, lat0 + dlat), (lon0 + dlon, lat0 + dlat), (lon0 + dlon, lat0)]
    meta = GridMetadata(crs="EPSG:4326", bounds=(lon0, lat0, lon0 + dlon, lat0 + dlat), meshsize=meshsize)

    classes = np.zeros((nx, ny, nz), dtype=np.int8)
    classes[:, :, 0] = GROUND_LC
    classes[2, 2, 1:bh + 1] = BUILDING  # building column, roof top at k=bh

    heights = np.zeros((nx, ny), dtype=float)
    heights[2, 2] = float(bh)
    ids = np.zeros((nx, ny), dtype=np.int32)
    ids[2, 2] = 101
    min_heights = np.empty((nx, ny), dtype=object)
    for i in range(nx):
        for j in range(ny):
            min_heights[i, j] = []
    min_heights[2, 2] = [[0.0, float(bh)]]
    dem = np.zeros((nx, ny), dtype=float)
    lc = np.full((nx, ny), GROUND_LC, dtype=np.int32)
    canopy = np.zeros((nx, ny), dtype=float)

    return VoxCity(
        voxels=VoxelGrid(classes=classes, meta=meta),
        buildings=BuildingGrid(heights=heights, min_heights=min_heights, ids=ids, meta=meta),
        land_cover=LandCoverGrid(classes=lc, meta=meta),
        dem=DemGrid(elevation=dem, meta=meta),
        tree_canopy=CanopyGrid(top=canopy, bottom=None, meta=meta),
        extras={"rectangle_vertices": rect},
    )


def make_voxcity_with_pilotis(nx=6, ny=6, nz=14, meshsize=1.0, gap=3, deck=4):
    """Flat ground with one pilotis column at (2, 2): open ground floor, an air
    gap, then an elevated building mass (the "roof"/top deck).

    Column (2,2): k0 = GROUND_LC (open floor, walkable), k1..gap = AIR,
    k(gap+1)..(gap+deck) = BUILDING (-3); roof top voxel = gap+deck.
    Street cells: k0 = GROUND_LC, rest AIR.
    """
    classes = np.zeros((nx, ny, nz), dtype=np.int8)
    classes[:, :, 0] = GROUND_LC
    roof_bottom = gap + 1
    roof_top = gap + deck                      # top building voxel index
    classes[2, 2, roof_bottom:roof_top + 1] = BUILDING

    lon0, lat0 = 0.0, 0.0
    dlat = (meshsize * nx) / 111320.0
    dlon = (meshsize * ny) / 111320.0
    rect = [(lon0, lat0), (lon0, lat0 + dlat), (lon0 + dlon, lat0 + dlat), (lon0 + dlon, lat0)]
    meta = GridMetadata(crs="EPSG:4326", bounds=(lon0, lat0, lon0 + dlon, lat0 + dlat), meshsize=meshsize)

    heights = np.zeros((nx, ny), dtype=float)
    heights[2, 2] = float(roof_top)
    ids = np.zeros((nx, ny), dtype=np.int32)
    ids[2, 2] = 101
    min_heights = np.empty((nx, ny), dtype=object)
    for i in range(nx):
        for j in range(ny):
            min_heights[i, j] = []
    min_heights[2, 2] = [[float(roof_bottom), float(roof_top + 1)]]
    dem = np.zeros((nx, ny), dtype=float)
    lc = np.full((nx, ny), GROUND_LC, dtype=np.int32)
    canopy = np.zeros((nx, ny), dtype=float)

    return VoxCity(
        voxels=VoxelGrid(classes=classes, meta=meta),
        buildings=BuildingGrid(heights=heights, min_heights=min_heights, ids=ids, meta=meta),
        land_cover=LandCoverGrid(classes=lc, meta=meta),
        dem=DemGrid(elevation=dem, meta=meta),
        tree_canopy=CanopyGrid(top=canopy, bottom=None, meta=meta),
        extras={"rectangle_vertices": rect},
    )
