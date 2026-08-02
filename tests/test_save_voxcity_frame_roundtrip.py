"""``save_voxcity`` → ``check_axes`` → ``GridProjector.from_h5``: content round trip.

``save_voxcity`` stamps ``axes = "north,east,up"`` (row 0 = south) on the file
it writes. ``check_axes`` verifies only that the attribute is present and
well-formed — earlier in this project it passed happily on a file whose voxel
grid was stored the other way up, because an attribute check cannot see the
data. The claim that actually matters, and that the app's session save/load
depends on (VoxCityApp's ``backend/test_frame_consumers.py`` pins it through
the FastAPI session path), is pinned here library-side:

    map a known feature's lon/lat through ``GridProjector.from_h5`` and the
    saved arrays must hold that feature at exactly that cell — and not at the
    cell's north<->south mirror.

Fixture discipline: the building block sits in the **southern third** of a
non-square grid, east of centre, and the DEM is a strict south-to-north ramp.
Every lookup is keyed by a lon/lat computed from the rectangle's geometry —
an absolute reference the stored array cannot drag along with itself when it
flips.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

h5py = pytest.importorskip("h5py")

from voxcity.geoprocessor.raster import compute_cell_center_coords
from voxcity.io import save_voxcity
from voxcity.models import (
    BuildingGrid,
    CanopyGrid,
    DemGrid,
    GridMetadata,
    LandCoverGrid,
    VoxCity,
    VoxelGrid,
)
from voxcity.utils import GridProjector
from voxcity.utils.orientation import check_axes

GROUND_CODE = -1
BUILDING_CODE = -3

# ── Geography: 120 m x 90 m at 5 m cells → a 24 x 18 (non-square) grid ─────
M_PER_DEG_LAT = 111320.0
MESHSIZE = 5.0
NX, NY, NZ = 24, 18, 10
LON0, LAT0 = 139.7000, 35.6800
DLAT = (NX * MESHSIZE) / M_PER_DEG_LAT
DLON = (NY * MESHSIZE) / (M_PER_DEG_LAT * math.cos(math.radians(LAT0)))
RECT = [
    (LON0, LAT0),                 # SW — row 0 lives here
    (LON0, LAT0 + DLAT),          # NW
    (LON0 + DLON, LAT0 + DLAT),   # NE
    (LON0 + DLON, LAT0),          # SE
]

# The marker building: rows 3..6 of 24 (southern third), cols 11..14 (east of
# centre). Mirrors: rows 18..21, cols 4..7 — both disjoint from the block.
ROW_LO, ROW_HI = 3, 6
COL_LO, COL_HI = 11, 14
BUILD_H = 15.0  # 3 voxels

# The DEM ramp: strictly increasing northward, so any row maps to one value.
DEM_BASE, DEM_SLOPE = 20.0, 0.5


def _make_city() -> VoxCity:
    meta = GridMetadata(
        crs="EPSG:4326",
        bounds=(LON0, LAT0, LON0 + DLON, LAT0 + DLAT),
        meshsize=MESHSIZE,
    )
    classes = np.zeros((NX, NY, NZ), dtype=np.int8)
    classes[:, :, 0] = GROUND_CODE
    classes[ROW_LO:ROW_HI, COL_LO:COL_HI, 1:4] = BUILDING_CODE

    heights = np.zeros((NX, NY))
    heights[ROW_LO:ROW_HI, COL_LO:COL_HI] = BUILD_H
    min_heights = np.empty((NX, NY), dtype=object)
    for idx in np.ndindex((NX, NY)):
        min_heights[idx] = []
    ids = np.zeros((NX, NY), dtype=np.int32)
    ids[ROW_LO:ROW_HI, COL_LO:COL_HI] = 1

    dem = DEM_BASE + DEM_SLOPE * np.arange(NX, dtype=float)[:, None] + np.zeros((NX, NY))

    return VoxCity(
        voxels=VoxelGrid(classes=classes, meta=meta),
        buildings=BuildingGrid(heights=heights, min_heights=min_heights, ids=ids, meta=meta),
        land_cover=LandCoverGrid(classes=np.zeros((NX, NY), dtype=np.int32), meta=meta),
        dem=DemGrid(elevation=dem, meta=meta),
        tree_canopy=CanopyGrid(top=np.zeros((NX, NY)), meta=meta),
        extras={"rectangle_vertices": RECT},
    )


@pytest.fixture(scope="module")
def saved(tmp_path_factory):
    path = str(tmp_path_factory.mktemp("frame") / "city.h5")
    save_voxcity(path, _make_city())
    return path


def _cell_lonlat(i: int, j: int) -> tuple[float, float]:
    """Cell-centre lon/lat from the rectangle alone — independent of the file."""
    cc = compute_cell_center_coords(RECT, MESHSIZE)
    assert cc["grid_size"] == (NX, NY), "fixture arithmetic drifted"
    return float(cc["lons"][i, j]), float(cc["lats"][i, j])


# ═══════════════════════ guards on the fixture itself ═══════════════════════

def test_the_marker_is_off_centre_and_clear_of_both_mirrors():
    assert ROW_HI <= NX - ROW_HI, "block overlaps its own north-south mirror"
    assert NY - COL_HI < COL_LO, "block must sit off-centre east" \
        " — and clear of its east-west mirror"
    assert not set(range(ROW_LO, ROW_HI)) & {NX - 1 - r for r in range(ROW_LO, ROW_HI)}
    assert not set(range(COL_LO, COL_HI)) & {NY - 1 - c for c in range(COL_LO, COL_HI)}


def test_the_dem_is_asymmetric():
    dem = _make_city().dem.elevation
    assert np.all(np.diff(dem[:, 0]) > 0)
    assert not np.array_equal(dem, np.flipud(dem))


# ═══════════════════════════ the round trip itself ══════════════════════════

def test_check_axes_passes_on_the_saved_file(saved):
    check_axes(saved)  # necessary — and, alone, nowhere near sufficient


def test_the_marker_lonlat_maps_to_a_cell_that_holds_it(saved):
    """The teeth: declared frame and stored content must be the same frame."""
    lon, lat = _cell_lonlat((ROW_LO + ROW_HI) // 2, (COL_LO + COL_HI) // 2)
    i, j = GridProjector.from_h5(saved).lon_lat_to_cell(lon, lat)
    with h5py.File(saved, "r") as f:
        voxels = np.asarray(f["voxcity/voxel_grid"][:])

    assert BUILDING_CODE in voxels[i, j, :].tolist(), (
        f"the building's lon/lat maps to saved cell ({i}, {j}), which holds "
        "no building voxel — the file declares axes='north,east,up' (row 0 = "
        "south) but its voxel grid is stored in some other frame")
    assert BUILDING_CODE not in voxels[NX - 1 - i, j, :].tolist(), (
        "the row-mirrored cell also holds a building — the fixture cannot "
        "detect a north<->south flip")
    assert BUILDING_CODE not in voxels[i, NY - 1 - j, :].tolist(), (
        "the column-mirrored cell also holds a building — the fixture cannot "
        "detect an east<->west flip")


@pytest.mark.parametrize("row", [2, NX - 3], ids=["near-south-edge", "near-north-edge"])
def test_the_saved_dem_reads_the_value_its_latitude_holds(saved, row):
    """Probed near both edges so a constant offset or a centre coincidence
    cannot pass; the expected value names the row, so the mirrored row's value
    (off by (NX-1-2*row)*DEM_SLOPE = 9.5 m at row 2) cannot either."""
    lon, lat = _cell_lonlat(row, NY // 2)
    i, j = GridProjector.from_h5(saved).lon_lat_to_cell(lon, lat)
    with h5py.File(saved, "r") as f:
        dem = np.asarray(f["voxcity/dem"][:], dtype=float)

    assert dem[i, j] == pytest.approx(DEM_BASE + DEM_SLOPE * row), (
        f"the saved DEM reads {dem[i, j]} m at lat {lat:.6f} (row {row} from "
        f"the south edge); the model holds {DEM_BASE + DEM_SLOPE * row} m "
        "there — declared axes and stored data disagree by a mirror")


def test_load_round_trip_preserves_the_marker_cell(saved):
    """Same claim through the public loader instead of raw h5py."""
    from voxcity.io import load_voxcity

    city = load_voxcity(saved)
    lon, lat = _cell_lonlat((ROW_LO + ROW_HI) // 2, (COL_LO + COL_HI) // 2)
    i, j = GridProjector.from_city(city).lon_lat_to_cell(lon, lat)
    assert BUILDING_CODE in np.asarray(city.voxels.classes)[i, j, :].tolist()
    assert city.buildings.heights[i, j] == pytest.approx(BUILD_H)
