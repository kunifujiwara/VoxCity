"""Orientation of the grid-derived overlays returned by ``/api/model/geo``.

``build_canopy_geojson`` and ``build_lc_geojson`` both walk row 0 of their grid
onto the *south* edge of the rectangle (``compute_grid_geometry`` anchors at
``rectangle_vertices[0]`` = SW and makes ``side_1`` = SW→NW, so the row index
grows northward). That is the display frame.

A voxcity-produced model keeps every 2-D grid in that one frame. A
VoxCityGML-produced PLATEAU LOD2 model does not: ``land_cover.classes`` stays
south-up while ``tree_canopy.top`` is north-up, because voxcitygml flips the
canopy it derives from that land cover to match the DEM and the voxel grid.
Drawn unflipped, the LOD2 canopy overlay lands mirrored north<->south.

The canopy grids below are deliberately **asymmetric** (a band near the south
edge, not centred): a symmetric canopy is flip-invariant and would pass whether
or not the endpoint flips.
"""
from __future__ import annotations

import json

import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend.main import app
from backend.state import app_state
from tests.importer.conftest import make_flat_voxcity
from voxcity.geoprocessor.draw._common import build_canopy_geojson, compute_grid_geometry
from voxcity.utils.lc import get_land_cover_classes

NX = NY = 20
MESHSIZE = 1.0
LC_SOURCE = "OpenEarthMapJapan"
_CLASS_NAMES = list(dict.fromkeys(get_land_cover_classes(LC_SOURCE).values()))
TREE_CLS = _CLASS_NAMES.index("Tree")

# Asymmetric: 4 of 20 rows, hugging the south edge. Its mirror is rows 14..18.
BAND_LO, BAND_HI = 2, 6
MIRROR_LO, MIRROR_HI = NX - BAND_HI, NX - BAND_LO
CANOPY_H = 9.0


@pytest.fixture
def client():
    return TestClient(app)


def _land_frame_canopy() -> np.ndarray:
    """Canopy in the south-up display/land-cover frame: a band near the south."""
    canopy = np.zeros((NX, NY), dtype=float)
    canopy[BAND_LO:BAND_HI, :] = CANOPY_H
    return canopy


def _install_model(*, lod2: bool) -> np.ndarray:
    """Put a model on ``app_state`` and return its stored canopy grid."""
    vc = make_flat_voxcity(nx=NX, ny=NY, nz=6, meshsize=MESHSIZE)

    lc = np.zeros((NX, NY), dtype=np.int32)
    lc[BAND_LO:BAND_HI, :] = TREE_CLS
    vc.land_cover.classes = lc

    land_frame = _land_frame_canopy()
    # A VoxCityGML LOD2 model stores the canopy already flipped into the voxel
    # frame; a voxcity model stores it in the same frame as the land cover.
    stored = np.flipud(land_frame).copy() if lod2 else land_frame
    vc.tree_canopy.top = stored
    if lod2:
        vc.extras["building_lod"] = 2

    app_state.voxcity = vc
    app_state.rectangle_vertices = vc.extras["rectangle_vertices"]
    app_state.land_cover_source = LC_SOURCE
    app_state.auxiliary_lines = []
    return stored


@pytest.fixture(autouse=True)
def _clean_state():
    yield
    app_state.voxcity = None
    app_state.rectangle_vertices = None
    app_state.auxiliary_lines = []


def _grid_geom():
    return compute_grid_geometry(app_state.rectangle_vertices, MESHSIZE)


def _row_lat(gg, row: int) -> float:
    """Latitude of the southern edge of grid row *row* in the display frame."""
    return float(gg["origin"][1] + row * gg["adj_mesh"][0] * gg["u_vec"][1])


def _lat_span(fc) -> tuple[float, float]:
    lats = [
        coord[1]
        for feature in fc["features"]
        for ring in feature["geometry"]["coordinates"]
        for coord in ring
    ]
    assert lats, "feature collection is empty — nothing to orient"
    return min(lats), max(lats)


def _as_json(fc: dict) -> dict:
    """Round-trip through JSON so shapely's coordinate tuples become lists."""
    return json.loads(json.dumps(fc))


def _tree_only(lc_fc) -> dict:
    return {
        "type": "FeatureCollection",
        "features": [
            f for f in lc_fc["features"] if (f["properties"] or {}).get("cls") == TREE_CLS
        ],
    }


def test_the_test_canopy_is_asymmetric():
    """Guard the guard: a flip-invariant canopy would make every check vacuous."""
    canopy = _land_frame_canopy()
    assert not np.array_equal(canopy, np.flipud(canopy))


@pytest.mark.parametrize("lod2", [True, False], ids=["lod2", "lod1"])
def test_canopy_overlay_lands_on_the_land_cover_side(client, lod2):
    """The canopy overlay must cover the same cells as the Tree land cover."""
    _install_model(lod2=lod2)
    gg = _grid_geom()

    body = client.get("/api/model/geo").json()
    canopy_span = _lat_span(body["canopy_geojson"])
    tree_span = _lat_span(_tree_only(body["land_cover_geojson"]))

    assert canopy_span == pytest.approx(tree_span, abs=1e-12)
    assert canopy_span == pytest.approx(
        (_row_lat(gg, BAND_LO), _row_lat(gg, BAND_HI)), abs=1e-12)
    # ... and emphatically not on the reflected side.
    mirrored = (_row_lat(gg, MIRROR_LO), _row_lat(gg, MIRROR_HI))
    assert canopy_span[1] < mirrored[0], (
        f"canopy overlay sits at {canopy_span}, the reflection of the tree "
        f"land cover ({mirrored}) — it is drawn mirrored north<->south")


def test_lod1_canopy_is_drawn_exactly_as_stored(client):
    """No flip for a voxcity model: same bytes as building it from the grid."""
    stored = _install_model(lod2=False)
    gg = _grid_geom()

    body = client.get("/api/model/geo").json()

    assert body["canopy_geojson"] == _as_json(build_canopy_geojson(stored, gg))


def test_lod2_canopy_is_drawn_flipped_from_storage(client):
    """The LOD2 flip actually happens (and is not what LOD1 gets)."""
    stored = _install_model(lod2=True)
    gg = _grid_geom()

    body = client.get("/api/model/geo").json()

    assert body["canopy_geojson"] == _as_json(build_canopy_geojson(np.flipud(stored), gg))
    assert body["canopy_geojson"] != _as_json(build_canopy_geojson(stored, gg))


@pytest.mark.parametrize("lod2", [True, False], ids=["lod2", "lod1"])
def test_geo_does_not_mutate_the_stored_canopy(client, lod2):
    """The voxel-frame invariant the nDSM canopy depends on must survive."""
    stored = _install_model(lod2=lod2)
    before = stored.copy()

    client.get("/api/model/geo")

    assert np.array_equal(np.asarray(app_state.voxcity.tree_canopy.top), before)
    assert app_state.voxcity.tree_canopy.top is stored
