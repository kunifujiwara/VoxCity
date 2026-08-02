"""Orientation and property contracts of ``voxcity.geoprocessor.geojson``.

``build_canopy_geojson`` and ``build_lc_geojson`` draw row 0 of their grid onto
the ``vertex[0]`` edge of the rectangle — for the app's ``[SW, NW, NE, SE]``
convention, the south edge. The app's ``/api/model/geo`` overlays consume the
builders exactly as stored (VoxCityApp's ``backend/test_model_geo_overlays.py``
pins the endpoint side); this module pins the builders themselves, in the
library's own suite, so a library-only checkout keeps the guard now that the
app lives in its own repository.

Contracts pinned here:

* **Orientation** — a canopy/land-cover band in the southern quarter of the
  grid must come out at southern latitudes, not at its mirror image. The
  fixture is asymmetric (band rows 2..6 of 20, mirror rows 14..18, disjoint)
  and the grid is non-square (20 x 15), so a flip or a transpose cannot pass.
* **``cls`` property** — ``build_lc_geojson`` stamps each feature with the
  *index into the deduplicated source class-name list* (the order
  ``get_land_cover_classes(source)`` yields). The app's land-cover editor and
  overlay legend depend on exactly that indexing.
* **``style`` colours** — features carry the source palette colour for their
  class, via ``get_lc_source_colors``.

Known, deliberately unpinned: ``build_lc_geojson(..., land_cover_source=None)``
raises ``UnboundLocalError`` inside ``voxcity/utils/lc.py`` (pre-existing bug,
ticket candidate — behaviour change is out of scope here). Every call below
passes a real source.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from voxcity.geoprocessor.geojson import (
    build_canopy_geojson,
    build_lc_geojson,
    get_lc_source_colors,
)
from voxcity.geoprocessor.raster import compute_grid_geometry
from voxcity.utils import get_land_cover_classes

# ── Geography: 200 m x 150 m at 10 m cells → a 20 x 15 (non-square) grid ──
M_PER_DEG_LAT = 111320.0
MESHSIZE = 10.0
NX, NY = 20, 15
LON0, LAT0 = 139.7000, 35.6800
DLAT = (NX * MESHSIZE) / M_PER_DEG_LAT
DLON = (NY * MESHSIZE) / (M_PER_DEG_LAT * math.cos(math.radians(LAT0)))
RECT = [
    (LON0, LAT0),                 # SW — vertex[0], the row-0 edge
    (LON0, LAT0 + DLAT),          # NW
    (LON0 + DLON, LAT0 + DLAT),   # NE
    (LON0 + DLON, LAT0),          # SE
]
MID_LAT = LAT0 + DLAT / 2

# The asymmetric band: rows 2..6 of 20 — the southern quarter, its mirror
# (rows 14..18) disjoint.
BAND_LO, BAND_HI = 2, 6
CANOPY_H = 8.0

LC_SOURCE = "OpenEarthMapJapan"
_CLASS_NAMES = list(dict.fromkeys(get_land_cover_classes(LC_SOURCE).values()))
TREE_CLS = _CLASS_NAMES.index("Tree")
# Any non-Tree class for the background, so the band is the only Tree area.
OTHER_CLS = _CLASS_NAMES.index("Bareland")


@pytest.fixture(scope="module")
def gg():
    g = compute_grid_geometry(RECT, MESHSIZE)
    assert g is not None
    assert g["grid_size"] == (NX, NY), "fixture arithmetic drifted"
    return g


def _band_canopy() -> np.ndarray:
    canopy = np.zeros((NX, NY), dtype=float)
    canopy[BAND_LO:BAND_HI, :] = CANOPY_H
    return canopy


def _band_land_cover() -> np.ndarray:
    lc = np.full((NX, NY), OTHER_CLS, dtype=np.int32)
    lc[BAND_LO:BAND_HI, :] = TREE_CLS
    return lc


def _coords(fc):
    return [
        coord
        for feature in fc["features"]
        for ring in feature["geometry"]["coordinates"]
        for coord in ring
    ]


def _lat_span(fc) -> tuple[float, float]:
    lats = [c[1] for c in _coords(fc)]
    assert lats, "feature collection is empty — nothing to orient"
    return min(lats), max(lats)


def _lon_span(fc) -> tuple[float, float]:
    lons = [c[0] for c in _coords(fc)]
    assert lons, "feature collection is empty — nothing to orient"
    return min(lons), max(lons)


def _row_edge_lat(gg, row: int) -> float:
    """Latitude of the ``row``-th cell boundary, walking north from vertex[0]."""
    return float(gg["origin"][1] + row * gg["adj_mesh"][0] * gg["u_vec"][1])


# ═══════════════════════ guards on the fixture itself ═══════════════════════

def test_the_fixture_grid_is_not_square():
    assert NX != NY, "a square grid cannot catch a transposed builder"


def test_the_band_is_asymmetric_and_clear_of_its_mirror():
    canopy = _band_canopy()
    assert not np.array_equal(canopy, np.flipud(canopy))
    assert BAND_HI <= NX - BAND_HI, "the band overlaps its own mirror image"
    # Whole band strictly south of the rectangle's mid-latitude, mirror
    # strictly north — so "south of MID_LAT" is a decisive verdict below.
    assert BAND_HI < NX / 2


# ═══════════════════════════ build_canopy_geojson ═══════════════════════════

class TestCanopyOrientation:
    def test_the_band_lands_between_its_own_row_edges(self, gg):
        span = _lat_span(build_canopy_geojson(_band_canopy(), gg))
        assert span == pytest.approx(
            (_row_edge_lat(gg, BAND_LO), _row_edge_lat(gg, BAND_HI)), abs=1e-12)

    def test_the_band_lands_south_of_the_rectangle_midline(self, gg):
        """The absolute half of the claim: real latitudes, not grid algebra."""
        lo, hi = _lat_span(build_canopy_geojson(_band_canopy(), gg))
        assert LAT0 < lo < hi < MID_LAT, (
            f"a canopy band in rows {BAND_LO}..{BAND_HI} of {NX} came out at "
            f"latitudes ({lo:.6f}, {hi:.6f}), north of the rectangle midline "
            f"{MID_LAT:.6f} — the overlay is mirrored north<->south")

    def test_a_flipped_canopy_moves_to_the_northern_half(self, gg):
        """The test above can actually fail: flip the input, watch it move."""
        lo, hi = _lat_span(build_canopy_geojson(np.flipud(_band_canopy()), gg))
        assert MID_LAT < lo < hi

    def test_the_band_spans_the_full_east_west_width(self, gg):
        """Transpose catch: rows are latitude, columns longitude — a builder
        that swapped axes would draw a 40 m-wide sliver instead of the full
        150 m width."""
        lo, hi = _lon_span(build_canopy_geojson(_band_canopy(), gg))
        assert lo == pytest.approx(LON0, abs=1e-12)
        assert (hi - lo) * M_PER_DEG_LAT * math.cos(math.radians(LAT0)) == (
            pytest.approx(NY * MESHSIZE, rel=0.01))

    def test_empty_canopy_gives_an_empty_collection(self, gg):
        assert build_canopy_geojson(np.zeros((NX, NY)), gg) == {
            "type": "FeatureCollection", "features": []}


# ═════════════════════════════ build_lc_geojson ═════════════════════════════

def _tree_features(fc):
    return [f for f in fc["features"] if f["properties"]["cls"] == TREE_CLS]


class TestLandCoverGeojson:
    @pytest.fixture(scope="class")
    def fc(self, gg):
        return build_lc_geojson(_band_land_cover(), gg, LC_SOURCE)

    def test_cls_is_the_index_into_the_source_class_list(self, fc):
        """The contract ``/api/model/geo`` consumers rely on: ``cls`` indexes
        the deduplicated ``get_land_cover_classes(source)`` name list."""
        seen = {f["properties"]["cls"] for f in fc["features"]}
        assert seen == {OTHER_CLS, TREE_CLS}
        for cls in seen:
            assert isinstance(cls, int)
            assert 0 <= cls < len(_CLASS_NAMES)
        assert _CLASS_NAMES[TREE_CLS] == "Tree"

    def test_tree_features_cover_exactly_the_band(self, fc, gg):
        tree_fc = {"type": "FeatureCollection", "features": _tree_features(fc)}
        assert tree_fc["features"], "no Tree features at all"
        span = _lat_span(tree_fc)
        assert span == pytest.approx(
            (_row_edge_lat(gg, BAND_LO), _row_edge_lat(gg, BAND_HI)), abs=1e-12)
        assert span[1] < MID_LAT, (
            "the Tree band is drawn in the northern half — the land-cover "
            "overlay is mirrored north<->south")

    def test_a_flipped_grid_moves_the_tree_band_north(self, gg):
        flipped = build_lc_geojson(np.flipud(_band_land_cover()), gg, LC_SOURCE)
        tree_fc = {"type": "FeatureCollection", "features": _tree_features(flipped)}
        lo, hi = _lat_span(tree_fc)
        assert MID_LAT < lo < hi

    def test_features_carry_the_source_palette_colour(self, fc):
        expected = get_lc_source_colors(LC_SOURCE)
        for f in fc["features"]:
            name = _CLASS_NAMES[f["properties"]["cls"]]
            style = f["properties"]["style"]
            assert style["color"] == style["fillColor"] == expected[name]

    def test_out_of_range_codes_are_skipped_not_drawn(self, gg):
        """Codes outside the source list must be dropped silently — the app
        feeds grids straight from models whose no-data cells hold -1."""
        lc = _band_land_cover()
        lc[10:12, :] = -1
        lc[12:14, :] = len(_CLASS_NAMES) + 5
        fc = build_lc_geojson(lc, gg, LC_SOURCE)
        assert {f["properties"]["cls"] for f in fc["features"]} == {OTHER_CLS, TREE_CLS}
