"""Tests for voxcity.exporter.palm (PALM static driver exporter)."""

import logging

import numpy as np
import pytest

import voxcity.exporter.palm as palm_module
from voxcity.exporter.palm import (
    BYTE_RANGES,
    DEFAULT_ASSIGNMENT,
    DYNAMIC_WORLD_CLASS_TO_PALM,
    ESA_CLASS_TO_PALM,
    ESRI_CLASS_TO_PALM,
    FILL_BYTE,
    FILL_FLOAT,
    FILL_INT,
    OEMJ_CLASS_TO_PALM,
    OSM_CLASS_TO_PALM,
    URBANWATCH_CLASS_TO_PALM,
    _build_buildings,
    _build_buildings_3d,
    _build_georeference,
    _build_index_to_palm_map,
    _build_zt,
    _get_source_name_mapping,
    _has_elevated_segments,
)
from voxcity.utils.lc import get_land_cover_classes

# Axis-aligned rectangle near Tokyo, canonical [SW, NW, NE, SE] (lon, lat)
RECT = [(139.7000, 35.6800), (139.7000, 35.6809),
        (139.7011, 35.6809), (139.7011, 35.6800)]

ALL_TABLES = [
    OSM_CLASS_TO_PALM,
    URBANWATCH_CLASS_TO_PALM,
    OEMJ_CLASS_TO_PALM,
    ESA_CLASS_TO_PALM,
    ESRI_CLASS_TO_PALM,
    DYNAMIC_WORLD_CLASS_TO_PALM,
]

# Source name -> table, for tests that need to cross-check a table against
# the live get_land_cover_classes() output for that source.
SOURCE_TABLE_PAIRS = [
    ("OpenStreetMap", OSM_CLASS_TO_PALM),
    ("Urbanwatch", URBANWATCH_CLASS_TO_PALM),
    ("OpenEarthMapJapan", OEMJ_CLASS_TO_PALM),
    ("ESA WorldCover", ESA_CLASS_TO_PALM),
    ("ESRI 10m Annual Land Cover", ESRI_CLASS_TO_PALM),
    ("Dynamic World V1", DYNAMIC_WORLD_CLASS_TO_PALM),
]

# BYTE_RANGES is the authority the future validator will use, so the test's
# valid ranges are derived from it rather than restated by hand.
VALID_RANGES = {
    "vegetation": BYTE_RANGES["vegetation_type"],
    "pavement": BYTE_RANGES["pavement_type"],
    "water": BYTE_RANGES["water_type"],
}


class TestMappingTables:
    def test_fill_values(self):
        # PIDS-mandated fill values for the static driver's _FillValue attrs.
        assert FILL_FLOAT == -9999.0
        assert FILL_INT == -9999
        assert FILL_BYTE == -127

    @pytest.mark.parametrize("table", ALL_TABLES)
    def test_entries_are_category_code_pairs(self, table):
        for name, (category, code) in table.items():
            assert category in ("vegetation", "pavement", "water", "building")
            if category == "building":
                assert code is None
            else:
                lo, hi = VALID_RANGES[category]
                assert lo <= code <= hi, f"{name}: {category} code {code} out of range"

    @pytest.mark.parametrize("source_name, table", SOURCE_TABLE_PAIRS)
    def test_mapping_table_covers_all_source_classes(self, source_name, table):
        names = list(get_land_cover_classes(source_name).values())
        assert set(names) == set(table.keys())

    def test_spec_pinned_codes(self):
        assert OSM_CLASS_TO_PALM["Road"] == ("pavement", 1)
        assert OSM_CLASS_TO_PALM["Developed space"] == ("pavement", 2)
        assert OSM_CLASS_TO_PALM["Tree"] == ("vegetation", 7)
        assert OSM_CLASS_TO_PALM["Shrub"] == ("vegetation", 16)
        assert OSM_CLASS_TO_PALM["Wet land"] == ("vegetation", 14)
        assert OSM_CLASS_TO_PALM["Snow and ice"] == ("vegetation", 13)
        assert OSM_CLASS_TO_PALM["Water"] == ("water", 1)
        assert OSM_CLASS_TO_PALM["Building"] == ("building", None)
        assert URBANWATCH_CLASS_TO_PALM["Sea"] == ("water", 3)

    def test_byte_ranges_match_pids(self):
        # PIDS-mandated valid class ranges; the validator and the table tests both derive from these.
        assert BYTE_RANGES == {
            "vegetation_type": (1, 18), "pavement_type": (1, 15), "water_type": (1, 5),
            "soil_type": (1, 6), "building_type": (1, 6),
        }


class TestSourceResolution:
    def test_known_sources(self):
        assert _get_source_name_mapping('OpenStreetMap') is OSM_CLASS_TO_PALM
        assert _get_source_name_mapping('Standard') is OSM_CLASS_TO_PALM
        assert _get_source_name_mapping('Urbanwatch') is URBANWATCH_CLASS_TO_PALM
        assert _get_source_name_mapping('OpenEarthMapJapan') is OEMJ_CLASS_TO_PALM
        assert _get_source_name_mapping('ESA WorldCover') is ESA_CLASS_TO_PALM
        assert _get_source_name_mapping('ESRI 10m Annual Land Cover') is ESRI_CLASS_TO_PALM
        assert _get_source_name_mapping('Dynamic World V1') is DYNAMIC_WORLD_CLASS_TO_PALM

    def test_unknown_source_falls_back_to_osm(self, caplog, propagate_voxcity_logs):
        with caplog.at_level(logging.WARNING, logger="voxcity"):
            result = _get_source_name_mapping('SomeUnknownSource')
        assert result is OSM_CLASS_TO_PALM
        assert 'SomeUnknownSource' in caplog.text

    def test_index_map_osm(self):
        index_to_assignment, class_names = _build_index_to_palm_map('OpenStreetMap')
        # OSM raw order: 0 Bareland ... 5 Tree ... 8 Water, 11 Road, 12 Building
        assert class_names[0] == 'Bareland'
        assert index_to_assignment[0] == ('vegetation', 1)
        assert index_to_assignment[5] == ('vegetation', 7)
        assert index_to_assignment[8] == ('water', 1)
        assert index_to_assignment[11] == ('pavement', 1)
        assert index_to_assignment[12] == ('building', None)

    def test_index_map_unknown_source_uses_osm_names(self):
        index_to_assignment, class_names = _build_index_to_palm_map('Nope')
        assert len(class_names) == 14
        assert class_names[12] == 'Building'
        assert index_to_assignment[0] == ('vegetation', 1)

    def test_default_assignment_used_for_unmapped_class_name(self, monkeypatch):
        """A class name absent from the resolved table falls back to DEFAULT_ASSIGNMENT.

        All six shipped tables are exhaustive for their source's class list,
        so this path is unreachable in practice; exercise it directly by
        stubbing out a table missing an entry the real class list still has.
        """
        stub_table = {k: v for k, v in OSM_CLASS_TO_PALM.items() if k != 'Bareland'}
        monkeypatch.setattr(palm_module, '_get_source_name_mapping', lambda source: stub_table)

        index_to_assignment, class_names = _build_index_to_palm_map('OpenStreetMap')

        assert class_names[0] == 'Bareland'
        assert index_to_assignment[0] == DEFAULT_ASSIGNMENT


class TestGeoreference:
    def test_tokyo_rectangle(self):
        geo = _build_georeference(RECT)
        assert geo["origin_lon"] == pytest.approx(139.7000)
        assert geo["origin_lat"] == pytest.approx(35.6800)
        assert geo["epsg"] == 32654  # UTM zone 54N
        assert abs(geo["rotation_angle"]) < 0.5
        # Tokyo UTM54N easting ~ 380-390 km, northing ~ 3.94-3.96 Mm
        assert 300_000 < geo["origin_x"] < 500_000
        assert 3_900_000 < geo["origin_y"] < 4_000_000

    def test_southern_hemisphere_epsg(self):
        rect = [(151.2, -33.87), (151.2, -33.86), (151.21, -33.86), (151.21, -33.87)]
        geo = _build_georeference(rect)
        assert geo["epsg"] == 32756  # Sydney, UTM zone 56S

    def test_non_canonical_order_still_resolves_sw_corner(self):
        # Same rectangle as RECT, but the vertex list is rotated to start at
        # NE (same winding). normalize_rectangle_vertices must reorder it
        # back to SW-first; without that call the SW corner would not be
        # vertices[0] and this would report the NE corner instead.
        rotated_start = RECT[2:] + RECT[:2]
        geo = _build_georeference(rotated_start)
        assert geo["origin_lon"] == pytest.approx(139.7000)
        assert geo["origin_lat"] == pytest.approx(35.6800)

    def test_rotated_rectangle_angle(self):
        # Built directly in Web Mercator space (the space compute_rotation_angle
        # itself uses) by rotating a rectangle 30 deg clockwise from north
        # around the Tokyo SW corner, then projecting back to WGS84.
        # The rectangle is 1000 m east-west x 900 m north-south, anchored at
        # the Tokyo SW corner, constructed in EPSG:3857 (so regenerating it
        # means: build a 1000 m (east-west) x 900 m (north-south)
        # axis-aligned box in EPSG:3857 at that corner, rotate it 30 deg
        # clockwise, then reproject to WGS84).
        # Empirically confirmed: compute_rotation_angle(rect) == 30.0 exactly
        # for this fixture (verified by direct call before pinning here).
        rotated_rect = [
            (139.7, 35.67999999999999),
            (139.7040424187785, 35.685687167858035),
            (139.71182205734505, 35.68203889459048),
            (139.70777963856654, 35.67635146668452),
        ]
        geo = _build_georeference(rotated_rect)
        assert geo["rotation_angle"] == pytest.approx(30.0, abs=0.01)


class TestBuildZt:
    def test_shift_to_zero_min(self):
        dem = np.array([[3.0, 5.0], [4.0, 7.0]])
        zt, origin_z = _build_zt(dem)
        assert zt.dtype == np.float32
        assert origin_z == pytest.approx(3.0)
        assert zt.min() == pytest.approx(0.0)
        assert zt[1, 1] == pytest.approx(4.0)

    def test_nan_replaced_with_min_before_shift(self):
        dem = np.array([[np.nan, 5.0], [4.0, 7.0]])
        zt, origin_z = _build_zt(dem)
        assert origin_z == pytest.approx(4.0)
        assert zt[0, 0] == pytest.approx(0.0)
        assert np.isfinite(zt).all()

    def test_all_nan_becomes_flat_zero(self):
        dem = np.full((2, 2), np.nan)
        zt, origin_z = _build_zt(dem)
        assert origin_z == 0.0
        assert (zt == 0.0).all()

    def test_inf_replaced_with_finite_min_before_shift(self):
        dem = np.array([[-np.inf, 5.0], [4.0, np.inf]])
        zt, origin_z = _build_zt(dem)
        assert origin_z == pytest.approx(4.0)
        assert zt[0, 0] == pytest.approx(0.0)  # -inf takes the finite min
        assert zt[1, 1] == pytest.approx(0.0)  # +inf takes the finite min too
        assert np.isfinite(zt).all()

    def test_does_not_mutate_input(self):
        # dem_grid is already float64, so np.asarray(..., dtype=float64) is a
        # no-op view of the same object -- .copy() is what keeps this
        # function from rewriting the caller's array in place.
        dem = np.array([[3.0, 5.0], [4.0, 7.0]], dtype=np.float64)
        original = dem.copy()
        _build_zt(dem)
        assert np.array_equal(dem, original)


def _empty_min_heights(ny, nx):
    mh = np.empty((ny, nx), dtype=object)
    for i in range(ny):
        for j in range(nx):
            mh[i, j] = []
    return mh


class TestBuildBuildings:
    def test_basic_fields(self):
        heights = np.array([[0.0, 10.0], [6.0, np.nan]])
        ids = np.array([[0, 7], [0, 0]])
        b2d, bid, btype = _build_buildings(heights, ids, building_type=3)
        assert b2d.dtype == np.float32
        assert b2d[0, 0] == np.float32(FILL_FLOAT)
        assert b2d[1, 1] == np.float32(FILL_FLOAT)  # NaN height -> no building
        assert b2d[0, 1] == np.float32(10.0)
        assert bid.dtype == np.int32
        assert bid[0, 1] == 7
        # (1,0) has height but id 0 -> generated id > existing max
        assert bid[1, 0] > 7
        assert btype.dtype == np.int8
        assert btype[0, 1] == 3
        assert btype[0, 0] == FILL_BYTE

    def test_none_ids_all_generated(self):
        heights = np.array([[5.0, 0.0]])
        b2d, bid, btype = _build_buildings(heights, None, building_type=2)
        assert bid[0, 0] >= 1
        assert bid[0, 1] == FILL_INT

    def test_generated_ids_unique_and_no_collision_with_existing(self):
        # Three id-less buildings alongside two pre-existing ids (5 and 10).
        # The real contract is uniqueness across the whole grid, not merely
        # "greater than the max existing id at that one cell".
        heights = np.array([[3.0, 4.0, 5.0], [6.0, 0.0, 7.0]])
        ids = np.array([[0, 10, 0], [5, 0, 0]])
        b2d, bid, btype = _build_buildings(heights, ids, building_type=1)
        mask = heights > 0.0
        non_fill_ids = bid[mask].tolist()
        assert len(non_fill_ids) == len(set(non_fill_ids)), non_fill_ids
        assert 5 in non_fill_ids
        assert 10 in non_fill_ids
        generated = [i for i in non_fill_ids if i not in (5, 10)]
        assert len(generated) == 3
        assert all(i > 10 for i in generated)

    def test_does_not_mutate_inputs(self):
        # ids is float64 deliberately, matching heights: for non-float64
        # input (e.g. int64), np.asarray(ids, dtype=np.float64) always
        # copies, so the in-place np.nan_to_num step inside _build_buildings
        # could never be observed corrupting the caller's array. float64 is
        # the aliasing-prone dtype where np.asarray returns the same object.
        heights = np.array([[3.0, 0.0], [6.0, np.nan]], dtype=np.float64)
        ids = np.array([[0.0, 0.0], [5.0, np.nan]], dtype=np.float64)
        heights_orig = heights.copy()
        ids_orig = ids.copy()
        _build_buildings(heights, ids, building_type=1)
        assert np.array_equal(heights, heights_orig, equal_nan=True)
        assert np.array_equal(ids, ids_orig, equal_nan=True)


class TestElevatedSegments:
    def test_no_segments(self):
        assert not _has_elevated_segments(None, 2.0)
        assert not _has_elevated_segments(_empty_min_heights(2, 2), 2.0)

    def test_ground_based_only(self):
        mh = _empty_min_heights(2, 2)
        mh[0, 0] = [[0.0, 10.0]]
        assert not _has_elevated_segments(mh, 2.0)

    def test_elevated_segment_detected(self):
        mh = _empty_min_heights(2, 2)
        mh[0, 0] = [[4.0, 10.0]]  # starts 2 voxels above ground
        assert _has_elevated_segments(mh, 2.0)

    def test_sub_voxel_min_rounds_to_ground(self):
        mh = _empty_min_heights(1, 1)
        mh[0, 0] = [[0.4, 10.0]]  # rounds to level 0 with meshsize 2.0
        assert not _has_elevated_segments(mh, 2.0)

    def test_elevated_segment_in_later_cell_after_empty_and_ground_cells(self):
        # Earlier cells are empty or ground-based (each hits the loop's
        # `continue`/false branch); only the last cell is elevated. Pins
        # that the scan keeps going rather than stopping at the first cell.
        mh = _empty_min_heights(2, 2)
        mh[0, 0] = []
        mh[0, 1] = [[0.0, 5.0]]
        mh[1, 0] = []
        mh[1, 1] = [[4.0, 10.0]]
        assert _has_elevated_segments(mh, 2.0)


class TestBuildBuildings3d:
    def test_overhang_column(self):
        heights = np.array([[10.0, 0.0]])
        mh = _empty_min_heights(1, 2)
        mh[0, 0] = [[4.0, 10.0]]
        b3d = _build_buildings_3d(heights, mh, meshsize=2.0)
        assert b3d.dtype == np.int8
        assert b3d.shape == (5, 1, 2)  # nz = round(10/2)
        col = b3d[:, 0, 0]
        assert list(col) == [0, 0, 1, 1, 1]  # levels 2..4 filled
        assert (b3d[:, 0, 1] == 0).all()

    def test_extrusion_fallback_when_no_segments(self):
        heights = np.array([[6.0]])
        b3d = _build_buildings_3d(heights, _empty_min_heights(1, 1), meshsize=2.0)
        assert list(b3d[:, 0, 0]) == [1, 1, 1]

    def test_multiple_disjoint_segments_leave_a_real_gap(self):
        # A bridge/overhang case: two segments in one cell with empty space
        # between them. Pins that the gap levels stay 0, not filled in.
        heights = np.array([[12.0]])
        mh = _empty_min_heights(1, 1)
        mh[0, 0] = [[0.0, 4.0], [8.0, 12.0]]
        b3d = _build_buildings_3d(heights, mh, meshsize=2.0)
        assert list(b3d[:, 0, 0]) == [1, 1, 0, 0, 1, 1]

    def test_segment_extending_above_heights_grows_nz_instead_of_truncating(self):
        # A segment's upper bound (20 m) exceeds heights.max() (5 m). nz is
        # sized from the taller of the two, so the full segment is
        # represented -- nothing above heights.max() gets silently clipped.
        heights = np.array([[5.0]])
        mh = _empty_min_heights(1, 1)
        mh[0, 0] = [[0.0, 20.0]]
        b3d = _build_buildings_3d(heights, mh, meshsize=2.0)
        assert b3d.shape == (10, 1, 1)  # nz from the segment top (20/2=10), not heights.max()
        assert list(b3d[:, 0, 0]) == [1] * 10

    def test_negative_segment_min_clamps_to_ground(self):
        # A segment starting below ground (e.g. CityGML LOD2 geometry dipping
        # below the terrain datum) must clamp to level 0, not wrap via a
        # negative-index slice. Unclamped: k0 = int(-6/2 + 0.5) = -2, so
        # b3d[-2:4] would set only index 3 (wrap-around), not levels 0..3.
        heights = np.array([[10.0]])
        mh = _empty_min_heights(1, 1)
        mh[0, 0] = [[-6.0, 8.0]]
        b3d = _build_buildings_3d(heights, mh, meshsize=2.0)
        assert b3d.shape == (5, 1, 1)
        assert list(b3d[:, 0, 0]) == [1, 1, 1, 1, 0]

    def test_empty_grid_does_not_raise(self):
        heights = np.zeros((0, 3))
        b3d = _build_buildings_3d(heights, None, meshsize=2.0)
        assert b3d.shape == (1, 0, 3)
