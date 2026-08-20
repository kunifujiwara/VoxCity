"""Tests for voxcity.exporter.palm (PALM static driver exporter)."""

import logging
import warnings

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
    IDX_PAVEMENT,
    IDX_VEGETATION,
    IDX_WATER,
    OEMJ_CLASS_TO_PALM,
    OSM_CLASS_TO_PALM,
    URBANWATCH_CLASS_TO_PALM,
    _build_building_mask,
    _build_buildings,
    _build_buildings_3d,
    _build_georeference,
    _build_index_to_palm_map,
    _build_lad,
    _build_surface_types,
    _build_zt,
    _get_source_name_mapping,
    _has_elevated_segments,
    _validate_static_fields,
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


class TestBuildingMask:
    def test_orphan_segment_cell_is_a_building(self):
        # Segment present, no recorded LOD1 height: the shared mask still
        # counts this as a building, and segment_top_m carries the height
        # buildings_2d should be repaired to.
        heights = np.array([[0.0]])
        mh = _empty_min_heights(1, 1)
        mh[0, 0] = [[4.0, 10.0]]
        mask, segment_top_m = _build_building_mask(heights, mh)
        assert mask[0, 0]
        assert segment_top_m[0, 0] == pytest.approx(10.0)

    def test_fully_below_ground_segment_is_not_a_building(self):
        heights = np.array([[0.0]])
        mh = _empty_min_heights(1, 1)
        mh[0, 0] = [[-10.0, -6.0]]
        mask, segment_top_m = _build_building_mask(heights, mh)
        assert not mask[0, 0]
        assert segment_top_m[0, 0] == 0.0

    def test_height_only_cell_is_a_building_without_segments(self):
        heights = np.array([[5.0]])
        mask, segment_top_m = _build_building_mask(heights, None)
        assert mask[0, 0]
        assert segment_top_m[0, 0] == 0.0

    def test_non_finite_segment_top_does_not_crash_and_reads_as_ground(self):
        # A NaN/inf seg[1] must be sanitized to ground level (0.0), not
        # reach float()/comparison in a way that silently misclassifies the
        # cell -- and it must agree with _build_buildings_3d, which routes
        # the same raw bound through _to_level (see the paired test in
        # TestBuildBuildings3d).
        heights = np.array([[0.0], [0.0]])
        mh = _empty_min_heights(2, 1)
        mh[0, 0] = [[0.0, np.inf]]
        mh[1, 0] = [[0.0, np.nan]]
        mask, segment_top_m = _build_building_mask(heights, mh)
        assert not mask[0, 0] and segment_top_m[0, 0] == 0.0
        assert not mask[1, 0] and segment_top_m[1, 0] == 0.0


class TestBuildBuildings:
    def test_basic_fields(self):
        heights = np.array([[0.0, 10.0], [6.0, np.nan]])
        ids = np.array([[0, 7], [0, 0]])
        mask, segment_top_m = _build_building_mask(heights, None)
        b2d, bid, btype = _build_buildings(
            heights, ids, building_type=3, segment_top_m=segment_top_m, building_mask=mask
        )
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
        mask, segment_top_m = _build_building_mask(heights, None)
        b2d, bid, btype = _build_buildings(
            heights, None, building_type=2, segment_top_m=segment_top_m, building_mask=mask
        )
        assert bid[0, 0] >= 1
        assert bid[0, 1] == FILL_INT

    def test_generated_ids_unique_and_no_collision_with_existing(self):
        # Three id-less buildings alongside two pre-existing ids (5 and 10).
        # The real contract is uniqueness across the whole grid, not merely
        # "greater than the max existing id at that one cell".
        heights = np.array([[3.0, 4.0, 5.0], [6.0, 0.0, 7.0]])
        ids = np.array([[0, 10, 0], [5, 0, 0]])
        mask, segment_top_m = _build_building_mask(heights, None)
        b2d, bid, btype = _build_buildings(
            heights, ids, building_type=1, segment_top_m=segment_top_m, building_mask=mask
        )
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
        mask, segment_top_m = _build_building_mask(heights, None)
        heights_orig = heights.copy()
        ids_orig = ids.copy()
        _build_buildings(
            heights, ids, building_type=1, segment_top_m=segment_top_m, building_mask=mask
        )
        assert np.array_equal(heights, heights_orig, equal_nan=True)
        assert np.array_equal(ids, ids_orig, equal_nan=True)

    def test_orphan_segment_repairs_buildings_2d_and_logs(
        self, caplog, propagate_voxcity_logs
    ):
        # A cell with LOD2 geometry but no recorded LOD1 height (h=0) must
        # get buildings_2d raised to the geometry's height, get a generated
        # id, and get a building_type -- and the repair must be logged at
        # WARNING (genuinely inconsistent input) with the discrepancy
        # magnitude (10.0 m here: segment top 10.0 minus recorded height
        # 0.0), so a malformed input grid is visible and actionable rather
        # than silently patched over.
        heights = np.array([[0.0]])
        mh = _empty_min_heights(1, 1)
        mh[0, 0] = [[4.0, 10.0]]
        mask, segment_top_m = _build_building_mask(heights, mh)
        with caplog.at_level(logging.WARNING, logger="voxcity"):
            b2d, bid, btype = _build_buildings(
                heights, None, building_type=1, segment_top_m=segment_top_m, building_mask=mask
            )
        assert b2d[0, 0] == np.float32(10.0)
        assert bid[0, 0] != FILL_INT
        assert btype[0, 0] == 1
        assert "repair" in caplog.text.lower()
        assert caplog.records[-1].levelname == "WARNING"
        assert "10.0 m" in caplog.text

    def test_infinite_ids_are_cleaned_like_heights(self):
        # heights gets full finite-cleaning via _clean_heights (nan/inf -> 0);
        # ids must be cleaned the same way (nan_to_num with posinf/neginf),
        # not just nan -- otherwise +-inf reaches .astype(np.int64) and
        # triggers "RuntimeWarning: invalid value encountered in cast".
        heights = np.array([[5.0, 6.0]])
        ids = np.array([[np.inf, -np.inf]])
        mask, segment_top_m = _build_building_mask(heights, None)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            b2d, bid, btype = _build_buildings(
                heights, ids, building_type=1, segment_top_m=segment_top_m, building_mask=mask
            )
        # both inf ids are cleaned to 0 (treated as "no id"), so both cells
        # get generated ids, and those generated ids must still be unique.
        assert bid[0, 0] != FILL_INT
        assert bid[0, 1] != FILL_INT
        assert bid[0, 0] != bid[0, 1]

    def test_stale_id_on_non_building_cell_becomes_fill_and_does_not_collide(self):
        # A positive id sitting on a cell with no height/geometry is not a
        # building: it must not get a building_id, but it must still count
        # toward the "existing max" that generated ids are placed above.
        heights = np.array([[5.0, 0.0]])
        ids = np.array([[0, 99]])
        mask, segment_top_m = _build_building_mask(heights, None)
        b2d, bid, btype = _build_buildings(
            heights, ids, building_type=1, segment_top_m=segment_top_m, building_mask=mask
        )
        assert bid[0, 1] == FILL_INT
        assert bid[0, 0] == 100


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

    def test_non_finite_minimum_does_not_crash_and_reads_as_ground(self):
        # A NaN/inf seg[0] must not reach _to_level unsanitized (int(nan)
        # raises ValueError, int(inf) raises OverflowError); routed through
        # _clean_segment_bound it reads as ground level (0), agreeing with
        # _segment_top_m and _build_buildings_3d instead of being the one
        # sibling that still raises on the same raw bound.
        mh = _empty_min_heights(2, 1)
        mh[0, 0] = [[np.nan, 10.0]]
        mh[1, 0] = [[np.inf, 10.0]]
        assert not _has_elevated_segments(mh, 2.0)


class TestBuildBuildings3d:
    def test_overhang_column(self):
        heights = np.array([[10.0, 0.0]])
        mh = _empty_min_heights(1, 2)
        mh[0, 0] = [[4.0, 10.0]]
        mask, segment_top_m = _build_building_mask(heights, mh)
        b3d = _build_buildings_3d(
            heights, mh, meshsize=2.0, segment_top_m=segment_top_m, building_mask=mask
        )
        assert b3d.dtype == np.int8
        assert b3d.shape == (5, 1, 2)  # nz = round(10/2)
        col = b3d[:, 0, 0]
        assert list(col) == [0, 0, 1, 1, 1]  # levels 2..4 filled
        assert (b3d[:, 0, 1] == 0).all()

    def test_extrusion_fallback_when_no_segments(self):
        heights = np.array([[6.0]])
        mh = _empty_min_heights(1, 1)
        mask, segment_top_m = _build_building_mask(heights, mh)
        b3d = _build_buildings_3d(
            heights, mh, meshsize=2.0, segment_top_m=segment_top_m, building_mask=mask
        )
        assert list(b3d[:, 0, 0]) == [1, 1, 1]

    def test_multiple_disjoint_segments_leave_a_real_gap(self):
        # A bridge/overhang case: two segments in one cell with empty space
        # between them. Pins that the gap levels stay 0, not filled in.
        heights = np.array([[12.0]])
        mh = _empty_min_heights(1, 1)
        mh[0, 0] = [[0.0, 4.0], [8.0, 12.0]]
        mask, segment_top_m = _build_building_mask(heights, mh)
        b3d = _build_buildings_3d(
            heights, mh, meshsize=2.0, segment_top_m=segment_top_m, building_mask=mask
        )
        assert list(b3d[:, 0, 0]) == [1, 1, 0, 0, 1, 1]

    def test_segment_extending_above_heights_grows_nz_instead_of_truncating(self):
        # A segment's upper bound (20 m) exceeds heights.max() (5 m). nz is
        # sized from the taller of the two, so the full segment is
        # represented -- nothing above heights.max() gets silently clipped.
        heights = np.array([[5.0]])
        mh = _empty_min_heights(1, 1)
        mh[0, 0] = [[0.0, 20.0]]
        mask, segment_top_m = _build_building_mask(heights, mh)
        b3d = _build_buildings_3d(
            heights, mh, meshsize=2.0, segment_top_m=segment_top_m, building_mask=mask
        )
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
        mask, segment_top_m = _build_building_mask(heights, mh)
        b3d = _build_buildings_3d(
            heights, mh, meshsize=2.0, segment_top_m=segment_top_m, building_mask=mask
        )
        assert b3d.shape == (5, 1, 1)
        assert list(b3d[:, 0, 0]) == [1, 1, 1, 1, 0]

    def test_fully_below_ground_segment_does_not_spuriously_fill(self):
        # Segment entirely below ground: seg[1] = -6.0 rounds to k1 = -2 (not
        # -1 or 0, per int(-2.5) == -2). If k1's lower bound isn't clamped
        # independently of k0, b3d[k0:k1] normalizes the negative stop index
        # and spuriously fills levels 0..2 instead of contributing nothing.
        # A second, taller building establishes nz=5 so the array is long
        # enough for that wraparound to be visible if the clamp regresses.
        heights = np.array([[0.0, 10.0]])
        mh = _empty_min_heights(1, 2)
        mh[0, 0] = [[-10.0, -6.0]]
        mask, segment_top_m = _build_building_mask(heights, mh)
        b3d = _build_buildings_3d(
            heights, mh, meshsize=2.0, segment_top_m=segment_top_m, building_mask=mask
        )
        assert b3d.shape == (5, 1, 2)
        assert list(b3d[:, 0, 0]) == [0, 0, 0, 0, 0]

    def test_segment_present_but_unfilled_falls_back_to_extrusion(self):
        # Same fully-below-ground segment as the test above, but this time
        # heights records a real LOD1 height (10 m) for the cell. The
        # segment loop still fills nothing (k1 <= k0 for [-10, -6]), but
        # unlike the mask-only case above, the cell IS a building (h > 0),
        # so it must fall back to extruding from heights instead of
        # collapsing to the forced single-level fill -- losing 8 of the
        # recorded 10 m would be a silent, avoidable height loss.
        heights = np.array([[10.0]])
        mh = _empty_min_heights(1, 1)
        mh[0, 0] = [[-10.0, -6.0]]
        mask, segment_top_m = _build_building_mask(heights, mh)
        b3d = _build_buildings_3d(
            heights, mh, meshsize=2.0, segment_top_m=segment_top_m, building_mask=mask
        )
        assert b3d.shape == (5, 1, 1)
        assert list(b3d[:, 0, 0]) == [1, 1, 1, 1, 1]

    def test_sub_voxel_building_gets_at_least_one_level(self):
        # 0.4 m at meshsize=2.0 rounds to level 0 (an empty extrusion range)
        # under the voxelizer's own rounding, but the mask still counts it
        # as a building, so buildings_3d must not drop the column entirely.
        heights = np.array([[0.4]])
        mask, segment_top_m = _build_building_mask(heights, None)
        b3d = _build_buildings_3d(
            heights, None, meshsize=2.0, segment_top_m=segment_top_m, building_mask=mask
        )
        assert b3d.shape[0] >= 1
        assert b3d[0, 0, 0] == 1

    def test_none_min_heights_extrudes_from_heights_directly(self):
        # Distinct from the "no segments" tests above, which pass an object
        # array of empty per-cell lists: this exercises the actual
        # `min_heights is None` branch (mh stays None, not an array).
        heights = np.array([[6.0]])
        mask, segment_top_m = _build_building_mask(heights, None)
        b3d = _build_buildings_3d(
            heights, None, meshsize=2.0, segment_top_m=segment_top_m, building_mask=mask
        )
        assert list(b3d[:, 0, 0]) == [1, 1, 1]

    def test_nan_height_cell_produces_empty_column(self):
        heights = np.array([[3.0, np.nan]])
        mask, segment_top_m = _build_building_mask(heights, None)
        b3d = _build_buildings_3d(
            heights, None, meshsize=2.0, segment_top_m=segment_top_m, building_mask=mask
        )
        assert not mask[0, 1]
        assert (b3d[:, 0, 1] == 0).all()
        assert (b3d[:, 0, 0] == 1).any()

    def test_empty_grid_does_not_raise(self):
        heights = np.zeros((0, 3))
        mask, segment_top_m = _build_building_mask(heights, None)
        b3d = _build_buildings_3d(
            heights, None, meshsize=2.0, segment_top_m=segment_top_m, building_mask=mask
        )
        assert b3d.shape == (1, 0, 3)

    def test_non_finite_segment_bounds_do_not_crash(self):
        # Without _clean_segment_bound, a NaN lower bound reaching
        # _to_level(nan, ...) raises ValueError (int(nan)) and an inf upper
        # bound raises OverflowError (int(inf)) -- both must instead read as
        # ground level (0.0), matching _segment_top_m's treatment of the
        # same raw bounds (see the paired test in TestBuildingMask).
        heights = np.array([[0.0, 0.0]])
        mh = _empty_min_heights(1, 2)
        mh[0, 0] = [[np.nan, 10.0]]
        mh[0, 1] = [[0.0, np.inf]]
        mask, segment_top_m = _build_building_mask(heights, mh)
        b3d = _build_buildings_3d(
            heights, mh, meshsize=2.0, segment_top_m=segment_top_m, building_mask=mask
        )
        # NaN lower bound -> ground (0): fills the full [0, 10] range
        assert list(b3d[:, 0, 0]) == [1, 1, 1, 1, 1]
        # inf upper bound -> ground (0): k1 == k0 == 0, no fill, and mask is
        # False here so nothing forces a fallback level either
        assert (b3d[:, 0, 1] == 0).all()


class TestBuildSurfaceTypes:
    def _run(self):
        # OSM raw indices: 0 Bareland, 8 Water, 11 Road, 12 Building
        lc = np.array([[0, 8], [11, 12], [0, 12]])
        building_mask = np.array([[False, False], [False, True], [False, False]])
        canopy_mask = np.array([[True, False], [False, False], [False, False]])
        # (1,1): building; (2,1): land cover says Building but no height
        return _build_surface_types(
            lc, 'OpenStreetMap', canopy_mask,
            under_tree_vegetation_type=3, soil_type_code=3, building_mask=building_mask,
        )

    def test_categories(self):
        f = self._run()
        veg, pav, wat = f["vegetation_type"], f["pavement_type"], f["water_type"]
        assert veg[0, 0] == 3          # canopy overrides Bareland
        assert wat[0, 1] == 1          # Water -> lake
        assert pav[1, 0] == 1          # Road -> asphalt
        assert veg[2, 0] == 1          # Bareland -> bare soil
        assert pav[2, 1] == 2          # Building class w/o height -> concrete
        # building cell: everything fill
        assert veg[1, 1] == FILL_BYTE
        assert pav[1, 1] == FILL_BYTE
        assert wat[1, 1] == FILL_BYTE

    def test_exclusivity(self):
        f = self._run()
        set_count = sum(
            (f[k] != FILL_BYTE).astype(int)
            for k in ("vegetation_type", "pavement_type", "water_type")
        )
        building_mask = np.array([[False, False], [False, True], [False, False]])
        assert (set_count[~building_mask] == 1).all()
        assert (set_count[building_mask] == 0).all()

    def test_soil_and_fraction(self):
        f = self._run()
        soil, sf = f["soil_type"], f["surface_fraction"]
        assert soil[2, 0] == 3                      # vegetation -> soil
        assert soil[1, 0] == 3                      # pavement -> soil
        assert soil[0, 1] == FILL_BYTE              # water -> no soil
        assert soil[1, 1] == FILL_BYTE              # building -> no soil
        assert sf[IDX_VEGETATION, 2, 0] == 1.0
        assert sf[IDX_PAVEMENT, 2, 0] == 0.0
        assert sf[IDX_WATER, 0, 1] == 1.0
        assert sf[:, 1, 1].tolist() == [np.float32(FILL_FLOAT)] * 3

    def test_dtypes(self):
        # This unit's return values are the writer's input contract: an
        # int8 -> int16 slip here produces a silently non-PIDS-conformant
        # file (`byte` vs `short`), and a regression must be attributed to
        # this builder rather than discovered later by Unit 5's validator.
        f = self._run()
        assert f["vegetation_type"].dtype == np.int8
        assert f["pavement_type"].dtype == np.int8
        assert f["water_type"].dtype == np.int8
        assert f["soil_type"].dtype == np.int8
        assert f["surface_fraction"].dtype == np.float32

    def test_classification_log_reports_building_share_and_breakdown(
        self, caplog, propagate_voxcity_logs
    ):
        # The classification summary is the primary user-facing signal for
        # whether land cover mapped sensibly; it had no test coverage at
        # all (deleting the accumulation and log loop was silent). _run()'s
        # fixture is a 3x2 grid (6 cells) with 1 building cell, so the
        # building-share line must read "1 cell(s) (16.7%)", and the
        # itemized per-class lines must still be present (they cover the
        # other 5 cells; percentages intentionally don't sum to 100% on
        # their own -- the building-share line is what explains the gap).
        with caplog.at_level(logging.INFO, logger="voxcity"):
            self._run()
        assert "summary" in caplog.text.lower()
        assert "1 cell(s) (16.7%) are buildings" in caplog.text
        assert "Bareland" in caplog.text
        assert "Water" in caplog.text

    def test_building_mask_shape_mismatch_raises(self):
        # A stale mask from a different grid must fail loudly at the point
        # that has the wrong input, not silently ignore extra rows via
        # NumPy broadcasting/fancy-indexing.
        lc = np.zeros((2, 2))
        bad_building_mask = np.zeros((4, 2), dtype=bool)
        good_canopy_mask = np.zeros((2, 2), dtype=bool)
        with pytest.raises(ValueError):
            _build_surface_types(
                lc, 'OpenStreetMap', good_canopy_mask,
                under_tree_vegetation_type=3, soil_type_code=3,
                building_mask=bad_building_mask,
            )

    def test_canopy_mask_shape_mismatch_raises(self):
        lc = np.zeros((2, 2))
        good_building_mask = np.zeros((2, 2), dtype=bool)
        bad_canopy_mask = np.zeros((4, 2), dtype=bool)
        with pytest.raises(ValueError):
            _build_surface_types(
                lc, 'OpenStreetMap', bad_canopy_mask,
                under_tree_vegetation_type=3, soil_type_code=3,
                building_mask=good_building_mask,
            )

    def test_out_of_range_raw_index_falls_back_to_default_assignment(self):
        # Negative and absurdly large indices have no entry in
        # index_to_assignment; .get(..., DEFAULT_ASSIGNMENT) must degrade to
        # the default rather than raising KeyError/IndexError.
        lc = np.array([[-1, 9999]])
        building_mask = np.zeros((1, 2), dtype=bool)
        canopy_mask = np.zeros((1, 2), dtype=bool)
        f = _build_surface_types(
            lc, 'OpenStreetMap', canopy_mask,
            under_tree_vegetation_type=3, soil_type_code=3, building_mask=building_mask,
        )
        category, code = DEFAULT_ASSIGNMENT
        assert category == 'vegetation'
        assert f["vegetation_type"][0, 0] == code
        assert f["vegetation_type"][0, 1] == code

    def test_unknown_land_cover_source_still_classifies(self):
        # An unrecognised source must still produce a usable classification
        # (via the OSM fallback name list inside _build_index_to_palm_map),
        # not raise.
        lc = np.array([[0]])  # index 0 in the OSM fallback order: Bareland
        building_mask = np.zeros((1, 1), dtype=bool)
        canopy_mask = np.zeros((1, 1), dtype=bool)
        f = _build_surface_types(
            lc, 'TotallyUnknownSource', canopy_mask,
            under_tree_vegetation_type=3, soil_type_code=3, building_mask=building_mask,
        )
        assert f["vegetation_type"][0, 0] == 1  # Bareland -> bare soil (OSM table)

    def test_canopy_and_building_overlap_building_wins(self):
        # A cell flagged as both canopy and building must stay all-fill:
        # building takes precedence per the function's documented order.
        lc = np.array([[5]])  # Tree -- would be vegetation if not a building
        building_mask = np.array([[True]])
        canopy_mask = np.array([[True]])
        f = _build_surface_types(
            lc, 'OpenStreetMap', canopy_mask,
            under_tree_vegetation_type=3, soil_type_code=3, building_mask=building_mask,
        )
        assert f["vegetation_type"][0, 0] == FILL_BYTE
        assert f["pavement_type"][0, 0] == FILL_BYTE
        assert f["water_type"][0, 0] == FILL_BYTE
        assert f["soil_type"][0, 0] == FILL_BYTE
        assert (f["surface_fraction"][:, 0, 0] == np.float32(FILL_FLOAT)).all()

    def test_canopy_over_water_keeps_water_type(self):
        # Overhanging/riparian canopy over water must NOT get the
        # under-tree vegetation override: the water surface stays, since
        # water differs from short grass in albedo/heat capacity/
        # evaporation by margins that dominate a microclimate result, and
        # the LAD field already resolves the trees independently of the
        # surface type below them.
        lc = np.array([[8]])  # OSM raw index 8: Water
        building_mask = np.array([[False]])
        canopy_mask = np.array([[True]])
        f = _build_surface_types(
            lc, 'OpenStreetMap', canopy_mask,
            under_tree_vegetation_type=3, soil_type_code=3, building_mask=building_mask,
        )
        assert f["water_type"][0, 0] == 1          # lake -- unchanged by canopy
        assert f["vegetation_type"][0, 0] == FILL_BYTE
        assert f["soil_type"][0, 0] == FILL_BYTE   # water gets no soil either
        sf = f["surface_fraction"][:, 0, 0]
        assert sf[IDX_WATER] == 1.0
        assert sf[IDX_VEGETATION] == 0.0
        assert sf[IDX_PAVEMENT] == 0.0

    def test_distinctive_parameter_values_are_read_not_defaulted(self):
        # Every other fixture in this class pins under_tree_vegetation_type
        # and soil_type_code at 3 -- which is also DEFAULT_ASSIGNMENT's code
        # and also the OSM code for Rangeland/No Data, so those fixtures
        # cannot tell "parameter read" apart from "parameter ignored, and 3
        # happened to be the answer anyway". Uses distinct, non-default
        # values and checks the soil write on both branches that make it
        # (vegetation and pavement can regress independently).
        lc = np.array([[0, 11]])  # Bareland (canopy cell), Road (pavement cell)
        building_mask = np.zeros((1, 2), dtype=bool)
        canopy_mask = np.array([[True, False]])
        f = _build_surface_types(
            lc, 'OpenStreetMap', canopy_mask,
            under_tree_vegetation_type=7, soil_type_code=5, building_mask=building_mask,
        )
        assert f["vegetation_type"][0, 0] == 7  # not Bareland's own code (1)
        assert f["soil_type"][0, 0] == 5         # vegetation-branch soil write
        assert f["pavement_type"][0, 1] == 1     # Road, sanity check
        assert f["soil_type"][0, 1] == 5         # pavement-branch soil write


class TestBuildLad:
    def test_crown_placement(self):
        top = np.array([[6.0, 0.0]])
        bottom = np.array([[1.8, 0.0]])
        building = np.zeros((1, 2), dtype=bool)
        lad, zlad = _build_lad(top, bottom, meshsize=2.0, lad_value=1.0,
                               building_mask=building)
        # zlad: surface 0, then centres 1, 3, 5 (max top 6, dz 2)
        assert zlad.tolist() == [0.0, 1.0, 3.0, 5.0]
        col = lad[:, 0, 0]
        assert col[0] == 0.0     # surface, below crown
        assert col[1] == 0.0     # z=1 < bottom 1.8
        assert col[2] == 1.0     # z=3 in crown
        assert col[3] == 1.0     # z=5 in crown
        # non-vegetated column: all fill
        assert (lad[:, 0, 1] == np.float32(FILL_FLOAT)).all()
        assert lad.dtype == np.float32
        assert zlad.dtype == np.float32

    def test_building_clears_canopy(self):
        top = np.array([[6.0]])
        bottom = np.array([[1.8]])
        building = np.array([[True]])
        lad, zlad = _build_lad(top, bottom, meshsize=2.0, lad_value=1.0,
                               building_mask=building)
        assert lad is None and zlad is None  # nothing left to resolve

    def test_thin_crown_gets_one_layer(self):
        # bottom == top == 2.0 with dz 2: the topmost level at/below top is forced
        top = np.array([[2.0]])
        bottom = np.array([[2.0]])
        lad, zlad = _build_lad(top, bottom, meshsize=2.0, lad_value=0.5,
                               building_mask=np.array([[False]]))
        assert (lad[:, 0, 0] == 0.5).sum() == 1

    def test_sub_dz_canopy_still_gets_one_centre(self):
        # top=0.5 with dz=2: np.ceil(0.5/2 - 0.5) = np.ceil(-0.25) = 0, which
        # without the max(..., 1) floor gives zero centres (zlad=[0.0], the
        # surface level only -- nowhere for a crown to land). With the
        # floor, one centre exists (zlad=[0.0, 1.0]). This is the LAD
        # analogue of the sub-voxel building case in _build_buildings_3d:
        # a real, if tiny, canopy must not be sized out of existence.
        top = np.array([[0.5]])
        bottom = np.array([[0.0]])
        lad, zlad = _build_lad(top, bottom, meshsize=2.0, lad_value=1.0,
                               building_mask=np.array([[False]]))
        assert zlad.tolist() == [0.0, 1.0]

    def test_no_canopy_returns_none(self):
        lad, zlad = _build_lad(np.zeros((2, 2)), np.zeros((2, 2)),
                               meshsize=1.0, lad_value=1.0,
                               building_mask=np.zeros((2, 2), dtype=bool))
        assert lad is None and zlad is None

    def test_does_not_mutate_inputs(self):
        # float64 is the aliasing-prone dtype: np.asarray(x, dtype=float64)
        # returns the same object for float64 input, so if nan_to_num or
        # np.where ever stopped copying, this would observe it -- but only
        # if the inputs actually force a substitution. Finite, in-range
        # values (e.g. top=6.0, bottom=1.8, an all-False mask) give
        # nan_to_num/np.where nothing to change, so a copy=False regression
        # would be a silent no-op on them and this test would pass either
        # way. Using +-inf/NaN and a True mask entry forces a real
        # substitution in every op that touches the caller's arrays.
        top = np.array([[6.0, np.inf]], dtype=np.float64)
        bottom = np.array([[np.nan, 1.0]], dtype=np.float64)
        top_orig, bottom_orig = top.copy(), bottom.copy()
        _build_lad(top, bottom, meshsize=2.0, lad_value=1.0,
                   building_mask=np.array([[False, True]]))
        assert np.array_equal(top, top_orig, equal_nan=True)
        assert np.array_equal(bottom, bottom_orig, equal_nan=True)

    def test_non_finite_top_is_sanitized_not_a_crash(self):
        # NaN and +-inf must not reach the n_centres/np.arange sizing below
        # (without posinf/neginf, nan_to_num's default substitutes the
        # largest finite float64 for +inf rather than leaving literal
        # infinity, so the observed failure is ValueError: Maximum allowed
        # size exceeded out of np.arange, not OverflowError out of int())
        # or corrupt a finite column; sanitized to 0.0, consistent with how
        # _clean_heights treats non-finite heights as "nothing here". NaN
        # is the one most likely to appear for real (canopy height rasters
        # commonly carry NaN as nodata; +-inf is essentially synthetic), so
        # it gets its own column rather than being assumed to behave like
        # +-inf just because they share one nan_to_num call.
        top = np.array([[np.nan, np.inf, 5.0, -np.inf]])
        bottom = np.array([[0.0, 0.0, 2.0, 0.0]])
        building = np.zeros((1, 4), dtype=bool)
        lad, zlad = _build_lad(top, bottom, meshsize=2.0, lad_value=1.0,
                               building_mask=building)
        assert lad is not None
        assert (lad[:, 0, 0] == np.float32(FILL_FLOAT)).all()  # NaN -> no canopy
        assert (lad[:, 0, 1] == np.float32(FILL_FLOAT)).all()  # +inf -> no canopy
        assert (lad[:, 0, 3] == np.float32(FILL_FLOAT)).all()  # -inf -> no canopy
        assert (lad[:, 0, 2] == np.float32(1.0)).any()         # finite column unaffected

    def test_inverted_crown_clamped_to_a_single_sane_layer(self):
        # bottom (100) > top (6): an inverted (bottom > top) range must
        # never leave an empty crown loose in the output -- the
        # empty-in_crown fallback below gives exactly one surviving level,
        # the highest level at/below top, without needing bottom clamped
        # down to top first (bottom is used raw; see _build_lad's
        # docstring for why no clamping is required).
        top = np.array([[6.0]])
        bottom = np.array([[100.0]])
        lad, zlad = _build_lad(top, bottom, meshsize=2.0, lad_value=1.0,
                               building_mask=np.array([[False]]))
        assert zlad.tolist() == [0.0, 1.0, 3.0, 5.0]
        col = lad[:, 0, 0]
        assert (col == np.float32(1.0)).sum() == 1
        assert col[3] == np.float32(1.0)  # z=5, the last level <= top

    def test_fill_above_own_top_not_zero(self):
        # Two columns with different tops: the short column's levels above
        # its OWN top must be FILL_FLOAT, not left at the default 0.0 --
        # every other fixture's tallest column is the domain max (the one
        # that sizes zlad), so no level is ever above its own top without a
        # second, taller column establishing a larger zlad array first.
        top = np.array([[6.0, 2.0]])
        bottom = np.array([[0.0, 0.0]])
        building = np.zeros((1, 2), dtype=bool)
        lad, zlad = _build_lad(top, bottom, meshsize=2.0, lad_value=1.0,
                               building_mask=building)
        assert zlad.tolist() == [0.0, 1.0, 3.0, 5.0]
        col1 = lad[:, 0, 1]
        assert list(col1) == [
            np.float32(1.0), np.float32(1.0),
            np.float32(FILL_FLOAT), np.float32(FILL_FLOAT),
        ]

    def test_crown_boundary_is_inclusive_at_bottom(self):
        # top=5, bottom=1, dz=2 -> zlad=[0, 1, 3]. Crown membership is
        # inclusive at the bottom boundary (>=): level 1 sits exactly on
        # bottom and must be included, not excluded by a strict
        # inequality. The thin-crown fallback (bottom == top, used
        # elsewhere) doesn't depend on bottom at all, so it can't catch a
        # boundary-strictness regression -- this needs bottom < top with a
        # level landing exactly on bottom. (Changing only the lower
        # inequality to strict, `zlad > b`, was confirmed to fail this
        # test; three other fixtures in this class also happen to use
        # bottom == 0 == zlad[0] and fail alongside it, which is expected
        # overlap, not a substitute for this test.)
        top = np.array([[5.0]])
        bottom = np.array([[1.0]])
        lad, zlad = _build_lad(top, bottom, meshsize=2.0, lad_value=1.0,
                               building_mask=np.array([[False]]))
        assert zlad.tolist() == [0.0, 1.0, 3.0]
        assert list(lad[:, 0, 0]) == [
            np.float32(0.0), np.float32(1.0), np.float32(1.0),
        ]

    def test_crown_boundary_is_inclusive_at_top(self):
        # Two columns: col0's top (10.0) sizes zlad=[0,1,3,5,7,9]; col1's
        # own top (3.0) lands exactly on one of those grid points. Crown
        # membership is inclusive at the top boundary (<=): level 3 sits
        # exactly on col1's own top and must be included in its crown, not
        # excluded by a strict inequality -- and it must NOT be caught by
        # the separate "fill above own top" `zlad > t` step either, since
        # 3 is not above 3. No single-column fixture can exercise this: a
        # column's own top only ever lands exactly on a zlad grid point
        # when a taller sibling column is the one sizing that grid.
        top = np.array([[10.0, 3.0]])
        bottom = np.array([[0.0, 0.0]])
        lad, zlad = _build_lad(top, bottom, meshsize=2.0, lad_value=1.0,
                               building_mask=np.zeros((1, 2), dtype=bool))
        assert zlad.tolist() == [0.0, 1.0, 3.0, 5.0, 7.0, 9.0]
        col1 = lad[:, 0, 1]
        assert list(col1) == [
            np.float32(1.0), np.float32(1.0), np.float32(1.0),
            np.float32(FILL_FLOAT), np.float32(FILL_FLOAT), np.float32(FILL_FLOAT),
        ]

    def test_non_finite_bottom_is_sanitized_not_left_huge(self):
        # NaN and +-inf on the bottom side must also be sanitized to 0.0
        # (ground), not left as nan_to_num's default huge-but-finite
        # substitute -- that would put bottom far above every zlad level,
        # forcing the column into the single-level fallback instead of
        # filling the whole crown from the ground up. Only the `top` side
        # was pinned by test_non_finite_top_is_sanitized_not_a_crash; this
        # covers the other nan_to_num call, with both NaN (the realistic
        # nodata case for a canopy-bottom raster) and +inf.
        top = np.array([[6.0, 6.0]])
        bottom = np.array([[np.inf, np.nan]])
        lad, zlad = _build_lad(top, bottom, meshsize=2.0, lad_value=1.0,
                               building_mask=np.array([[False, False]]))
        assert zlad.tolist() == [0.0, 1.0, 3.0, 5.0]
        assert list(lad[:, 0, 0]) == [np.float32(1.0)] * 4  # inf bottom -> ground
        assert list(lad[:, 0, 1]) == [np.float32(1.0)] * 4  # NaN bottom -> ground


def _valid_fields():
    """Minimal internally consistent 1x2 field set: veg cell + building cell."""
    zt = np.array([[0.0, 1.0]], dtype=np.float32)
    b2d = np.array([[FILL_FLOAT, 8.0]], dtype=np.float32)
    bid = np.array([[FILL_INT, 5]], dtype=np.int32)
    btype = np.array([[FILL_BYTE, 3]], dtype=np.int8)
    veg = np.array([[3, FILL_BYTE]], dtype=np.int8)
    pav = np.full((1, 2), FILL_BYTE, dtype=np.int8)
    wat = np.full((1, 2), FILL_BYTE, dtype=np.int8)
    soil = np.array([[3, FILL_BYTE]], dtype=np.int8)
    sf = np.full((3, 1, 2), FILL_FLOAT, dtype=np.float32)
    sf[:, 0, 0] = [1.0, 0.0, 0.0]
    return {
        "zt": zt, "buildings_2d": b2d, "building_id": bid,
        "building_type": btype, "vegetation_type": veg,
        "pavement_type": pav, "water_type": wat, "soil_type": soil,
        "surface_fraction": sf, "buildings_3d": None, "lad": None,
    }


class TestValidator:
    def test_valid_fields_pass(self):
        _validate_static_fields(_valid_fields())  # no raise

    def test_zt_min_nonzero_fails(self):
        f = _valid_fields()
        f["zt"] = f["zt"] + 1.0
        with pytest.raises(RuntimeError, match="zt"):
            _validate_static_fields(f)

    def test_two_surface_types_fail(self):
        f = _valid_fields()
        f["pavement_type"][0, 0] = 1  # cell now veg AND pavement
        with pytest.raises(RuntimeError, match="exactly one"):
            _validate_static_fields(f)

    def test_surface_type_on_building_fails(self):
        f = _valid_fields()
        f["vegetation_type"][0, 1] = 3
        with pytest.raises(RuntimeError, match="building cells"):
            _validate_static_fields(f)

    def test_missing_soil_fails(self):
        f = _valid_fields()
        f["soil_type"][0, 0] = FILL_BYTE
        with pytest.raises(RuntimeError, match="soil_type"):
            _validate_static_fields(f)

    def test_bad_fraction_sum_fails(self):
        f = _valid_fields()
        f["surface_fraction"][0, 0, 0] = 0.5
        with pytest.raises(RuntimeError, match="surface_fraction"):
            _validate_static_fields(f)

    def test_building_without_id_fails(self):
        f = _valid_fields()
        f["building_id"][0, 1] = FILL_INT
        with pytest.raises(RuntimeError, match="building_id"):
            _validate_static_fields(f)

    def test_out_of_range_code_fails(self):
        f = _valid_fields()
        f["vegetation_type"][0, 0] = 99
        with pytest.raises(RuntimeError, match="vegetation_type"):
            _validate_static_fields(f)

    def test_buildings_3d_column_without_2d_fails(self):
        f = _valid_fields()
        b3d = np.zeros((2, 1, 2), dtype=np.int8)
        b3d[0, 0, 0] = 1  # column (0,0) has no buildings_2d
        f["buildings_3d"] = b3d
        with pytest.raises(RuntimeError, match="buildings_3d"):
            _validate_static_fields(f)
