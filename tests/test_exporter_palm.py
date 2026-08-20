"""Tests for voxcity.exporter.palm (PALM static driver exporter)."""

import logging
import re
import warnings
from pathlib import Path

import numpy as np
import pytest
from netCDF4 import Dataset
from pyproj import Transformer

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
    PalmExporter,
    _FIELD_SPECS,
    _SHAPE_CHECKED_GRID_ACCESSORS,
    _build_building_mask,
    _build_buildings,
    _build_buildings_3d,
    _build_georeference,
    _build_index_to_palm_map,
    _build_lad,
    _build_surface_types,
    _build_zt,
    _check_export_inputs,
    _get_source_name_mapping,
    _has_elevated_segments,
    _validate_static_fields,
    _write_static_driver,
    export_palm,
)
from voxcity.models import (
    BuildingGrid,
    CanopyGrid,
    DemGrid,
    GridMetadata,
    LandCoverGrid,
    VoxCity,
    VoxelGrid,
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


# field name -> a same-shape, wrong-dtype copy of the value _valid_fields()
# assigns it. Values are preserved exactly (only .astype changes), so each
# case trips only the dtype check -- not any value-based rule -- proving the
# dtype check is independently exercised rather than riding along with one
# of the value checks above.
_WRONG_DTYPE_CASES = [
    ("zt", np.float64),
    ("buildings_2d", np.float64),
    ("surface_fraction", np.float64),
    ("building_id", np.int64),
    ("building_type", np.int16),
    ("vegetation_type", np.int16),
    ("pavement_type", np.int16),
    ("water_type", np.int16),
    ("soil_type", np.int16),
]


class TestValidatorDtypes:
    """Dtype is the actual file contract (byte vs short); assert it directly.

    Every case below only changes dtype (via .astype, values unchanged), so
    a failure here can only be the dtype check -- verified per-field by
    breaking _REQUIRED_DTYPES/_OPTIONAL_DTYPES and confirming the matching
    case here (and only that case) starts failing.
    """

    @pytest.mark.parametrize("field, wrong_dtype", _WRONG_DTYPE_CASES)
    def test_wrong_dtype_fails(self, field, wrong_dtype):
        f = _valid_fields()
        f[field] = f[field].astype(wrong_dtype)
        with pytest.raises(RuntimeError, match=field):
            _validate_static_fields(f)

    def test_buildings_3d_wrong_dtype_fails(self):
        f = _valid_fields()
        # Column present exactly where buildings_2d/building_id/building_type
        # already are (cell (0,1)), so no presence rule fires -- isolates
        # the dtype check from TestValidatorBuildings3dPresence below.
        b3d = np.zeros((1, 1, 2), dtype=np.int16)
        b3d[0, 0, 1] = 1
        f["buildings_3d"] = b3d
        with pytest.raises(RuntimeError, match="buildings_3d"):
            _validate_static_fields(f)

    def test_lad_wrong_dtype_fails(self):
        f = _valid_fields()
        # All-fill: the lad value-range rule only inspects non-fill cells,
        # so an all-fill array of the wrong dtype trips only the dtype check.
        f["lad"] = np.full((1, 1, 2), FILL_FLOAT, dtype=np.float64)
        with pytest.raises(RuntimeError, match="lad"):
            _validate_static_fields(f)


def _presence_fields(bld, col, has_id, has_type):
    """1x1 field set isolating exactly one of the six LOD1/LOD2 presence
    directions the validator checks between buildings_3d, buildings_2d,
    building_id, and building_type (see _validate_static_fields'
    ``presence_checks``). Surface-type fields are left all-fill throughout
    (irrelevant to these checks); a resulting "exactly one surface type"
    complaint on a non-building (bld=False) cell is an expected, harmless
    side effect of that unrelated pre-existing rule -- the match= patterns
    used against this fixture are the six presence-direction messages
    verbatim and cannot be satisfied by that rule's text.
    """
    zt = np.zeros((1, 1), dtype=np.float32)
    b2d = np.array([[8.0 if bld else FILL_FLOAT]], dtype=np.float32)
    bid = np.array([[5 if has_id else FILL_INT]], dtype=np.int32)
    btype = np.array([[3 if has_type else FILL_BYTE]], dtype=np.int8)
    b3d = np.array([[[1 if col else 0]]], dtype=np.int8)
    return {
        "zt": zt, "buildings_2d": b2d, "building_id": bid,
        "building_type": btype,
        "vegetation_type": np.full((1, 1), FILL_BYTE, dtype=np.int8),
        "pavement_type": np.full((1, 1), FILL_BYTE, dtype=np.int8),
        "water_type": np.full((1, 1), FILL_BYTE, dtype=np.int8),
        "soil_type": np.full((1, 1), FILL_BYTE, dtype=np.int8),
        "surface_fraction": np.full((3, 1, 1), FILL_FLOAT, dtype=np.float32),
        "buildings_3d": b3d, "lad": None,
    }


class TestValidatorBuildings3dPresence:
    """The LOD1/LOD2 presence invariant is bidirectional (see the spec's
    validation section): buildings_3d column presence, buildings_2d,
    building_id, and building_type must all agree per cell. Each of the six
    directions below flips exactly one of (bld, col, has_id, has_type) away
    from a fully-consistent baseline, chosen so that direction -- and only
    that direction -- fires among the six; verified by hand (see the
    per-case comments) rather than by running the other five as negative
    assertions, since the point is a regression in any single direction is
    caught by its own dedicated test rather than only riding along with a
    sibling direction's test (e.g. the original col-without-2d case above
    also happens to leave building_id/building_type unset, so on its own it
    would not prove those two directions are independently reachable).

    Isolated only among these six: three of the combos below (bld=False)
    also trip the always-on "stray building_id/building_type on a
    non-building cell" checks (see TestValidatorStrayPresenceWithoutBuildings3d),
    since those run unconditionally now, not only when buildings_3d is
    present. That is expected and correct, not a leak -- match= still
    finds each test's own target message regardless of what else is also
    reported alongside it.
    """

    def test_buildings_3d_column_without_2d_fails(self):
        f = _presence_fields(bld=False, col=True, has_id=True, has_type=True)
        with pytest.raises(
            RuntimeError, match="buildings_3d column present without a buildings_2d height"
        ):
            _validate_static_fields(f)

    def test_2d_without_buildings_3d_column_fails(self):
        f = _presence_fields(bld=True, col=False, has_id=False, has_type=False)
        with pytest.raises(
            RuntimeError, match="buildings_2d height present without a buildings_3d column"
        ):
            _validate_static_fields(f)

    def test_buildings_3d_column_without_building_id_fails(self):
        f = _presence_fields(bld=True, col=True, has_id=False, has_type=True)
        with pytest.raises(
            RuntimeError, match="buildings_3d column present without a building_id"
        ):
            _validate_static_fields(f)

    def test_building_id_without_buildings_3d_column_fails(self):
        f = _presence_fields(bld=False, col=False, has_id=True, has_type=False)
        with pytest.raises(
            RuntimeError, match="building_id present without a buildings_3d column"
        ):
            _validate_static_fields(f)

    def test_buildings_3d_column_without_building_type_fails(self):
        f = _presence_fields(bld=True, col=True, has_id=True, has_type=False)
        with pytest.raises(
            RuntimeError, match="buildings_3d column present without a building_type"
        ):
            _validate_static_fields(f)

    def test_building_type_without_buildings_3d_column_fails(self):
        f = _presence_fields(bld=False, col=False, has_id=False, has_type=True)
        with pytest.raises(
            RuntimeError, match="building_type present without a buildings_3d column"
        ):
            _validate_static_fields(f)


class TestValidatorAgainstRealBuilders:
    """The _valid_fields() fixture above is the linchpin of every validator
    test: if it were subtly invalid in a way no rule catches, every test
    that mutates it would be weakened. Cross-check the validator against
    fields produced by the real builders (not the hand-rolled fixture) for
    a small synthetic city, proving the builders and validator actually
    agree -- which is the point of this unit.
    """

    def test_builder_output_passes_validation(self):
        # 3x3 grid: one plain building (extruded, no segments), one
        # overhang building (LOD2 geometry rising above ground, forcing
        # buildings_3d), one canopy cell, one water cell, one road cell,
        # bare ground elsewhere.
        heights = np.array([
            [0.0, 0.0, 0.0],
            [0.0, 8.0, 0.0],
            [0.0, 0.0, 6.0],
        ])
        ids = np.array([
            [0, 0, 0],
            [0, 3, 0],
            [0, 0, 0],
        ])
        min_heights = np.empty((3, 3), dtype=object)
        for i in range(3):
            for j in range(3):
                min_heights[i, j] = []
        min_heights[2, 2] = [[4.0, 6.0]]  # overhang: starts above ground

        land_cover = np.zeros((3, 3), dtype=int)  # 0 = Bareland (OSM)
        land_cover[0, 0] = 8   # Water
        land_cover[0, 1] = 11  # Road
        land_cover[1, 2] = 5   # Tree

        canopy_top = np.zeros((3, 3))
        canopy_top[1, 2] = 5.0
        canopy_bottom = canopy_top * 0.3

        meshsize = 2.0
        building_mask, segment_top_m = _build_building_mask(heights, min_heights)
        buildings_2d, building_id, building_type = _build_buildings(
            heights, ids, building_type=3,
            segment_top_m=segment_top_m, building_mask=building_mask,
        )
        buildings_3d = _build_buildings_3d(
            heights, min_heights, meshsize,
            segment_top_m=segment_top_m, building_mask=building_mask,
        )
        canopy_mask = (canopy_top > 0.0) & ~building_mask
        surfaces = _build_surface_types(
            land_cover, 'OpenStreetMap', canopy_mask,
            under_tree_vegetation_type=3, soil_type_code=3,
            building_mask=building_mask,
        )
        lad, zlad = _build_lad(
            canopy_top, canopy_bottom, meshsize, lad_value=1.0,
            building_mask=building_mask,
        )
        zt, _origin_z = _build_zt(np.zeros((3, 3)))

        fields = {
            "zt": zt,
            "buildings_2d": buildings_2d,
            "building_id": building_id,
            "building_type": building_type,
            "buildings_3d": buildings_3d,
            "lad": lad,
            **surfaces,
        }
        _validate_static_fields(fields)  # no raise -- builders and validator agree


class TestValidatorLadRange:
    """The lad value-range rule (negative/non-finite lad values, checked
    only on non-fill cells) has no coverage anywhere in the plan's given
    TestValidator tests -- found during the (a) audit of this unit's test
    suite. Isolated the same way as the dtype tests: every other lad cell
    stays fill, so only the one deliberately-bad value can trip a rule.
    """

    def test_lad_negative_value_fails(self):
        f = _valid_fields()
        f["lad"] = np.full((1, 1, 2), FILL_FLOAT, dtype=np.float32)
        f["lad"][0, 0, 0] = -1.0
        with pytest.raises(RuntimeError, match="lad contains negative or non-finite"):
            _validate_static_fields(f)

    def test_lad_non_finite_value_fails(self):
        f = _valid_fields()
        f["lad"] = np.full((1, 1, 2), FILL_FLOAT, dtype=np.float32)
        f["lad"][0, 0, 0] = np.nan
        with pytest.raises(RuntimeError, match="lad contains negative or non-finite"):
            _validate_static_fields(f)


class TestValidatorRemainingRuleDirections:
    """A second audit pass (after the buildings_3d/lad gaps found above)
    found five more rules -- and one rule direction each for two rules
    already partly covered -- with no test anywhere in this module:
    zt's non-finite branch, the building_type twin of
    test_building_without_id_fails, the "soil set where neither
    vegetation nor pavement is" direction of the soil-coverage rule, the
    "zero surface types" direction of the exactly-one-surface-type rule,
    and the BYTE_RANGES lower bound. Each test below is built to fail
    under the exact narrow mutation that let it slip through undetected
    (see the per-test comments), not just under a wholesale rule removal.
    """

    def test_zt_non_finite_fails(self):
        # The real risk this rule guards against: zt's isfinite branch is
        # the only thing that can catch a NaN, because a NaN silently
        # defeats the elif -- abs(nan) > 1e-4 is False, not True, so a NaN
        # terrain would pass the min-check branch even if it were reached.
        f = _valid_fields()
        f["zt"][0, 0] = np.nan
        with pytest.raises(RuntimeError, match="zt contains non-finite"):
            _validate_static_fields(f)

    def test_building_without_type_fails(self):
        # Twin of test_building_without_id_fails above, which covers only
        # the building_id half of the two symmetric "building cells
        # missing X" checks; building_type's own check had no test.
        f = _valid_fields()
        f["building_type"][0, 1] = FILL_BYTE
        with pytest.raises(RuntimeError, match="building cells missing building_type"):
            _validate_static_fields(f)

    def test_soil_set_without_vegetation_or_pavement_fails(self):
        # The reverse direction of test_missing_soil_fails. That test
        # (soil missing where vegetation IS set) also happens to satisfy
        # an AND-based mutation of the rule (`(veg|pav) & ~soil`), because
        # vegetation=True/soil=False still trips `True & True`. Only a
        # cell where vegetation/pavement are BOTH absent but soil is
        # nonetheless set exposes that mutation: `False & ~True == False`,
        # silently missing it. Cell (0,1) is the building cell -- soil
        # has no business being set there at all.
        f = _valid_fields()
        f["soil_type"][0, 1] = 3
        with pytest.raises(RuntimeError, match="soil_type"):
            _validate_static_fields(f)

    def test_zero_surface_types_on_non_building_cell_fails(self):
        # The reverse direction of test_two_surface_types_fail, which only
        # exercises the "more than one type" half of `!= 1`. A non-building
        # cell with NO surface type set (n_types == 0) is the other way
        # `!= 1` can fire, and `> 1` (a plausible-looking mutation) misses
        # it entirely: 0 > 1 is False. soil_type is cleared alongside
        # vegetation_type so the soil-coverage rule doesn't also fire here,
        # keeping this test's failure attributable to this rule alone.
        f = _valid_fields()
        f["vegetation_type"][0, 0] = FILL_BYTE
        f["soil_type"][0, 0] = FILL_BYTE
        with pytest.raises(RuntimeError, match="exactly one"):
            _validate_static_fields(f)

    def test_below_range_code_fails(self):
        # test_out_of_range_code_fails above only ever uses a value above
        # the upper bound (99); dropping the `(vals < lo).any()` half of
        # the BYTE_RANGES check survives that test untouched. 0 is below
        # vegetation_type's lower bound (1) but is not FILL_BYTE, so it is
        # still "set" as far as every other rule is concerned.
        f = _valid_fields()
        f["vegetation_type"][0, 0] = 0
        with pytest.raises(RuntimeError, match="vegetation_type"):
            _validate_static_fields(f)


class TestValidatorBuildings2dFinite:
    """buildings_2d had no NaN/inf check at all, though the spec's
    validation section lists "all written arrays free of NaN/inf" -- and
    it compounds badly here specifically: `bld = buildings_2d !=
    FILL_FLOAT` is True for NaN (NaN != anything is True, including
    FILL_FLOAT), so an unsanitized NaN height would count as a building
    and go on to satisfy every downstream presence rule rather than
    getting caught anywhere. Unreachable today because _clean_heights
    sanitizes upstream in the real builder pipeline, but this function's
    stated purpose is to be the backstop for exactly that.
    """

    def test_nan_height_fails(self):
        f = _valid_fields()
        f["buildings_2d"][0, 1] = np.nan
        with pytest.raises(RuntimeError, match="buildings_2d contains non-finite"):
            _validate_static_fields(f)

    def test_inf_height_fails(self):
        f = _valid_fields()
        f["buildings_2d"][0, 1] = np.inf
        with pytest.raises(RuntimeError, match="buildings_2d contains non-finite"):
            _validate_static_fields(f)


class TestValidatorStrayPresenceWithoutBuildings3d:
    """All six LOD1/LOD2 presence directions live inside `if b3d is not
    None:`. Outside it -- the common path, since buildings_3d="auto"
    means most exports never build one -- only "building cell missing
    building_id/building_type" was checked; a stray building_id or
    building_type sitting on a non-building cell was accepted. Both new
    fixtures use _valid_fields() with buildings_3d left at its default
    None, exercising exactly the regime every other non-presence test in
    this module already runs in.
    """

    def test_stray_building_id_without_buildings_3d_fails(self):
        f = _valid_fields()
        assert f["buildings_3d"] is None
        f["building_id"][0, 0] = 42  # cell (0,0) is not a building
        with pytest.raises(RuntimeError, match="building_id present on a non-building cell"):
            _validate_static_fields(f)

    def test_stray_building_type_without_buildings_3d_fails(self):
        f = _valid_fields()
        assert f["buildings_3d"] is None
        f["building_type"][0, 0] = 3
        with pytest.raises(RuntimeError, match="building_type present on a non-building cell"):
            _validate_static_fields(f)


class TestValidatorZtTolerance:
    def test_small_offset_above_tolerance_still_fails(self):
        # test_zt_min_nonzero_fails (TestValidator, above) shifts by a
        # full +1.0 m, so it cannot distinguish the documented 1e-4 m
        # tolerance from anything up to just under 1.0 m -- changing
        # `> 1e-4` to `> 0.999` passes that test unchanged. 2e-4 is safely
        # above 1e-4 but far below a mistaken ~1 m tolerance, so this
        # pins the constant itself rather than merely "some threshold".
        f = _valid_fields()
        f["zt"] = f["zt"] + np.float32(2e-4)
        with pytest.raises(RuntimeError, match="zt"):
            _validate_static_fields(f)


class TestValidatorErrorAggregation:
    def test_multiple_problems_all_reported(self):
        # Confirms _validate_static_fields reports every problem it finds
        # in one exception, not just the first (or last). Two rules
        # trip here from independent causes -- zt's minimum and the
        # surface-type exclusivity rule -- and the exception message
        # must name both, not just one of them.
        f = _valid_fields()
        f["zt"] = f["zt"] + 1.0
        f["pavement_type"][0, 0] = 1  # cell now veg AND pavement
        with pytest.raises(RuntimeError) as exc_info:
            _validate_static_fields(f)
        message = str(exc_info.value)
        assert "zt" in message
        assert "exactly one" in message


class TestValidatorSurfaceFractionShape:
    def test_wrong_leading_dimension_fails(self):
        # A (2, y, x) surface_fraction (e.g. a missing water slice) sums
        # fine along axis 0 regardless of how many slices it has, so the
        # existing sum-to-1 check does not catch this on its own -- it
        # needs its own explicit shape check. Without one, this reaches
        # the writer, which fails on a raw broadcast/shape error instead
        # of the clean RuntimeError every other malformed input gets here.
        f = _valid_fields()
        f["surface_fraction"] = f["surface_fraction"][:2]
        with pytest.raises(RuntimeError, match="surface_fraction"):
            _validate_static_fields(f)


class TestWriter:
    def _write(self, tmp_path, with_3d=False, with_lad=True):
        fields = _valid_fields()
        if with_3d:
            b3d = np.zeros((2, 1, 2), dtype=np.int8)
            b3d[:, 0, 1] = 1
            fields["buildings_3d"] = b3d
        if with_lad:
            fields["lad"] = np.full((2, 1, 2), FILL_FLOAT, dtype=np.float32)
            fields["lad"][:, 0, 0] = [0.0, 1.0]
        coords = {
            "x": np.array([1.0, 3.0], dtype=np.float32),
            "y": np.array([1.0], dtype=np.float32),
        }
        if with_3d:
            coords["z"] = np.array([1.0, 3.0], dtype=np.float32)
        if with_lad:
            coords["zlad"] = np.array([0.0, 1.0], dtype=np.float32)
        attrs = {
            "Conventions": "CF-1.7",
            "origin_time": "2000-01-01 00:00:00 +00",
            "origin_lon": 139.7, "origin_lat": 35.68,
            "origin_x": 382000.0, "origin_y": 3949000.0,
            "origin_z": 0.0, "rotation_angle": 0.0,
        }
        path = tmp_path / "dom_static"
        _write_static_driver(path, fields, coords, attrs)
        return path

    def test_dims_dtypes_and_fills(self, tmp_path):
        path = self._write(tmp_path, with_3d=True)
        with Dataset(path) as nc:
            nc.set_auto_mask(False)
            assert nc.dimensions["x"].size == 2
            assert nc.dimensions["y"].size == 1
            assert nc.dimensions["z"].size == 2
            assert nc.dimensions["zlad"].size == 2
            assert nc.dimensions["nsurface_fraction"].size == 3
            zt = nc.variables["zt"]
            assert zt.dtype == np.float32
            assert zt._FillValue == np.float32(FILL_FLOAT)
            b2d = nc.variables["buildings_2d"]
            assert b2d.lod == 1
            assert b2d[0, 1] == np.float32(8.0)
            bid = nc.variables["building_id"]
            assert bid.dtype == np.int32
            assert bid._FillValue == FILL_INT
            for name in ("building_type", "vegetation_type", "pavement_type",
                         "water_type", "soil_type"):
                var = nc.variables[name]
                assert var.dtype == np.int8
                assert var._FillValue == FILL_BYTE
            b3d = nc.variables["buildings_3d"]
            assert b3d.lod == 2
            assert b3d.dimensions == ("z", "y", "x")
            lad = nc.variables["lad"]
            assert lad.dimensions == ("zlad", "y", "x")
            sf = nc.variables["surface_fraction"]
            assert sf.dimensions == ("nsurface_fraction", "y", "x")
            assert nc.rotation_angle == 0.0
            assert nc.origin_lon == pytest.approx(139.7)

    def test_optional_vars_absent(self, tmp_path):
        path = self._write(tmp_path, with_3d=False, with_lad=False)
        with Dataset(path) as nc:
            assert "buildings_3d" not in nc.variables
            assert "lad" not in nc.variables
            assert "z" not in nc.dimensions
            assert "zlad" not in nc.dimensions

    def test_round_trip_values_and_fill_placement(self, tmp_path):
        # Structure and dtypes are covered above; this checks the actual
        # data values read back, including fill placement -- the thing the
        # read-side nc.set_auto_mask(False) below exists to protect (fill
        # values otherwise come back masked, not literal), and the single
        # most likely thing to break silently (e.g. a transposed write).
        path = self._write(tmp_path, with_3d=True, with_lad=True)
        with Dataset(path) as nc:
            nc.set_auto_mask(False)
            assert nc.variables["zt"][:].tolist() == [[0.0, 1.0]]
            b2d = nc.variables["buildings_2d"][:]
            assert b2d[0, 0] == np.float32(FILL_FLOAT)
            assert b2d[0, 1] == np.float32(8.0)
            assert nc.variables["building_id"][:].tolist() == [[FILL_INT, 5]]
            assert nc.variables["building_type"][:].tolist() == [[FILL_BYTE, 3]]
            assert nc.variables["vegetation_type"][:].tolist() == [[3, FILL_BYTE]]
            assert nc.variables["soil_type"][:].tolist() == [[3, FILL_BYTE]]
            sf = nc.variables["surface_fraction"][:]
            assert sf[:, 0, 0].tolist() == [1.0, 0.0, 0.0]
            assert sf[:, 0, 1].tolist() == [np.float32(FILL_FLOAT)] * 3
            b3d = nc.variables["buildings_3d"][:]
            assert b3d[:, 0, 1].tolist() == [1, 1]
            assert b3d[:, 0, 0].tolist() == [0, 0]
            lad = nc.variables["lad"][:]
            assert lad[:, 0, 0].tolist() == [0.0, 1.0]
            assert lad[:, 0, 1].tolist() == [np.float32(FILL_FLOAT)] * 2

    def test_variable_attributes(self, tmp_path):
        # PALM reads `lod` to decide how to interpret buildings_2d vs
        # buildings_3d, so these are the attributes most load-bearing for
        # a real PALM run, not just descriptive metadata.
        path = self._write(tmp_path, with_3d=True, with_lad=True)
        with Dataset(path) as nc:
            assert nc.variables["buildings_2d"].lod == 1
            assert nc.variables["buildings_2d"].units == "m"
            assert nc.variables["buildings_2d"].long_name == "building height"
            assert nc.variables["buildings_3d"].lod == 2
            assert nc.variables["buildings_3d"].units == "1"
            assert nc.variables["buildings_3d"].long_name == "building structure in 3d"
            assert nc.variables["soil_type"].lod == 1
            assert nc.variables["soil_type"].long_name == "soil type classification"
            assert nc.variables["lad"].units == "m2 m-3"
            assert nc.variables["lad"].long_name == "leaf area density"
            assert nc.variables["surface_fraction"].units == "1"
            assert nc.variables["surface_fraction"].long_name == "surface fraction"
            assert nc.variables["zt"].units == "m"
            assert nc.variables["zt"].long_name == "terrain height"
            assert nc.variables["x"].units == "m"
            assert nc.variables["x"].long_name == "distance to origin in x-direction"
            assert nc.variables["y"].long_name == "distance to origin in y-direction"
            assert nc.variables["z"].long_name == "height above origin"
            assert nc.variables["zlad"].long_name == "height above ground"

    def test_reopenable_and_self_consistent(self, tmp_path):
        # nsurface_fraction's own values must match the IDX_* constants
        # (not just have the right size), and coordinate variables must
        # hold exactly the values passed in coords -- otherwise a PALM run
        # would silently misinterpret which fraction slot is which surface.
        path = self._write(tmp_path, with_3d=True, with_lad=True)
        with Dataset(path) as nc:
            nc.set_auto_mask(False)
            assert nc.variables["nsurface_fraction"][:].tolist() == [
                IDX_VEGETATION, IDX_PAVEMENT, IDX_WATER
            ]
            assert nc.variables["x"][:].tolist() == [1.0, 3.0]
            assert nc.variables["y"][:].tolist() == [1.0]
            assert nc.variables["z"][:].tolist() == [1.0, 3.0]
            assert nc.variables["zlad"][:].tolist() == [0.0, 1.0]
        # Re-open with a fresh Dataset handle (not the writer's own) to
        # prove the file is a complete, standalone artifact on disk.
        with Dataset(path) as nc2:
            assert nc2.dimensions["x"].size == 2
            assert nc2.dimensions["nsurface_fraction"].size == 3

    def test_fill_value_declared_and_observed_via_auto_mask(self, tmp_path):
        # test_round_trip_values_and_fill_placement above reads with
        # nc.set_auto_mask(False), which returns the raw stored values
        # and therefore cannot observe whether the file actually
        # DECLARES a _FillValue attribute for each variable -- it only
        # confirms the literal sentinel landed in the right cells, which
        # is independent of the declaration PALM itself honours. Leaving
        # auto-mask ON (the default, not touched here) is what exercises
        # the declared _FillValue: netCDF4 substitutes a masked entry
        # only where a value matches the variable's own _FillValue.
        path = self._write(tmp_path, with_3d=True, with_lad=True)
        with Dataset(path) as nc:
            assert np.ma.getmaskarray(nc.variables["buildings_2d"][:]).tolist() == [[True, False]]
            assert np.ma.getmaskarray(nc.variables["building_id"][:]).tolist() == [[True, False]]
            assert np.ma.getmaskarray(nc.variables["building_type"][:]).tolist() == [[True, False]]
            assert np.ma.getmaskarray(nc.variables["vegetation_type"][:]).tolist() == [[False, True]]
            assert np.ma.getmaskarray(nc.variables["soil_type"][:]).tolist() == [[False, True]]
            sf_mask = np.ma.getmaskarray(nc.variables["surface_fraction"][:])
            assert sf_mask[:, 0, 0].tolist() == [False, False, False]
            assert sf_mask[:, 0, 1].tolist() == [True, True, True]
            lad_mask = np.ma.getmaskarray(nc.variables["lad"][:])
            assert lad_mask[:, 0, 0].tolist() == [False, False]
            assert lad_mask[:, 0, 1].tolist() == [True, True]


def _write_sample_driver(tmp_path, with_3d=True, with_lad=True):
    """Write a small static driver exercising every field, for tests that
    need a real file on disk without going through TestWriter's own
    fixture method (kept separate so TestWriter's given _write stays
    untouched -- see Task 9's Step 1 test block)."""
    fields = _valid_fields()
    if with_3d:
        b3d = np.zeros((2, 1, 2), dtype=np.int8)
        b3d[:, 0, 1] = 1
        fields["buildings_3d"] = b3d
    if with_lad:
        fields["lad"] = np.full((2, 1, 2), FILL_FLOAT, dtype=np.float32)
        fields["lad"][:, 0, 0] = [0.0, 1.0]
    coords = {
        "x": np.array([1.0, 3.0], dtype=np.float32),
        "y": np.array([1.0], dtype=np.float32),
    }
    if with_3d:
        coords["z"] = np.array([1.0, 3.0], dtype=np.float32)
    if with_lad:
        coords["zlad"] = np.array([0.0, 1.0], dtype=np.float32)
    attrs = {
        "Conventions": "CF-1.7",
        "origin_time": "2000-01-01 00:00:00 +00",
        "origin_lon": 139.7, "origin_lat": 35.68,
        "origin_x": 382000.0, "origin_y": 3949000.0,
        "origin_z": 0.0, "rotation_angle": 0.0,
    }
    path = tmp_path / "dom_static"
    _write_static_driver(path, fields, coords, attrs)
    return path


class TestWriterFieldSpecs:
    """Drive dtype/_FillValue/dims/units/long_name/lod assertions directly
    off _FIELD_SPECS, the same table _write_static_driver iterates (see
    the field-declaration consolidation) -- so a new field, or a changed
    declaration for an existing one, cannot ship without its on-disk
    contract being pinned. test_variable_attributes and
    test_dims_dtypes_and_fills above spot-check a hand-picked subset
    (originally ~10 of ~30 (variable, attribute) pairs, missing long_name
    for 5 of the 11 fields entirely); this is exhaustive over all 11
    fields x 5 properties by construction, not by manual upkeep.
    """

    @pytest.mark.parametrize("name", sorted(_FIELD_SPECS))
    def test_field_matches_its_spec(self, tmp_path, name):
        spec = _FIELD_SPECS[name]
        path = _write_sample_driver(tmp_path)
        with Dataset(path) as nc:
            nc.set_auto_mask(False)
            var = nc.variables[name]
            assert var.dtype == np.dtype(spec.nc_type)
            assert var.dimensions == spec.dims
            assert var._FillValue == np.dtype(spec.nc_type).type(spec.fill)
            assert var.units == spec.units
            assert var.long_name == spec.long_name
            if spec.lod is not None:
                assert var.lod == spec.lod
            else:
                assert not hasattr(var, "lod")

    def test_coordinate_and_index_variable_dtypes(self, tmp_path):
        # _FIELD_SPECS covers the 11 data fields; x/y/z/zlad and
        # nsurface_fraction are structurally different (dimension-
        # defining coordinate/index variables built by the separate
        # coord()/nsf special-case code, not the write() loop) and so
        # were never in scope for that table -- but their on-disk dtype
        # was just as unpinned.
        path = _write_sample_driver(tmp_path)
        with Dataset(path) as nc:
            nc.set_auto_mask(False)
            for name in ("x", "y", "z", "zlad"):
                assert nc.variables[name].dtype == np.float32
                assert nc.variables[name].units == "m"
            assert nc.variables["nsurface_fraction"].dtype == np.int32

    def test_file_format_is_netcdf4(self, tmp_path):
        path = _write_sample_driver(tmp_path)
        with Dataset(path) as nc:
            assert nc.data_model == "NETCDF4"


class TestWriterAttrsGuard:
    def test_none_valued_attr_is_skipped(self, tmp_path):
        # attrs.items() with a None value (e.g. export_palm's optional
        # author/comment) must be skipped, not passed to setattr: netCDF4
        # rejects None outright (TypeError: illegal data type for
        # attribute ..., got O -- confirmed directly), so without the
        # `if value is not None` guard this call would raise instead of
        # writing a file at all.
        fields = _valid_fields()
        coords = {
            "x": np.array([1.0, 3.0], dtype=np.float32),
            "y": np.array([1.0], dtype=np.float32),
        }
        attrs = {"Conventions": "CF-1.7", "author": None}
        path = tmp_path / "dom_static"
        _write_static_driver(path, fields, coords, attrs)  # must not raise
        with Dataset(path) as nc:
            assert "author" not in nc.ncattrs()
            assert nc.Conventions == "CF-1.7"


class TestWriterCompression:
    def test_compression_enabled_and_shrinks_a_sparse_array(self, tmp_path):
        # Measured independently (see the commit message): a 50x200x200
        # buildings_3d array with one small contiguous building block is
        # ~1961 KiB uncompressed and ~13 KiB with zlib's default
        # complevel=4 -- real buildings_3d arrays are exactly this
        # profile (overwhelmingly zero) and real domains are larger, so
        # this is not a synthetic worst case. This array must be large
        # enough that its own compressed size dominates the file (fixed
        # per-file/per-variable netCDF4 overhead from the other ten
        # small fields written alongside it, ~40 KiB, would otherwise
        # swamp the comparison at a much smaller array size).
        fields = _valid_fields()
        nz, ny, nx = 50, 150, 150
        b3d = np.zeros((nz, ny, nx), dtype=np.int8)
        b3d[:5, 10:13, 10:13] = 1  # one small contiguous "building"
        fields["buildings_2d"] = np.full((ny, nx), FILL_FLOAT, dtype=np.float32)
        fields["buildings_2d"][10:13, 10:13] = 5.0
        fields["building_id"] = np.full((ny, nx), FILL_INT, dtype=np.int32)
        fields["building_id"][10:13, 10:13] = 1
        fields["building_type"] = np.full((ny, nx), FILL_BYTE, dtype=np.int8)
        fields["building_type"][10:13, 10:13] = 3
        fields["vegetation_type"] = np.full((ny, nx), FILL_BYTE, dtype=np.int8)
        fields["pavement_type"] = np.full((ny, nx), FILL_BYTE, dtype=np.int8)
        fields["water_type"] = np.full((ny, nx), FILL_BYTE, dtype=np.int8)
        fields["soil_type"] = np.full((ny, nx), FILL_BYTE, dtype=np.int8)
        fields["surface_fraction"] = np.full((3, ny, nx), FILL_FLOAT, dtype=np.float32)
        fields["zt"] = np.zeros((ny, nx), dtype=np.float32)
        fields["buildings_3d"] = b3d
        coords = {
            "x": np.arange(nx, dtype=np.float32),
            "y": np.arange(ny, dtype=np.float32),
            "z": np.arange(nz, dtype=np.float32),
        }
        attrs = {"Conventions": "CF-1.7"}
        path = tmp_path / "dom_static"
        _write_static_driver(path, fields, coords, attrs)

        raw_size = b3d.nbytes  # nz*ny*nx bytes, uncompressed int8 data
        on_disk = path.stat().st_size
        assert on_disk < raw_size / 5, (
            f"expected substantial compression on a sparse array: "
            f"{on_disk} bytes on disk vs {raw_size} bytes of raw data"
        )

        with Dataset(path) as nc:
            assert nc.variables["buildings_3d"].filters()["zlib"] is True
            nc.set_auto_mask(False)
            assert (nc.variables["buildings_3d"][:5, 10:13, 10:13] == 1).all()
            assert (nc.variables["buildings_3d"][5:, :, :] == 0).all()


class TestWriterOverwrite:
    def test_rewriting_the_same_path_truncates_stale_content(self, tmp_path):
        # Decision, pinned: Dataset(path, "w") truncates rather than merging.
        # Kept deliberately (not guarded against) because export_palm always
        # regenerates the full static driver from the current VoxCity model
        # in one call -- there is no partial-update use case -- and every
        # other exporter in this package (cityles.py's `open(filename, 'w')`
        # writers) truncates the same way. A stale variable surviving a
        # re-export (e.g. a leftover `lad` from a run that had canopy, on a
        # domain that no longer does) would be silent wrong data, which is
        # worse than requiring a full re-export.
        fields = _valid_fields()
        fields["lad"] = np.full((1, 1, 2), FILL_FLOAT, dtype=np.float32)
        coords = {
            "x": np.array([1.0, 3.0], dtype=np.float32),
            "y": np.array([1.0], dtype=np.float32),
            "zlad": np.array([0.0], dtype=np.float32),
        }
        attrs = {"Conventions": "CF-1.7"}
        path = tmp_path / "dom_static"
        _write_static_driver(path, fields, coords, attrs)
        with Dataset(path) as nc:
            assert "lad" in nc.variables
            assert "zlad" in nc.dimensions

        fields_no_lad = _valid_fields()  # lad stays None
        _write_static_driver(
            path, fields_no_lad,
            {"x": coords["x"], "y": coords["y"]}, attrs,
        )
        with Dataset(path) as nc:
            assert "lad" not in nc.variables
            assert "zlad" not in nc.dimensions


def make_city(ny=4, nx=4, meshsize=2.0, with_overhang=False, with_canopy=True,
              with_buildings=True, with_orphan_segment=False, extras=None):
    """Small synthetic VoxCity in the internal south-up orientation.

    Cell layout (row, col); valid for any ny, nx >= 3:
    - (0,0): Water, no canopy.
    - (0,1): Road.
    - (0,2): Tree, canopy top 6.0 m over bare ground (no building).
    - (1,1): Building (height 10.0, id 7) when with_buildings; also carries
      a canopy_top value when with_canopy, which must be cleared by the
      building.
    - (2,2): Building (height 6.0; LOD2 segment [4, 6] when with_overhang,
      else [0, 6]) when with_buildings.
    - (ny-2, 0): orphan-segment cell when with_orphan_segment -- no
      recorded LOD1 height, but an LOD2 segment topping at 8 m. Exercises
      the shared building_mask/segment_top_m predicate end-to-end (see
      _build_building_mask's docstring and export_palm's own inline
      comment): without segment_top_m actually reaching both
      _build_buildings and _build_buildings_3d, this cell is invisible as
      a building and buildings_2d stays fill instead of being repaired to
      8.0. Independent of with_buildings (heights stays 0.0 here either
      way).
    - (ny-1, nx-1): Water WITH canopy when with_canopy -- the one cell the
      plan's original fixture never exercised: canopy must NOT override a
      water surface (precedence is building > canopy-over-non-water >
      land-cover mapping; see _build_surface_types' docstring).
    - (ny-1, :): DEM 1.5 m; 0 elsewhere.

    with_buildings=False drops both buildings entirely (heights/ids/
    min_heights/land-cover-Building all revert to "nothing here"), for
    tests that need a domain with no buildings at all.
    """
    meta = GridMetadata(crs="EPSG:4326",
                        bounds=(139.7000, 35.6800, 139.7011, 35.6809),
                        meshsize=meshsize)
    heights = np.zeros((ny, nx))
    ids = np.zeros((ny, nx), dtype=int)
    min_heights = np.empty((ny, nx), dtype=object)
    for i in range(ny):
        for j in range(nx):
            min_heights[i, j] = []
    land_cover = np.zeros((ny, nx), dtype=int)  # 0 = Bareland
    if with_buildings:
        heights[1, 1] = 10.0
        heights[2, 2] = 6.0
        ids[1, 1] = 7
        ids[2, 2] = 8
        min_heights[1, 1] = [[0.0, 10.0]]
        min_heights[2, 2] = [[4.0, 6.0]] if with_overhang else [[0.0, 6.0]]
        land_cover[1, 1] = 12             # Building (has height)
        land_cover[2, 2] = 12
    if with_orphan_segment:
        min_heights[ny - 2, 0] = [[0.0, 8.0]]
    land_cover[0, 0] = 8              # Water
    land_cover[0, 1] = 11             # Road
    land_cover[0, 2] = 5              # Tree
    land_cover[ny - 1, nx - 1] = 8    # Water, also under canopy below
    dem = np.zeros((ny, nx))
    dem[ny - 1, :] = 1.5
    canopy_top = np.zeros((ny, nx))
    if with_canopy:
        canopy_top[0, 2] = 6.0
        canopy_top[1, 1] = 5.0             # over the building -> must be cleared
        canopy_top[ny - 1, nx - 1] = 4.0   # over water -> must not override it
    voxels = VoxelGrid(classes=np.zeros((ny, nx, 8), dtype=np.int32), meta=meta)
    if extras is None:
        extras = {"rectangle_vertices": RECT,
                  "land_cover_source": "OpenStreetMap"}
    return VoxCity(
        voxels=voxels,
        buildings=BuildingGrid(heights=heights, min_heights=min_heights,
                               ids=ids, meta=meta),
        land_cover=LandCoverGrid(classes=land_cover, meta=meta),
        dem=DemGrid(elevation=dem, meta=meta),
        tree_canopy=CanopyGrid(top=canopy_top, bottom=None, meta=meta),
        extras=extras,
    )


class TestExportPalm:
    def test_file_contract(self, tmp_path):
        out = export_palm(make_city(), output_directory=str(tmp_path),
                          domain_name="testdom")
        path = Path(out)
        assert path.name == "testdom_static"
        with Dataset(path) as nc:
            nc.set_auto_mask(False)
            assert nc.dimensions["x"].size == 4
            assert nc.dimensions["y"].size == 4
            # coords are cell centres
            assert nc.variables["x"][:].tolist() == [1.0, 3.0, 5.0, 7.0]
            # georeference
            assert nc.origin_lon == pytest.approx(139.7000)
            assert nc.origin_lat == pytest.approx(35.6800)
            assert abs(nc.rotation_angle) < 0.5
            assert nc.origin_z == pytest.approx(0.0)
            assert nc.origin_time == "2000-01-01 00:00:00 +00"
            # terrain: dem row 3 = 1.5
            assert nc.variables["zt"][3, 0] == pytest.approx(1.5)
            # buildings
            assert nc.variables["buildings_2d"][1, 1] == np.float32(10.0)
            assert nc.variables["building_id"][1, 1] == 7
            assert nc.variables["building_type"][1, 1] == 3
            # no overhang -> LOD1 only, and no dangling z dimension left
            # behind for a variable that was never written
            assert "buildings_3d" not in nc.variables
            assert "z" not in nc.dimensions
            # surface classification
            assert nc.variables["water_type"][0, 0] == 1
            assert nc.variables["pavement_type"][0, 1] == 1
            assert nc.variables["vegetation_type"][0, 2] == 3   # under canopy
            assert nc.variables["vegetation_type"][1, 1] == FILL_BYTE
            # lad: canopy at (0,2) top 6, default ratio 0.3 -> bottom 1.8
            lad = nc.variables["lad"][:]
            zlad = nc.variables["zlad"][:].tolist()
            assert zlad == [0.0, 1.0, 3.0, 5.0]
            assert lad[2, 0, 2] == np.float32(1.0)
            assert lad[1, 0, 2] == np.float32(0.0)
            # canopy over building cleared
            assert (lad[:, 1, 1] == np.float32(FILL_FLOAT)).all()

    def test_canopy_over_water_keeps_water_type(self, tmp_path):
        # The highest-value fixture cell: (ny-1, nx-1) is both water (per
        # land cover) and under canopy. Water must survive unchanged (Unit
        # 4's precedence: building > canopy-over-non-water > land-cover
        # mapping -- canopy never overrides water), while lad still
        # represents the tree independently of the surface type below it.
        # This is the one place the orchestrator's wiring could silently
        # undo that deliberate physics decision.
        out = export_palm(make_city(), output_directory=str(tmp_path))
        with Dataset(out) as nc:
            nc.set_auto_mask(False)
            assert nc.variables["water_type"][3, 3] == 1
            assert nc.variables["vegetation_type"][3, 3] == FILL_BYTE
            assert nc.variables["soil_type"][3, 3] == FILL_BYTE
            sf = nc.variables["surface_fraction"][:, 3, 3]
            assert sf.tolist() == [0.0, 0.0, 1.0]
            # lad still resolves the tree here, independently of the water
            # surface below it.
            assert (nc.variables["lad"][:, 3, 3] != np.float32(FILL_FLOAT)).any()

    def test_exclusivity_invariant_whole_grid(self, tmp_path):
        out = export_palm(make_city(), output_directory=str(tmp_path))
        with Dataset(out) as nc:
            nc.set_auto_mask(False)
            veg = nc.variables["vegetation_type"][:] != FILL_BYTE
            pav = nc.variables["pavement_type"][:] != FILL_BYTE
            wat = nc.variables["water_type"][:] != FILL_BYTE
            bld = nc.variables["buildings_2d"][:] != np.float32(FILL_FLOAT)
            n = veg.astype(int) + pav.astype(int) + wat.astype(int)
            assert (n[~bld] == 1).all()
            assert (n[bld] == 0).all()

    def test_buildings_3d_auto(self, tmp_path):
        out = export_palm(make_city(with_overhang=True),
                          output_directory=str(tmp_path))
        with Dataset(out) as nc:
            nc.set_auto_mask(False)
            b3d = nc.variables["buildings_3d"]
            assert b3d.lod == 2
            assert "z" in nc.dimensions
            col = b3d[:, 2, 2]
            assert col[0] == 0 and col[1] == 0  # below the overhang
            assert col[2] == 1                  # level 2 (4-6 m)
            # z's own values were previously unpinned (only its dimension's
            # existence was checked): z and zlad deliberately use different
            # vertical conventions (z is edge-based, cell index * meshsize;
            # zlad is cell-centre-based -- see _build_lad's docstring), so
            # the one place the two could be confused had only one side
            # (zlad, via test_file_contract) actually pinned to exact
            # values. nz=5 here (heights.max()=10 and the tallest segment
            # top 10 m both give level 5 at meshsize=2.0).
            assert nc.variables["z"][:].tolist() == [1.0, 3.0, 5.0, 7.0, 9.0]

    def test_buildings_3d_forced_off(self, tmp_path):
        out = export_palm(make_city(with_overhang=True), buildings_3d=False,
                          output_directory=str(tmp_path))
        with Dataset(out) as nc:
            assert "buildings_3d" not in nc.variables
            assert "z" not in nc.dimensions

    def test_trunk_ratio_recompute(self, tmp_path):
        out = export_palm(make_city(), trunk_height_ratio=0.5,
                          output_directory=str(tmp_path))
        with Dataset(out) as nc:
            nc.set_auto_mask(False)
            lad = nc.variables["lad"][:]
            # bottom = 3.0 now: z=1 below crown -> 0; z=3 in crown -> 1
            assert lad[1, 0, 2] == np.float32(0.0)
            assert lad[2, 0, 2] == np.float32(1.0)

    def test_missing_rectangle_vertices_raises(self, tmp_path):
        city = make_city(extras={"land_cover_source": "OpenStreetMap"})
        with pytest.raises(ValueError, match="rectangle_vertices"):
            export_palm(city, output_directory=str(tmp_path))
        # The pre-check is a ValueError raised before _validate_static_fields
        # and before the writer -- nothing must land on disk.
        assert list(tmp_path.iterdir()) == []

    def test_shape_mismatch_raises(self, tmp_path):
        city = make_city()
        city.dem.elevation = np.zeros((2, 2))
        with pytest.raises(ValueError, match="shape"):
            export_palm(city, output_directory=str(tmp_path))
        assert list(tmp_path.iterdir()) == []

    def test_voxel_grid_shape_mismatch_raises(self, tmp_path):
        # The other half of the pre-check pair: a mismatched voxel grid
        # horizontal shape, distinct from the 2D-grid case above.
        city = make_city()
        city.voxels.classes = np.zeros((2, 2, 8), dtype=np.int32)
        with pytest.raises(ValueError, match="shape"):
            export_palm(city, output_directory=str(tmp_path))
        assert list(tmp_path.iterdir()) == []

    def test_creates_missing_nested_output_directory(self, tmp_path):
        # Without makedirs(..., exist_ok=True), a missing parent surfaces as
        # netCDF4's confusing PermissionError [Errno 13] on Windows.
        nested = tmp_path / "a" / "b" / "c"
        assert not nested.exists()
        out = export_palm(make_city(), output_directory=str(nested))
        assert Path(out).exists()

    def test_validator_failure_prevents_file_creation(self, tmp_path, monkeypatch):
        # _validate_static_fields has no other caller -- this proves
        # export_palm actually wires it in, and that a failure there runs
        # before the writer, not just that the standalone validator raises
        # in isolation (already covered by TestValidator).
        def boom(fields):
            raise RuntimeError("boom")

        monkeypatch.setattr(palm_module, "_validate_static_fields", boom)
        with pytest.raises(RuntimeError, match="boom"):
            export_palm(make_city(), output_directory=str(tmp_path))
        assert list(tmp_path.iterdir()) == []

    def test_unexpected_keyword_raises(self, tmp_path):
        # export_palm deliberately takes no **kwargs (see PalmExporter's
        # docstring for the reasoning): a typo'd keyword must fail loudly
        # rather than being silently dropped, especially with 14 parameters.
        with pytest.raises(TypeError):
            export_palm(make_city(), output_directory=str(tmp_path),
                        buildings3d=True)

    def test_non_square_grid_coords_and_fields_agree(self, tmp_path):
        # ny != nx: a coordinate/field shape mix-up (e.g. sizing "x" off ny
        # instead of nx) would slip through a square fixture unnoticed but
        # is directly observable here.
        out = export_palm(make_city(ny=3, nx=5), output_directory=str(tmp_path))
        with Dataset(out) as nc:
            nc.set_auto_mask(False)
            assert nc.dimensions["x"].size == 5
            assert nc.dimensions["y"].size == 3
            assert nc.variables["x"][:].tolist() == [1.0, 3.0, 5.0, 7.0, 9.0]
            assert nc.variables["y"][:].tolist() == [1.0, 3.0, 5.0]
            assert nc.variables["buildings_2d"][1, 1] == np.float32(10.0)
            assert nc.variables["buildings_2d"][2, 2] == np.float32(6.0)


class TestPalmExporterAdapter:
    def test_export_via_adapter(self, tmp_path):
        exporter = PalmExporter()
        out = exporter.export(make_city(), str(tmp_path), "mydom")
        assert Path(out).name == "mydom_static"
        # Not just "the file exists somewhere": a mutant hardcoding
        # "output/palm" as the write location would still create a file
        # that exists, just in the wrong place (the repo's own working
        # directory) -- caught only by pinning the actual parent.
        assert Path(out).parent == tmp_path
        assert Path(out).exists()

    def test_rejects_non_voxcity(self, tmp_path):
        with pytest.raises(TypeError):
            PalmExporter().export(object(), str(tmp_path), "x")

    def test_registered_in_package(self):
        import voxcity.exporter as ex
        assert "PalmExporter" in ex.__all__
        assert "export_palm" in ex.__all__
        assert ex.PalmExporter is PalmExporter
        # Dropping "export_palm" from __all__ (while leaving the name
        # itself defined) survived without this: the two lines above only
        # ever checked __all__'s contents, never that the star-imported
        # name in voxcity.exporter actually resolves to this module's
        # export_palm.
        assert ex.export_palm is export_palm

    def test_forwards_kwargs_to_export_palm(self, tmp_path):
        # Distinctive, non-default building_type (5, not the default 3)
        # proves kwargs are actually forwarded to export_palm, not merely
        # accepted by the adapter and dropped.
        out = PalmExporter().export(make_city(), str(tmp_path), "mydom",
                                    building_type=5)
        with Dataset(out) as nc:
            assert nc.variables["building_type"][1, 1] == 5

    def test_unexpected_keyword_raises_via_adapter(self, tmp_path):
        # PalmExporter.export's own **kwargs is pure pass-through: an
        # unrecognised keyword must still fail (from export_palm's
        # signature), not be silently absorbed by the adapter.
        with pytest.raises(TypeError):
            PalmExporter().export(make_city(), str(tmp_path), "mydom",
                                  buildings3d=True)


class TestExportPalmCanopyBottomResolution:
    """Pins all four branches of the canopy-bottom resolution logic
    (export_palm's replica of export_cityles' sentinel semantics: see
    export_palm's own docstring) and the precedence between them.

    Uses meshsize=1.0 (finer than the 2.0 m default fixture) so canopy_top
    (fixed at 6.0 m by make_city's own (0,2) cell) yields
    zlad=[0, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5], and each branch's resolved
    bottom is chosen so its crown-level COUNT is distinct from every other
    branch's: explicit ratio 0.75 -> bottom 4.5 (2 levels in crown),
    explicit canopy_bottom_height_grid -> bottom 3.0 (3 levels),
    city.tree_canopy.bottom -> bottom 0.8 (5 levels), default ratio 0.3 ->
    bottom 1.8 (4 levels). A regression collapsing one branch into another
    (or into the wrong precedence order) is therefore observable from the
    crown-level count alone, without needing to inspect the raw bottom
    value directly.
    """

    def _crown_count(self, path):
        with Dataset(path) as nc:
            nc.set_auto_mask(False)
            lad = nc.variables["lad"][:]
            return int((lad[:, 0, 2] == np.float32(1.0)).sum())

    def test_branch1_explicit_ratio_used(self, tmp_path):
        out = export_palm(make_city(meshsize=1.0), trunk_height_ratio=0.75,
                          output_directory=str(tmp_path))
        assert self._crown_count(out) == 2

    def test_branch2_explicit_grid_used_when_no_ratio(self, tmp_path):
        city = make_city(meshsize=1.0)
        grid = np.zeros_like(city.tree_canopy.top)
        grid[0, 2] = 3.0
        out = export_palm(city, canopy_bottom_height_grid=grid,
                          output_directory=str(tmp_path))
        assert self._crown_count(out) == 3

    def test_branch3_model_bottom_used_when_no_ratio_or_grid(self, tmp_path):
        city = make_city(meshsize=1.0)
        bottom = np.zeros_like(city.tree_canopy.top)
        bottom[0, 2] = 0.8
        city.tree_canopy.bottom = bottom
        out = export_palm(city, output_directory=str(tmp_path))
        assert self._crown_count(out) == 5

    def test_branch4_default_ratio_used_when_nothing_else_given(self, tmp_path):
        out = export_palm(make_city(meshsize=1.0), output_directory=str(tmp_path))
        assert self._crown_count(out) == 4

    def test_explicit_ratio_beats_explicit_grid_and_model_bottom(self, tmp_path):
        # All three optional inputs given at once: the explicit ratio must
        # win over both of the others simultaneously.
        city = make_city(meshsize=1.0)
        grid = np.zeros_like(city.tree_canopy.top)
        grid[0, 2] = 3.0
        bottom = np.zeros_like(city.tree_canopy.top)
        bottom[0, 2] = 0.8
        city.tree_canopy.bottom = bottom
        out = export_palm(city, trunk_height_ratio=0.75,
                          canopy_bottom_height_grid=grid,
                          output_directory=str(tmp_path))
        assert self._crown_count(out) == 2  # branch 1's pattern, not 3 or 5

    def test_explicit_ratio_beats_model_bottom_alone(self, tmp_path):
        city = make_city(meshsize=1.0)
        bottom = np.zeros_like(city.tree_canopy.top)
        bottom[0, 2] = 0.8
        city.tree_canopy.bottom = bottom
        out = export_palm(city, trunk_height_ratio=0.75,
                          output_directory=str(tmp_path))
        assert self._crown_count(out) == 2  # branch 1's pattern, not 5

    def test_explicit_grid_beats_model_bottom(self, tmp_path):
        # No ratio; grid and model bottom both given: the explicit grid
        # must win.
        city = make_city(meshsize=1.0)
        grid = np.zeros_like(city.tree_canopy.top)
        grid[0, 2] = 3.0
        bottom = np.zeros_like(city.tree_canopy.top)
        bottom[0, 2] = 0.8
        city.tree_canopy.bottom = bottom
        out = export_palm(city, canopy_bottom_height_grid=grid,
                          output_directory=str(tmp_path))
        assert self._crown_count(out) == 3  # branch 2's pattern, not 5


class TestExportPalmParameterSurface:
    """Every optional parameter must actually reach the written file, not
    merely be accepted and dropped. test_file_contract (TestExportPalm)
    only ever exercises defaults (origin_time, under_tree_vegetation_type
    at its default value 3, etc.), which cannot distinguish "the parameter
    was read" from "the parameter was ignored and the default happened to
    be the answer anyway" -- see this module's non-vacuity convention.
    """

    def test_distinctive_non_default_parameters_land_in_file(self, tmp_path):
        city = make_city()
        # DEM minimum shifted well off zero so origin_z is observably
        # non-zero and non-default.
        city.dem.elevation = city.dem.elevation + 100.0
        out = export_palm(
            city,
            output_directory=str(tmp_path),
            origin_time="2020-06-15 12:00:00 +00",
            lad=2.5,
            # 13 and 6 are chosen to differ from: the default (3), each
            # other, and the land-cover mapping's own code for the (0,2)
            # Tree cell (OSM 'Tree' -> vegetation 7) -- so a passing
            # assertion can only mean the parameter was actually used to
            # override the canopy cell, not that the land-cover code or
            # the default leaked through by coincidence.
            under_tree_vegetation_type=13,
            soil_type=6,
            author="Test Author",
            comment="distinctive comment",
        )
        with Dataset(out) as nc:
            nc.set_auto_mask(False)
            assert nc.origin_time == "2020-06-15 12:00:00 +00"
            assert nc.author == "Test Author"
            assert "distinctive comment" in nc.comment
            assert nc.origin_z == pytest.approx(100.0)
            # (0,2): Tree, under canopy, non-building.
            assert nc.variables["vegetation_type"][0, 2] == 13
            assert nc.variables["soil_type"][0, 2] == 6
            # lad: z=3 sits inside the default-ratio crown (bottom 1.8) at
            # (0,2) -- must carry the passed lad value, not the default 1.0.
            lad = nc.variables["lad"][:]
            assert lad[2, 0, 2] == np.float32(2.5)

    def test_explicit_land_cover_source_overrides_extras(self, tmp_path):
        # extras['land_cover_source'] is 'OpenStreetMap' by default (see
        # make_city); passing a different source explicitly must win (the
        # resolution is `land_cover_source or extras.get(...)`, matching
        # export_cityles). OSM and Urbanwatch disagree on raw index 0:
        # OSM's is 'Bareland' (-> vegetation 1); Urbanwatch's is 'Building'
        # with no recorded height here (-> pavement 2, per the "Building
        # class without height becomes pavement" rule), so the two sources
        # produce observably different output for the same raw grid.
        # (ny-1, 0) is plain Bareland (raw index 0), not water/canopy/
        # building, in the default fixture.
        out = export_palm(make_city(), land_cover_source="Urbanwatch",
                          output_directory=str(tmp_path))
        with Dataset(out) as nc:
            nc.set_auto_mask(False)
            assert nc.variables["vegetation_type"][3, 0] == FILL_BYTE
            assert nc.variables["pavement_type"][3, 0] == 2


class TestExportPalmBuildings3dDecision:
    """The buildings_3d="auto"/True/False decision has two independent
    conditions (`buildings_3d is True` and `building_mask.any()`) ANDed
    with the "auto" branch's own condition; TestExportPalm's existing
    coverage (test_buildings_3d_auto, test_buildings_3d_forced_off) only
    ever exercises "auto" and False, leaving the `is True` branch and the
    building_mask.any() guard both unpinned.
    """

    def test_forced_true_without_elevated_segments_still_emits_lod2(self, tmp_path):
        # buildings_3d="auto" would NOT build a LOD2 mask here (no segment
        # starts above ground -- with_overhang=False), isolating the
        # `buildings_3d is True` branch from "auto" entirely.
        out = export_palm(make_city(with_overhang=False), buildings_3d=True,
                          output_directory=str(tmp_path))
        with Dataset(out) as nc:
            assert "buildings_3d" in nc.variables
            assert nc.variables["buildings_3d"].lod == 2

    def test_forced_true_with_no_buildings_at_all_emits_nothing(self, tmp_path):
        # want_3d is True, but building_mask.any() is False: nothing for a
        # LOD2 mask to represent, so it must not be written even though the
        # caller explicitly asked for it.
        out = export_palm(make_city(with_buildings=False), buildings_3d=True,
                          output_directory=str(tmp_path))
        with Dataset(out) as nc:
            assert "buildings_3d" not in nc.variables
            assert "z" not in nc.dimensions


class TestExportPalmAdditionalShapeChecks:
    """Three inputs export_palm reads but did not shape-check: a
    mismatched buildings.ids raises a raw numpy broadcast ValueError deep
    in _build_buildings; a mismatched buildings.min_heights or
    canopy_bottom_height_grid each raise IndexError (not even ValueError)
    deep in _build_buildings_3d/_build_lad's per-cell loops -- confirmed
    directly against the unpatched module before adding these checks.
    zt/DEM is already covered via test_shape_mismatch_raises.
    """

    def test_ids_shape_mismatch_raises(self, tmp_path):
        # match= is the specific "buildings.ids grid shape" prefix this
        # module's own pre-check produces, not just "shape" -- the raw
        # numpy broadcast error this check replaces
        # ("operands could not be broadcast together with shapes ...") is
        # also a ValueError and its message also contains the substring
        # "shape" (inside "shapes"), so a loose match="shape" pattern would
        # pass whether or not the dedicated pre-check actually ran.
        city = make_city()
        city.buildings.ids = np.zeros((2, 2), dtype=int)
        with pytest.raises(ValueError, match="buildings.ids grid shape"):
            export_palm(city, output_directory=str(tmp_path))
        assert list(tmp_path.iterdir()) == []

    def test_min_heights_shape_mismatch_raises(self, tmp_path):
        city = make_city()
        mh = np.empty((2, 2), dtype=object)
        for i in range(2):
            for j in range(2):
                mh[i, j] = []
        city.buildings.min_heights = mh
        with pytest.raises(ValueError, match="buildings.min_heights grid shape"):
            export_palm(city, output_directory=str(tmp_path))
        assert list(tmp_path.iterdir()) == []

    def test_canopy_bottom_height_grid_shape_mismatch_raises(self, tmp_path):
        with pytest.raises(ValueError, match="canopy_bottom_height_grid grid shape"):
            export_palm(make_city(), canopy_bottom_height_grid=np.zeros((2, 2)),
                        output_directory=str(tmp_path))
        assert list(tmp_path.iterdir()) == []


# Independently computed (NOT copy-pasted from export_palm's own
# _build_georeference call): a swapped origin_x/origin_y in the attrs dict
# relocates the simulated domain by ~3.5 million metres for this fixture,
# a silently catastrophic scientific error that 177 tests previously did
# not catch (the adjacent origin_lon/origin_lat swap already was caught,
# which is what made the gap look like an oversight rather than a
# decision). Comparing against a value nothing in export_palm's own code
# path produced is the whole point.
_EXPECTED_ORIGIN_X, _EXPECTED_ORIGIN_Y = Transformer.from_crs(
    "EPSG:4326", "EPSG:32654", always_xy=True
).transform(139.7000, 35.6800)

# Every PIDS global attribute export_palm writes with a value independent
# of wall-clock time, parametrized rather than asserted one-by-one -- see
# the module's non-vacuity convention and this unit's review history
# (_FIELD_SPECS, then the parameter surface, now this): a flat table
# tested entry-by-entry is exactly the shape that lets one entry go
# unpinned indefinitely.
_EXPECTED_GLOBAL_ATTRS = [
    ("Conventions", "CF-1.7"),
    ("title", "VoxCity PALM static driver: testdom"),
    ("source", "VoxCity"),
    ("origin_time", "2000-01-01 00:00:00 +00"),
    ("origin_lon", pytest.approx(139.7000)),
    ("origin_lat", pytest.approx(35.6800)),
    ("origin_x", pytest.approx(_EXPECTED_ORIGIN_X)),
    ("origin_y", pytest.approx(_EXPECTED_ORIGIN_Y)),
    ("origin_z", pytest.approx(0.0)),
]

# Attribute names present in the written file but excluded from
# _EXPECTED_GLOBAL_ATTRS because they need their own check, not a single
# expected value: creation_time varies with wall-clock time (format only),
# comment is a formatted string built from origin_x/origin_y's own EPSG
# zone (prefix only), and rotation_angle is only near-zero for this
# axis-aligned fixture, not exactly zero.
_GLOBAL_ATTRS_WITH_DEDICATED_CHECKS = {"creation_time", "comment", "rotation_angle"}


class TestExportPalmGlobalAttributes:
    """The full PIDS global-attribute block was previously unpinned: not
    one of the 177 tests as of the last review round asserted origin_x,
    origin_y, Conventions, title, source, or creation_time, and swapping
    origin_x/origin_y in the attrs dict -- which relocates the PALM
    simulation domain by thousands of kilometres -- passed every test.
    """

    @pytest.fixture(scope="class")
    def static_path(self, tmp_path_factory):
        out_dir = tmp_path_factory.mktemp("palm_attrs")
        return export_palm(make_city(), output_directory=str(out_dir),
                           domain_name="testdom")

    @pytest.fixture(scope="class")
    def nc_attrs(self, static_path):
        with Dataset(static_path) as nc:
            return {name: getattr(nc, name) for name in nc.ncattrs()}

    @pytest.mark.parametrize("name, expected", _EXPECTED_GLOBAL_ATTRS,
                             ids=[name for name, _ in _EXPECTED_GLOBAL_ATTRS])
    def test_attribute_matches_independent_expectation(self, nc_attrs, name, expected):
        assert nc_attrs[name] == expected

    def test_every_written_attribute_is_accounted_for(self, nc_attrs):
        # Catches what the parametrized cases above cannot: a mutant
        # introducing a brand-new, silently-unpinned attribute, or one
        # that drops author when it IS provided (author/comment aren't
        # both always present -- author is None-guarded away by the
        # writer when not passed, matching this call's defaults).
        expected_names = (
            {name for name, _ in _EXPECTED_GLOBAL_ATTRS}
            | _GLOBAL_ATTRS_WITH_DEDICATED_CHECKS
        )
        assert set(nc_attrs) == expected_names

    def test_comment_names_the_epsg_zone(self, nc_attrs):
        assert nc_attrs["comment"].startswith("origin_x/origin_y in EPSG:32654")

    def test_creation_time_matches_documented_format(self, nc_attrs):
        assert re.match(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} \+00$",
                        nc_attrs["creation_time"])

    def test_rotation_angle_is_near_zero_for_axis_aligned_rect(self, nc_attrs):
        assert abs(nc_attrs["rotation_angle"]) < 0.5

    def test_author_and_comment_suffix_appear_when_provided(self, tmp_path):
        # author/comment are None-guarded out of the file when absent
        # (TestWriterAttrsGuard already pins that at the writer level);
        # this is the orchestrator-level proof that passing them actually
        # reaches the written file.
        out = export_palm(make_city(), output_directory=str(tmp_path),
                          author="Test Author", comment="extra detail")
        with Dataset(out) as nc:
            assert nc.author == "Test Author"
            assert nc.comment.startswith("origin_x/origin_y in EPSG:32654")
            assert nc.comment.endswith("; extra detail")


class TestExportPalmUserInputValueChecks:
    """building_type/soil_type/under_tree_vegetation_type/lad/meshsize/
    buildings_3d previously escaped the ValueError pre-check boundary
    entirely: an out-of-range class code either reached
    _validate_static_fields (a RuntimeError telling the user the exporter
    itself is broken, when they typed the bad value) or, for
    building_type=200 specifically, raised a raw OverflowError from the
    np.int8 cast -- exactly the class of confusing failure the pre-check
    block exists to eliminate. meshsize=0 raised ZeroDivisionError and
    meshsize<0 exported "successfully" with negative coordinates.
    """

    @pytest.mark.parametrize("kwarg, value", [
        ("building_type", 99),
        ("building_type", 200),   # the exact OverflowError case reported
        ("building_type", 0),     # below range, not just above
        ("soil_type", 99),
        ("under_tree_vegetation_type", 99),
    ])
    def test_out_of_range_class_code_raises_value_error(self, tmp_path, kwarg, value):
        with pytest.raises(ValueError, match=f"{kwarg}={value}"):
            export_palm(make_city(), output_directory=str(tmp_path), **{kwarg: value})
        assert list(tmp_path.iterdir()) == []

    def test_negative_lad_raises(self, tmp_path):
        with pytest.raises(ValueError, match="lad"):
            export_palm(make_city(), output_directory=str(tmp_path), lad=-1.0)
        assert list(tmp_path.iterdir()) == []

    def test_non_finite_lad_raises(self, tmp_path):
        with pytest.raises(ValueError, match="lad"):
            export_palm(make_city(), output_directory=str(tmp_path), lad=float("nan"))
        assert list(tmp_path.iterdir()) == []

    def test_zero_meshsize_raises_instead_of_zero_division(self, tmp_path):
        city = make_city()
        city.voxels.meta.meshsize = 0.0
        with pytest.raises(ValueError, match="meshsize"):
            export_palm(city, output_directory=str(tmp_path))
        assert list(tmp_path.iterdir()) == []

    def test_negative_meshsize_raises_instead_of_exporting(self, tmp_path):
        city = make_city()
        city.voxels.meta.meshsize = -2.0
        with pytest.raises(ValueError, match="meshsize"):
            export_palm(city, output_directory=str(tmp_path))
        assert list(tmp_path.iterdir()) == []

    @pytest.mark.parametrize("bad_value", ["yes", "Auto", "AUTO", 1, 0])
    def test_unrecognised_buildings_3d_value_raises(self, tmp_path, bad_value):
        # 1 and 0 specifically: `buildings_3d in (True, False, "auto")`
        # would silently accept both (bool is an int subtype, so 1 == True
        # and 0 == False) -- confirmed directly before this was written as
        # an isinstance-based check instead. Without the fix, buildings_3d
        #="yes"/"Auto" already export with no 3D mask and no warning
        # (`"Auto" is True` is False), directly contradicting the loud-
        # failure stance PalmExporter's own docstring argues for.
        with pytest.raises(ValueError, match="buildings_3d"):
            export_palm(make_city(), output_directory=str(tmp_path),
                        buildings_3d=bad_value)
        assert list(tmp_path.iterdir()) == []

    def test_valid_buildings_3d_values_still_work(self, tmp_path):
        # Sanity check alongside the invalid-value tests above: True,
        # False, and "auto" must NOT raise.
        for value in (True, False, "auto"):
            out_dir = tmp_path / str(value)
            export_palm(make_city(), output_directory=str(out_dir),
                        buildings_3d=value)  # must not raise


class TestExportPalmAtomicWrite:
    """netCDF4's Dataset(path, "w") truncates its target immediately, so a
    failure partway through a re-export would previously destroy a
    previously-valid, stable <domain_name>_static file -- replacing good
    data with something that opens cleanly as NetCDF4 but has zero
    variables. export_palm now writes to a temp sibling and os.replace()s
    it into place only on success.
    """

    def test_failed_write_does_not_destroy_existing_good_file(self, tmp_path, monkeypatch):
        good_out = export_palm(make_city(), output_directory=str(tmp_path))
        original_bytes = Path(good_out).read_bytes()

        def crash_mid_write(path, fields, coords, attrs):
            # Simulates netCDF4 crashing after creating the file but before
            # finishing writing it -- the exact failure mode the reviewer
            # reproduced (a 3 KB file that opens as valid, empty NetCDF4).
            Path(path).write_bytes(b"garbage-partial-write")
            raise RuntimeError("simulated mid-write crash")

        monkeypatch.setattr(palm_module, "_write_static_driver", crash_mid_write)
        with pytest.raises(RuntimeError, match="simulated mid-write crash"):
            export_palm(make_city(), output_directory=str(tmp_path))

        # The previously-written good file must be completely untouched.
        assert Path(good_out).read_bytes() == original_bytes
        # No stray temp file left behind either.
        assert list(Path(tmp_path).glob("*.tmp")) == []

    def test_failed_write_on_first_export_leaves_no_stray_file(self, tmp_path, monkeypatch):
        # Same crash, but with no prior good file: the directory must end
        # up completely clean, not holding a partial/garbage file under
        # the final <domain_name>_static name or any temp name.
        def crash_mid_write(path, fields, coords, attrs):
            Path(path).write_bytes(b"garbage-partial-write")
            raise RuntimeError("simulated mid-write crash")

        monkeypatch.setattr(palm_module, "_write_static_driver", crash_mid_write)
        with pytest.raises(RuntimeError, match="simulated mid-write crash"):
            export_palm(make_city(), output_directory=str(tmp_path))
        assert list(tmp_path.iterdir()) == []


# (name, mutator) pairs mirroring palm_module._SHAPE_CHECKED_GRID_ACCESSORS'
# city-derived entries (canopy_bottom_height_grid is a call-time kwarg, not
# a city attribute, and has its own dedicated test above), each shrinking
# that one grid to an incompatible shape.
def _small_min_heights():
    mh = np.empty((2, 2), dtype=object)
    for i in range(2):
        for j in range(2):
            mh[i, j] = []
    return mh


_SHAPE_CHECK_MUTATORS = {
    "land_cover": lambda city: setattr(city.land_cover, "classes",
                                       np.zeros((2, 2), dtype=int)),
    "dem": lambda city: setattr(city.dem, "elevation", np.zeros((2, 2))),
    "canopy_top": lambda city: setattr(city.tree_canopy, "top", np.zeros((2, 2))),
    "buildings.ids": lambda city: setattr(city.buildings, "ids",
                                          np.zeros((2, 2), dtype=int)),
    "buildings.min_heights": lambda city: setattr(city.buildings, "min_heights",
                                                   _small_min_heights()),
}


class TestExportPalmShapePreChecksParametrized:
    """Parametrized over every city-derived grid export_palm shape-checks,
    driven by the same _SHAPE_CHECKED_GRID_ACCESSORS table production
    reads from -- so a newly added grid is covered by construction rather
    than by someone remembering to add a one-off test. Before this class,
    2 of the 6 named_grids checks (land_cover, canopy_top) had no
    dedicated test anywhere in this module: only dem, buildings.ids,
    buildings.min_heights, and canopy_bottom_height_grid did.
    """

    def test_mutator_table_matches_every_city_derived_grid_the_production_code_checks(self):
        # If a new grid is added to palm_module._SHAPE_CHECKED_GRID_ACCESSORS
        # without a matching entry in _SHAPE_CHECK_MUTATORS, this fails --
        # closing the loop so this test class cannot silently fall behind
        # production's own list of checked grids.
        city_derived_names = {
            name for name, _ in _SHAPE_CHECKED_GRID_ACCESSORS
            if name != "canopy_bottom_height_grid"
        }
        assert set(_SHAPE_CHECK_MUTATORS) == city_derived_names

    @pytest.mark.parametrize("name", sorted(_SHAPE_CHECK_MUTATORS))
    def test_mismatched_grid_raises_naming_it(self, tmp_path, name):
        # match= is the exact "<name> grid shape" prefix
        # _check_export_inputs' own named_grids loop produces, not just
        # the bare grid name: for "land_cover" specifically, a bare-name
        # match is satisfiable by a completely different, coincidental
        # source -- _build_surface_types has its own internal shape guard
        # whose message happens to contain the substring "land_cover_grid"
        # too, so a bare match="land_cover" would still pass even with
        # _check_export_inputs' own land_cover check deleted (confirmed
        # directly). The full prefix is unique to this pre-check.
        city = make_city()
        _SHAPE_CHECK_MUTATORS[name](city)
        with pytest.raises(ValueError, match=re.escape(f"{name} grid shape")):
            export_palm(city, output_directory=str(tmp_path))
        assert list(tmp_path.iterdir()) == []


class TestExportPalmMiscCoverage:
    """Smaller gaps found in the same review pass: each is cheap and
    isolated, so grouped in one class rather than scattered.
    """

    def test_default_trunk_height_ratio_constant_is_0_3(self):
        # Direct assertion on the named constant, not just inferable
        # through crown geometry (TestExportPalmCanopyBottomResolution's
        # branch4 test already proves the constant is actually USED; this
        # proves the constant itself is the documented 0.3, matching
        # export_cityles's own inline default).
        assert palm_module._DEFAULT_TRUNK_HEIGHT_RATIO == 0.3

    def test_missing_land_cover_source_in_extras_falls_back_to_standard(self, tmp_path):
        # extras has no 'land_cover_source' key at all (distinct from
        # make_city's own default extras, which always sets one) --
        # exercises the `extras.get("land_cover_source", "Standard")`
        # fallback, and 'Standard' must itself resolve without raising
        # (it aliases OpenStreetMap; see _get_source_name_mapping).
        city = make_city(extras={"rectangle_vertices": RECT})
        out = export_palm(city, output_directory=str(tmp_path))  # must not raise
        with Dataset(out) as nc:
            nc.set_auto_mask(False)
            # OSM/Standard raw index 0 is Bareland -> vegetation code 1.
            assert nc.variables["vegetation_type"][3, 0] == 1

    def test_none_extras_treated_as_missing_rectangle_vertices(self, tmp_path):
        # city.extras is a plain dict by the VoxCity dataclass's own
        # default_factory, but nothing stops a caller from setting it to
        # None outright; `extras = city.extras or {}` must degrade to the
        # ordinary "missing rectangle_vertices" ValueError rather than
        # crashing with AttributeError on `None.get(...)`.
        city = make_city()
        city.extras = None
        with pytest.raises(ValueError, match="rectangle_vertices"):
            export_palm(city, output_directory=str(tmp_path))
        assert list(tmp_path.iterdir()) == []

    def test_inf_canopy_top_reads_as_no_canopy_not_spuriously_covered(self, tmp_path):
        # (3,2) is plain Bareland in the default fixture, untouched by
        # with_canopy -- setting it to +inf here is a deliberate, isolated
        # mutation exercising canopy_present's nan_to_num sanitization (see
        # export_palm's own inline comment above canopy_present). +inf, not
        # NaN: `np.nan > 0.0` is already False in plain numpy (NaN
        # comparisons are always False), so a NaN cell reads as "no
        # canopy" whether or not nan_to_num runs at all -- it cannot
        # distinguish this line from a no-op (confirmed directly: removing
        # the nan_to_num call still passed a NaN-based version of this
        # test). +inf is the case that actually flips: `np.inf > 0.0` is
        # True, so without sanitization this cell would be spuriously
        # counted as canopy-covered here while _build_lad -- which
        # sanitizes its own `top` input the same way -- treats it as bare,
        # the exact disagreement the shared sanitization exists to prevent.
        city = make_city()
        city.tree_canopy.top[3, 2] = np.inf
        out = export_palm(city, output_directory=str(tmp_path))  # must not raise
        with Dataset(out) as nc:
            nc.set_auto_mask(False)
            # Bareland's own code (1), NOT under_tree_vegetation_type's
            # default (3) -- proving the +inf cell was excluded from
            # canopy_mask rather than spuriously counted as canopy-covered.
            assert nc.variables["vegetation_type"][3, 2] == 1

    def test_no_tree_canopy_model_treated_as_no_canopy(self, tmp_path):
        # city.tree_canopy is typed as required (CanopyGrid, not Optional)
        # but nothing enforces that at runtime; `canopy_top = ... if
        # city.tree_canopy is not None else None` and the canopy-bottom
        # resolution's own `city.tree_canopy is not None` guard both cover
        # this, and both are exercised by the same single city here.
        city = make_city(with_canopy=False)
        city.tree_canopy = None
        out = export_palm(city, output_directory=str(tmp_path))  # must not raise
        with Dataset(out) as nc:
            assert "lad" not in nc.variables

    def test_domain_name_with_path_separator_creates_nested_directory(self, tmp_path):
        # Regression for the makedirs fix: makedirs was keyed on
        # output_directory, not on out_path.parent, so a domain_name
        # containing a path separator hit the same PermissionError
        # [Errno 13] the makedirs call exists to prevent.
        out = export_palm(make_city(), output_directory=str(tmp_path),
                          domain_name="a/b")
        assert Path(out) == tmp_path / "a" / "b_static"
        assert Path(out).exists()

    def test_orphan_segment_cell_is_repaired_end_to_end(
        self, tmp_path, caplog, propagate_voxcity_logs
    ):
        # The shared building_mask/segment_top_m predicate
        # (_build_building_mask's docstring, and export_palm's own inline
        # comment above building_mask) had no end-to-end test: make_city's
        # existing buildings all have heights >= their segment tops, so
        # passing np.zeros(shape) as segment_top_m to either builder, or
        # building the mask from heights alone, both previously survived.
        with caplog.at_level(logging.WARNING, logger="voxcity"):
            out = export_palm(make_city(with_orphan_segment=True),
                              output_directory=str(tmp_path))
        with Dataset(out) as nc:
            nc.set_auto_mask(False)
            # (ny-2, 0) = (2, 0) for the default 4x4 fixture.
            assert nc.variables["buildings_2d"][2, 0] == np.float32(8.0)
            assert nc.variables["building_id"][2, 0] != FILL_INT
            assert nc.variables["building_type"][2, 0] == 3  # default building_type
        assert "repair" in caplog.text.lower()
