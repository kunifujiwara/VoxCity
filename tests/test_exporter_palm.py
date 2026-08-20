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
    _build_georeference,
    _build_index_to_palm_map,
    _get_source_name_mapping,
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
