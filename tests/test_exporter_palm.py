"""Tests for voxcity.exporter.palm (PALM static driver exporter)."""

import numpy as np
import pytest
from pathlib import Path

from voxcity.exporter.palm import (
    FILL_FLOAT,
    FILL_INT,
    FILL_BYTE,
    OSM_CLASS_TO_PALM,
    URBANWATCH_CLASS_TO_PALM,
    OEMJ_CLASS_TO_PALM,
    ESA_CLASS_TO_PALM,
    ESRI_CLASS_TO_PALM,
    DYNAMIC_WORLD_CLASS_TO_PALM,
)

ALL_TABLES = [
    OSM_CLASS_TO_PALM,
    URBANWATCH_CLASS_TO_PALM,
    OEMJ_CLASS_TO_PALM,
    ESA_CLASS_TO_PALM,
    ESRI_CLASS_TO_PALM,
    DYNAMIC_WORLD_CLASS_TO_PALM,
]

VALID_RANGES = {"vegetation": (1, 18), "pavement": (1, 15), "water": (1, 5)}


class TestMappingTables:
    def test_fill_values(self):
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

    def test_osm_covers_all_standard_classes(self):
        from voxcity.utils.lc import get_land_cover_classes
        names = list(get_land_cover_classes("OpenStreetMap").values())
        assert set(names) == set(OSM_CLASS_TO_PALM.keys())

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
