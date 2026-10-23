"""Tests for voxcity.downloader.aws_opendata (offline)."""
import os

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from voxcity.downloader.aws_opendata import (
    CHMV2_BASE_URL,
    _quadkey,
    _quadkeys_for_bbox,
)


class TestQuadkey:
    def test_zoom1_corners(self):
        # Bing quadkey digits: 0=NW, 1=NE, 2=SW, 3=SE
        assert _quadkey(0, 0, 1) == "0"
        assert _quadkey(1, 0, 1) == "1"
        assert _quadkey(0, 1, 1) == "2"
        assert _quadkey(1, 1, 1) == "3"

    def test_length_equals_zoom(self):
        assert len(_quadkey(523, 341, 10)) == 10


class TestQuadkeysForBbox:
    def test_berlin_point_matches_live_tile_key(self):
        # Verified against the bucket's tiles.geojson on 2026-10-23.
        assert _quadkeys_for_bbox(13.3, 52.1, 13.3, 52.1) == ["1202120103"]

    def test_bbox_straddling_tile_edge_yields_two_keys(self):
        assert _quadkeys_for_bbox(13.35, 52.1, 13.37, 52.1) == [
            "1202120103",
            "1202120112",
        ]

    def test_bbox_straddling_both_axes_yields_four_sorted_unique_keys(self):
        keys = _quadkeys_for_bbox(13.35, 52.26, 13.37, 52.27)
        assert keys == ["1202120101", "1202120103", "1202120110", "1202120112"]
        assert keys == sorted(set(keys))
        assert all(len(k) == 10 for k in keys)


def test_base_url_has_no_trailing_slash():
    assert not CHMV2_BASE_URL.endswith("/")
