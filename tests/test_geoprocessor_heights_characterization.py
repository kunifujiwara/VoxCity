"""Characterization tests for extract_building_heights_from_gdf.

Written against the pre-optimization implementation; they pin exact
weighted-average values, NaN rules, invalid-geometry exclusion, and log
counts so the optimized version can be proven output-equivalent.
Assertions use approx(rel=1e-9): candidate iteration order (rtree vs
sindex) may legally differ, changing float summation order in the last ulp.
"""
import logging

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import Polygon

from voxcity.geoprocessor.heights import extract_building_heights_from_gdf


def _rect(x0, y0, x1, y1):
    return Polygon([(x0, y0), (x1, y0), (x1, y1), (x0, y1)])


UNIT_SQUARE = _rect(0, 0, 1, 1)


def test_weighted_average_exact_value():
    primary = gpd.GeoDataFrame(
        {"height": [0.0], "geometry": [UNIT_SQUARE]}, crs="EPSG:4326"
    )
    # ref1 overlaps the left half (area 0.5, h=10); ref2 the right quarter
    # (area 0.25, h=40): (10*0.5 + 40*0.25) / (0.5 + 0.25) = 20.0
    ref = gpd.GeoDataFrame(
        {
            "height": [10.0, 40.0],
            "geometry": [_rect(0, 0, 0.5, 1), _rect(0.75, 0, 1, 1)],
        },
        crs="EPSG:4326",
    )
    out = extract_building_heights_from_gdf(primary, ref)
    assert out.iloc[0]["height"] == pytest.approx(20.0, rel=1e-9)


def test_zero_height_ref_dilutes_average():
    primary = gpd.GeoDataFrame(
        {"height": [np.nan], "geometry": [UNIT_SQUARE]}, crs="EPSG:4326"
    )
    # h=30 over half, h=0 over the other half: (30*0.5 + 0*0.5) / 1.0 = 15.0
    ref = gpd.GeoDataFrame(
        {
            "height": [30.0, 0.0],
            "geometry": [_rect(0, 0, 0.5, 1), _rect(0.5, 0, 1, 1)],
        },
        crs="EPSG:4326",
    )
    out = extract_building_heights_from_gdf(primary, ref)
    assert out.iloc[0]["height"] == pytest.approx(15.0, rel=1e-9)


def test_all_zero_height_refs_yield_nan():
    primary = gpd.GeoDataFrame(
        {"height": [0.0], "geometry": [UNIT_SQUARE]}, crs="EPSG:4326"
    )
    ref = gpd.GeoDataFrame(
        {"height": [0.0], "geometry": [_rect(0, 0, 1, 1)]}, crs="EPSG:4326"
    )
    out = extract_building_heights_from_gdf(primary, ref)
    assert pd.isna(out.iloc[0]["height"])


def test_invalid_ref_geometry_is_excluded():
    # Self-intersecting bowtie is invalid -> never enters the spatial index,
    # so only the valid half-overlap ref (h=20) contributes.
    bowtie = Polygon([(0, 0), (1, 1), (1, 0), (0, 1)])
    assert not bowtie.is_valid  # fixture sanity
    primary = gpd.GeoDataFrame(
        {"height": [0.0], "geometry": [UNIT_SQUARE]}, crs="EPSG:4326"
    )
    ref = gpd.GeoDataFrame(
        {"height": [100.0, 20.0], "geometry": [bowtie, _rect(0, 0, 0.5, 1)]},
        crs="EPSG:4326",
    )
    out = extract_building_heights_from_gdf(primary, ref)
    assert out.iloc[0]["height"] == pytest.approx(20.0, rel=1e-9)


def test_existing_positive_heights_untouched():
    primary = gpd.GeoDataFrame(
        {"height": [12.5], "geometry": [UNIT_SQUARE]}, crs="EPSG:4326"
    )
    ref = gpd.GeoDataFrame(
        {"height": [99.0], "geometry": [_rect(0, 0, 1, 1)]}, crs="EPSG:4326"
    )
    out = extract_building_heights_from_gdf(primary, ref)
    assert out.iloc[0]["height"] == 12.5


def test_no_overlap_yields_nan_and_logs(caplog, propagate_voxcity_logs):
    primary = gpd.GeoDataFrame(
        {"height": [0.0], "geometry": [UNIT_SQUARE]}, crs="EPSG:4326"
    )
    ref = gpd.GeoDataFrame(
        {"height": [50.0], "geometry": [_rect(10, 10, 11, 11)]}, crs="EPSG:4326"
    )
    with caplog.at_level(logging.INFO, logger="voxcity"):
        out = extract_building_heights_from_gdf(primary, ref)
    assert pd.isna(out.iloc[0]["height"])
    # count_1 == 0 assigned, count_2 == 1 without complementary data
    assert "For 0 of these building footprints" in caplog.text
    assert "For 1 of these building footprints" in caplog.text
