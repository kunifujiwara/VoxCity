"""Extended tests for voxcity.utils.lc module."""
import pytest
import numpy as np

from voxcity.utils.lc import (
    rgb_distance,
    get_land_cover_classes,
    get_source_class_descriptions,
    convert_land_cover,
    get_class_priority,
)


class TestRgbDistance:
    def test_same_color_zero_distance(self):
        assert rgb_distance((255, 0, 0), (255, 0, 0)) == 0
        assert rgb_distance((0, 0, 0), (0, 0, 0)) == 0
        assert rgb_distance((128, 128, 128), (128, 128, 128)) == 0

    def test_black_to_white(self):
        dist = rgb_distance((0, 0, 0), (255, 255, 255))
        expected = np.sqrt(255**2 * 3)
        assert dist == pytest.approx(expected)

    def test_primary_colors(self):
        # Red to green
        dist = rgb_distance((255, 0, 0), (0, 255, 0))
        expected = np.sqrt(255**2 + 255**2)
        assert dist == pytest.approx(expected)

    def test_small_differences(self):
        dist = rgb_distance((100, 100, 100), (101, 100, 100))
        assert dist == pytest.approx(1.0)


class TestGetLandCoverClasses:
    def test_urbanwatch(self):
        classes = get_land_cover_classes("Urbanwatch")
        assert (255, 0, 0) in classes
        assert classes[(255, 0, 0)] == "Building"
        assert classes[(133, 133, 133)] == "Road"

    def test_open_earth_map_japan(self):
        classes = get_land_cover_classes("OpenEarthMapJapan")
        assert (222, 31, 7) in classes
        assert classes[(222, 31, 7)] == "Building"
        assert classes[(34, 97, 38)] == "Tree"

    def test_esri(self):
        classes = get_land_cover_classes("ESRI 10m Annual Land Cover")
        assert (237, 2, 42) in classes
        assert classes[(237, 2, 42)] == "Built Area"

    def test_esa_worldcover(self):
        classes = get_land_cover_classes("ESA WorldCover")
        assert (230, 0, 0) in classes
        assert classes[(230, 0, 0)] == "Built-up"

    def test_dynamic_world(self):
        classes = get_land_cover_classes("Dynamic World V1")
        assert (196, 40, 27) in classes
        assert classes[(196, 40, 27)] == "Built"

    def test_standard(self):
        classes = get_land_cover_classes("Standard")
        assert (222, 31, 7) in classes
        assert classes[(222, 31, 7)] == "Building"

    def test_openstreetmap(self):
        classes = get_land_cover_classes("OpenStreetMap")
        assert (222, 31, 7) in classes
        assert classes[(222, 31, 7)] == "Building"


class TestGetSourceClassDescriptions:
    def test_urbanwatch_description(self):
        desc = get_source_class_descriptions("Urbanwatch")
        assert "Urbanwatch" in desc
        assert "Land Cover Classes" in desc
        assert "Building" in desc

    def test_openstreetmap_note(self):
        desc = get_source_class_descriptions("OpenStreetMap")
        assert "OpenStreetMap" in desc
        assert "standard" in desc.lower()


class TestConvertLandCover:
    def test_urbanwatch_conversion(self):
        arr = np.array([[0, 1, 2]], dtype=np.uint8)
        result = convert_land_cover(arr, "Urbanwatch")
        # 0->13 (Building), 1->12 (Road), 2->11 (Developed space)
        assert result.tolist() == [[13, 12, 11]]

    def test_preserves_dtype(self):
        arr = np.array([[0, 1]], dtype=np.uint8)
        result = convert_land_cover(arr, "Urbanwatch")
        assert result.dtype == arr.dtype

    def test_esri_conversion(self):
        arr = np.array([[0, 1, 2]], dtype=np.uint8)
        result = convert_land_cover(arr, "ESRI 10m Annual Land Cover")
        # Check mapping: 0->14, 1->9, 2->5
        assert result[0, 0] == 14  # No Data
        assert result[0, 1] == 9   # Water
        assert result[0, 2] == 5   # Tree

    def test_esa_worldcover_conversion(self):
        arr = np.array([[0, 1, 2]], dtype=np.uint8)
        result = convert_land_cover(arr, "ESA WorldCover")
        # 0->5 (Tree), 1->3 (Shrub), 2->2 (Rangeland)
        assert result[0, 0] == 5
        assert result[0, 1] == 3
        assert result[0, 2] == 2

    def test_dynamic_world_conversion(self):
        arr = np.array([[0, 6]], dtype=np.uint8)
        result = convert_land_cover(arr, "Dynamic World V1")
        # 0->9 (Water), 6->11 (Built)
        assert result[0, 0] == 9
        assert result[0, 1] == 11

    def test_open_earth_map_japan_conversion(self):
        arr = np.array([[0, 7]], dtype=np.uint8)
        result = convert_land_cover(arr, "OpenEarthMapJapan")
        # 0->1 (Bareland), 7->13 (Building)
        assert result[0, 0] == 1
        assert result[0, 1] == 13

    def test_unknown_source_adds_one(self):
        arr = np.array([[0, 1, 2]], dtype=np.uint8)
        result = convert_land_cover(arr, "Unknown Source")
        # Should add 1 to each value
        assert result.tolist() == [[1, 2, 3]]

    def test_2d_array(self):
        arr = np.array([[0, 1], [2, 3]], dtype=np.uint8)
        result = convert_land_cover(arr, "Urbanwatch")
        assert result.shape == (2, 2)


class TestGetClassPriority:
    def test_osm_priority(self):
        priority = get_class_priority("OpenStreetMap")
        assert isinstance(priority, dict)
        assert "Building" in priority
        assert "Road" in priority
        assert "Tree" in priority

    def test_building_high_priority(self):
        priority = get_class_priority("OpenStreetMap")
        # Building should have higher priority than most other classes
        assert priority["Building"] < priority["Tree"]
        assert priority["Building"] < priority["Rangeland"]

    def test_no_data_lowest_priority(self):
        priority = get_class_priority("OpenStreetMap")
        # No Data should have lowest priority (highest number in this system)
        assert priority["No Data"] == max(priority.values())
