"""Tests for voxcity.utils.lc module - land cover utilities."""
import pytest
import numpy as np

from voxcity.utils.lc import (
    rgb_distance,
    get_land_cover_classes,
    get_nearest_class,
    get_dominant_class,
    convert_land_cover_array,
    get_class_priority,
    create_land_cover_polygons,
)


class TestRgbDistance:
    """Tests for rgb_distance function."""

    def test_identical_colors(self):
        """Test distance between identical colors is zero."""
        assert rgb_distance((255, 0, 0), (255, 0, 0)) == 0.0
        assert rgb_distance((100, 100, 100), (100, 100, 100)) == 0.0

    def test_different_colors(self):
        """Test distance between different colors."""
        # Black to white should be sqrt(3*255^2)
        dist = rgb_distance((0, 0, 0), (255, 255, 255))
        assert dist == pytest.approx(np.sqrt(3 * 255**2))

    def test_single_channel_difference(self):
        """Test with only one channel different."""
        # Red channel only: sqrt(100^2) = 100
        assert rgb_distance((100, 0, 0), (200, 0, 0)) == pytest.approx(100.0)

    def test_symmetry(self):
        """Test that distance is symmetric."""
        color1 = (50, 100, 150)
        color2 = (200, 50, 75)
        assert rgb_distance(color1, color2) == rgb_distance(color2, color1)


class TestGetLandCoverClasses:
    """Tests for get_land_cover_classes function."""

    def test_urbanwatch_returns_dict(self):
        """Test Urbanwatch source returns dictionary."""
        result = get_land_cover_classes("Urbanwatch")
        assert isinstance(result, dict)
        assert len(result) > 0

    def test_urbanwatch_has_building(self):
        """Test Urbanwatch includes Building class."""
        result = get_land_cover_classes("Urbanwatch")
        assert "Building" in result.values()

    def test_openearthmapjapan(self):
        """Test OpenEarthMapJapan source."""
        result = get_land_cover_classes("OpenEarthMapJapan")
        assert isinstance(result, dict)
        assert "Tree" in result.values()

    def test_esri_10m(self):
        """Test ESRI 10m Annual Land Cover source."""
        result = get_land_cover_classes("ESRI 10m Annual Land Cover")
        assert isinstance(result, dict)
        assert "Trees" in result.values()

    def test_esa_worldcover(self):
        """Test ESA WorldCover source."""
        result = get_land_cover_classes("ESA WorldCover")
        assert isinstance(result, dict)
        assert "Trees" in result.values()

    def test_dynamic_world_v1(self):
        """Test Dynamic World V1 source."""
        result = get_land_cover_classes("Dynamic World V1")
        assert isinstance(result, dict)
        assert "Water" in result.values()

    def test_standard(self):
        """Test Standard source."""
        result = get_land_cover_classes("Standard")
        assert isinstance(result, dict)
        assert "Building" in result.values()

    def test_openstreetmap(self):
        """Test OpenStreetMap source."""
        result = get_land_cover_classes("OpenStreetMap")
        assert isinstance(result, dict)
        # Should be same as Standard
        standard = get_land_cover_classes("Standard")
        assert result == standard

    def test_rgb_keys(self):
        """Test that keys are RGB tuples."""
        result = get_land_cover_classes("Urbanwatch")
        for key in result.keys():
            assert isinstance(key, tuple)
            assert len(key) == 3
            assert all(isinstance(v, int) for v in key)


class TestGetNearestClass:
    """Tests for get_nearest_class function."""

    def test_exact_match(self):
        """Test exact color match."""
        classes = {(255, 0, 0): 'Red', (0, 255, 0): 'Green'}
        assert get_nearest_class((255, 0, 0), classes) == 'Red'

    def test_nearest_match(self):
        """Test finding nearest color."""
        classes = {(255, 0, 0): 'Red', (0, 255, 0): 'Green', (0, 0, 255): 'Blue'}
        # (250, 5, 5) is closest to red
        assert get_nearest_class((250, 5, 5), classes) == 'Red'

    def test_single_class(self):
        """Test with single class always returns that class."""
        classes = {(100, 100, 100): 'Gray'}
        assert get_nearest_class((0, 0, 0), classes) == 'Gray'
        assert get_nearest_class((255, 255, 255), classes) == 'Gray'


class TestGetDominantClass:
    """Tests for get_dominant_class function."""

    def test_uniform_cell(self):
        """Test cell with uniform color."""
        classes = {(255, 0, 0): 'Building', (0, 255, 0): 'Tree'}
        cell_data = np.full((3, 3, 3), [255, 0, 0], dtype=np.uint8)
        result = get_dominant_class(cell_data, classes)
        assert result == 'Building'

    def test_empty_cell(self):
        """Test empty cell returns No Data."""
        classes = {(255, 0, 0): 'Building'}
        cell_data = np.array([]).reshape(0, 0, 3)
        result = get_dominant_class(cell_data, classes)
        assert result == 'No Data'

    def test_majority_class(self):
        """Test that majority class is returned."""
        classes = {(255, 0, 0): 'Building', (0, 255, 0): 'Tree'}
        # 5 red pixels, 4 green pixels
        cell_data = np.array([
            [[255, 0, 0], [255, 0, 0], [255, 0, 0]],
            [[255, 0, 0], [255, 0, 0], [0, 255, 0]],
            [[0, 255, 0], [0, 255, 0], [0, 255, 0]]
        ], dtype=np.uint8)
        result = get_dominant_class(cell_data, classes)
        assert result == 'Building'


class TestConvertLandCoverArray:
    """Tests for convert_land_cover_array function."""

    def test_basic_conversion(self):
        """Test basic class name to index conversion."""
        classes = {(255, 0, 0): 'Building', (0, 255, 0): 'Tree'}
        arr = np.array(['Building', 'Tree', 'Building'])
        result = convert_land_cover_array(arr, classes)
        np.testing.assert_array_equal(result, [0, 1, 0])

    def test_unknown_class(self):
        """Test unknown class returns -1."""
        classes = {(255, 0, 0): 'Building'}
        arr = np.array(['Building', 'Unknown'])
        result = convert_land_cover_array(arr, classes)
        assert result[0] == 0
        assert result[1] == -1

    def test_2d_array(self):
        """Test 2D array conversion."""
        classes = {(255, 0, 0): 'Building', (0, 255, 0): 'Tree', (0, 0, 255): 'Water'}
        arr = np.array([['Building', 'Tree'], ['Water', 'Building']])
        result = convert_land_cover_array(arr, classes)
        assert result.shape == (2, 2)
        assert result[0, 0] == 0  # Building
        assert result[0, 1] == 1  # Tree
        assert result[1, 0] == 2  # Water


class TestGetClassPriority:
    """Tests for get_class_priority function."""

    def test_returns_dict(self):
        """Test get_class_priority returns dictionary."""
        result = get_class_priority("OpenStreetMap")
        assert isinstance(result, dict)

    def test_building_has_high_priority(self):
        """Test Building has high priority (low number = high priority)."""
        priorities = get_class_priority("OpenStreetMap")
        # Building should have priority 2 (second highest)
        assert 'Building' in priorities
        assert priorities['Building'] == 2

    def test_road_has_highest_priority(self):
        """Test Road has highest priority."""
        priorities = get_class_priority("OpenStreetMap")
        assert 'Road' in priorities
        assert priorities['Road'] == 1

    def test_no_data_has_lowest_priority(self):
        """Test No Data has lowest priority."""
        priorities = get_class_priority("OpenStreetMap")
        assert 'No Data' in priorities
        # Should be highest number (lowest priority)
        max_priority = max(priorities.values())
        assert priorities['No Data'] == max_priority

    def test_all_standard_classes_covered(self):
        """Test all standard land cover classes have priorities."""
        priorities = get_class_priority("OpenStreetMap")
        expected_classes = [
            'Road', 'Building', 'Water', 'Bareland', 'Agriculture land',
            'Rangeland', 'Tree', 'Developed space', 'Wet land', 'Mangrove',
            'Shrub', 'Moss and lichen', 'Snow and ice', 'No Data'
        ]
        for cls in expected_classes:
            assert cls in priorities, f"Missing priority for {cls}"


class TestCreateLandCoverPolygons:
    """Tests for create_land_cover_polygons function."""

    def test_creates_polygons_and_index(self):
        """Test that function returns polygons and spatial index."""
        geojson = [
            {
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]]
                },
                "properties": {"class": "Building"}
            }
        ]
        polygons, idx = create_land_cover_polygons(geojson)
        
        assert len(polygons) == 1
        assert polygons[0][1] == "Building"
        assert idx is not None

    def test_multiple_polygons(self):
        """Test with multiple land cover features."""
        geojson = [
            {
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]]
                },
                "properties": {"class": "Building"}
            },
            {
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[2, 2], [3, 2], [3, 3], [2, 3], [2, 2]]]
                },
                "properties": {"class": "Tree"}
            }
        ]
        polygons, idx = create_land_cover_polygons(geojson)
        
        assert len(polygons) == 2
        assert polygons[0][1] == "Building"
        assert polygons[1][1] == "Tree"

    def test_spatial_index_works(self):
        """Test that spatial index returns correct results."""
        geojson = [
            {
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]]
                },
                "properties": {"class": "Building"}
            },
            {
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[10, 10], [11, 10], [11, 11], [10, 11], [10, 10]]]
                },
                "properties": {"class": "Tree"}
            }
        ]
        polygons, idx = create_land_cover_polygons(geojson)
        
        # Query for features intersecting with (0, 0, 1, 1) should return first polygon
        results = list(idx.intersection((0, 0, 1, 1)))
        assert 0 in results
        assert 1 not in results

    def test_empty_input(self):
        """Test with empty input."""
        geojson = []
        polygons, idx = create_land_cover_polygons(geojson)
        
        assert len(polygons) == 0
