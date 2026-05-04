"""
Tests for voxcity.exporter.envimet helper functions to improve coverage.
"""

import numpy as np
import pytest

from voxcity.exporter.envimet import (
    array_to_string,
    array_to_string_with_value,
    array_to_string_int,
    create_xml_content,
    prepare_grids,
)


class TestArrayToString:
    def test_simple_2x2(self):
        arr = np.array([[1, 2], [3, 4]])
        result = array_to_string(arr)
        lines = result.strip().split('\n')
        assert len(lines) == 2
        assert '1,2' in lines[0]
        assert '3,4' in lines[1]

    def test_indentation(self):
        arr = np.array([[10, 20]])
        result = array_to_string(arr)
        assert result.startswith('     ')

    def test_single_element(self):
        arr = np.array([[42]])
        result = array_to_string(arr)
        assert '42' in result

    def test_float_values(self):
        arr = np.array([[1.5, 2.7]])
        result = array_to_string(arr)
        assert '1.5' in result
        assert '2.7' in result

    def test_3x3(self):
        arr = np.arange(9).reshape(3, 3)
        result = array_to_string(arr)
        lines = result.strip().split('\n')
        assert len(lines) == 3

    def test_no_trailing_comma(self):
        arr = np.array([[1, 2, 3]])
        result = array_to_string(arr)
        stripped = result.strip()
        assert not stripped.endswith(',')


class TestArrayToStringWithValue:
    def test_uniform_value(self):
        arr = np.zeros((2, 3))
        result = array_to_string_with_value(arr, '0')
        lines = result.strip().split('\n')
        assert len(lines) == 2
        for line in lines:
            values = line.strip().split(',')
            assert all(v == '0' for v in values)
            assert len(values) == 3

    def test_string_value(self):
        arr = np.ones((2, 2))
        result = array_to_string_with_value(arr, '000000')
        assert '000000' in result

    def test_numeric_value(self):
        arr = np.ones((3, 3))
        result = array_to_string_with_value(arr, 5)
        assert '5' in result


class TestArrayToStringInt:
    def test_rounding(self):
        arr = np.array([[1.6, 2.3], [3.7, 4.1]])
        result = array_to_string_int(arr)
        lines = result.strip().split('\n')
        # 1.6+0.5=2.1->2, 2.3+0.5=2.8->2, 3.7+0.5=4.2->4, 4.1+0.5=4.6->4
        assert '2,2' in lines[0].strip()
        assert '4,4' in lines[1].strip()

    def test_zero_values(self):
        arr = np.zeros((2, 2))
        result = array_to_string_int(arr)
        assert '0,0' in result

    def test_large_values(self):
        arr = np.array([[100.9, 200.1]])
        result = array_to_string_int(arr)
        assert '101' in result
        assert '200' in result

    def test_indentation(self):
        arr = np.array([[5]])
        result = array_to_string_int(arr)
        assert result.startswith('     ')


def fail_if_orientation_conversion_is_used(*args, **kwargs):
    raise AssertionError("ENVI-met exporter should process SOUTH_UP grids directly")


class TestEnvimetSouthUpProcessing:
    def test_prepare_grids_preserves_south_up_layout_without_conversion(self, monkeypatch):
        from voxcity.exporter import envimet

        monkeypatch.setattr(envimet, "ensure_orientation", fail_if_orientation_conversion_is_used, raising=False)

        building_height = np.array(
            [
                [0.0, 0.0, 0.0],
                [0.0, 5.0, 0.0],
                [0.0, 7.0, 0.0],
                [0.0, 10.0, 0.0],
                [0.0, 0.0, 0.0],
            ]
        )
        building_id = np.array(
            [
                [0, 0, 0],
                [0, 2, 0],
                [0, 7, 0],
                [0, 3, 0],
                [0, 0, 0],
            ]
        )
        canopy_height = np.zeros_like(building_height)
        land_cover = np.zeros_like(building_id)
        dem = np.array(
            [
                [0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 1.5, 0.0],
                [0.0, 2.0, 0.0],
                [0.0, 0.0, 0.0],
            ]
        )

        bh, bid, _, _, _, dem_out = prepare_grids(
            building_height,
            building_id,
            canopy_height,
            land_cover,
            dem,
            meshsize=1.0,
            land_cover_source="OpenStreetMap",
        )

        assert bh[1, 1] == pytest.approx(5.0)
        assert bh[3, 1] == pytest.approx(10.0)
        assert bid[1, 1] == 2
        assert bid[3, 1] == 3
        assert dem_out[1, 1] == pytest.approx(1.0)
        assert dem_out[3, 1] == pytest.approx(2.0)

    def test_create_xml_content_writes_matrices_north_first_from_south_up_inputs(self, monkeypatch):
        from voxcity.exporter import envimet

        monkeypatch.setattr(envimet, "ensure_orientation", fail_if_orientation_conversion_is_used, raising=False)

        building_height = np.array(
            [
                [0.0, 0.0, 0.0],
                [0.0, 5.0, 0.0],
                [0.0, 7.0, 0.0],
                [0.0, 10.0, 0.0],
                [0.0, 0.0, 0.0],
            ]
        )
        building_id = np.array(
            [
                [0, 0, 0],
                [0, 2, 0],
                [0, 7, 0],
                [0, 3, 0],
                [0, 0, 0],
            ]
        )
        land_cover_veg = np.full(building_height.shape, "", dtype=object)
        land_cover_mat = np.full(building_height.shape, "000000", dtype=object)
        canopy_height = np.zeros_like(building_height)
        dem = np.zeros_like(building_height)

        xml = create_xml_content(
            building_height,
            building_id,
            land_cover_veg,
            land_cover_mat,
            canopy_height,
            dem,
            meshsize=1.0,
            rectangle_vertices=[(0.0, 0.0), (0.0, 1.0), (1.0, 1.0), (1.0, 0.0)],
            min_grids_Z=3,
        )

        ztop = xml.split("<zTop", 1)[1].split("</zTop>", 1)[0]
        rows = [line.strip() for line in ztop.splitlines() if line.strip() and "," in line]
        assert rows[1] == "0.0,10.0,0.0"
        assert rows[-2] == "0.0,5.0,0.0"
