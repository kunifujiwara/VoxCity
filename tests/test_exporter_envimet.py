"""
Tests for voxcity.exporter.envimet helper functions to improve coverage.
"""

import numpy as np
import pytest

from voxcity.exporter.envimet import (
    array_to_string,
    array_to_string_with_value,
    array_to_string_int,
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
