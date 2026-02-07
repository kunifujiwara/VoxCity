"""
Tests for voxcity.simulator.common.raytracing functions.
"""

import numpy as np
import pytest

from voxcity.simulator.common.raytracing import (
    calculate_transmittance,
    trace_ray_generic,
)


class TestCalculateTransmittance:
    def test_zero_length(self):
        result = calculate_transmittance(0.0)
        assert result == pytest.approx(1.0)

    def test_positive_length(self):
        result = calculate_transmittance(1.0, tree_k=0.6, tree_lad=1.0)
        expected = np.exp(-0.6 * 1.0 * 1.0)
        assert result == pytest.approx(expected)

    def test_large_length(self):
        result = calculate_transmittance(100.0)
        assert result < 0.01

    def test_custom_parameters(self):
        result = calculate_transmittance(2.0, tree_k=0.3, tree_lad=0.5)
        expected = np.exp(-0.3 * 0.5 * 2.0)
        assert result == pytest.approx(expected)

    def test_decreasing_with_length(self):
        t1 = calculate_transmittance(1.0)
        t2 = calculate_transmittance(2.0)
        t3 = calculate_transmittance(5.0)
        assert t1 > t2 > t3


class TestTraceRayGeneric:
    def test_empty_voxel_space(self):
        voxels = np.zeros((5, 5, 5), dtype=np.int32)
        origin = np.array([2.0, 2.0, 2.0])
        direction = np.array([1.0, 0.0, 0.0])
        hit_values = np.array([-3], dtype=np.int32)

        hit, trans = trace_ray_generic(voxels, origin, direction, hit_values,
                                       meshsize=1.0, tree_k=0.6, tree_lad=1.0,
                                       inclusion_mode=True)
        assert hit is False
        assert trans == pytest.approx(1.0)

    def test_hit_building(self):
        voxels = np.zeros((10, 10, 10), dtype=np.int32)
        voxels[5, 2, 2] = -3  # Building voxel

        origin = np.array([0.0, 2.0, 2.0])
        direction = np.array([1.0, 0.0, 0.0])
        hit_values = np.array([-3], dtype=np.int32)

        hit, trans = trace_ray_generic(voxels, origin, direction, hit_values,
                                       meshsize=1.0, tree_k=0.6, tree_lad=1.0,
                                       inclusion_mode=True)
        assert hit is True

    def test_hit_tree_transmittance(self):
        voxels = np.zeros((10, 10, 10), dtype=np.int32)
        voxels[3, 2, 2] = -2  # Tree voxel
        voxels[4, 2, 2] = -2  # Tree voxel

        origin = np.array([0.0, 2.0, 2.0])
        direction = np.array([1.0, 0.0, 0.0])
        hit_values = np.array([-3], dtype=np.int32)

        hit, trans = trace_ray_generic(voxels, origin, direction, hit_values,
                                       meshsize=1.0, tree_k=0.6, tree_lad=1.0,
                                       inclusion_mode=True)
        # Transmittance should be reduced after passing through trees
        assert trans < 1.0

    def test_zero_direction(self):
        voxels = np.zeros((5, 5, 5), dtype=np.int32)
        origin = np.array([2.0, 2.0, 2.0])
        direction = np.array([0.0, 0.0, 0.0])
        hit_values = np.array([-3], dtype=np.int32)

        hit, trans = trace_ray_generic(voxels, origin, direction, hit_values,
                                       meshsize=1.0, tree_k=0.6, tree_lad=1.0)
        assert hit is False

    def test_diagonal_direction(self):
        voxels = np.zeros((10, 10, 10), dtype=np.int32)
        voxels[5, 5, 5] = -3

        origin = np.array([0.0, 0.0, 0.0])
        direction = np.array([1.0, 1.0, 1.0])
        hit_values = np.array([-3], dtype=np.int32)

        hit, trans = trace_ray_generic(voxels, origin, direction, hit_values,
                                       meshsize=1.0, tree_k=0.6, tree_lad=1.0,
                                       inclusion_mode=True)
        assert hit is True

    def test_exclusion_mode(self):
        voxels = np.zeros((10, 10, 10), dtype=np.int32)
        voxels[3, 2, 2] = -1  # Ground code

        origin = np.array([0.0, 2.0, 2.0])
        direction = np.array([1.0, 0.0, 0.0])
        hit_values = np.array([-3], dtype=np.int32)

        hit, trans = trace_ray_generic(voxels, origin, direction, hit_values,
                                       meshsize=1.0, tree_k=0.6, tree_lad=1.0,
                                       inclusion_mode=False)
        # In exclusion mode, hitting something NOT in hit_values returns True
        assert hit is True

    def test_negative_direction(self):
        voxels = np.zeros((10, 10, 10), dtype=np.int32)
        voxels[0, 5, 5] = -3

        origin = np.array([9.0, 5.0, 5.0])
        direction = np.array([-1.0, 0.0, 0.0])
        hit_values = np.array([-3], dtype=np.int32)

        hit, trans = trace_ray_generic(voxels, origin, direction, hit_values,
                                       meshsize=1.0, tree_k=0.6, tree_lad=1.0,
                                       inclusion_mode=True)
        assert hit is True
