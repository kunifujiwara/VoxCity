"""
Comprehensive tests for voxcity.simulator.common.raytracing to improve coverage.
Covers: compute_vi_generic, compute_vi_map_generic, _prepare_masks_for_vi,
_trace_ray_inclusion_masks, _trace_ray_exclusion_masks,
_compute_vi_map_generic_fast, _precompute_observer_base_z, _trace_ray
"""

import numpy as np
import pytest

from voxcity.simulator.common.raytracing import (
    calculate_transmittance,
    trace_ray_generic,
    compute_vi_generic,
    compute_vi_map_generic,
    _prepare_masks_for_vi,
    _trace_ray_inclusion_masks,
    _trace_ray_exclusion_masks,
    _precompute_observer_base_z,
    _trace_ray,
)


# --- Helpers ---

def _make_ground_grid(nx=10, ny=10, nz=10):
    """Ground at z=0, air above."""
    grid = np.zeros((nx, ny, nz), dtype=np.int32)
    grid[:, :, 0] = 1  # ground
    return grid


def _make_building_grid():
    """10x10x10 grid, ground at z=0, building at (5,5,1:4)."""
    grid = _make_ground_grid()
    grid[5, 5, 1:4] = -3  # building voxels
    return grid


def _make_tree_grid():
    """10x10x10 grid, ground at z=0, tree at (5,5,1:3)."""
    grid = _make_ground_grid()
    grid[5, 5, 1:3] = -2  # tree voxels
    return grid


# --- Tests ---

class TestComputeViGeneric:
    def test_open_sky_full_visibility(self):
        grid = _make_ground_grid()
        obs = np.array([5.0, 5.0, 2.0])
        # Directions pointing upward
        dirs = np.array([[0.0, 0.0, 1.0],
                         [0.1, 0.0, 1.0],
                         [-0.1, 0.0, 1.0]], dtype=np.float64)
        hit_values = np.array([-3], dtype=np.int32)
        vi = compute_vi_generic(obs, grid, dirs, hit_values, 1.0, 0.6, 1.0, inclusion_mode=True)
        # No buildings => no hits in inclusion mode => vi = 0
        assert vi == 0.0

    def test_building_visible(self):
        grid = _make_building_grid()
        # Observer at (3,5,2), looking toward building at (5,5,x)
        obs = np.array([3.0, 5.0, 2.0])
        dirs = np.array([[1.0, 0.0, 0.0]], dtype=np.float64)  # +x direction
        hit_values = np.array([-3], dtype=np.int32)
        vi = compute_vi_generic(obs, grid, dirs, hit_values, 1.0, 0.6, 1.0, inclusion_mode=True)
        assert vi == 1.0  # One ray, hits building => vi = 1.0

    def test_exclusion_mode(self):
        grid = _make_ground_grid()
        obs = np.array([5.0, 5.0, 2.0])
        dirs = np.array([[0.0, 0.0, 1.0]], dtype=np.float64)
        hit_values = np.array([0], dtype=np.int32)  # Air is allowed
        vi = compute_vi_generic(obs, grid, dirs, hit_values, 1.0, 0.6, 1.0, inclusion_mode=False)
        # In exclusion mode, if not hit => visibility_sum += value
        assert vi > 0


class TestComputeViMapGeneric:
    def test_basic_map(self):
        grid = _make_ground_grid()
        dirs = np.array([[0.0, 0.0, 1.0],
                         [0.1, 0.0, 1.0]], dtype=np.float64)
        hit_values = np.array([-3], dtype=np.int32)
        vi_map = compute_vi_map_generic(grid, dirs, 0, hit_values, 1.0, 0.6, 1.0, inclusion_mode=True)
        assert vi_map.shape == (10, 10)
        # All open sky, no buildings => vi = 0 everywhere except water/invalid
        assert np.nanmax(vi_map) == 0.0

    def test_map_with_building(self):
        grid = _make_building_grid()
        dirs = np.array([[1.0, 0.0, 0.0],
                         [-1.0, 0.0, 0.0],
                         [0.0, 1.0, 0.0],
                         [0.0, -1.0, 0.0],
                         [0.0, 0.0, 1.0]], dtype=np.float64)
        hit_values = np.array([-3], dtype=np.int32)
        vi_map = compute_vi_map_generic(grid, dirs, 0, hit_values, 1.0, 0.6, 1.0)
        assert vi_map.shape == (10, 10)
        # Some cells should see the building
        assert not np.all(np.isnan(vi_map))


class TestPrepareMasksForVi:
    def test_inclusion_mode(self):
        grid = _make_building_grid()
        hit_values = np.array([-3], dtype=np.int32)
        is_tree, is_target, is_allowed, is_blocker = _prepare_masks_for_vi(grid, hit_values, True)
        assert is_tree.shape == grid.shape
        assert is_target.shape == grid.shape
        assert is_allowed is None
        assert is_blocker is not None
        # Trees at grid == -2 => no trees in this grid
        assert not np.any(is_tree)
        # Building at (5,5,1:4) should be target
        assert is_target[5, 5, 1]

    def test_exclusion_mode(self):
        grid = _make_building_grid()
        hit_values = np.array([0, 1], dtype=np.int32)  # Air + ground allowed
        is_tree, is_target, is_allowed, is_blocker = _prepare_masks_for_vi(grid, hit_values, False)
        assert is_target is None
        assert is_blocker is None
        assert is_allowed is not None
        # Air (0) and ground (1) should be allowed
        assert is_allowed[0, 0, 5]  # Air cell


class TestTraceRayInclusionMasks:
    def test_hit_target(self):
        grid = np.zeros((10, 10, 10), dtype=np.int32)
        grid[:, :, 0] = 1
        grid[5, 5, 2] = -3
        hit_values = np.array([-3], dtype=np.int32)
        is_tree, is_target, _, is_blocker = _prepare_masks_for_vi(grid, hit_values, True)
        origin = np.array([3.0, 5.0, 2.0])
        direction = np.array([1.0, 0.0, 0.0])
        hit, trans = _trace_ray_inclusion_masks(is_tree, is_target, is_blocker, origin, direction, 1.0, 0.6, 1.0)
        assert hit is True
        assert trans == pytest.approx(1.0)

    def test_miss(self):
        grid = _make_ground_grid()
        hit_values = np.array([-3], dtype=np.int32)
        is_tree, is_target, _, is_blocker = _prepare_masks_for_vi(grid, hit_values, True)
        origin = np.array([5.0, 5.0, 2.0])
        direction = np.array([0.0, 0.0, 1.0])
        hit, trans = _trace_ray_inclusion_masks(is_tree, is_target, is_blocker, origin, direction, 1.0, 0.6, 1.0)
        assert hit is False

    def test_through_tree(self):
        grid = _make_ground_grid()
        grid[4, 5, 2] = -2  # tree
        grid[6, 5, 2] = -3  # building behind tree
        hit_values = np.array([-3], dtype=np.int32)
        is_tree, is_target, _, is_blocker = _prepare_masks_for_vi(grid, hit_values, True)
        origin = np.array([3.0, 5.0, 2.0])
        direction = np.array([1.0, 0.0, 0.0])
        hit, trans = _trace_ray_inclusion_masks(is_tree, is_target, is_blocker, origin, direction, 1.0, 0.6, 1.0)
        assert hit is True
        assert trans < 1.0  # Tree reduces transmittance

    def test_zero_direction(self):
        grid = _make_ground_grid()
        hit_values = np.array([-3], dtype=np.int32)
        is_tree, is_target, _, is_blocker = _prepare_masks_for_vi(grid, hit_values, True)
        origin = np.array([5.0, 5.0, 2.0])
        direction = np.array([0.0, 0.0, 0.0])
        hit, trans = _trace_ray_inclusion_masks(is_tree, is_target, is_blocker, origin, direction, 1.0, 0.6, 1.0)
        assert hit is False
        assert trans == 1.0


class TestTraceRayExclusionMasks:
    def test_open_path(self):
        grid = _make_ground_grid()
        hit_values = np.array([0], dtype=np.int32)  # Air allowed
        is_tree, _, is_allowed, _ = _prepare_masks_for_vi(grid, hit_values, False)
        origin = np.array([5.0, 5.0, 2.0])
        direction = np.array([0.0, 0.0, 1.0])
        hit, trans = _trace_ray_exclusion_masks(is_tree, is_allowed, origin, direction, 1.0, 0.6, 1.0)
        assert hit is False
        assert trans == pytest.approx(1.0)

    def test_blocked_by_non_allowed(self):
        grid = _make_ground_grid()
        grid[5, 5, 2] = -3  # Building (not in allowed set)
        hit_values = np.array([0, 1], dtype=np.int32)
        is_tree, _, is_allowed, _ = _prepare_masks_for_vi(grid, hit_values, False)
        origin = np.array([5.0, 5.0, 3.0])
        direction = np.array([0.0, 0.0, -1.0])
        hit, trans = _trace_ray_exclusion_masks(is_tree, is_allowed, origin, direction, 1.0, 0.6, 1.0)
        assert hit is True  # Blocked

    def test_zero_direction(self):
        grid = _make_ground_grid()
        hit_values = np.array([0], dtype=np.int32)
        is_tree, _, is_allowed, _ = _prepare_masks_for_vi(grid, hit_values, False)
        origin = np.array([5.0, 5.0, 2.0])
        direction = np.array([0.0, 0.0, 0.0])
        hit, trans = _trace_ray_exclusion_masks(is_tree, is_allowed, origin, direction, 1.0, 0.6, 1.0)
        assert hit is False
        assert trans == 1.0


class TestPrecomputeObserverBaseZ:
    def test_ground_level(self):
        grid = _make_ground_grid()
        base_z = _precompute_observer_base_z(grid)
        assert base_z.shape == (10, 10)
        # Ground at z=0, air at z=1 => base_z should be 0
        assert np.all(base_z == 0)

    def test_no_ground(self):
        grid = np.zeros((5, 5, 5), dtype=np.int32)  # All air
        base_z = _precompute_observer_base_z(grid)
        # No transition found => should be -1
        assert np.all(base_z == -1)

    def test_elevated_ground(self):
        grid = np.zeros((5, 5, 5), dtype=np.int32)
        grid[:, :, 0:3] = 1  # Ground occupies z=0,1,2
        base_z = _precompute_observer_base_z(grid)
        # Transition at z=3 (air) with z=2 below => base_z = 2
        assert np.all(base_z == 2)


class TestTraceRay:
    def test_clear_path(self):
        is_tree = np.zeros((10, 10, 10), dtype=np.bool_)
        is_opaque = np.zeros((10, 10, 10), dtype=np.bool_)
        origin = np.array([2.0, 5.0, 5.0])
        target = np.array([7.0, 5.0, 5.0])
        result = _trace_ray(is_tree, is_opaque, origin, target, 0.5, 0.01)
        assert result is True

    def test_blocked_by_opaque(self):
        is_tree = np.zeros((10, 10, 10), dtype=np.bool_)
        is_opaque = np.zeros((10, 10, 10), dtype=np.bool_)
        is_opaque[5, 5, 5] = True
        origin = np.array([2.0, 5.0, 5.0])
        target = np.array([7.0, 5.0, 5.0])
        result = _trace_ray(is_tree, is_opaque, origin, target, 0.5, 0.01)
        assert result is False

    def test_attenuated_by_trees(self):
        is_tree = np.zeros((10, 10, 10), dtype=np.bool_)
        is_opaque = np.zeros((10, 10, 10), dtype=np.bool_)
        # Fill path with many trees
        for i in range(3, 8):
            is_tree[i, 5, 5] = True
        origin = np.array([2.0, 5.0, 5.0])
        target = np.array([8.0, 5.0, 5.0])
        result = _trace_ray(is_tree, is_opaque, origin, target, 0.1, 0.01)
        # Many trees with low attenuation => likely blocked
        assert result is False

    def test_out_of_bounds(self):
        is_tree = np.zeros((5, 5, 5), dtype=np.bool_)
        is_opaque = np.zeros((5, 5, 5), dtype=np.bool_)
        origin = np.array([2.0, 2.0, 2.0])
        target = np.array([10.0, 2.0, 2.0])  # Target outside grid
        result = _trace_ray(is_tree, is_opaque, origin, target, 0.5, 0.01)
        assert result is False  # Goes out of bounds before reaching target

    def test_zero_distance(self):
        is_tree = np.zeros((5, 5, 5), dtype=np.bool_)
        is_opaque = np.zeros((5, 5, 5), dtype=np.bool_)
        origin = np.array([2.0, 2.0, 2.0])
        target = np.array([2.0, 2.0, 2.0])  # Same point
        result = _trace_ray(is_tree, is_opaque, origin, target, 0.5, 0.01)
        assert result is True  # Zero-distance => True
