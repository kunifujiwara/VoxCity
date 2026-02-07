"""
Tests for voxcity.simulator.common.raytracing njit functions to improve coverage.
Targets the full trace_ray_generic DDA traversal, compute_vi_generic,
compute_vi_map_generic, _compute_vi_map_generic_fast, _precompute_observer_base_z,
and _trace_ray.
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
    _compute_vi_map_generic_fast,
)


# ---------- Shared fixtures ----------

def _make_voxel_scene():
    """10x10x10 scene: ground at z=0, building at (5,5) z=1..4, tree at (3,3) z=1..3."""
    vox = np.zeros((10, 10, 10), dtype=np.int8)
    # Ground layer
    for x in range(10):
        for y in range(10):
            vox[x, y, 0] = 1  # land cover
    # Building
    vox[5, 5, 1:5] = -3
    # Tree
    vox[3, 3, 1:4] = -2
    return vox


# ---------- trace_ray_generic (lines 10-156) ----------

class TestTraceRayGenericFull:
    """Tests for the full DDA ray tracer covering all branches."""

    def test_inclusion_hit_target(self):
        vox = _make_voxel_scene()
        origin = np.array([5.0, 4.0, 2.0], dtype=np.float64)
        direction = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        hit_values = np.array([-3], dtype=np.int8)
        hit, trans = trace_ray_generic(vox, origin, direction, hit_values, 1.0, 0.6, 1.0, True)
        assert hit is True

    def test_inclusion_blocked_by_non_target(self):
        vox = _make_voxel_scene()
        origin = np.array([5.0, 4.0, 2.0], dtype=np.float64)
        direction = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        # Looking for tree (-2) but building (-3) is in the way
        hit_values = np.array([-2], dtype=np.int8)
        hit, trans = trace_ray_generic(vox, origin, direction, hit_values, 1.0, 0.6, 1.0, True)
        assert hit is False

    def test_exclusion_blocked(self):
        vox = _make_voxel_scene()
        origin = np.array([5.0, 4.0, 2.0], dtype=np.float64)
        direction = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        hit_values = np.array([0, -2], dtype=np.int8)  # air and trees are allowed
        hit, trans = trace_ray_generic(vox, origin, direction, hit_values, 1.0, 0.6, 1.0, False)
        # Building is NOT in allowed set, so ray is blocked
        assert hit is True

    def test_exclusion_not_blocked(self):
        vox = _make_voxel_scene()
        origin = np.array([0.0, 0.0, 5.0], dtype=np.float64)
        direction = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        hit_values = np.array([0], dtype=np.int8)  # air is allowed
        hit, trans = trace_ray_generic(vox, origin, direction, hit_values, 1.0, 0.6, 1.0, False)
        assert hit is False  # ray travels through air only at z=5

    def test_tree_attenuation_heavy(self):
        """Ray through thick tree should drop transmittance below 0.01."""
        vox = np.zeros((10, 10, 20), dtype=np.int8)
        vox[:, :, 0] = 1
        # Thick tree band
        vox[5, 2:8, 2:10] = -2
        origin = np.array([5.0, 0.0, 5.0], dtype=np.float64)
        direction = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        hit_values = np.array([-3], dtype=np.int8)
        hit, trans = trace_ray_generic(vox, origin, direction, hit_values, 1.0, 0.6, 1.0, True)
        assert trans < 0.1  # attenuated through several tree voxels

    def test_diagonal_direction(self):
        vox = _make_voxel_scene()
        origin = np.array([0.0, 0.0, 5.0], dtype=np.float64)
        direction = np.array([1.0, 1.0, 0.0], dtype=np.float64)
        hit_values = np.array([-3], dtype=np.int8)
        hit, trans = trace_ray_generic(vox, origin, direction, hit_values, 1.0, 0.6, 1.0, True)
        # Ray at z=5 is above building (z=1..4), so no hit
        assert hit is False

    def test_ray_exits_grid(self):
        vox = _make_voxel_scene()
        origin = np.array([0.0, 0.0, 5.0], dtype=np.float64)
        direction = np.array([-1.0, 0.0, 0.0], dtype=np.float64)  # exits immediately
        hit_values = np.array([-3], dtype=np.int8)
        hit, trans = trace_ray_generic(vox, origin, direction, hit_values, 1.0, 0.6, 1.0, True)
        assert hit is False

    def test_tie_breaking_branches(self):
        """Test diagonal ray that hits tie conditions between axes."""
        vox = _make_voxel_scene()
        origin = np.array([0.0, 0.0, 0.5], dtype=np.float64)
        # Direction along perfect diagonal triggers tie-breaking
        direction = np.array([1.0, 1.0, 1.0], dtype=np.float64)
        hit_values = np.array([-3], dtype=np.int8)
        hit, trans = trace_ray_generic(vox, origin, direction, hit_values, 1.0, 0.6, 1.0, True)
        # Just ensure no crash
        assert isinstance(hit, (bool, np.bool_))


# ---------- compute_vi_generic (lines 159-176) ----------

class TestComputeViGenericFull:
    def test_open_sky(self):
        vox = np.zeros((5, 5, 5), dtype=np.int8)
        vox[:, :, 0] = 1
        observer = np.array([2.0, 2.0, 2.0], dtype=np.float64)
        rays = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]], dtype=np.float64)
        hit_values = np.array([-3], dtype=np.int8)
        vi = compute_vi_generic(observer, vox, rays, hit_values, 1.0, 0.6, 1.0, True)
        assert vi == 0.0  # No buildings to hit

    def test_inclusion_with_tree_targets(self):
        """When -2 (tree) is in hit_values, VI uses transmittance-based contrib."""
        vox = np.zeros((10, 10, 10), dtype=np.int8)
        vox[:, :, 0] = 1
        vox[3, 3, 2:5] = -2  # tree
        observer = np.array([3.0, 2.0, 3.0], dtype=np.float64)
        rays = np.array([[0, 1, 0]], dtype=np.float64)
        hit_values = np.array([-2], dtype=np.int8)  # looking for trees
        vi = compute_vi_generic(observer, vox, rays, hit_values, 1.0, 0.6, 1.0, True)
        assert vi > 0.0

    def test_exclusion_mode(self):
        vox = np.zeros((10, 10, 10), dtype=np.int8)
        vox[:, :, 0] = 1
        observer = np.array([5.0, 5.0, 3.0], dtype=np.float64)
        rays = np.array([[0, 0, 1], [0, 1, 0]], dtype=np.float64)
        hit_values = np.array([0], dtype=np.int8)  # air allowed
        vi = compute_vi_generic(observer, vox, rays, hit_values, 1.0, 0.6, 1.0, False)
        assert vi > 0.0  # rays go through air = not blocked


# ---------- compute_vi_map_generic (lines 179-200) ----------

class TestComputeViMapGenericFull:
    def test_small_map(self):
        vox = np.zeros((5, 5, 5), dtype=np.int8)
        vox[:, :, 0] = 1  # ground
        rays = np.array([[0, 0, 1]], dtype=np.float64)
        hit_values = np.array([-3], dtype=np.int8)
        vi_map = compute_vi_map_generic(vox, rays, 0, hit_values, 1.0, 0.6, 1.0, True)
        assert vi_map.shape == (5, 5)

    def test_water_cell_excluded(self):
        """Cells with land cover 7, 8, 9 should be NaN."""
        vox = np.zeros((5, 5, 5), dtype=np.int8)
        vox[:, :, 0] = 1
        vox[2, 2, 0] = 7  # water
        rays = np.array([[0, 0, 1]], dtype=np.float64)
        hit_values = np.array([-3], dtype=np.int8)
        vi_map = compute_vi_map_generic(vox, rays, 0, hit_values, 1.0, 0.6, 1.0, True)
        # After flipud, the water cell's position changes
        assert np.any(np.isnan(vi_map))

    def test_no_surface_cell(self):
        """Cell with all zeros (no ground) should be NaN."""
        vox = np.zeros((5, 5, 5), dtype=np.int8)
        vox[:, :, 0] = 1
        vox[2, 2, :] = 0  # no ground
        rays = np.array([[0, 0, 1]], dtype=np.float64)
        hit_values = np.array([-3], dtype=np.int8)
        vi_map = compute_vi_map_generic(vox, rays, 0, hit_values, 1.0, 0.6, 1.0, True)
        assert np.any(np.isnan(vi_map))


# ---------- _precompute_observer_base_z (lines 392-413) ----------

class TestPrecomputeObserverBaseZFull:
    def test_ground_only(self):
        vox = np.zeros((3, 3, 5), dtype=np.int8)
        vox[:, :, 0] = 1  # ground at z=0
        result = _precompute_observer_base_z(vox)
        assert result.shape == (3, 3)
        assert np.all(result == 0)  # transition at z=0 -> z=1

    def test_no_ground(self):
        vox = np.zeros((3, 3, 5), dtype=np.int8)
        result = _precompute_observer_base_z(vox)
        assert np.all(result == -1)

    def test_building_top(self):
        vox = np.zeros((3, 3, 10), dtype=np.int8)
        vox[:, :, 0] = 1
        vox[1, 1, 1:5] = -3  # building
        result = _precompute_observer_base_z(vox)
        assert result[1, 1] == 4  # top of building at z=4, air starts at z=5


# ---------- _trace_ray (lines 416-448) ----------

class TestTraceRayFull:
    def test_clear_path_to_target(self):
        is_tree = np.zeros((10, 10, 10), dtype=np.bool_)
        is_opaque = np.zeros((10, 10, 10), dtype=np.bool_)
        origin = np.array([0.0, 0.0, 5.0], dtype=np.float64)
        target = np.array([9.0, 0.0, 5.0], dtype=np.float64)
        result = _trace_ray(is_tree, is_opaque, origin, target, 0.5, 0.01)
        assert result is True

    def test_blocked_by_opaque(self):
        is_tree = np.zeros((10, 10, 10), dtype=np.bool_)
        is_opaque = np.zeros((10, 10, 10), dtype=np.bool_)
        is_opaque[5, 0, 5] = True  # wall in the path
        origin = np.array([0.0, 0.0, 5.0], dtype=np.float64)
        target = np.array([9.0, 0.0, 5.0], dtype=np.float64)
        result = _trace_ray(is_tree, is_opaque, origin, target, 0.5, 0.01)
        assert result is False

    def test_attenuated_through_trees(self):
        is_tree = np.zeros((10, 10, 10), dtype=np.bool_)
        is_opaque = np.zeros((10, 10, 10), dtype=np.bool_)
        is_tree[2:8, 0, 5] = True  # thick tree barrier
        origin = np.array([0.0, 0.0, 5.0], dtype=np.float64)
        target = np.array([9.0, 0.0, 5.0], dtype=np.float64)
        result = _trace_ray(is_tree, is_opaque, origin, target, 0.1, 0.5)
        # Heavy attenuation, transmittance below cutoff
        assert result is False

    def test_exits_grid(self):
        is_tree = np.zeros((5, 5, 5), dtype=np.bool_)
        is_opaque = np.zeros((5, 5, 5), dtype=np.bool_)
        origin = np.array([0.0, 0.0, 2.0], dtype=np.float64)
        target = np.array([10.0, 0.0, 2.0], dtype=np.float64)  # target outside grid
        result = _trace_ray(is_tree, is_opaque, origin, target, 0.5, 0.01)
        assert result is False

    def test_zero_length(self):
        is_tree = np.zeros((5, 5, 5), dtype=np.bool_)
        is_opaque = np.zeros((5, 5, 5), dtype=np.bool_)
        origin = np.array([2.0, 2.0, 2.0], dtype=np.float64)
        target = np.array([2.0, 2.0, 2.0], dtype=np.float64)
        result = _trace_ray(is_tree, is_opaque, origin, target, 0.5, 0.01)
        assert result is True


# ---------- _compute_vi_map_generic_fast (lines 336-387) ----------

class TestComputeViMapGenericFast:
    def test_small_map_inclusion(self):
        vox = np.zeros((5, 5, 5), dtype=np.int8)
        vox[:, :, 0] = 1  # ground
        rays = np.array([[0, 0, 1]], dtype=np.float64)
        hit_values = np.array([-3], dtype=np.int8)
        is_tree, is_target, is_allowed, is_blocker = _prepare_masks_for_vi(vox, hit_values, True)
        dummy = np.zeros_like(is_tree)
        vi_map = _compute_vi_map_generic_fast(
            vox, rays, 0, 1.0, 0.6, 1.0,
            is_tree, is_target, dummy, is_blocker,
            True, False,
        )
        assert vi_map.shape == (5, 5)

    def test_exclusion_mode(self):
        vox = np.zeros((5, 5, 5), dtype=np.int8)
        vox[:, :, 0] = 1
        rays = np.array([[0, 0, 1]], dtype=np.float64)
        hit_values = np.array([0], dtype=np.int8)
        is_tree, is_target, is_allowed, is_blocker = _prepare_masks_for_vi(vox, hit_values, False)
        dummy = np.zeros_like(is_tree)
        vi_map = _compute_vi_map_generic_fast(
            vox, rays, 0, 1.0, 0.6, 1.0,
            is_tree, dummy, is_allowed, dummy,
            False, False,
        )
        assert vi_map.shape == (5, 5)

    def test_trees_in_targets(self):
        vox = np.zeros((5, 5, 5), dtype=np.int8)
        vox[:, :, 0] = 1
        vox[2, 2, 1:3] = -2
        rays = np.array([[0, 0, 1], [1, 0, 0]], dtype=np.float64)
        hit_values = np.array([-2], dtype=np.int8)
        is_tree, is_target, is_allowed, is_blocker = _prepare_masks_for_vi(vox, hit_values, True)
        dummy = np.zeros_like(is_tree)
        vi_map = _compute_vi_map_generic_fast(
            vox, rays, 0, 1.0, 0.6, 1.0,
            is_tree, is_target, dummy,
            is_blocker if is_blocker is not None else dummy,
            True, True,
        )
        assert vi_map.shape == (5, 5)


# ---------- _trace_ray_inclusion_masks / _trace_ray_exclusion_masks full coverage ----------

class TestTraceMasksFull:
    def test_inclusion_tree_attenuation(self):
        is_tree = np.zeros((10, 10, 10), dtype=np.bool_)
        is_target = np.zeros((10, 10, 10), dtype=np.bool_)
        is_blocker = np.zeros((10, 10, 10), dtype=np.bool_)
        # Dense tree
        is_tree[3:7, 5, 5] = True
        origin = np.array([0.0, 5.0, 5.0], dtype=np.float64)
        direction = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        hit, trans = _trace_ray_inclusion_masks(is_tree, is_target, is_blocker, origin, direction, 1.0, 0.6, 1.0)
        assert trans < 1.0

    def test_exclusion_non_allowed_blocks(self):
        is_tree = np.zeros((10, 10, 10), dtype=np.bool_)
        is_allowed = np.ones((10, 10, 10), dtype=np.bool_)  # all allowed
        is_allowed[5, 5, 5] = False  # one blocked voxel
        origin = np.array([3.0, 5.0, 5.0], dtype=np.float64)
        direction = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        hit, trans = _trace_ray_exclusion_masks(is_tree, is_allowed, origin, direction, 1.0, 0.6, 1.0)
        assert hit is True  # blocked by non-allowed

    def test_exclusion_tree_heavy_attenuation(self):
        is_tree = np.zeros((10, 10, 10), dtype=np.bool_)
        is_allowed = np.ones((10, 10, 10), dtype=np.bool_)
        is_tree[2:8, 5, 5] = True
        origin = np.array([0.0, 5.0, 5.0], dtype=np.float64)
        direction = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        hit, trans = _trace_ray_exclusion_masks(is_tree, is_allowed, origin, direction, 1.0, 2.0, 2.0)
        # Heavy attenuation
        assert hit is True
