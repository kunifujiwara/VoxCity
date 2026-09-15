"""CPU parity: every CPU inclusion-mode scorer must weight a non-tree target hit by the
surviving canopy transmittance, matching the GPU kernels."""
import math

import numpy as np
import pytest

from voxcity.simulator.common.raytracing import (
    _compute_vi_map_generic_fast,
    compute_vi_generic,
)
from voxcity.simulator.visibility.view import _ray_visibility_contrib

TARGET = -31
TREE = -2
K, LAD = 0.6, 1.0


def _ground_grid():
    g = np.zeros((10, 14, 10), dtype=np.int32)
    g[:, :, 0] = 1
    g[:, 11, 1:] = TARGET
    return g


def _generic(grid, hit_values):
    observer = np.array([2.0, 1.0, 2.0])
    direction = np.array([[0.0, 1.0, 0.0]])
    return compute_vi_generic(observer, grid, direction, np.array(hit_values), 1.0, K, LAD, True)


def _fast(grid, hit_values):
    is_tree = grid == TREE
    is_target = np.isin(grid, hit_values)
    is_blocker = (grid != 0) & ~is_tree & ~is_target
    vi = _compute_vi_map_generic_fast(
        grid, np.array([[0.0, 1.0, 0.0]]), 1, 1.0, K, LAD,
        is_tree, is_target, is_target.copy(), is_blocker, True, TREE in hit_values,
    )
    return float(vi[2, 1])


def _surface(grid, target_values):
    is_tree = grid == TREE
    is_target = np.isin(grid, target_values)
    is_opaque = (grid != 0) & ~is_tree & ~is_target
    att = math.exp(-K * LAD * 1.0)
    return _ray_visibility_contrib(
        np.array([1.51, 2.5, 2.5]), np.array([1.0, 0.0, 0.0]),
        is_tree, is_target, is_target.copy(), is_opaque,
        att, 0.01, True, TREE in target_values,
    )


def _surface_grid():
    g = np.zeros((14, 10, 6), dtype=np.int32)
    g[:, :, 0] = 1
    g[11, :, 1:] = TARGET
    return g


@pytest.mark.parametrize("scorer, make_grid, tree_slice", [
    (_generic, _ground_grid, (slice(None), 4, slice(1, None))),
    (_fast, _ground_grid, (slice(None), 4, slice(1, None))),
    (_surface, _surface_grid, (4, slice(None), slice(1, None))),
])
def test_cpu_one_canopy_voxel_scores_transmittance(scorer, make_grid, tree_slice):
    clear = make_grid()
    assert scorer(clear, [TARGET]) == pytest.approx(1.0)
    g = make_grid()
    g[tree_slice] = TREE
    assert scorer(g, [TARGET]) == pytest.approx(math.exp(-K * LAD), abs=1e-6)


@pytest.mark.parametrize("scorer, make_grid, tree_slice", [
    (_generic, _ground_grid, (slice(None), 4, slice(1, None))),
    (_fast, _ground_grid, (slice(None), 4, slice(1, None))),
    (_surface, _surface_grid, (4, slice(None), slice(1, None))),
])
def test_cpu_green_mode_unchanged(scorer, make_grid, tree_slice):
    g = make_grid()
    g[tree_slice] = TREE
    assert scorer(g, [TREE]) == pytest.approx(1.0 - math.exp(-K * LAD), abs=1e-6)
