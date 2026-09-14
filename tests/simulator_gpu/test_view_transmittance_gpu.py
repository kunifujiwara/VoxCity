"""Single-ray tests pinning Beer-Lambert canopy attenuation in the view kernels.

Each test fires ONE ray so the expected value is analytic:
    T = exp(-tree_k * tree_lad * canopy_path_length_m)
with tree_k = 0.6, tree_lad = 1.0 (the privacy defaults).
"""
import math

import numpy as np
import pytest

pytest.importorskip("taichi")
import taichi as ti

from voxcity.simulator_gpu.visibility.integration import _get_or_create_domain
from voxcity.simulator_gpu.visibility.view import (
    SurfaceViewFactorCalculator,
    ViewCalculator,
)

TARGET = -31
TREE = -2
K, LAD = 0.6, 1.0

# ── surface fixture: face on x=1 (normal +x), target wall on x=11 ──────────
SNX, SNY, SNZ = 14, 10, 6


def _surface_grid():
    g = np.zeros((SNX, SNY, SNZ), dtype=np.int32)
    g[:, :, 0] = 1               # walkable ground
    g[11, :, 1:] = TARGET        # target wall, every y and z above ground
    return g


def _surface_value(grid, local_dir, target_values=(TARGET,)):
    """View factor of one face for one local ray direction (a, 0, b) -> world (b, a, 0)."""
    domain = _get_or_create_domain(SNX, SNY, SNZ, 1.0)
    calc = SurfaceViewFactorCalculator(domain, precompute_directions=False)
    calc._hemisphere_dirs = ti.Vector.field(3, dtype=ti.f32, shape=(1,))
    calc._hemisphere_dirs.from_numpy(np.asarray([local_dir], dtype=np.float32))
    calc._n_hemisphere_dirs = 1
    centers = np.array([[1.0, 2.5, 2.5]], dtype=np.float32)
    normals = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
    vals = calc.compute_surface_view_factor(
        centers, normals, grid,
        target_values=target_values, inclusion_mode=True, tree_k=K, tree_lad=LAD,
    )
    return float(vals[0])


STRAIGHT = (0.0, 0.0, 1.0)                       # world +x


def test_surface_clear_air_hit_scores_one():
    assert _surface_value(_surface_grid(), STRAIGHT) == pytest.approx(1.0)


def test_surface_one_canopy_voxel_scores_exp_minus_k_lad():
    g = _surface_grid()
    g[4, :, 1:] = TREE                            # 1 m of canopy on the ray
    assert _surface_value(g, STRAIGHT) == pytest.approx(math.exp(-K * LAD * 1.0), abs=2e-3)


def test_surface_two_canopy_voxels_scores_exp_minus_two_k_lad():
    g = _surface_grid()
    g[4:6, :, 1:] = TREE                          # 2 m of canopy on the ray
    assert _surface_value(g, STRAIGHT) == pytest.approx(math.exp(-K * LAD * 2.0), abs=2e-3)


def test_surface_green_mode_tree_hit_unchanged():
    """Trees-as-targets keeps its 1 - T semantics: this pins that Task 1 did not touch it."""
    g = _surface_grid()
    g[4, :, 1:] = TREE
    assert _surface_value(g, STRAIGHT, target_values=(TREE,)) == pytest.approx(
        1.0 - math.exp(-K * LAD * 1.0), abs=2e-3
    )
