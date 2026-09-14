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

from voxcity.models import (
    BuildingGrid, CanopyGrid, DemGrid, GridMetadata, LandCoverGrid, VoxCity, VoxelGrid,
)
from voxcity.simulator_gpu.visibility.integration import _get_or_create_domain
from voxcity.simulator_gpu.visibility.integration import get_view_index as get_view_index_public
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


def _surface_value(grid, local_dir, target_values=(TARGET,), meshsize=1.0):
    """View factor of one face for one local ray direction (a, 0, b) -> world (b, a, 0)."""
    domain = _get_or_create_domain(SNX, SNY, SNZ, meshsize)
    calc = SurfaceViewFactorCalculator(domain, precompute_directions=False)
    calc._hemisphere_dirs = ti.Vector.field(3, dtype=ti.f32, shape=(1,))
    calc._hemisphere_dirs.from_numpy(np.asarray([local_dir], dtype=np.float32))
    calc._n_hemisphere_dirs = 1
    # Face centers are in world metres; scale by meshsize so the ray origin still
    # lands in the same voxel indices the grid fixture was built for.
    centers = np.array([[1.0 * meshsize, 2.5 * meshsize, 2.5 * meshsize]], dtype=np.float32)
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


# ── ground fixture: observer at (2, 1), target wall on y=11 ──────────────────
GNX, GNY, GNZ = 10, 14, 10


def _ground_grid():
    g = np.zeros((GNX, GNY, GNZ), dtype=np.int32)
    g[:, :, 0] = 1
    g[:, 11, 1:] = TARGET
    return g


def _ground_value(grid, hit_values, elevation_deg=0.0, meshsize=1.0):
    domain = _get_or_create_domain(GNX, GNY, GNZ, meshsize)
    calc = ViewCalculator(domain, n_azimuth=1, n_elevation=1)
    mask = np.zeros((GNX, GNY), dtype=bool)
    mask[2, 1] = True
    vi = calc.compute_view_index(
        voxel_data=grid, hit_values=hit_values, inclusion_mode=True,
        view_point_height=1.5,
        elevation_min_degrees=elevation_deg, elevation_max_degrees=elevation_deg,
        tree_k=K, tree_lad=LAD, computation_mask=mask,
    )
    # Guards the module's single-ray analytic assumption: if this private name
    # drifts (e.g. n_azimuth/n_elevation stop producing exactly one direction),
    # every expected value above becomes wrong silently. Fail loudly instead.
    assert calc._n_ray_dirs == 1
    return float(vi[2, 1])


def test_ground_clear_air_hit_scores_one():
    assert _ground_value(_ground_grid(), (TARGET,)) == pytest.approx(1.0)


def test_ground_one_canopy_voxel_scores_exp_minus_k_lad():
    g = _ground_grid()
    g[:, 4, 1:] = TREE
    assert _ground_value(g, (TARGET,)) == pytest.approx(math.exp(-K * LAD * 1.0), abs=2e-3)


def test_ground_two_canopy_voxels_scores_exp_minus_two_k_lad():
    g = _ground_grid()
    g[:, 4:6, 1:] = TREE
    assert _ground_value(g, (TARGET,)) == pytest.approx(math.exp(-K * LAD * 2.0), abs=2e-3)


def test_ground_green_mode_tree_hit_unchanged():
    """hit_values containing -2 keeps the flat 1.0 per hit (GVI semantics untouched)."""
    g = _ground_grid()
    g[:, 4, 1:] = TREE
    assert _ground_value(g, (TREE,)) == pytest.approx(1.0)


# ── green mode through the PUBLIC entry point ─────────────────────────────
# The tests above call ViewCalculator.compute_view_index directly, which is not
# how the two live apps reach this code. This one goes through
# voxcity.simulator_gpu.visibility.integration.get_view_index with mode='green'
# so the derivation of trees_are_targets from mode/hit_values is exercised
# end to end (workspace caching included), not just the kernel it feeds.
def _make_voxcity_ground_canopy():
    """Same ground/observer geometry as `_ground_grid`, wrapped in a VoxCity
    object: ground plane at z=0, one canopy column at y=4 that the single ray
    from observer (2, 1) crosses before it would reach anything else."""
    nx, ny, nz = GNX, GNY, GNZ
    meshsize = 1.0
    classes = np.zeros((nx, ny, nz), dtype=np.int32)
    classes[:, :, 0] = 1
    classes[:, 4, 1:] = TREE

    lon0, lat0 = 0.0, 0.0
    dlat = (meshsize * nx) / 111320.0
    dlon = (meshsize * ny) / 111320.0
    rect = [(lon0, lat0), (lon0, lat0 + dlat), (lon0 + dlon, lat0 + dlat), (lon0 + dlon, lat0)]
    meta = GridMetadata(crs="EPSG:4326", bounds=(lon0, lat0, lon0 + dlon, lat0 + dlat), meshsize=meshsize)

    heights = np.zeros((nx, ny), dtype=float)
    ids = np.zeros((nx, ny), dtype=np.int32)
    min_heights = np.empty((nx, ny), dtype=object)
    for i in range(nx):
        for j in range(ny):
            min_heights[i, j] = []
    dem = np.zeros((nx, ny), dtype=float)
    lc = np.ones((nx, ny), dtype=np.int32)
    canopy = np.zeros((nx, ny), dtype=float)
    canopy[:, 4] = float(nz - 1)

    return VoxCity(
        voxels=VoxelGrid(classes=classes, meta=meta),
        buildings=BuildingGrid(heights=heights, min_heights=min_heights, ids=ids, meta=meta),
        land_cover=LandCoverGrid(classes=lc, meta=meta),
        dem=DemGrid(elevation=dem, meta=meta),
        tree_canopy=CanopyGrid(top=canopy, bottom=None, meta=meta),
        extras={"rectangle_vertices": rect},
    )


def test_green_mode_through_public_api_scores_flat_one_through_canopy():
    """mode='green' must derive trees_are_targets=True end to end through the
    public get_view_index() entry point, not just when the calculator is
    called directly.

    In green mode trees ARE targets, so the ray terminates at the first tree
    voxel it steps into -- but `_trace_ray_vi` multiplies `trans` by the tree
    attenuation BEFORE the target check runs on that same voxel, so a wrongly
    derived (or missing) trees_are_targets flag is observable here: the score
    would come out as exp(-k*lad) (~0.549) instead of the flat 1.0. That is
    exactly what green view index must NOT do -- both apps that import this
    package live depend on it staying flat through canopy.
    """
    voxcity = _make_voxcity_ground_canopy()
    mask = np.zeros((GNX, GNY), dtype=bool)
    mask[2, 1] = True
    vi_map = get_view_index_public(
        voxcity, mode='green',
        n_azimuth=1, n_elevation=1,
        elevation_min_degrees=0.0, elevation_max_degrees=0.0,
        view_point_height=1.5, tree_k=K, tree_lad=LAD,
        computation_mask=mask, show_plot=False,
    )
    assert vi_map[2, 1] == pytest.approx(1.0)


OBLIQUE_LOCAL = (0.4472136, 0.0, 0.8944272)       # world (0.894, 0.447, 0), tan = 0.5
CHORD = 1.0 / 0.8944272                            # 1.118 m through a 1 m column


def test_surface_oblique_ray_charges_chord_length():
    g = _surface_grid()
    g[4, :, 1:] = TREE
    assert _surface_value(g, OBLIQUE_LOCAL) == pytest.approx(math.exp(-K * LAD * CHORD), abs=2e-3)


def test_ground_oblique_ray_charges_chord_length():
    g = _ground_grid()
    g[:, 4, 1:] = TREE
    elev = math.degrees(math.atan(0.5))
    assert _ground_value(g, (TARGET,), elevation_deg=elev) == pytest.approx(
        math.exp(-K * LAD * CHORD), abs=2e-3)


# ── meshsize scaling (see docstrings below) ───────────────────────────────
def test_surface_meshsize_scales_the_charge():
    """Every other surface test in this module runs at meshsize=1.0, where a
    missing `* self.meshsize` factor in compute_surface_view_factor's tree_ext
    would go unnoticed. At meshsize=2.0 one tree voxel must cost exp(-K*LAD*2.0),
    twice the 1 m charge."""
    g = _surface_grid()
    g[4, :, 1:] = TREE
    assert _surface_value(g, STRAIGHT, meshsize=2.0) == pytest.approx(
        math.exp(-K * LAD * 2.0), abs=2e-3)


def test_ground_meshsize_scales_the_charge():
    """Every other ground test in this module runs at meshsize=1.0, where a
    missing `* self.dz` factor in compute_view_index's tree_ext would go
    unnoticed. At meshsize=2.0 one tree voxel must cost exp(-K*LAD*2.0), twice
    the 1 m charge."""
    g = _ground_grid()
    g[:, 4, 1:] = TREE
    assert _ground_value(g, (TARGET,), meshsize=2.0) == pytest.approx(
        math.exp(-K * LAD * 2.0), abs=2e-3)
