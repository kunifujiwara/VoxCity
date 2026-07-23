"""Benchmark a CPU visibility/ray-tracing kernel: compute_vi_map_generic.

compute_vi_map_generic (src/voxcity/simulator/common/raytracing.py) is
@njit(parallel=True, cache=True) and is the kernel behind sky-view-factor /
green-view-index computation (see voxcity.simulator.visibility.view.get_view_index).
We build a realistic voxel grid with the real Voxelizer kernel (not timed),
then benchmark compute_vi_map_generic directly on a small ~48x48 grid with a
modest ray count, matching "sky view" usage (hit_values=(0,), inclusion_mode=False).
"""

import numpy as np

from voxcity.generator.voxelizer import Voxelizer
from voxcity.simulator.common.geometry import _generate_ray_directions_grid
from voxcity.simulator.common.raytracing import compute_vi_map_generic

SIZE = 48
VOXEL_SIZE = 2.0


def _make_voxel_grid(size=SIZE, seed=1):
    rng = np.random.default_rng(seed)

    has_building = rng.random((size, size)) < 0.2
    building_height_grid = np.where(has_building, rng.uniform(5.0, 30.0, (size, size)), 0.0)

    building_min_height_grid = np.empty((size, size), dtype=object)
    building_id_grid = np.zeros((size, size), dtype=np.int32)
    next_id = 1
    for i in range(size):
        for j in range(size):
            if has_building[i, j]:
                building_min_height_grid[i, j] = [[0.0, float(building_height_grid[i, j])]]
                building_id_grid[i, j] = next_id
                next_id += 1
            else:
                building_min_height_grid[i, j] = []

    land_cover_grid = rng.integers(0, 7, (size, size)).astype(np.int32)
    dem_grid = rng.uniform(0.0, 2.0, (size, size))

    has_tree = rng.random((size, size)) < 0.15
    tree_grid = np.where(has_tree, rng.uniform(3.0, 10.0, (size, size)), 0.0)

    voxelizer = Voxelizer(voxel_size=VOXEL_SIZE, land_cover_source="OpenStreetMap")
    voxel_data = voxelizer.generate_combined(
        building_height_grid,
        building_min_height_grid,
        building_id_grid,
        land_cover_grid,
        dem_grid,
        tree_grid,
        print_class_info=False,
    )
    # Cap vertical extent to keep the benchmark seconds-scale.
    return voxel_data[:, :, :24].copy()


def test_bench_compute_vi_map_generic(benchmark):
    voxel_data = _make_voxel_grid()

    # Modest ray count: 16 azimuths x 6 elevations = 96 rays (sky-view style).
    ray_directions = _generate_ray_directions_grid(
        N_azimuth=16, N_elevation=6, elevation_min_degrees=0, elevation_max_degrees=90
    )

    view_height_voxel = 1
    hit_values = (0,)  # sky mode: unobstructed sky voxel
    meshsize = VOXEL_SIZE
    tree_k = 0.6
    tree_lad = 1.0
    inclusion_mode = False

    kwargs = dict(
        voxel_data=voxel_data,
        ray_directions=ray_directions,
        view_height_voxel=view_height_voxel,
        hit_values=hit_values,
        meshsize=meshsize,
        tree_k=tree_k,
        tree_lad=tree_lad,
        inclusion_mode=inclusion_mode,
        include_building_roofs=False,
    )

    # Warm up numba JIT compilation outside the timed region.
    compute_vi_map_generic(**kwargs)

    result = benchmark(lambda: compute_vi_map_generic(**kwargs))

    assert result.shape == (SIZE, SIZE)
