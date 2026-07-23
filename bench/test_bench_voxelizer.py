"""Benchmark the numba voxelization kernel (Voxelizer.generate_combined).

Exercises the real @jit(nopython=True, parallel=True) kernel in
src/voxcity/generator/voxelizer.py (_voxelize_kernel) via the public
Voxelizer.generate_combined API, on a synthetic ~200x200 grid with a mix of
buildings, trees, land cover, and terrain -- the same code path used by the
real generation pipeline.
"""

import numpy as np

from voxcity.generator.voxelizer import Voxelizer

SIZE = 200
VOXEL_SIZE = 2.0


def _make_grids(size=SIZE, seed=0):
    rng = np.random.default_rng(seed)

    # ~20% of cells have a building; heights 5-50 m.
    has_building = rng.random((size, size)) < 0.2
    building_height_grid = np.where(has_building, rng.uniform(5.0, 50.0, (size, size)), 0.0)

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
    dem_grid = rng.uniform(0.0, 5.0, (size, size))

    has_tree = rng.random((size, size)) < 0.15
    tree_grid = np.where(has_tree, rng.uniform(3.0, 15.0, (size, size)), 0.0)

    return (
        building_height_grid,
        building_min_height_grid,
        building_id_grid,
        land_cover_grid,
        dem_grid,
        tree_grid,
    )


def test_bench_voxelizer_generate_combined(benchmark):
    grids = _make_grids()
    voxelizer = Voxelizer(voxel_size=VOXEL_SIZE, land_cover_source="OpenStreetMap")

    # Warm up numba JIT compilation outside the timed region.
    voxelizer.generate_combined(*grids, print_class_info=False)

    result = benchmark(voxelizer.generate_combined, *grids, print_class_info=False)

    assert result.shape[:2] == (SIZE, SIZE)
