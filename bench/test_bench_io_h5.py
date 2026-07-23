"""Benchmark HDF5 VoxCity round-trip: save_h5 / load_h5 (voxcity.io).

save_h5/load_h5 are the recommended (non-pickle) persistence path for a plain
VoxCity model (see src/voxcity/io.py). Benchmarked separately on the shared
small_city fixture (bench/conftest.py), writing into tmp_path so no repo
files are touched and no network is used.
"""

from voxcity.io import save_h5, load_h5


def test_bench_save_h5(benchmark, small_city, tmp_path):
    counter = {"n": 0}

    def _save():
        counter["n"] += 1
        path = tmp_path / f"bench_save_{counter['n']}.h5"
        save_h5(path, small_city)
        return path

    result = benchmark(_save)
    assert result.exists()


def test_bench_load_h5(benchmark, small_city, tmp_path):
    path = tmp_path / "bench_load.h5"
    save_h5(path, small_city)

    loaded = benchmark(load_h5, path)

    assert loaded.voxels.classes.shape == small_city.voxels.classes.shape
