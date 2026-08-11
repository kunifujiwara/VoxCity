import numpy as np
import pytest

ti = pytest.importorskip("taichi")


@pytest.fixture(scope="module")
def _ti():
    from voxcity.simulator_gpu.init_taichi import ensure_initialized
    ensure_initialized()


def _empty_domain(n=6, nz=4):
    from voxcity.simulator_gpu.solar.domain import Domain
    d = Domain(nx=n, ny=n, nz=nz, dx=1.0, dy=1.0, dz=1.0,
               origin_lat=35.0, origin_lon=139.0)
    occ = np.zeros((n, n, nz), dtype=np.int32)
    # One solid cell so occupancy-based extraction yields >0 surfaces.
    # RadiationModel allocates its (self.n_surfaces,) buffers -- and, when
    # surface_reflections/cache_svf_matrix are enabled (both default True),
    # its SVF/CSF sparse-matrix buffers -- eagerly in __init__; a fully air
    # domain (0 surfaces) makes those zero-sized, and Taichi rejects
    # zero-sized field allocation outright. That is a pre-existing gap in
    # RadiationModel unrelated to what this test checks (cell_patch
    # defaulting/upload), so we sidestep it here rather than widen this
    # dispatch's scope to make zero-surface domains generally supported.
    occ[0, 0, 0] = 1
    d.is_solid.from_numpy(occ)
    return d


def test_injected_surfaces_size_the_internal_buffers(_ti):
    from voxcity.simulator_gpu.solar.domain import surfaces_from_override
    from voxcity.simulator_gpu.solar.radiation import RadiationModel, RadiationConfig
    from voxcity.simulator_gpu.solar.surface_override import SurfaceOverride

    n_rows = 7
    ov = SurfaceOverride(
        cell=np.tile([[2, 2, 1]], (n_rows, 1)).astype(np.int32),
        face=np.full(n_rows, 4, dtype=np.int8),
        origin=np.tile([[3.0, 2.5, 1.5]], (n_rows, 1)).astype(np.float32),
        normal=np.tile([[1.0, 0.0, 0.0]], (n_rows, 1)).astype(np.float32),
        patch=np.arange(n_rows, dtype=np.int32),
        area=np.ones(n_rows, dtype=np.float32),
    )
    surfaces = surfaces_from_override(ov)
    model = RadiationModel(_empty_domain(), RadiationConfig(skip_svf=True),
                           surfaces=surfaces)
    assert model.n_surfaces == n_rows
    assert model._surfinswdir.shape[0] == n_rows      # buffers sized to injection


def test_cell_patch_defaults_to_minus_one(_ti):
    """No grid supplied -> every cell reports NO_PATCH -> skip never fires."""
    from voxcity.simulator_gpu.solar.radiation import RadiationModel, RadiationConfig

    model = RadiationModel(_empty_domain(), RadiationConfig(skip_svf=True))
    assert (model.cell_patch.to_numpy() == -1).all()


def test_supplied_cell_patch_is_uploaded(_ti):
    from voxcity.simulator_gpu.solar.radiation import RadiationModel, RadiationConfig

    grid = np.full((6, 6, 4), -1, dtype=np.int32)
    grid[2, 2, 1] = 42
    model = RadiationModel(_empty_domain(), RadiationConfig(skip_svf=True),
                           cell_patch=grid)
    assert int(model.cell_patch.to_numpy()[2, 2, 1]) == 42
