import numpy as np
import pytest

ti = pytest.importorskip("taichi")


@pytest.fixture(scope="module")
def _ti():
    from voxcity.simulator_gpu.init_taichi import ensure_initialized
    ensure_initialized()


def test_patch_defaults_to_no_patch(_ti):
    """Surfaces built the old way must report NO_PATCH, so nothing is skipped."""
    from voxcity.simulator_gpu.solar.domain import Surfaces
    from voxcity.simulator_gpu.solar.surface_override import NO_PATCH

    s = Surfaces(16)

    @ti.kernel
    def add():
        s.add_surface(1, 2, 3, 0,
                      ti.Vector([1.5, 2.5, 4.0]), ti.Vector([0.0, 0.0, 1.0]),
                      1.0, 0.2)

    add()
    assert s.count == 1
    assert int(s.patch_id.to_numpy()[0]) == NO_PATCH
