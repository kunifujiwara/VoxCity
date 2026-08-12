import numpy as np
import pytest

from voxcity.simulator_gpu.solar.integration.caching import (
    CachedBuildingRadiationModel,
)
from voxcity.simulator_gpu.solar.surface_override import (
    SurfaceOverride, surface_override_signature, NO_SURFACE_OVERRIDE,
)


def _table(nz_value=1.0):
    return SurfaceOverride(
        cell=np.array([[1, 1, 1]], dtype=np.int32),
        face=np.array([0], dtype=np.int8),
        origin=np.array([[1.5, 1.5, 2.0]], dtype=np.float32),
        normal=np.array([[0.0, 0.0, nz_value]], dtype=np.float32),
        patch=np.array([3], dtype=np.int32),
        area=np.array([1.0], dtype=np.float32),
    )


def test_cache_dataclass_defaults_to_absent():
    c = CachedBuildingRadiationModel(
        model=None, voxcity_shape=(4, 4, 4), meshsize=1.0, n_reflection_steps=0,
        n_azimuth=40, n_elevation=10, is_building_surf=np.zeros(0, bool),
        building_svf_mesh=None,
    )
    assert c.surface_override_signature == NO_SURFACE_OVERRIDE
    assert c.surface_override_signature == surface_override_signature(None)


def test_different_tables_have_different_signatures():
    assert (surface_override_signature(_table(1.0))
            != surface_override_signature(_table(-1.0)))


# ---------------------------------------------------------------------------
# Live-model regression tests.
#
# The predicate test this replaced hand-copied the boolean expression from
# get_or_create_building_radiation_model (caching.py) rather than exercising
# the real code path: deleting `and cache.surface_override_signature ==
# ov_sig` from that function would not have made the old test fail, since the
# old test carried its own independent copy of the condition. These drive the
# real factory end to end (real Domain, real RadiationModel -- no
# monkeypatching) on a small grid with n_reflection_steps=0, which a prior
# reviewer measured at a few seconds on a 6x6x6 grid.
# ---------------------------------------------------------------------------

pytest.importorskip("taichi")

from tests.simulator._roof_helpers import make_voxcity_with_building  # noqa: E402
from voxcity.simulator_gpu.solar.integration import caching  # noqa: E402


def _override_table(cell=(2, 2, 2), nz_value=1.0):
    """A single-row table on a cell inside the building fixture's column."""
    return SurfaceOverride(
        cell=np.array([cell], dtype=np.int32),
        face=np.array([0], dtype=np.int8),
        origin=(np.array([cell], dtype=np.float32) + 0.5),
        normal=np.array([[0.0, 0.0, nz_value]], dtype=np.float32),
        patch=np.array([0], dtype=np.int32),
        area=np.array([1.0], dtype=np.float32),
    )


def test_different_tables_on_same_voxels_do_not_reuse_model():
    """Same voxel content, two different override tables -> cold-create both
    times; the second call must not reuse the first table's model."""
    caching.clear_building_radiation_model_cache()
    vc = make_voxcity_with_building(nx=6, ny=6, nz=6, bh=3)

    model1, _ = caching.get_or_create_building_radiation_model(
        vc, n_reflection_steps=0, surface_override=_override_table(nz_value=1.0))
    model2, _ = caching.get_or_create_building_radiation_model(
        vc, n_reflection_steps=0, surface_override=_override_table(nz_value=-1.0))

    assert model2 is not model1
    caching.clear_building_radiation_model_cache()


def test_same_table_on_same_voxels_reuses_model():
    """Same voxel content, same override table (a fresh but equal-content
    array each call, as a real caller would build) -> the second call must
    reuse the first model."""
    caching.clear_building_radiation_model_cache()
    vc = make_voxcity_with_building(nx=6, ny=6, nz=6, bh=3)

    model1, _ = caching.get_or_create_building_radiation_model(
        vc, n_reflection_steps=0, surface_override=_override_table(nz_value=1.0))
    model2, _ = caching.get_or_create_building_radiation_model(
        vc, n_reflection_steps=0, surface_override=_override_table(nz_value=1.0))

    assert model2 is model1
    caching.clear_building_radiation_model_cache()


def test_same_table_changed_voxels_refreshes_in_place():
    """Same override table, changed voxel content (taller building) -> the
    warm-refresh path fires: same model object, refreshed hash."""
    caching.clear_building_radiation_model_cache()
    vc1 = make_voxcity_with_building(nx=6, ny=6, nz=6, bh=3)
    vc2 = make_voxcity_with_building(nx=6, ny=6, nz=6, bh=4)

    model1, _ = caching.get_or_create_building_radiation_model(
        vc1, n_reflection_steps=0, surface_override=_override_table(nz_value=1.0))
    hash1 = caching.get_building_radiation_model_cache().voxel_data_hash

    model2, _ = caching.get_or_create_building_radiation_model(
        vc2, n_reflection_steps=0, surface_override=_override_table(nz_value=1.0))
    hash2 = caching.get_building_radiation_model_cache().voxel_data_hash

    assert model2 is model1, "same table + changed voxels must warm-refresh, not cold-create"
    assert hash1 != hash2
    caching.clear_building_radiation_model_cache()


def test_different_table_and_changed_voxels_cold_creates():
    """Different override table AND changed voxel content -> neither reuse
    nor warm-refresh applies; must cold-create a new model."""
    caching.clear_building_radiation_model_cache()
    vc1 = make_voxcity_with_building(nx=6, ny=6, nz=6, bh=3)
    vc2 = make_voxcity_with_building(nx=6, ny=6, nz=6, bh=4)

    model1, _ = caching.get_or_create_building_radiation_model(
        vc1, n_reflection_steps=0, surface_override=_override_table(nz_value=1.0))
    model2, _ = caching.get_or_create_building_radiation_model(
        vc2, n_reflection_steps=0, surface_override=_override_table(nz_value=-1.0))

    assert model2 is not model1
    caching.clear_building_radiation_model_cache()
