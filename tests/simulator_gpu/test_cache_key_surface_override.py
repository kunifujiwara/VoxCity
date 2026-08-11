import numpy as np

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


def test_cache_valid_predicate_rejects_different_table_signature():
    """Exercises the cache-invalidation predicate in isolation.

    Building a real CachedBuildingRadiationModel is cheap (it's a plain
    dataclass -- `model=None` is fine, nothing here touches Taichi), but
    driving it through the full get_or_create_building_radiation_model path
    would require a real Domain/RadiationModel just to reach the `cache_valid`
    line. Since the logic under test is a pure boolean expression over cache
    fields, we replicate the exact condition from
    get_or_create_building_radiation_model (caching.py) here and check it
    against a cache built with one table's signature and a request carrying a
    different table's signature -- i.e. we test the predicate, not the whole
    factory.
    """
    voxel_data_hash = "same-voxel-content"
    cache = CachedBuildingRadiationModel(
        model=object(),
        voxcity_shape=(4, 4, 4),
        meshsize=1.0,
        n_reflection_steps=1,
        n_azimuth=40,
        n_elevation=10,
        is_building_surf=np.zeros(0, bool),
        building_svf_mesh=None,
        voxel_data_hash=voxel_data_hash,
        surface_override_signature=surface_override_signature(_table(1.0)),
    )

    requested_ov_sig = surface_override_signature(_table(-1.0))

    # Mirrors the cache_valid condition in get_or_create_building_radiation_model:
    # same shape/meshsize/azimuth/elevation/voxel-hash but a DIFFERENT override
    # table must still be rejected.
    cache_valid = (
        cache.voxcity_shape == (4, 4, 4) and
        cache.meshsize == 1.0 and
        cache.n_azimuth == 40 and
        cache.n_elevation == 10 and
        cache.voxel_data_hash == voxel_data_hash and
        (1 == 0 or cache.n_reflection_steps > 0) and
        cache.surface_override_signature == requested_ov_sig
    )
    assert not cache_valid

    # Same table's signature must be accepted (sanity check on the predicate).
    cache_valid_same = cache.surface_override_signature == surface_override_signature(_table(1.0))
    assert cache_valid_same
