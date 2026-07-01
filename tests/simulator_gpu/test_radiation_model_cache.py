import numpy as np
import pytest

pytest.importorskip("taichi")

from tests.simulator._roof_helpers import make_voxcity_with_building
from voxcity.simulator_gpu.solar.integration import caching
from voxcity.simulator_gpu.solar.integration.caching import (
    get_or_create_radiation_model, clear_radiation_model_cache,
)


def test_radiation_model_cache_refreshes_on_content_change():
    clear_radiation_model_cache()
    vc1 = make_voxcity_with_building(bh=4)
    m1, _, _ = get_or_create_radiation_model(vc1, n_reflection_steps=1)
    hash1 = caching.get_radiation_model_cache().voxel_data_hash

    # Same shape, different content (taller building) must invalidate the cache.
    vc2 = make_voxcity_with_building(bh=6)
    m2, _, _ = get_or_create_radiation_model(vc2, n_reflection_steps=1)
    hash2 = caching.get_radiation_model_cache().voxel_data_hash

    assert hash1 != hash2                 # cache was refreshed, not reused
