import numpy as np
from voxcity.simulator_gpu.solar.integration.caching import _voxel_content_hash


def test_same_content_same_hash():
    a = np.zeros((4, 4, 4), dtype=np.int8)
    b = np.zeros((4, 4, 4), dtype=np.int8)
    assert _voxel_content_hash(a) == _voxel_content_hash(b)


def test_different_content_different_hash():
    a = np.zeros((4, 4, 4), dtype=np.int8)
    b = a.copy(); b[1, 1, 1] = -2
    assert _voxel_content_hash(a) != _voxel_content_hash(b)
