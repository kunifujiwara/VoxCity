"""Window/glass (-16) cells must count as building surface in the GPU path.

These pin the two GPU-side fixes that let building-surface simulation include
windows: (1) the solid classification treats -16 as solid so the radiation
domain generates surfaces there, and (2) the building-surface class group
includes -16 so those surfaces are marked (and valued) as building.
"""
import numpy as np

from voxcity.simulator_gpu.solar.integration.utils import (
    convert_voxel_data_to_arrays,
    VOXCITY_WINDOW_CODE,
)
from voxcity.simulator_gpu.solar.integration.caching import BUILDING_SURFACE_CLASSES


def test_window_code_is_minus_16():
    assert VOXCITY_WINDOW_CODE == -16


def test_window_voxels_classified_solid():
    vox = np.zeros((3, 3, 3), dtype=np.int32)
    vox[1, 1, 0] = -3   # building
    vox[1, 1, 1] = -16  # window/glass on top
    vox[0, 0, 0] = -2   # tree (must stay non-solid)
    is_solid, lad = convert_voxel_data_to_arrays(vox)
    assert is_solid[1, 1, 1] == 1   # window is solid
    assert is_solid[1, 1, 0] == 1   # building is solid
    assert is_solid[0, 0, 0] == 0   # tree is not solid
    assert lad[0, 0, 0] > 0         # tree still carries LAD


def test_building_surface_group_includes_window():
    assert -3 in BUILDING_SURFACE_CLASSES
    assert -16 in BUILDING_SURFACE_CLASSES
