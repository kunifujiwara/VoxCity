import numpy as np
import pytest

pytest.importorskip("taichi")

from tests.simulator._roof_helpers import make_voxcity_with_building
from voxcity.simulator_gpu.visibility.integration import get_landmark_visibility_map_gpu


def _vis(vc, include_roofs):
    return get_landmark_visibility_map_gpu(
        vc, landmark_building_ids=[101], view_point_height=1.5,
        show_plot=False, include_building_roofs=include_roofs,
    )


def test_gpu_landmark_excludes_other_buildings_by_default():
    vc = make_voxcity_with_building()
    # Add a second building at (4,4) to use as observer; building 101 is the landmark.
    vc.voxels.classes[4, 4, 1:3] = -3
    vc.buildings.ids[4, 4] = 202
    m, _ = _vis(vc, include_roofs=False)
    assert np.isnan(m[4, 4])     # non-landmark building roof excluded by default


def test_gpu_landmark_includes_building_roof_when_enabled():
    vc = make_voxcity_with_building()
    vc.voxels.classes[4, 4, 1:3] = -3
    vc.buildings.ids[4, 4] = 202
    m, _ = _vis(vc, include_roofs=True)
    assert np.isfinite(m[4, 4])  # non-landmark building roof is now a valid observer
