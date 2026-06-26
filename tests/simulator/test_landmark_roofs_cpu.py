import matplotlib
matplotlib.use("Agg")  # non-interactive backend — compute_landmark_visibility calls plt.show()

import numpy as np

from tests.simulator._roof_helpers import make_voxcity_with_building
from voxcity.simulator.visibility.landmark import compute_landmark_visibility


def _vis(voxel, include_roofs):
    # Mark a far cell as the landmark (value -30) so other cells can "see" it.
    voxel = voxel.copy()
    voxel[5, 5, 1] = -30
    return compute_landmark_visibility(
        voxel, target_value=-30, view_height_voxel=1,
        include_building_roofs=include_roofs,
    )


def test_landmark_excludes_building_by_default():
    vc = make_voxcity_with_building()
    m = _vis(vc.voxels.classes, include_roofs=False)
    assert np.isnan(m[2, 2])
    assert not np.isnan(m[0, 0])


def test_landmark_includes_building_roof_when_enabled():
    vc = make_voxcity_with_building()
    m = _vis(vc.voxels.classes, include_roofs=True)
    assert not np.isnan(m[2, 2])
