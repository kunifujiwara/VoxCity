import numpy as np

from voxcity.importer.integrate import stamp_buildings
from tests.importer.conftest import make_flat_voxcity

BUILDING_CODE = -3
GROUND_CODE = -1


def test_stamps_voxels_and_assigns_new_id():
    vc = make_flat_voxcity(nx=10, ny=10, nz=6, meshsize=1.0)
    # one building occupying column (2,3) at k=1,2,3
    occ = {"b1": np.array([[2, 3, 1], [2, 3, 2], [2, 3, 3]], dtype=np.int64)}
    out = stamp_buildings(vc, occ)
    assert out.voxels.classes[2, 3, 1] == BUILDING_CODE
    assert out.voxels.classes[2, 3, 3] == BUILDING_CODE
    # ground untouched
    assert out.voxels.classes[2, 3, 0] == GROUND_CODE
    # new id assigned at that column
    assert out.buildings.ids[2, 3] == 1
    # height grid = top k * meshsize (k=3 -> 3.0... top span end)
    assert out.buildings.heights[2, 3] > 0


def test_grows_z_when_taller_than_grid():
    vc = make_flat_voxcity(nx=8, ny=8, nz=4, meshsize=1.0)
    occ = {"tower": np.array([[1, 1, k] for k in range(1, 7)], dtype=np.int64)}
    out = stamp_buildings(vc, occ)
    assert out.voxels.classes.shape[2] >= 7
    assert out.voxels.classes[1, 1, 6] == BUILDING_CODE


def test_overwrite_false_yields_to_existing():
    vc = make_flat_voxcity(nx=8, ny=8, nz=6, meshsize=1.0)
    vc.voxels.classes[1, 1, 1] = BUILDING_CODE  # pre-existing building
    occ = {"b": np.array([[1, 1, 1], [1, 1, 2]], dtype=np.int64)}
    out = stamp_buildings(vc, occ, overwrite=False)
    # existing cell stays building, new cell added
    assert out.voxels.classes[1, 1, 1] == BUILDING_CODE
    assert out.voxels.classes[1, 1, 2] == BUILDING_CODE


def test_unique_ids_per_group_above_existing():
    vc = make_flat_voxcity(nx=8, ny=8, nz=6, meshsize=1.0)
    vc.buildings.ids[0, 0] = 7  # existing max id
    occ = {
        "a": np.array([[2, 2, 1]], dtype=np.int64),
        "b": np.array([[3, 3, 1]], dtype=np.int64),
    }
    out = stamp_buildings(vc, occ)
    ids = {int(out.buildings.ids[2, 2]), int(out.buildings.ids[3, 3])}
    assert ids == {8, 9}


def test_provenance_recorded():
    vc = make_flat_voxcity(nx=8, ny=8, nz=6, meshsize=1.0)
    occ = {"a": np.array([[2, 2, 1]], dtype=np.int64)}
    out = stamp_buildings(vc, occ, source="model.obj")
    assert "imported_buildings" in out.extras
    man = out.extras["imported_buildings"][-1]
    assert man["source"] == "model.obj"
    assert "id_map" in man
