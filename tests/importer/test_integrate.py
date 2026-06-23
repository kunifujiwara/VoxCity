import logging

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


def test_cross_group_column_collision_logs_warning(caplog, propagate_voxcity_logs):
    """Two groups in the same call that both touch column (i, j) must not
    silently clobber the ids_grid entry -- a warning must be logged, and
    (per last-group-wins, dict insertion order) the *second* group inserted
    ("b") wins the column since occupied_by_name.items() iterates in
    insertion order in Python 3.7+."""
    vc = make_flat_voxcity(nx=8, ny=8, nz=6, meshsize=1.0)
    occ = {
        "a": np.array([[2, 2, 1]], dtype=np.int64),
        "b": np.array([[2, 2, 3]], dtype=np.int64),  # same (i, j), different k
    }

    with caplog.at_level(logging.WARNING, logger="voxcity"):
        out = stamp_buildings(vc, occ)

    man = out.extras["imported_buildings"][-1]
    id_map = man["id_map"]
    assert id_map["a"] != id_map["b"]
    # "b" was inserted after "a", so it is processed second and wins the column.
    assert int(out.buildings.ids[2, 2]) == id_map["b"]
    assert "collision" in caplog.text.lower() or "already" in caplog.text.lower()


def test_out_of_bounds_group_skipped_without_consuming_id(caplog, propagate_voxcity_logs):
    """A group entirely outside (i, j) bounds must not consume an id or
    appear in the manifest's id_map, and the next real group must still get
    the id that would have gone to it had the bad group never run."""
    vc = make_flat_voxcity(nx=8, ny=8, nz=6, meshsize=1.0)
    expected_next_id = int(vc.buildings.ids.max()) + 1

    occ = {
        "bad": np.array([[100, 100, 1], [-5, -5, 1]], dtype=np.int64),  # all out of bounds
        "good": np.array([[3, 3, 1]], dtype=np.int64),
    }

    with caplog.at_level(logging.WARNING, logger="voxcity"):
        out = stamp_buildings(vc, occ)

    man = out.extras["imported_buildings"][-1]
    id_map = man["id_map"]
    assert "bad" not in id_map
    assert id_map["good"] == expected_next_id
    assert int(out.buildings.ids[3, 3]) == expected_next_id
    assert "bad" in caplog.text
