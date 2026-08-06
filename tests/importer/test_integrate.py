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


def test_heights_stored_above_ground_with_nonzero_dem():
    """heights/min_heights must be stored above ground level, not as absolute
    voxel-index heights. With a non-zero, non-uniform DEM, ground_level at
    column (2, 3) is int(5.0/1.0 + 0.5) + 1 = 6. A building stamped at
    k=6,7,8 (i.e. starting exactly at ground_level) should report an
    above-ground span of [0.0, 3.0] and a top height of 3.0 -- not the old
    buggy absolute values of [6.0, 9.0] / 9.0.
    """
    vc = make_flat_voxcity(nx=10, ny=10, nz=12, meshsize=1.0)
    vc.dem.elevation[2, 3] = 5.0  # ground_level = int(5.0/1.0 + 0.5) + 1 = 6

    occ = {"b1": np.array([[2, 3, 6], [2, 3, 7], [2, 3, 8]], dtype=np.int64)}
    out = stamp_buildings(vc, occ)

    assert out.buildings.min_heights[2, 3] == [[0.0, 3.0]]
    assert out.buildings.heights[2, 3] == 3.0


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


def _building_ks(classes, i, j):
    """Voxel k indices occupied by a building in column (i, j)."""
    return np.flatnonzero(classes[i, j, :] == BUILDING_CODE).tolist()


def test_stamped_metadata_survives_regenerate_voxels():
    """stamp_buildings must derive heights/min_heights against the SAME ground
    datum the voxelizer uses, so that stamping and then regenerating the voxel
    grid is a fixed point. With a uniform non-zero DEM the old code was off by
    dem_min/meshsize cells, which made the imported building sink (or vanish)
    on the first edit."""
    from voxcity.generator.update import regenerate_voxels

    vc = make_flat_voxcity(nx=12, ny=12, nz=20, meshsize=1.0)
    vc.dem.elevation[:] = 10.0  # non-zero DEM minimum

    occ = {"b1": np.array([[4, 5, k] for k in range(1, 5)], dtype=np.int64)}
    out = stamp_buildings(vc, occ)

    before = _building_ks(out.voxels.classes, 4, 5)
    assert before == [1, 2, 3, 4]

    regenerate_voxels(out, inplace=True)
    assert _building_ks(out.voxels.classes, 4, 5) == before


def test_ground_datum_matches_voxelizer_on_sloped_dem():
    """On a sloped DEM the voxelizer flattens each building's footprint to that
    building's mean DEM via process_grid. stamp_buildings must measure its spans
    against that same flattened datum, not the raw per-cell DEM -- verified here
    by an actual round-trip through the real voxelizer, not by recomputing the
    expected value with the implementation's own process_grid call (which would
    be blind to drift in the datum's definition)."""
    from voxcity.generator.update import regenerate_voxels

    vc = make_flat_voxcity(nx=12, ny=12, nz=30, meshsize=1.0)
    for i in range(12):
        vc.dem.elevation[i, :] = 10.0 + i  # sloped, min = 10.0

    # One building straddling two DEM steps: the raw per-cell DEM gives the two
    # columns different ground levels, the voxelizer's footprint-mean gives them
    # the same one. Only the latter round-trips.
    occ = {"b1": np.array(
        [[4, 5, k] for k in range(6, 10)] + [[5, 5, k] for k in range(6, 10)],
        dtype=np.int64,
    )}
    out = stamp_buildings(vc, occ)

    before = {c: _building_ks(out.voxels.classes, *c) for c in [(4, 5), (5, 5)]}
    assert before[(4, 5)] and before[(5, 5)]  # non-vacuous: something was actually stamped

    regenerate_voxels(out, inplace=True)
    assert {c: _building_ks(out.voxels.classes, *c) for c in [(4, 5), (5, 5)]} == before


def test_flat_zero_dem_metadata_unchanged():
    """Guard on the no-op case: with a flat zero DEM the correction is
    identically zero, so the values match the raw-DEM computation."""
    vc = make_flat_voxcity(nx=10, ny=10, nz=8, meshsize=1.0)
    occ = {"b1": np.array([[2, 3, 1], [2, 3, 2], [2, 3, 3]], dtype=np.int64)}
    out = stamp_buildings(vc, occ)
    # ground_level = int(0/1 + 0.5) + 1 = 1; spans are k - 1.
    assert out.buildings.min_heights[2, 3] == [[0.0, 3.0]]
    assert out.buildings.heights[2, 3] == 3.0


def test_sequential_stamps_first_building_still_round_trips():
    """The ground datum is derived from the cumulative ids_grid as of each call,
    so a second, unrelated stamp_buildings call must not disturb the first
    building's already-written metadata. Uses a non-zero DEM so the datum
    correction is actually exercised."""
    from voxcity.generator.update import regenerate_voxels

    vc = make_flat_voxcity(nx=14, ny=14, nz=20, meshsize=1.0)
    vc.dem.elevation[:] = 10.0  # non-zero DEM minimum

    occ1 = {"b1": np.array([[3, 3, k] for k in range(1, 5)], dtype=np.int64)}
    out = stamp_buildings(vc, occ1)

    occ2 = {"b2": np.array([[9, 9, k] for k in range(1, 6)], dtype=np.int64)}
    out = stamp_buildings(out, occ2)

    before = _building_ks(out.voxels.classes, 3, 3)
    assert before  # non-vacuous

    regenerate_voxels(out, inplace=True)
    assert _building_ks(out.voxels.classes, 3, 3) == before


def test_non_float_dem_dtype_matches_float_equivalent():
    """np.asarray(..., dtype=float) is the only thing preventing integer
    truncation of the ground level when the DEM's own dtype is integral.

    A *uniform* int DEM would not exercise this at all: dem - dem.min() is zero
    regardless of dtype. The truncation only bites when a building's footprint
    straddles two DEM steps, because process_grid then writes that footprint's
    fractional mean back into the grid -- and an int grid floors it. Here the
    footprint spans dem 5 and 6, so the mean is 0.5 above the global min:
    float keeps ground_level = int(0.5 + 0.5) + 1 = 2, an int grid truncates the
    mean to 0 and yields 1, shifting the building a whole voxel.
    """
    def _model(dtype):
        vc = make_flat_voxcity(nx=10, ny=10, nz=16, meshsize=1.0)
        dem = np.full((10, 10), 5, dtype=dtype)
        dem[3, :] = 6
        vc.dem.elevation = dem
        return vc

    # Footprint straddles the 5/6 DEM step -> footprint mean is fractional.
    occ = {"b1": np.array(
        [[2, 3, k] for k in range(6, 9)] + [[3, 3, k] for k in range(6, 9)],
        dtype=np.int64,
    )}
    out_int = stamp_buildings(_model(np.int32), occ)
    out_float = stamp_buildings(_model(float), occ)

    # Non-vacuous: the float path must land on the fractional-mean datum, i.e.
    # ground_level 2, not the truncated 1.
    assert out_float.buildings.min_heights[2, 3] == [[4.0, 7.0]]

    for i, j in [(2, 3), (3, 3)]:
        assert out_int.buildings.min_heights[i, j] == out_float.buildings.min_heights[i, j]
        assert out_int.buildings.heights[i, j] == out_float.buildings.heights[i, j]
