import numpy as np
import pytest
import trimesh

from voxcity.importer import add_buildings_from_obj
from voxcity.importer.transform import grid_geom_from_voxcity
from tests.importer.conftest import make_flat_voxcity

BUILDING_CODE = -3
GLASS_CODE = -16


def test_end_to_end_box_import(box_obj_factory):
    vc = make_flat_voxcity(nx=30, ny=30, nz=10, meshsize=1.0)
    geom = grid_geom_from_voxcity(vc)
    # anchor the model origin a few meters into the domain
    proj_origin = geom["origin"]
    obj = box_obj_factory(origin=(0.0, 0.0, 0.0), size=(3.0, 3.0, 4.0), name="b1")
    out = add_buildings_from_obj(
        vc, obj,
        anchor_lonlat=(float(proj_origin[0]), float(proj_origin[1])),
        anchor_elevation=0.0,
        anchor_model_point=(0.0, 0.0, 0.0),
        move=(5.0, 5.0, 0.0),   # 5 m east, 5 m north
        rotation=0.0, units="m",
    )
    # building voxels should appear near columns (5..7 north, 5..7 east), above ground
    sub = out.voxels.classes[5:8, 5:8, 1:5]
    assert np.any(sub == BUILDING_CODE)
    # ids assigned somewhere
    assert out.buildings.ids.max() >= 1


def test_missing_file_raises(flat_voxcity, tmp_path):
    with pytest.raises(FileNotFoundError):
        add_buildings_from_obj(
            flat_voxcity, tmp_path / "nope.obj",
            anchor_lonlat=(0.0, 0.0), anchor_elevation=0.0,
        )


def test_invalid_units_raises(flat_voxcity, box_obj_factory):
    obj = box_obj_factory()
    with pytest.raises(ValueError, match="Unknown units"):
        add_buildings_from_obj(
            flat_voxcity, obj, anchor_lonlat=(0.0, 0.0),
            anchor_elevation=0.0, units="furlong",
        )


def test_meshlib_backend_not_installed_raises(flat_voxcity, box_obj_factory, monkeypatch):
    obj = box_obj_factory()
    # simulate meshlib missing
    import builtins
    real_import = builtins.__import__

    def fake_import(name, *a, **k):
        if name.startswith("meshlib"):
            raise ImportError("no meshlib")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(ImportError, match="meshlib"):
        add_buildings_from_obj(
            flat_voxcity, obj, anchor_lonlat=(0.0, 0.0),
            anchor_elevation=0.0, backend="meshlib",
        )


def test_original_object_not_mutated(box_obj_factory):
    vc = make_flat_voxcity(nx=30, ny=30, nz=10, meshsize=1.0)
    before = vc.voxels.classes.copy()
    geom = grid_geom_from_voxcity(vc)
    obj = box_obj_factory(size=(3.0, 3.0, 4.0))
    add_buildings_from_obj(
        vc, obj,
        anchor_lonlat=(float(geom["origin"][0]), float(geom["origin"][1])),
        anchor_elevation=0.0, move=(5.0, 5.0, 0.0),
    )
    assert np.array_equal(vc.voxels.classes, before)  # input untouched (copy returned)


def _box_with_window_obj(tmp_path):
    """OBJ with a solid box 'BuildingA' and a planar window pane 'Windows' on
    the box's -Y face (model y=0), inset from the edges."""
    box = trimesh.creation.box(extents=(3.0, 3.0, 4.0))
    box.apply_translation((1.5, 1.5, 2.0))  # min corner at origin
    pane_v = np.array(
        [[0.5, 0.0, 0.5], [2.5, 0.0, 0.5], [2.5, 0.0, 3.5], [0.5, 0.0, 3.5]],
        dtype=float,
    )
    pane = trimesh.Trimesh(vertices=pane_v, faces=np.array([[0, 1, 2], [0, 2, 3]]),
                           process=False)
    scene = trimesh.Scene()
    scene.add_geometry(box, node_name="BuildingA", geom_name="BuildingA")
    scene.add_geometry(pane, node_name="Windows", geom_name="Windows")
    path = tmp_path / "bld_with_window.obj"
    scene.export(str(path))
    return path


def test_import_window_group_produces_glass_voxels(tmp_path):
    vc = make_flat_voxcity(nx=30, ny=30, nz=10, meshsize=1.0)
    geom = grid_geom_from_voxcity(vc)
    obj = _box_with_window_obj(tmp_path)
    out = add_buildings_from_obj(
        vc, obj,
        anchor_lonlat=(float(geom["origin"][0]), float(geom["origin"][1])),
        anchor_elevation=0.0, anchor_model_point=(0.0, 0.0, 0.0),
        move=(5.0, 5.0, 0.0), rotation=0.0, units="m",
    )
    assert np.any(out.voxels.classes == BUILDING_CODE)
    assert np.any(out.voxels.classes == GLASS_CODE)
    manifest = out.extras["imported_buildings"][-1]
    assert manifest["n_window_voxels"] > 0


def test_building_only_import_has_no_glass(box_obj_factory):
    vc = make_flat_voxcity(nx=30, ny=30, nz=10, meshsize=1.0)
    geom = grid_geom_from_voxcity(vc)
    obj = box_obj_factory(size=(3.0, 3.0, 4.0))
    out = add_buildings_from_obj(
        vc, obj,
        anchor_lonlat=(float(geom["origin"][0]), float(geom["origin"][1])),
        anchor_elevation=0.0, move=(5.0, 5.0, 0.0),
    )
    assert not np.any(out.voxels.classes == GLASS_CODE)
    assert out.extras["imported_buildings"][-1].get("n_window_voxels", 0) == 0


def _window_only_obj(tmp_path):
    """OBJ with only window-role panes, no building geometry at all.

    Needs >=2 named groups: trimesh collapses a single-group OBJ to one
    unnamed Trimesh (see load_obj_groups's docstring), which would lose the
    "Windows" name entirely and default to the building role.
    """
    pane_v = np.array(
        [[0.5, 0.0, 0.5], [2.5, 0.0, 0.5], [2.5, 0.0, 3.5], [0.5, 0.0, 3.5]],
        dtype=float,
    )
    pane_a = trimesh.Trimesh(vertices=pane_v, faces=np.array([[0, 1, 2], [0, 2, 3]]),
                             process=False)
    pane_b = trimesh.Trimesh(vertices=pane_v + [3.0, 0.0, 0.0],
                             faces=np.array([[0, 1, 2], [0, 2, 3]]), process=False)
    scene = trimesh.Scene()
    scene.add_geometry(pane_a, node_name="Windows_A", geom_name="Windows_A")
    scene.add_geometry(pane_b, node_name="Windows_B", geom_name="Windows_B")
    path = tmp_path / "window_only.obj"
    scene.export(str(path))
    return path


def test_window_only_import_skipped_with_warning(tmp_path):
    import logging as _logging

    # voxcity's package logger has propagate=False (see utils/logging.py), so
    # caplog (which hooks the root logger) never sees these records; attach a
    # handler directly to the package logger to observe them instead.
    records = []
    handler = _logging.Handler()
    handler.emit = lambda record: records.append(record)
    pkg_logger = _logging.getLogger("voxcity")
    pkg_logger.addHandler(handler)
    try:
        vc = make_flat_voxcity(nx=30, ny=30, nz=10, meshsize=1.0)
        before = vc.voxels.classes.copy()
        geom = grid_geom_from_voxcity(vc)
        obj = _window_only_obj(tmp_path)
        out = add_buildings_from_obj(
            vc, obj,
            anchor_lonlat=(float(geom["origin"][0]), float(geom["origin"][1])),
            anchor_elevation=0.0, move=(5.0, 5.0, 0.0),
        )
    finally:
        pkg_logger.removeHandler(handler)
    assert np.array_equal(out.voxels.classes, before)
    assert out.extras.get("imported_buildings") in (None, [])
    assert any("no building-role geometry" in r.getMessage().lower() for r in records)
