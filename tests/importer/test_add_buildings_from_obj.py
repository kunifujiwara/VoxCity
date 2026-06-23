import numpy as np
import pytest

from voxcity.importer import add_buildings_from_obj
from voxcity.importer.transform import grid_geom_from_voxcity
from tests.importer.conftest import make_flat_voxcity

BUILDING_CODE = -3


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
