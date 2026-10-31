"""Tests for the ``voxcity`` command line (voxcity.cli)."""

import numpy as np
import pytest

from tests.conftest import make_city
from voxcity import cli
from voxcity.io import save_voxcity


def _city_with_content():
    """A small city with a building, a tree and some terrain, so exports write data."""
    city = make_city(shape=(6, 7, 8), meshsize=2.0)
    city.voxels.classes[1:3, 1:3, 0:4] = -3        # building
    city.voxels.classes[4, 5, 0:2] = -2            # tree
    city.voxels.classes[:, :, 0][city.voxels.classes[:, :, 0] == 0] = 1
    city.buildings.heights[1:3, 1:3] = 8.0
    city.buildings.ids[1:3, 1:3] = 7
    city.dem.elevation[:] = np.linspace(0, 5, 6)[:, None]
    # 12 m north by 14 m east at the equator: a 6 x 7 grid of 2 m cells
    city.extras["rectangle_vertices"] = [(0.0, 0.0), (0.0, 0.0001085), (0.0001258, 0.0001085), (0.0001258, 0.0)]
    return city


@pytest.fixture
def model_h5(tmp_path):
    path = tmp_path / "model.h5"
    save_voxcity(str(path), _city_with_content())
    return path


# --- argument parsing --------------------------------------------------------

def test_bbox_becomes_sw_nw_ne_se_rectangle():
    assert cli.rectangle_from_bbox(-1.0, 50.0, -0.5, 50.5) == [(-1.0, 50.0), (-1.0, 50.5), (-0.5, 50.5), (-0.5, 50.0)]


def test_bbox_rejects_reversed_bounds():
    with pytest.raises(ValueError, match="west < east"):
        cli.rectangle_from_bbox(1.0, 50.0, -1.0, 50.5)


def test_center_and_size_make_a_square_of_that_size():
    from geopy.distance import distance
    rect = cli.rectangle_from_center(139.765, 35.681, 500)
    sw, nw, ne, _ = rect
    assert distance((sw[1], sw[0]), (nw[1], nw[0])).m == pytest.approx(500, rel=0.01)
    assert distance((nw[1], nw[0]), (ne[1], ne[0])).m == pytest.approx(500, rel=0.01)


def test_formats_are_parsed_deduplicated_and_validated():
    assert cli._formats("obj, ENVIMET,obj") == ["obj", "envimet"]
    with pytest.raises(Exception, match="unknown format"):
        cli._formats("obj,stl")


def test_generate_needs_an_area(capsys):
    with pytest.raises(SystemExit) as exc:
        cli.main(["generate"])
    assert exc.value.code == 2


def test_no_command_prints_usage():
    with pytest.raises(SystemExit) as exc:
        cli.main([])
    assert exc.value.code == 2


# --- generate ----------------------------------------------------------------

def test_generate_calls_get_voxcity_and_saves_and_exports(tmp_path, monkeypatch):
    import voxcity.generator
    calls = {}

    def fake_get_voxcity(rect, meshsize, **kwargs):
        calls["rect"], calls["meshsize"], calls["kwargs"] = rect, meshsize, kwargs
        return _city_with_content()

    monkeypatch.setattr(voxcity.generator, "get_voxcity", fake_get_voxcity)
    out = tmp_path / "seattle.h5"
    rc = cli.main(["generate", "--bbox", "-122.34", "47.60", "-122.33", "47.61", "--mesh", "4",
                   "--out", str(out), "--dem-source", "FABDEM", "--export", "obj"])
    assert rc == 0
    assert calls["rect"][0] == (-122.34, 47.60) and calls["meshsize"] == 4.0
    assert calls["kwargs"]["dem_source"] == "FABDEM"
    assert "building_source" not in calls["kwargs"]          # unset sources stay auto
    assert calls["kwargs"]["save_voxcity_data"] is False      # the CLI saves the file itself
    assert out.is_file()
    assert (tmp_path / "export" / "voxcity.obj").is_file()


def test_generate_center_without_size_is_an_error(tmp_path, capsys):
    rc = cli.main(["generate", "--center", "139.7", "35.6", "--out", str(tmp_path / "x.h5")])
    assert rc == 1
    assert "--size" in capsys.readouterr().err


# --- export ------------------------------------------------------------------

def test_export_writes_each_requested_format(model_h5, tmp_path):
    out = tmp_path / "exp"
    rc = cli.main(["export", str(model_h5), "--to", "obj,vox,netcdf,geotiff", "--export-dir", str(out), "--name", "m"])
    assert rc == 0
    assert (out / "m.obj").is_file() and (out / "m.mtl").is_file()
    assert (out / "m.nc").is_file()
    assert any(p.suffix == ".vox" for p in out.iterdir())
    assert any(p.name.startswith("m_") and p.suffix == ".tif" for p in out.iterdir())


def test_export_default_dir_is_next_to_the_model(model_h5):
    rc = cli.main(["export", str(model_h5), "--to", "obj"])
    assert rc == 0
    assert (model_h5.parent / "export" / "voxcity.obj").is_file()


def test_export_calls_every_writer_by_name(model_h5, tmp_path, monkeypatch):
    seen = []
    for fmt in cli.FORMATS:
        monkeypatch.setitem(cli.FORMATS, fmt, lambda city, out, name, fmt=fmt: seen.append((fmt, name)))
    rc = cli.main(["export", str(model_h5), "--to", ",".join(cli.FORMATS), "--export-dir", str(tmp_path / "e")])
    assert rc == 0
    assert [f for f, _ in seen] == list(cli.FORMATS)
    assert {n for _, n in seen} == {"voxcity"}


def test_export_missing_model_is_an_error(tmp_path, capsys):
    rc = cli.main(["export", str(tmp_path / "nope.h5"), "--to", "obj"])
    assert rc == 1
    assert "no such model file" in capsys.readouterr().err


# --- examples ----------------------------------------------------------------

def test_examples_list_prints_bundled_catalog(capsys):
    assert cli.main(["examples", "list"]) == 0
    out = capsys.readouterr().out
    assert "tokyo" in out and "new_york" in out


def test_examples_fetch_copies_and_exports(model_h5, tmp_path, monkeypatch):
    from voxcity import examples
    monkeypatch.setattr(examples, "example_city_path", lambda name, **kw: model_h5)
    out = tmp_path / "t.h5"
    rc = cli.main(["examples", "fetch", "Tokyo", "--out", str(out), "--export", "obj", "--export-dir", str(tmp_path / "e")])
    assert rc == 0
    assert out.is_file() and (tmp_path / "e" / "voxcity.obj").is_file()


def test_examples_fetch_unknown_city_is_an_error(capsys):
    rc = cli.main(["examples", "fetch", "atlantis"])
    assert rc == 1
    assert "atlantis" in capsys.readouterr().err


# --- info --------------------------------------------------------------------

def test_info_prints_grid_buildings_and_extent(model_h5, capsys):
    assert cli.main(["info", str(model_h5)]) == 0
    out = capsys.readouterr().out
    assert "6 x 7 x 8 voxels at 2.0 m" in out
    assert "buildings:  1" in out
    assert "lon 0.00000 to 0.00013" in out
