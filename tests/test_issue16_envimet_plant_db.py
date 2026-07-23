"""Regression test for issue #16: export_inx should also emit the ENVI-met
plant database (projectdatabase.edb) so the HxxW01 plant IDs in the INX
resolve on import."""

from voxcity.exporter.envimet import export_inx
from tests._envimet_helpers import make_minimal_city_with_trees


def test_export_inx_generates_plant_db(tmp_path):
    city = make_minimal_city_with_trees()
    export_inx(city, output_directory=str(tmp_path), file_basename="voxcity")
    assert (tmp_path / "voxcity.INX").exists()
    assert (tmp_path / "projectdatabase.edb").exists()


def test_export_inx_plant_db_opt_out(tmp_path):
    city = make_minimal_city_with_trees()
    export_inx(city, output_directory=str(tmp_path), file_basename="voxcity",
               generate_plant_db=False)
    assert (tmp_path / "voxcity.INX").exists()
    assert not (tmp_path / "projectdatabase.edb").exists()
