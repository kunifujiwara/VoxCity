import numpy as np
import pandas as pd
from pathlib import Path

from voxcity.utils.lc import rgb_distance, get_land_cover_classes, convert_land_cover
from voxcity.utils.material import get_material_dict, get_modulo_numbers, set_building_material_by_id
from voxcity.utils.weather import read_epw_for_solar_simulation


def test_lc_rgb_distance_and_classes():
    # Exact same color -> distance 0
    assert rgb_distance((255, 0, 0), (255, 0, 0)) == 0

    # Known class exists in mapping
    classes = get_land_cover_classes("OpenEarthMapJapan")
    assert classes[(222, 31, 7)] == "Building"


def test_lc_convert_land_cover_basic():
    # Urbanwatch simple mapping check (1-based indices)
    arr = np.array([[0, 1, 2]], dtype=np.uint8)
    converted = convert_land_cover(arr, land_cover_source="Urbanwatch")
    # Mapping: 0->13, 1->12, 2->11 (1-based: Building, Road, Developed space)
    assert converted.dtype == arr.dtype
    assert converted.tolist() == [[13, 12, 11]]


def test_material_dict_and_modulo_numbers():
    mat = get_material_dict()
    assert mat["brick"] == -11
    assert mat["glass"] == -16

    # Window ratio thresholds
    assert get_modulo_numbers(0.10) == (2, 2, 2)
    assert get_modulo_numbers(0.95) == (1, 1, 1)


def test_set_building_material_by_id_top_floor_no_glass():
    # Create a small grid (x,y,z) and a building id grid
    voxel = np.full((3, 3, 5), fill_value=-3, dtype=int)  # unknown everywhere
    building_ids = np.zeros((3, 3), dtype=int)
    building_ids[1, 1] = 1  # target building at center (flipud keeps row index 1)

    brick_id = get_material_dict()["brick"]
    glass_id = get_material_dict()["glass"]

    out = set_building_material_by_id(
        voxelcity_grid=voxel,
        building_id_grid_ori=building_ids,
        ids=[1],
        mark=brick_id,
        window_ratio=0.95,  # dense windows -> (1,1,1)
        glass_id=glass_id,
    )

    # At (1,1), all z except top should be glass; top should remain brick
    col = out[1, 1, :]
    assert col[-1] == brick_id
    assert np.all(col[:-1] == glass_id)


def test_weather_read_epw_for_solar_simulation(tmp_path: Path):
    # Minimal EPW with 8 headers and 2 data lines (ensure >30 columns per data line)
    epw_lines = []
    epw_lines.append("LOCATION,TestCity,TestState,TC,SRC,690070,36.68300,-121.7670,-8.0,43.0\n")
    for _ in range(7):
        epw_lines.append("HEADER,DUMMY\n")

    # Data columns: ensure at least 16 columns; indices used: 0..3, 14 (DNI), 15 (DHI)
    # Fill unused with zeros
    def make_data_line(year, month, day, hour, dni, dhi):
        vals = [str(year), str(month), str(day), str(hour)]
        # pad columns 4..13 with zeros
        vals += ["0"] * 10
        vals += [str(dni), str(dhi)]  # indices 14,15
        # pad a few more to exceed 30 columns
        vals += ["0"] * 16
        return ",".join(vals) + "\n"

    epw_lines.append(make_data_line(2020, 1, 1, 1, 100, 50))
    epw_lines.append(make_data_line(2020, 1, 1, 2, 200, 75))

    epw_path = tmp_path / "test.epw"
    epw_path.write_text("".join(epw_lines), encoding="utf-8")

    df, lon, lat, tz, elev = read_epw_for_solar_simulation(str(epw_path))
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2
    # LOCATION parsed values
    assert abs(lon - (-121.7670)) < 1e-6
    assert abs(lat - 36.68300) < 1e-6
    assert tz == -8.0

