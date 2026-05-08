import numpy as np

from voxcity.generator.pipeline import _flatten_water_dem_by_component


def test_flatten_water_dem_flattens_connected_bodies_independently():
    dem = np.array(
        [
            [10.0, 11.0, 50.0, 51.0],
            [12.0, 13.0, 52.0, 53.0],
            [90.0, 91.0, 20.0, 21.0],
        ]
    )
    land_cover = np.array(
        [
            [6, 6, 0, 0],
            [6, 6, 0, 0],
            [0, 0, 6, 6],
        ]
    )

    flattened, info = _flatten_water_dem_by_component(
        dem, land_cover, "Urbanwatch", connectivity=4
    )

    assert np.all(flattened[:2, :2] == 10.0)
    assert np.all(flattened[2, 2:] == 20.0)
    assert flattened[0, 2] == 50.0
    assert info["applied"] is True
    assert info["water_body_count"] == 2
    assert info["water_cell_count"] == 6
    assert info["water_dem_min_values"] == [10.0, 20.0]


def test_flatten_water_dem_respects_connectivity():
    dem = np.array(
        [
            [4.0, 100.0],
            [100.0, 8.0],
        ]
    )
    land_cover = np.array(
        [
            [6, 0],
            [0, 6],
        ]
    )

    flattened_4, info_4 = _flatten_water_dem_by_component(
        dem, land_cover, "Urbanwatch", connectivity=4
    )
    flattened_8, info_8 = _flatten_water_dem_by_component(
        dem, land_cover, "Urbanwatch", connectivity=8
    )

    assert info_4["water_body_count"] == 2
    assert flattened_4[0, 0] == 4.0
    assert flattened_4[1, 1] == 8.0
    assert info_8["water_body_count"] == 1
    assert flattened_8[0, 0] == 4.0
    assert flattened_8[1, 1] == 4.0


def test_flatten_water_dem_can_be_disabled():
    dem = np.array([[10.0, 12.0]])
    land_cover = np.array([[6, 6]])

    flattened, info = _flatten_water_dem_by_component(
        dem, land_cover, "Urbanwatch", enabled=False
    )

    assert np.array_equal(flattened, dem)
    assert info["applied"] is False
    assert info["reason"] == "disabled"


def test_flatten_water_dem_ignores_nan_when_finding_component_minimum():
    dem = np.array([[np.nan, 3.0, 9.0]])
    land_cover = np.array([[6, 6, 0]])

    flattened, info = _flatten_water_dem_by_component(dem, land_cover, "Urbanwatch")

    assert flattened[0, 0] == 3.0
    assert flattened[0, 1] == 3.0
    assert flattened[0, 2] == 9.0
    assert info["water_dem_min_values"] == [3.0]