from types import SimpleNamespace

import numpy as np
import pytest


def _fake_voxcity(voxel_data, meshsize=1.0):
    return SimpleNamespace(
        voxels=SimpleNamespace(
            classes=voxel_data,
            meta=SimpleNamespace(meshsize=meshsize),
        ),
        extras={"rotation_angle": 0},
        dem=None,
    )


def _ground_grid(shape=(4, 1, 4)):
    voxel_data = np.zeros(shape, dtype=np.int32)
    voxel_data[:, :, 0] = 1
    return voxel_data


def test_gpu_compute_sun_direction_uses_u_north_v_east():
    from voxcity.simulator_gpu.solar.integration.utils import compute_sun_direction

    du, dv, dz, _ = compute_sun_direction(0, 45, 0)
    assert du > 0
    assert abs(dv) < 1e-12
    assert dz > 0

    du, dv, dz, _ = compute_sun_direction(90, 45, 0)
    assert abs(du) < 1e-12
    assert dv > 0
    assert dz > 0


def test_gpu_calc_zenith_direction_matches_u_north_v_east_azimuth():
    from voxcity.simulator_gpu.solar.solar import calc_zenith

    pos = calc_zenith(day_of_year=120, second_of_day=9 * 3600, latitude=35.0, longitude=0.0)
    azimuth = np.deg2rad(pos.azimuth_angle)
    elevation = np.deg2rad(pos.elevation_angle)

    assert pos.direction[0] == pytest.approx(np.cos(elevation) * np.cos(azimuth))
    assert pos.direction[1] == pytest.approx(np.cos(elevation) * np.sin(azimuth))


def test_gpu_direct_solar_preserves_uv_layout():
    from voxcity.simulator_gpu.solar.integration.caching import clear_all_caches
    from voxcity.simulator_gpu.solar.integration.ground import get_direct_solar_irradiance_map

    voxel_data = _ground_grid()
    voxel_data[3, 0, 1:3] = -3
    voxcity = _fake_voxcity(voxel_data)

    clear_all_caches()
    result = get_direct_solar_irradiance_map(
        voxcity,
        azimuth_degrees_ori=0,
        elevation_degrees=45,
        direct_normal_irradiance=1000,
        view_point_height=0,
        with_reflections=False,
    )

    assert result.shape == voxel_data.shape[:2]
    assert result[2, 0] == 0
    assert result[1, 0] > 0
    assert result[0, 0] > 0


def test_gpu_visibility_preserves_uv_layout():
    from voxcity.simulator_gpu.domain import Domain
    from voxcity.simulator_gpu.visibility.view import ViewCalculator

    voxel_data = _ground_grid()
    voxel_data[3, 0, 1:3] = -3
    domain = Domain(nx=4, ny=1, nz=4, dx=1.0, dy=1.0, dz=1.0)
    calc = ViewCalculator(domain, n_azimuth=4, n_elevation=1)

    result = calc.compute_view_index(
        voxel_data=voxel_data,
        mode="sky",
        view_point_height=0,
        elevation_min_degrees=90,
        elevation_max_degrees=90,
    )

    assert result.shape == voxel_data.shape[:2]
    assert np.isnan(result[3, 0])
    assert result[2, 0] == pytest.approx(1.0)
    assert result[0, 0] == pytest.approx(1.0)


def test_gpu_landmark_mark_building_by_id_preserves_uv_layout():
    from voxcity.simulator_gpu.visibility.landmark import mark_building_by_id

    voxel_data = np.zeros((4, 1, 3), dtype=np.int32)
    voxel_data[2, 0, 1:] = -3
    building_id_grid = np.zeros((4, 1), dtype=np.int32)
    building_id_grid[2, 0] = 99

    marked = mark_building_by_id(voxel_data, building_id_grid, [99], -30)

    assert np.all(marked[2, 0, 1:] == -30)
    assert not np.any(marked[1, 0, :] == -30)


def test_gpu_apply_computation_mask_to_faces_requires_uv_shape_and_no_flip():
    from voxcity.simulator_gpu.solar.integration.utils import apply_computation_mask_to_faces

    values = np.array([1.0, 2.0], dtype=np.float64)
    centers = np.array([[0.5, 0.5, 1.0], [1.5, 0.5, 1.0]], dtype=np.float64)
    mask = np.array([[False, True]], dtype=bool)

    masked = apply_computation_mask_to_faces(
        values,
        centers,
        mask,
        meshsize=1.0,
        grid_shape=(1, 2),
    )

    assert np.isnan(masked[0])
    assert masked[1] == 2.0

    with pytest.raises(ValueError, match="computation_mask"):
        apply_computation_mask_to_faces(
            values,
            centers,
            mask.T,
            meshsize=1.0,
            grid_shape=(1, 2),
        )


def test_gpu_global_solar_rejects_transposed_computation_mask(monkeypatch):
    from voxcity.simulator_gpu.solar.integration import ground

    voxel_data = _ground_grid()
    voxcity = _fake_voxcity(voxel_data)
    direct = np.ones(voxel_data.shape[:2], dtype=np.float64)
    diffuse = np.full(voxel_data.shape[:2], 2.0, dtype=np.float64)

    monkeypatch.setattr(ground, "get_direct_solar_irradiance_map", lambda *args, **kwargs: direct)
    monkeypatch.setattr(ground, "get_diffuse_solar_irradiance_map", lambda *args, **kwargs: diffuse)

    with pytest.raises(ValueError, match="computation_mask"):
        ground.get_global_solar_irradiance_map(
            voxcity,
            azimuth_degrees_ori=0,
            elevation_degrees=45,
            direct_normal_irradiance=1000,
            diffuse_irradiance=100,
            computation_mask=np.ones((1, 4), dtype=bool),
        )
