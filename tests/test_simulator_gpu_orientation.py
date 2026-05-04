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


def _fake_voxcity_with_buildings(voxel_data, building_ids, meshsize=1.0):
    voxcity = _fake_voxcity(voxel_data, meshsize=meshsize)
    voxcity.buildings = SimpleNamespace(ids=building_ids)
    return voxcity


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


def test_gpu_building_solar_maps_solver_surfaces_to_scene_mesh_coordinates():
    from voxcity.simulator_gpu.solar.integration.building import get_building_solar_irradiance
    from voxcity.simulator_gpu.solar.integration.caching import (
        clear_all_caches,
        get_building_radiation_model_cache,
    )

    voxel_data = np.zeros((4, 2, 4), dtype=np.int32)
    voxel_data[3, 1, 1:3] = -3
    building_ids = np.zeros((4, 2), dtype=np.int32)
    building_ids[3, 1] = 99
    voxcity = _fake_voxcity_with_buildings(voxel_data, building_ids)

    clear_all_caches()
    mesh = get_building_solar_irradiance(
        voxcity,
        azimuth_degrees_ori=0,
        elevation_degrees=45,
        direct_normal_irradiance=0,
        diffuse_irradiance=100,
        with_reflections=False,
    )

    cache = get_building_radiation_model_cache()
    assert cache is not None
    assert cache.mesh_to_surface_idx is not None

    n_surfaces = cache.model.surfaces.count
    solver_centers = cache.model.surfaces.center.to_numpy()[:n_surfaces]
    solver_normals = cache.model.surfaces.normal.to_numpy()[:n_surfaces]

    valid_mapping = cache.mesh_to_surface_idx >= 0
    mapped_centers = solver_centers[cache.mesh_to_surface_idx[valid_mapping]].copy()
    mapped_centers[:, [0, 1]] = mapped_centers[:, [1, 0]]
    mapped_normals = solver_normals[cache.mesh_to_surface_idx[valid_mapping]].copy()
    mapped_normals[:, [0, 1]] = mapped_normals[:, [1, 0]]

    distances = np.linalg.norm(mapped_centers - mesh.triangles_center[valid_mapping], axis=1)
    normal_dots = np.einsum("ij,ij->i", mapped_normals, mesh.face_normals[valid_mapping])

    assert distances.max() < 0.75
    assert np.nanmin(normal_dots) > 0.99
    assert np.all(mesh.face_normals[~valid_mapping, 2] < -0.9)
    assert np.all(np.isnan(mesh.metadata["global"][~valid_mapping]))


# ---------------------------------------------------------------------------
# Volumetric sun-direction orientation tests
# ---------------------------------------------------------------------------

class _FakeSolidField:
    def __init__(self, shape):
        self._shape = shape

    def to_numpy(self):
        return np.zeros(self._shape, dtype=np.int32)


class _FakeVolumetricCalculator:
    def __init__(self, shape):
        self.shape = shape
        self.calls = []
        self.c2s_matrix_cached = False

    def compute_swflux_vol(self, *, sw_direct, sw_diffuse, cos_zenith, sun_direction, lad):
        self.calls.append(
            {
                "sw_direct": sw_direct,
                "sw_diffuse": sw_diffuse,
                "cos_zenith": cos_zenith,
                "sun_direction": sun_direction,
                "lad": lad,
            }
        )

    def get_swflux_vol(self):
        return np.ones(self.shape, dtype=np.float32)

    def init_cumulative_accumulation(self, **kwargs):
        pass

    def accumulate_terrain_following_slice_gpu(self, weight=1.0):
        pass

    def finalize_cumulative_map(self, apply_nan_mask=True):
        return np.ones(self.shape[:2], dtype=np.float32)


def test_gpu_volumetric_sun_direction_uses_u_north_v_east(monkeypatch):
    from voxcity.simulator_gpu.solar.integration import volumetric

    voxel_data = _ground_grid(shape=(4, 3, 4))
    voxcity = _fake_voxcity(voxel_data)
    calc = _FakeVolumetricCalculator(voxel_data.shape)
    domain = SimpleNamespace(is_solid=_FakeSolidField(voxel_data.shape), lad=None)

    monkeypatch.setattr(volumetric, "get_or_create_volumetric_calculator", lambda *args, **kwargs: (calc, domain))
    monkeypatch.setattr(volumetric, "_compute_ground_k_from_voxels", lambda _voxel_data: np.zeros(_voxel_data.shape[:2], dtype=np.int32))

    result = volumetric.get_volumetric_solar_irradiance_map(
        voxcity,
        azimuth_degrees_ori=0,
        elevation_degrees=45,
        direct_normal_irradiance=1000,
        diffuse_irradiance=0,
        volumetric_height=1.0,
        with_reflections=False,
    )

    assert result.shape == voxel_data.shape[:2]
    sun_direction = calc.calls[0]["sun_direction"]
    assert sun_direction[0] > 0
    assert abs(sun_direction[1]) < 1e-12
    assert sun_direction[2] > 0


def test_gpu_volumetric_sun_direction_applies_grid_rotation(monkeypatch):
    from voxcity.simulator_gpu.solar.integration import volumetric

    voxel_data = _ground_grid(shape=(4, 3, 4))
    voxcity = _fake_voxcity(voxel_data)
    voxcity.extras["rotation_angle"] = 90
    calc = _FakeVolumetricCalculator(voxel_data.shape)
    domain = SimpleNamespace(is_solid=_FakeSolidField(voxel_data.shape), lad=None)

    monkeypatch.setattr(volumetric, "get_or_create_volumetric_calculator", lambda *args, **kwargs: (calc, domain))
    monkeypatch.setattr(volumetric, "_compute_ground_k_from_voxels", lambda _voxel_data: np.zeros(_voxel_data.shape[:2], dtype=np.int32))

    volumetric.get_volumetric_solar_irradiance_map(
        voxcity,
        azimuth_degrees_ori=90,
        elevation_degrees=45,
        direct_normal_irradiance=1000,
        diffuse_irradiance=0,
        volumetric_height=1.0,
        with_reflections=False,
    )

    sun_direction = calc.calls[0]["sun_direction"]
    assert sun_direction[0] > 0
    assert abs(sun_direction[1]) < 1e-12


# ---------------------------------------------------------------------------
# Cumulative volumetric coverage tests
# ---------------------------------------------------------------------------

def test_gpu_volumetric_sun_direction_helper_cardinals():
    from voxcity.simulator_gpu.solar.integration.volumetric import _compute_volumetric_sun_direction

    voxel_data = _ground_grid(shape=(4, 3, 4))
    voxcity = _fake_voxcity(voxel_data)

    north, _ = _compute_volumetric_sun_direction(voxcity, 0, 45)
    east, _ = _compute_volumetric_sun_direction(voxcity, 90, 45)
    south, _ = _compute_volumetric_sun_direction(voxcity, 180, 45)
    west, _ = _compute_volumetric_sun_direction(voxcity, 270, 45)

    assert north[0] > 0 and abs(north[1]) < 1e-12
    assert east[1] > 0 and abs(east[0]) < 1e-12
    assert south[0] < 0 and abs(south[1]) < 1e-12
    assert west[1] < 0 and abs(west[0]) < 1e-12


def test_gpu_cumulative_volumetric_reuses_phase3_sun_direction(monkeypatch):
    import pandas as pd
    from voxcity.simulator_gpu.solar.integration import volumetric

    voxel_data = _ground_grid(shape=(4, 3, 4))
    voxcity = _fake_voxcity(voxel_data)
    calc = _FakeVolumetricCalculator(voxel_data.shape)
    domain = SimpleNamespace(is_solid=_FakeSolidField(voxel_data.shape), lad=None)

    def fake_solar_positions(index, lon, lat):
        return pd.DataFrame({"azimuth": [0.0], "elevation": [45.0]}, index=index)

    monkeypatch.setattr(volumetric, "get_or_create_volumetric_calculator", lambda *args, **kwargs: (calc, domain))
    monkeypatch.setattr(volumetric, "get_solar_positions_astral", fake_solar_positions)
    monkeypatch.setattr(volumetric, "_compute_ground_k_from_voxels", lambda _voxel_data: np.zeros(_voxel_data.shape[:2], dtype=np.int32))

    df = pd.DataFrame(
        {"DNI": [1000.0], "DHI": [0.0]},
        index=pd.DatetimeIndex(["2026-01-01 12:00:00"]),
    )

    volumetric.get_cumulative_volumetric_solar_irradiance(
        voxcity,
        df,
        lon=0.0,
        lat=0.0,
        tz=0.0,
        volumetric_height=1.0,
        with_reflections=False,
        use_sky_patches=False,
    )

    sun_direction = calc.calls[0]["sun_direction"]
    assert sun_direction[0] > 0
    assert abs(sun_direction[1]) < 1e-12
