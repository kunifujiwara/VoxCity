from types import SimpleNamespace

import numpy as np
import pytest

from voxcity.simulator.common.coordinates import scene_points_to_uv_domain, scene_vectors_to_uv_domain
from voxcity.simulator.common.geometry import _generate_ray_directions_grid


def _fake_voxcity(voxel_data, building_ids, meshsize=1.0):
    return SimpleNamespace(
        voxels=SimpleNamespace(classes=voxel_data, meta=SimpleNamespace(meshsize=meshsize)),
        buildings=SimpleNamespace(ids=building_ids),
        extras={"rotation_angle": 0},
        dem=None,
    )


def _target_face_mask(mesh):
    centers = mesh.triangles_center
    return (
        (centers[:, 0] >= 5.99) & (centers[:, 0] <= 7.01) &
        (centers[:, 1] >= 5.99) & (centers[:, 1] <= 7.01)
    )


def _mean_by_scene_normal(mesh, values, normal):
    normals = np.rint(mesh.face_normals).astype(int)
    mask = _target_face_mask(mesh) & np.all(normals == np.asarray(normal), axis=1)
    finite = values[mask][np.isfinite(values[mask])]
    assert finite.size > 0
    return float(np.mean(finite))


def _skip_if_gpu_unavailable(error):
    message = str(error).lower()
    if "taichi" in message or "cuda" in message or "gpu" in message:
        pytest.skip(f"GPU/Taichi unavailable for this test: {error}")
    raise error


def _ray_is_clear(voxel_data, origin, direction):
    nx, ny, nz = voxel_data.shape
    x, y, z = origin.astype(float)
    dx, dy, dz = direction.astype(float)
    norm = np.linalg.norm(direction)
    assert norm > 0.0
    dx, dy, dz = dx / norm, dy / norm, dz / norm

    i, j, k = int(np.floor(x)), int(np.floor(y)), int(np.floor(z))
    step_x = 1 if dx >= 0 else -1
    step_y = 1 if dy >= 0 else -1
    step_z = 1 if dz >= 0 else -1
    big = 1e30

    t_max_x = (((i + (step_x > 0)) - x) / dx) if abs(dx) > 1e-12 else big
    t_max_y = (((j + (step_y > 0)) - y) / dy) if abs(dy) > 1e-12 else big
    t_max_z = (((k + (step_z > 0)) - z) / dz) if abs(dz) > 1e-12 else big
    t_delta_x = abs(1.0 / dx) if abs(dx) > 1e-12 else big
    t_delta_y = abs(1.0 / dy) if abs(dy) > 1e-12 else big
    t_delta_z = abs(1.0 / dz) if abs(dz) > 1e-12 else big

    while 0 <= i < nx and 0 <= j < ny and 0 <= k < nz:
        if voxel_data[i, j, k] not in (0, -2):
            return False

        t_next = min(t_max_x, t_max_y, t_max_z)
        if abs(t_max_x - t_next) <= 1e-12:
            t_max_x += t_delta_x
            i += step_x
        if abs(t_max_y - t_next) <= 1e-12:
            t_max_y += t_delta_y
            j += step_y
        if abs(t_max_z - t_next) <= 1e-12:
            t_max_z += t_delta_z
            k += step_z

    return True


def _reference_sky_diffuse(mesh, voxel_data, face_mask, dhi=100.0, n_azimuth=96, n_elevation=24):
    centers = scene_points_to_uv_domain(mesh.triangles_center)
    normals = scene_vectors_to_uv_domain(mesh.face_normals)
    directions = _generate_ray_directions_grid(n_azimuth, n_elevation, 0.0, 90.0)
    weights = np.sqrt(np.clip(1.0 - directions[:, 2] * directions[:, 2], 0.0, 1.0))
    denominator = float(np.sum(directions[:, 2] * weights))

    values = np.full(len(mesh.faces), np.nan, dtype=np.float64)
    for face_idx in np.where(face_mask)[0]:
        normal = normals[face_idx].astype(np.float64)
        normal /= np.linalg.norm(normal)
        origin = centers[face_idx].astype(np.float64) + normal * 1e-4
        numerator = 0.0
        for direction, weight in zip(directions, weights):
            if direction[2] <= 0.0:
                continue
            surface_cos = float(np.dot(normal, direction))
            if surface_cos <= 0.0:
                continue
            if _ray_is_clear(voxel_data, origin, direction):
                numerator += surface_cos * weight
        values[face_idx] = 0.0 if denominator <= 1e-12 else dhi * numerator / denominator
    return values


def test_gpu_building_diffuse_unobstructed_vertical_is_half_dhi():
    from voxcity.simulator_gpu.solar.integration.building import get_building_solar_irradiance
    from voxcity.simulator_gpu.solar.integration.caching import clear_all_caches

    voxel_data = np.zeros((12, 12, 8), dtype=np.int32)
    building_ids = np.zeros((12, 12), dtype=np.int32)
    voxel_data[6, 6, 1:5] = -3
    building_ids[6, 6] = 1
    voxcity = _fake_voxcity(voxel_data, building_ids)

    clear_all_caches()
    try:
        mesh = get_building_solar_irradiance(
            voxcity,
            azimuth_degrees_ori=0,
            elevation_degrees=5,
            direct_normal_irradiance=0,
            diffuse_irradiance=100,
            with_reflections=False,
        )
    except Exception as error:
        _skip_if_gpu_unavailable(error)

    for normal in [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0)]:
        assert _mean_by_scene_normal(mesh, mesh.metadata["diffuse"], normal) == pytest.approx(50.0, abs=5.0)
    assert _mean_by_scene_normal(mesh, mesh.metadata["diffuse"], (0, 0, 1)) == pytest.approx(100.0, abs=5.0)


@pytest.mark.parametrize(
    ("neighbor_uv", "scene_normal"),
    [
        ((6, 8), (1, 0, 0)),
        ((6, 4), (-1, 0, 0)),
        ((8, 6), (0, 1, 0)),
        ((4, 6), (0, -1, 0)),
    ],
)
def test_gpu_building_diffuse_obstructed_vertical_matches_face_reference(neighbor_uv, scene_normal):
    from voxcity.simulator_gpu.solar.integration.building import get_building_solar_irradiance
    from voxcity.simulator_gpu.solar.integration.caching import clear_all_caches

    voxel_data = np.zeros((12, 12, 8), dtype=np.int32)
    building_ids = np.zeros((12, 12), dtype=np.int32)
    voxel_data[6, 6, 1:5] = -3
    building_ids[6, 6] = 1
    voxel_data[neighbor_uv[0], neighbor_uv[1], 1:7] = -3
    building_ids[neighbor_uv[0], neighbor_uv[1]] = 2
    voxcity = _fake_voxcity(voxel_data, building_ids)

    clear_all_caches()
    try:
        mesh = get_building_solar_irradiance(
            voxcity,
            azimuth_degrees_ori=0,
            elevation_degrees=5,
            direct_normal_irradiance=0,
            diffuse_irradiance=100,
            with_reflections=False,
            n_azimuth=96,
            n_elevation=24,
        )
    except Exception as error:
        _skip_if_gpu_unavailable(error)

    normals = np.rint(mesh.face_normals).astype(int)
    face_mask = _target_face_mask(mesh) & np.all(normals == np.asarray(scene_normal), axis=1)
    reference = _reference_sky_diffuse(mesh, voxel_data, face_mask, dhi=100.0)

    gpu_value = _mean_by_scene_normal(mesh, mesh.metadata["diffuse"], scene_normal)
    ref_value = _mean_by_scene_normal(mesh, reference, scene_normal)
    assert gpu_value < 45.0
    assert gpu_value == pytest.approx(ref_value, abs=8.0)


def test_gpu_building_diffuse_honors_svf_sampling_kwargs():
    from voxcity.simulator_gpu.solar.integration.building import get_building_solar_irradiance
    from voxcity.simulator_gpu.solar.integration.caching import clear_all_caches, get_building_radiation_model_cache

    voxel_data = np.zeros((8, 8, 6), dtype=np.int32)
    building_ids = np.zeros((8, 8), dtype=np.int32)
    voxel_data[4, 4, 1:4] = -3
    building_ids[4, 4] = 1
    voxcity = _fake_voxcity(voxel_data, building_ids)

    clear_all_caches()
    try:
        get_building_solar_irradiance(
            voxcity,
            azimuth_degrees_ori=0,
            elevation_degrees=5,
            direct_normal_irradiance=0,
            diffuse_irradiance=100,
            with_reflections=False,
            n_azimuth=96,
            n_elevation=24,
        )
    except Exception as error:
        _skip_if_gpu_unavailable(error)
    model = get_building_radiation_model_cache().model
    assert model.config.n_azimuth == 96
    assert model.config.n_elevation == 24


def test_gpu_building_diffuse_default_sampling_stays_current_fast_default():
    from voxcity.simulator_gpu.solar.integration.building import get_building_solar_irradiance
    from voxcity.simulator_gpu.solar.integration.caching import clear_all_caches, get_building_radiation_model_cache

    voxel_data = np.zeros((8, 8, 6), dtype=np.int32)
    building_ids = np.zeros((8, 8), dtype=np.int32)
    voxel_data[4, 4, 1:4] = -3
    building_ids[4, 4] = 1
    voxcity = _fake_voxcity(voxel_data, building_ids)

    clear_all_caches()
    try:
        get_building_solar_irradiance(
            voxcity,
            azimuth_degrees_ori=0,
            elevation_degrees=5,
            direct_normal_irradiance=0,
            diffuse_irradiance=100,
            with_reflections=False,
        )
    except Exception as error:
        _skip_if_gpu_unavailable(error)
    model = get_building_radiation_model_cache().model
    assert model.config.n_azimuth == 40
    assert model.config.n_elevation == 10
