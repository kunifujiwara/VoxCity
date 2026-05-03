"""GPU surface view factor / landmark visibility coordinate-frame regression tests.

Phase 3 contract: voxel arrays are uv-domain (axis 0 = north/u, axis 1 = east/v);
mesh face geometry from `create_voxel_mesh` is in scene coordinates (x = v, y = u).
The CPU `get_surface_view_factor` converts scene -> uv before ray tracing via
`scene_points_to_uv_domain` / `scene_vectors_to_uv_domain`. The GPU equivalents
must do the same, otherwise the kernel reads scene-x as uv-u and produces a
transposed result.
"""
from types import SimpleNamespace

import numpy as np
import pytest


def _fake_voxcity(voxel_data, building_ids, meshsize=1.0):
    return SimpleNamespace(
        voxels=SimpleNamespace(
            classes=voxel_data,
            meta=SimpleNamespace(meshsize=meshsize),
        ),
        buildings=SimpleNamespace(ids=building_ids),
        extras={"rotation_angle": 0},
        dem=None,
    )


def _skip_if_gpu_unavailable(error):
    msg = str(error).lower()
    if "taichi" in msg or "cuda" in msg or "gpu" in msg or "device" in msg:
        pytest.skip(f"GPU/Taichi unavailable: {error}")
    raise error


def _asymmetric_layout():
    """Target at uv(u=2, v=2); tall obstructor only on the north side at uv(u=4, v=2).

    The target's north-facing wall must see less sky than its south/east/west walls.
    The asymmetry is along axis 0 (u/north) only, so any axis-swap bug in the GPU
    path will visibly disagree with the CPU oracle.
    """
    voxel_data = np.zeros((6, 6, 6), dtype=np.int32)
    voxel_data[2, 2, 1:3] = -3      # target building
    voxel_data[4, 2, 1:5] = -3      # tall north-side obstructor
    building_ids = np.zeros((6, 6), dtype=np.int32)
    building_ids[2, 2] = 1
    building_ids[4, 2] = 2
    return voxel_data, building_ids


def _target_face_mask(mesh):
    """Mask for faces belonging to the target building (centred at scene (2.5, 2.5))."""
    centers = mesh.triangles_center
    return (
        (centers[:, 0] >= 1.99) & (centers[:, 0] <= 3.01) &
        (centers[:, 1] >= 1.99) & (centers[:, 1] <= 3.01)
    )


def test_gpu_surface_view_factor_blocking_pattern_matches_cpu():
    """Pattern-level CPU/GPU agreement: the axis-swap bug shifts which faces
    are blocked/unblocked, so the per-face boolean ``vf > 0.5`` mask differs
    drastically. After the scene->uv conversion is applied at the call site,
    the mask must match. Magnitude differences on heavily-blocked faces are a
    separate ray-origin-offset issue and are not asserted here.
    """
    pytest.importorskip("taichi")
    from voxcity.simulator.visibility import get_surface_view_factor as cpu_svf
    from voxcity.simulator_gpu.visibility import get_surface_view_factor as gpu_svf

    voxel_data, building_ids = _asymmetric_layout()
    voxcity = _fake_voxcity(voxel_data, building_ids)

    cpu_mesh = cpu_svf(
        voxcity,
        target_values=(0,),
        inclusion_mode=False,
        N_azimuth=72,
        N_elevation=18,
    )
    try:
        gpu_mesh = gpu_svf(
            voxcity,
            target_values=(0,),
            inclusion_mode=False,
            N_azimuth=72,
            N_elevation=18,
        )
    except Exception as error:
        _skip_if_gpu_unavailable(error)

    cpu_values = cpu_mesh.metadata["view_factor_values"]
    gpu_values = gpu_mesh.metadata["view_factor_values"]
    assert cpu_values.shape == gpu_values.shape

    target_mask = _target_face_mask(cpu_mesh)
    finite = target_mask & np.isfinite(cpu_values) & np.isfinite(gpu_values)
    assert finite.any()

    np.testing.assert_array_equal(gpu_values[finite] > 0.5, cpu_values[finite] > 0.5)


def test_gpu_surface_view_factor_north_wall_blocked_by_north_obstructor():
    """Direct geometric check independent of CPU: the wall whose scene normal
    is +y (= +u in uv) must see less sky than the wall with scene normal -y.

    Scene mesh layout: face_normals[:, 1] is the y-component, which corresponds
    to uv axis 0 (north). With the tall obstructor at higher u, the +y face is
    blocked while -y face is open.
    """
    pytest.importorskip("taichi")
    from voxcity.simulator_gpu.visibility import get_surface_view_factor as gpu_svf

    voxel_data, building_ids = _asymmetric_layout()
    voxcity = _fake_voxcity(voxel_data, building_ids)

    try:
        mesh = gpu_svf(
            voxcity,
            target_values=(0,),
            inclusion_mode=False,
            N_azimuth=72,
            N_elevation=18,
        )
    except Exception as error:
        _skip_if_gpu_unavailable(error)

    values = mesh.metadata["view_factor_values"]
    target_mask = _target_face_mask(mesh)
    normals = np.rint(mesh.face_normals).astype(int)

    north_mask = target_mask & np.all(normals == np.array([0, 1, 0]), axis=1)
    south_mask = target_mask & np.all(normals == np.array([0, -1, 0]), axis=1)
    assert north_mask.any() and south_mask.any()

    north_vf = float(np.nanmean(values[north_mask]))
    south_vf = float(np.nanmean(values[south_mask]))

    assert north_vf < south_vf - 0.1


