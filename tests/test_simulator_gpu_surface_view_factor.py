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


def test_reuse_reference_mesh_skips_create_voxel_mesh(monkeypatch):
    """reuse_reference_mesh=True must bypass create_voxel_mesh and trace the
    supplied reference geometry (used by the optimizer where buildings/windows
    are static and only trees move)."""
    pytest.importorskip("taichi")
    from voxcity.simulator_gpu.visibility import integration

    voxel_data, building_ids = _asymmetric_layout()
    voxcity = _fake_voxcity(voxel_data, building_ids)

    ref = integration.get_surface_view_factor(
        voxcity, target_values=(0,), inclusion_mode=False,
        N_azimuth=24, N_elevation=6,
    )
    assert ref is not None and len(ref.faces) > 0

    def _boom(*args, **kwargs):
        raise AssertionError("create_voxel_mesh should not be called when reusing")

    monkeypatch.setattr("voxcity.geoprocessor.mesh.create_voxel_mesh", _boom)

    mesh = integration.get_surface_view_factor(
        voxcity, target_values=(0,), inclusion_mode=False,
        N_azimuth=24, N_elevation=6,
        reuse_reference_mesh=True, reference_mesh=ref,
    )
    assert mesh is ref
    assert "view_factor_values" in mesh.metadata





def _glazed_slot(same_building: bool):
    """Two parallel glazed walls (window class -16) facing each other across a
    ~4-voxel gap. When same_building, both share footprint id 1, so a window
    face on wall A seeing wall B is a SELF view the guard must drop. When not,
    the walls are buildings 1 and 2 and the guard must keep the cross view."""
    voxel_data = np.zeros((12, 12, 8), dtype=np.int32)
    voxel_data[3, 2:10, 1:6] = -16   # wall A (glazed), faces +u toward B
    voxel_data[8, 2:10, 1:6] = -16   # wall B (glazed), faces -u toward A
    building_ids = np.zeros((12, 12), dtype=np.int32)
    building_ids[3, 2:10] = 1
    building_ids[8, 2:10] = 1 if same_building else 2
    return voxel_data, building_ids


def test_surface_self_occlusion_guard_drops_same_building_targets():
    pytest.importorskip("taichi")
    from voxcity.simulator_gpu.visibility import get_surface_view_factor as gpu_svf

    def run(same_building, **extra):
        voxel_data, building_ids = _glazed_slot(same_building)
        voxcity = _fake_voxcity(voxel_data, building_ids)
        try:
            mesh = gpu_svf(voxcity, target_values=(-16,), inclusion_mode=True,
                           N_azimuth=48, N_elevation=12, **extra)
        except Exception as error:
            _skip_if_gpu_unavailable(error)
        return mesh.metadata["view_factor_values"]

    base = run(True)                                  # no guard arg
    off = run(True, self_occlusion_guard=False)       # explicit off -> equals base
    same_on = run(True, self_occlusion_guard=True)    # same building -> self view dropped
    cross_on = run(False, self_occlusion_guard=True)  # different buildings -> view kept

    np.testing.assert_allclose(np.nan_to_num(off), np.nan_to_num(base), rtol=0, atol=0)
    assert np.nanmean(base) > 0.05, "scene should have a non-trivial across-gap window view"
    assert np.nanmean(same_on) < np.nanmean(base) - 0.02, "guard must drop same-building self view"
    assert np.nanmean(cross_on) > np.nanmean(base) - 0.005, "guard must keep cross-building view"


def test_surface_guard_reuses_workspace_building_ids_field(monkeypatch):
    """The guarded surface path must not allocate a new Taichi field per call.

    The optimizer evaluates ``get_surface_view_factor`` once per candidate
    through the cached ``SurfaceViewWorkspace``; allocating a fresh ``(nx, ny)``
    building-id field on every call leaks GPU memory (Taichi cannot free fields
    without ``ti.reset()``). The building-id grid must therefore live in the
    workspace and be repopulated in place, like ``is_target``/``is_opaque``.
    """
    pytest.importorskip("taichi")
    import taichi as ti
    from voxcity.simulator_gpu.visibility import get_surface_view_factor as gpu_svf

    voxel_data, building_ids = _glazed_slot(True)
    voxcity = _fake_voxcity(voxel_data, building_ids)

    call = dict(target_values=(-16,), inclusion_mode=True,
               N_azimuth=24, N_elevation=6, self_occlusion_guard=True)

    # Warm-up: legitimately allocates the domain + workspace fields once.
    try:
        gpu_svf(voxcity, **call)
    except Exception as error:
        _skip_if_gpu_unavailable(error)

    # A second identical call reuses the cached workspace, so it must allocate
    # zero new Taichi fields.
    n_new = {"count": 0}
    real_field = ti.field

    def counting_field(*args, **kwargs):
        n_new["count"] += 1
        return real_field(*args, **kwargs)

    monkeypatch.setattr(ti, "field", counting_field)
    gpu_svf(voxcity, **call)

    assert n_new["count"] == 0, (
        f"guarded surface path allocated {n_new['count']} new Taichi field(s) on a "
        "cached-workspace call; building_ids must be reused from the workspace"
    )
