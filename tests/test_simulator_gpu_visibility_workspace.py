"""Tests for VoxCity GPU visibility workspace caching.

All tests monkeypatch the workspace constructor so no real Taichi fields
are allocated.  CUDA is not required.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class FakeWorkspace:
    key: object


@dataclass
class FakeSurfaceWorkspace:
    key: object


# ─────────────────────────────────────────────────────────────────────────────
# Chunk 1 – workspace cache-key and cache-management
# ─────────────────────────────────────────────────────────────────────────────

def test_visibility_workspace_reuses_same_config(monkeypatch):
    from voxcity.simulator_gpu.visibility import integration

    created = []

    def fake_factory(*, key, nx, ny, nz, n_azimuth, n_elevation, ray_sampling,
                     n_rays, elevation_min_degrees, elevation_max_degrees):
        workspace = FakeWorkspace(key=key)
        created.append(workspace)
        return workspace

    monkeypatch.setattr(integration, "ViewWorkspace", fake_factory, raising=False)
    integration.clear_visibility_cache()

    first = integration._get_or_create_view_workspace(
        nx=4, ny=5, nz=6, meshsize=1.0,
        n_azimuth=60, n_elevation=10, ray_sampling="grid", n_rays=None,
        elevation_min_degrees=0.0, elevation_max_degrees=90.0,
    )
    second = integration._get_or_create_view_workspace(
        nx=4, ny=5, nz=6, meshsize=1.0,
        n_azimuth=60, n_elevation=10, ray_sampling="grid", n_rays=None,
        elevation_min_degrees=0.0, elevation_max_degrees=90.0,
    )

    assert first is second
    assert len(created) == 1


def test_visibility_workspace_key_changes_for_ray_config(monkeypatch):
    from voxcity.simulator_gpu.visibility import integration

    created = []

    def fake_factory(*, key, nx, ny, nz, n_azimuth, n_elevation, ray_sampling,
                     n_rays, elevation_min_degrees, elevation_max_degrees):
        workspace = FakeWorkspace(key=key)
        created.append(workspace)
        return workspace

    monkeypatch.setattr(integration, "ViewWorkspace", fake_factory, raising=False)
    integration.clear_visibility_cache()

    first = integration._get_or_create_view_workspace(
        nx=4, ny=5, nz=6, meshsize=1.0,
        n_azimuth=60, n_elevation=10, ray_sampling="grid", n_rays=None,
        elevation_min_degrees=0.0, elevation_max_degrees=90.0,
    )
    second = integration._get_or_create_view_workspace(
        nx=4, ny=5, nz=6, meshsize=1.0,
        n_azimuth=120, n_elevation=10, ray_sampling="grid", n_rays=None,
        elevation_min_degrees=0.0, elevation_max_degrees=90.0,
    )

    assert first is not second
    assert len(created) == 2


def test_clear_visibility_cache_clears_workspace_cache(monkeypatch):
    from voxcity.simulator_gpu.visibility import integration

    created = []

    def fake_factory(*, key, nx, ny, nz, n_azimuth, n_elevation, ray_sampling,
                     n_rays, elevation_min_degrees, elevation_max_degrees):
        workspace = FakeWorkspace(key=key)
        created.append(workspace)
        return workspace

    monkeypatch.setattr(integration, "ViewWorkspace", fake_factory, raising=False)
    integration.clear_visibility_cache()

    kwargs = dict(
        nx=4, ny=5, nz=6, meshsize=1.0,
        n_azimuth=60, n_elevation=10, ray_sampling="grid", n_rays=None,
        elevation_min_degrees=0.0, elevation_max_degrees=90.0,
    )
    first = integration._get_or_create_view_workspace(**kwargs)
    integration.clear_visibility_cache()
    second = integration._get_or_create_view_workspace(**kwargs)

    assert first is not second
    assert len(created) == 2


def test_visibility_workspace_reuses_across_modes_when_allocation_shape_matches(monkeypatch):
    from voxcity.simulator_gpu.visibility import integration

    created = []

    def fake_factory(*, key, nx, ny, nz, n_azimuth, n_elevation, ray_sampling,
                     n_rays, elevation_min_degrees, elevation_max_degrees):
        workspace = FakeWorkspace(key=key)
        created.append(workspace)
        return workspace

    monkeypatch.setattr(integration, "ViewWorkspace", fake_factory, raising=False)
    integration.clear_visibility_cache()

    common = dict(
        nx=4, ny=5, nz=6, meshsize=1.0,
        n_azimuth=60, n_elevation=10, ray_sampling="grid", n_rays=None,
    )
    sky = integration._get_or_create_view_workspace(
        **common,
        elevation_min_degrees=0.0,
        elevation_max_degrees=90.0,
    )
    green = integration._get_or_create_view_workspace(
        **common,
        elevation_min_degrees=0.0,
        elevation_max_degrees=90.0,
    )

    assert sky is green
    assert len(created) == 1


def test_surface_view_workspace_reuses_same_config(monkeypatch):
    from voxcity.simulator_gpu.visibility import integration

    created = []

    def fake_factory(**kwargs):
        workspace = FakeSurfaceWorkspace(key=kwargs["key"])
        created.append(workspace)
        return workspace

    monkeypatch.setattr(integration, "SurfaceViewWorkspace", fake_factory, raising=False)
    integration.clear_visibility_cache()

    kwargs = dict(
        nx=4, ny=5, nz=6, meshsize=1.0,
        n_faces=5790,
        n_azimuth=60, n_elevation=10,
        ray_sampling="grid", n_rays=None,
    )

    first = integration._get_or_create_surface_view_workspace(**kwargs)
    second = integration._get_or_create_surface_view_workspace(**kwargs)

    assert first is second
    assert len(created) == 1


def test_surface_view_workspace_key_changes_for_face_count(monkeypatch):
    from voxcity.simulator_gpu.visibility import integration

    created = []

    def fake_factory(**kwargs):
        workspace = FakeSurfaceWorkspace(key=kwargs["key"])
        created.append(workspace)
        return workspace

    monkeypatch.setattr(integration, "SurfaceViewWorkspace", fake_factory, raising=False)
    integration.clear_visibility_cache()

    common = dict(
        nx=4, ny=5, nz=6, meshsize=1.0,
        n_azimuth=60, n_elevation=10,
        ray_sampling="grid", n_rays=None,
    )

    first = integration._get_or_create_surface_view_workspace(n_faces=5790, **common)
    second = integration._get_or_create_surface_view_workspace(n_faces=6000, **common)

    assert first is not second
    assert len(created) == 2


def test_clear_visibility_cache_clears_surface_view_workspace(monkeypatch):
    from voxcity.simulator_gpu.visibility import integration

    created = []

    def fake_factory(**kwargs):
        workspace = FakeSurfaceWorkspace(key=kwargs["key"])
        created.append(workspace)
        return workspace

    monkeypatch.setattr(integration, "SurfaceViewWorkspace", fake_factory, raising=False)
    integration.clear_visibility_cache()

    kwargs = dict(
        nx=4, ny=5, nz=6, meshsize=1.0,
        n_faces=5790,
        n_azimuth=60, n_elevation=10,
        ray_sampling="grid", n_rays=None,
    )

    first = integration._get_or_create_surface_view_workspace(**kwargs)
    integration.clear_visibility_cache()
    second = integration._get_or_create_surface_view_workspace(**kwargs)

    assert first is not second
    assert len(created) == 2


def test_domain_recreation_does_not_fallback_to_cpu(monkeypatch):
    import pytest
    from voxcity.simulator_gpu.visibility import integration

    calls = []

    class FakeTi:
        def reset(self):
            calls.append(("reset", None))

    class FakeDomain:
        def __init__(self, nx, ny, nz, dx, dy, dz):
            self.nx = nx
            self.ny = ny
            self.nz = nz
            self.dx = dx
            self.dy = dy
            self.dz = dz

    # Patch via integration._init_taichi_module to avoid __init__.py shadowing issue
    fake_init_taichi_mod = type("FakeInitTaichiMod", (), {})()
    fake_init_taichi_mod.reset = lambda: calls.append(("init_flag_reset", None))

    def fake_init_taichi(arch, honor_env=True):
        calls.append(("init", arch, honor_env))
        if arch == "cuda":
            raise RuntimeError("cuda init failed")

    fake_init_taichi_mod.init_taichi = fake_init_taichi

    integration.clear_visibility_cache()
    monkeypatch.setenv("TAICHI_ARCH", "cpu")
    monkeypatch.setattr(integration, "Domain", FakeDomain)
    monkeypatch.setattr(integration, "ti", FakeTi(), raising=False)
    monkeypatch.setattr(integration, "_init_taichi_module", fake_init_taichi_mod)

    integration._get_or_create_domain(4, 5, 6, 1.0)

    with pytest.raises(RuntimeError, match="cuda init failed"):
        integration._get_or_create_domain(7, 8, 9, 1.0)

    assert ("init", "cuda", False) in calls
    assert not any(call[0] == "init" and call[1] == "cpu" for call in calls)


def test_init_taichi_can_ignore_environment_arch(monkeypatch):
    import importlib
    # Use importlib to get the actual module, bypassing __init__.py re-export shadowing
    init_mod = importlib.import_module("voxcity.simulator_gpu.init_taichi")

    calls = []

    class FakeTi:
        f32 = "f32"
        i32 = "i32"
        ERROR = "error"
        cpu = "cpu"
        cuda = "cuda"
        gpu = "gpu"
        vulkan = "vulkan"
        metal = "metal"

        def init(self, **kwargs):
            calls.append(kwargs)

    # Track module-level flags so they are fully restored after this test,
    # preventing contamination of subsequent tests that call ensure_initialized().
    monkeypatch.setattr(init_mod, "_TAICHI_INITIALIZED", init_mod._TAICHI_INITIALIZED)

    monkeypatch.setattr(init_mod, "ti", FakeTi())
    monkeypatch.setenv("TAICHI_ARCH", "cpu")
    init_mod.reset()

    init_mod.init_taichi(arch="cuda", honor_env=False)

    assert calls[-1]["arch"] == "cuda"


def test_reset_visibility_taichi_cache_forces_cuda(monkeypatch):
    from voxcity.simulator_gpu.visibility import integration

    calls = []

    class FakeTi:
        def reset(self):
            calls.append(("ti_reset", None))

    # Patch via integration._init_taichi_module (the object integration.py actually calls)
    fake_init_taichi_mod = type("FakeInitTaichiMod", (), {})()
    fake_init_taichi_mod.reset = lambda: calls.append(("init_flag_reset", None))
    fake_init_taichi_mod.init_taichi = lambda arch, honor_env=True: calls.append(("init", arch, honor_env))

    monkeypatch.setenv("TAICHI_ARCH", "cpu")
    monkeypatch.setattr(integration, "ti", FakeTi(), raising=False)
    monkeypatch.setattr(integration, "_init_taichi_module", fake_init_taichi_mod)

    integration.reset_visibility_taichi_cache()

    assert ("init", "cuda", False) in calls
    assert not any(call[0] == "init" and call[1] == "cpu" for call in calls)


def test_domain_recreation_clears_workspace_cache(monkeypatch):
    from voxcity.simulator_gpu.visibility import integration

    created = []

    class FakeDomain:
        def __init__(self, nx, ny, nz, dx, dy, dz):
            self.nx = nx
            self.ny = ny
            self.nz = nz
            self.dx = dx
            self.dy = dy
            self.dz = dz

    def fake_workspace_factory(**kwargs):
        workspace = object()
        created.append(workspace)
        return workspace

    # Patch via integration._init_taichi_module to avoid __init__.py shadowing issue
    fake_init_taichi_mod = type("FakeInitTaichiMod", (), {})()
    fake_init_taichi_mod.reset = lambda: None
    fake_init_taichi_mod.init_taichi = lambda arch, honor_env=True: None

    monkeypatch.setattr(integration, "Domain", FakeDomain)
    monkeypatch.setattr(integration, "ViewWorkspace", fake_workspace_factory, raising=False)
    monkeypatch.setattr(integration, "ti", type("FakeTi", (), {"reset": lambda self: None})(), raising=False)
    monkeypatch.setattr(integration, "_init_taichi_module", fake_init_taichi_mod)

    integration.clear_visibility_cache()
    integration._get_or_create_domain(4, 5, 6, 1.0)
    first = integration._get_or_create_view_workspace(
        nx=4, ny=5, nz=6, meshsize=1.0,
        n_azimuth=60, n_elevation=10, ray_sampling="grid", n_rays=None,
        elevation_min_degrees=0.0, elevation_max_degrees=90.0,
    )
    integration._get_or_create_domain(7, 8, 9, 1.0)
    second = integration._get_or_create_view_workspace(
        nx=4, ny=5, nz=6, meshsize=1.0,
        n_azimuth=60, n_elevation=10, ray_sampling="grid", n_rays=None,
        elevation_min_degrees=0.0, elevation_max_degrees=90.0,
    )

    assert first is not second
    assert len(created) == 2


# ─────────────────────────────────────────────────────────────────────────────
# Chunk 2 – workspace wiring through get_view_index_gpu
# ─────────────────────────────────────────────────────────────────────────────

def test_get_view_index_gpu_passes_cached_workspace(monkeypatch):
    import numpy as np
    from voxcity.simulator_gpu.visibility import integration

    fake_domain = type(
        "FakeDomain",
        (),
        {"nx": 4, "ny": 5, "nz": 6, "dx": 1.0, "dy": 1.0, "dz": 1.0,
         "get_max_dist": lambda self: 10.0},
    )()
    fake_workspace = object()
    captured = {}

    monkeypatch.setattr(integration, "_get_or_create_domain", lambda *args, **kwargs: fake_domain)
    monkeypatch.setattr(integration, "_get_or_create_view_workspace", lambda **kwargs: fake_workspace)

    class FakeCalculator:
        def __init__(self, domain, **kwargs):
            captured["domain"] = domain
            captured["init_kwargs"] = kwargs

        def compute_view_index(self, **kwargs):
            captured["compute_kwargs"] = kwargs
            return np.ones((4, 5), dtype=float)

    monkeypatch.setattr(integration, "ViewCalculator", FakeCalculator)

    voxcity = type(
        "FakeVoxCity",
        (),
        {
            "voxels": type(
                "Voxels",
                (),
                {
                    "classes": np.zeros((4, 5, 6), dtype=np.int8),
                    "meta": type("Meta", (), {"meshsize": 1.0})(),
                },
            )()
        },
    )()

    result = integration.get_view_index_gpu(
        voxcity,
        mode="sky",
        n_azimuth=60,
        n_elevation=10,
        elevation_min_degrees=0.0,
        elevation_max_degrees=90.0,
        show_plot=False,
    )

    assert result.shape == (4, 5)
    assert captured["compute_kwargs"]["workspace"] is fake_workspace


def test_get_view_index_gpu_reuses_workspace_for_sky_and_green(monkeypatch):
    import numpy as np
    from voxcity.simulator_gpu.visibility import integration

    fake_domain = type(
        "FakeDomain",
        (),
        {"nx": 4, "ny": 5, "nz": 6, "dx": 1.0, "dy": 1.0, "dz": 1.0,
         "get_max_dist": lambda self: 10.0},
    )()
    created = []

    class FakeWorkspace:
        def __init__(self, **kwargs):
            created.append(self)

    class FakeCalculator:
        def __init__(self, domain, **kwargs):
            pass

        def compute_view_index(self, **kwargs):
            return np.ones((4, 5), dtype=float)

    monkeypatch.setattr(integration, "_get_or_create_domain", lambda *args, **kwargs: fake_domain)
    monkeypatch.setattr(integration, "ViewWorkspace", FakeWorkspace, raising=False)
    monkeypatch.setattr(integration, "ViewCalculator", FakeCalculator)
    integration.clear_visibility_cache()

    voxcity = type(
        "FakeVoxCity",
        (),
        {
            "voxels": type(
                "Voxels",
                (),
                {
                    "classes": np.zeros((4, 5, 6), dtype=np.int8),
                    "meta": type("Meta", (), {"meshsize": 1.0})(),
                },
            )()
        },
    )()

    integration.get_view_index_gpu(voxcity, mode="sky", n_azimuth=60, n_elevation=10, show_plot=False)
    integration.get_view_index_gpu(voxcity, mode="green", n_azimuth=60, n_elevation=10, show_plot=False)

    assert len(created) == 1
