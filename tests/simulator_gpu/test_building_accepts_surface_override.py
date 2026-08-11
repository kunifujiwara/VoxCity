"""The new argument is keyword-only in effect and must reach the factory.

The positional-argument juggling at the top of get_building_solar_irradiance is
load-bearing for API compatibility and must not be disturbed.
"""
import inspect

import numpy as np

from voxcity.simulator_gpu.solar.integration import building as B


def test_positional_order_is_unchanged():
    p = list(inspect.signature(B.get_building_solar_irradiance).parameters)
    assert p[:6] == ["voxcity", "building_svf_mesh", "azimuth_degrees_ori",
                     "elevation_degrees", "direct_normal_irradiance",
                     "diffuse_irradiance"]


class _DummyVoxels:
    class meta:
        meshsize = 1.0
    classes = np.zeros((4, 4, 4), dtype=np.int32)


class _DummyBuildings:
    ids = None


class _DummyCity:
    voxels = _DummyVoxels()
    buildings = _DummyBuildings()
    extras = {}


def test_surface_override_reaches_the_model_factory(monkeypatch):
    seen = {}

    def fake(voxcity, n_reflection_steps=2, progress_report=False,
             building_class_id=None, surface_override=None, **kwargs):
        seen["override"] = surface_override
        raise RuntimeError("stop here")

    monkeypatch.setattr(B, "get_or_create_building_radiation_model", fake)

    sentinel = object()
    try:
        B.get_building_solar_irradiance(
            voxcity=_DummyCity(), azimuth_degrees_ori=180.0, elevation_degrees=45.0,
            direct_normal_irradiance=800.0, diffuse_irradiance=100.0,
            surface_override=sentinel)
    except RuntimeError:
        pass
    assert seen["override"] is sentinel
