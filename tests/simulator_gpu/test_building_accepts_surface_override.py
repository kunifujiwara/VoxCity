"""The new argument is keyword-only in effect and must reach the factory.

The positional-argument juggling at the top of get_building_solar_irradiance is
load-bearing for API compatibility and must not be disturbed.
"""
import inspect

import numpy as np
import pandas as pd

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


class _FakeMesh:
    """Minimal stand-in for a Trimesh: just enough for the pre-loop checks
    (`len(result_mesh.faces)`) in get_cumulative_building_solar_irradiance
    and get_building_sunlight_hours. Deliberately has no `.copy()` or
    `.metadata` attribute, so both functions take the "use as-is" branch."""
    faces = [0]


def _fake_positions(times, lon, lat):
    """Stand-in for get_solar_positions_astral: a fixed sunlit position for
    every timestamp, so the calling function reaches its
    get_building_solar_irradiance call without needing the real astral
    library or real solar geometry."""
    return pd.DataFrame({"azimuth": [180.0] * len(times),
                         "elevation": [45.0] * len(times)}, index=times)


def _fake_irradiance_capturing(seen):
    def fake(voxcity, building_svf_mesh=None, azimuth_degrees_ori=None,
             elevation_degrees=None, direct_normal_irradiance=None,
             diffuse_irradiance=None, **kwargs):
        seen["override"] = kwargs.get("surface_override")
        raise RuntimeError("stop here")
    return fake


def test_cumulative_irradiance_forwards_surface_override(monkeypatch):
    """get_cumulative_building_solar_irradiance calls
    get_building_solar_irradiance internally (once per active sky patch or,
    as here with use_sky_patches=False, once per sunlit timestep) via
    **kwargs -- surface_override must ride along un-popped."""
    seen = {}
    monkeypatch.setattr(B, "get_building_solar_irradiance",
                        _fake_irradiance_capturing(seen))
    monkeypatch.setattr(B, "get_solar_positions_astral", _fake_positions)

    weather_df = pd.DataFrame(
        {"DNI": [800.0], "DHI": [100.0]},
        index=pd.date_range("2024-06-21 12:00:00", periods=1, freq="h"))

    sentinel = object()
    try:
        B.get_cumulative_building_solar_irradiance(
            voxcity=None, building_svf_mesh=_FakeMesh(), weather_df=weather_df,
            lon=0.0, lat=0.0, tz=0.0, use_sky_patches=False,
            surface_override=sentinel)
    except RuntimeError:
        pass
    assert seen["override"] is sentinel


def test_sunlight_hours_forwards_surface_override(monkeypatch):
    """get_building_sunlight_hours calls get_building_solar_irradiance
    internally via **kwargs, same as the cumulative-irradiance path --
    surface_override must reach it there too."""
    seen = {}
    monkeypatch.setattr(B, "get_building_solar_irradiance",
                        _fake_irradiance_capturing(seen))
    monkeypatch.setattr(B, "get_solar_positions_astral", _fake_positions)

    sentinel = object()
    try:
        B.get_building_sunlight_hours(
            voxcity=None, building_svf_mesh=_FakeMesh(), mode="DSH",
            lon=0.0, lat=0.0, tz=0.0, use_sky_patches=False,
            surface_override=sentinel)
    except RuntimeError:
        pass
    assert seen["override"] is sentinel
