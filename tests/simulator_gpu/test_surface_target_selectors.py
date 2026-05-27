"""Tests for target_selectors restriction in the building-surface sims."""

from dataclasses import dataclass, field
from typing import Any, Dict
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from voxcity.geoprocessor.surface_meta import resolve_target_face_mask
from voxcity.simulator_gpu.solar.integration import building as building_integration
from voxcity.simulator_gpu.visibility import integration as visibility_integration


@dataclass
class _Mesh:
    vertices: Any
    faces: Any
    metadata: Dict[str, Any] = field(default_factory=dict)


def _mesh_with_meta(face_meta):
    n = len(face_meta)
    verts = np.zeros((n * 3, 3), dtype=np.float32)
    faces = np.zeros((n, 3), dtype=np.int32)
    for i in range(n):
        verts[i * 3 : i * 3 + 3] = [[0, 0, 0], [1, 0, 0], [0, 1, 0]]
        faces[i] = [i * 3, i * 3 + 1, i * 3 + 2]
    mesh = _Mesh(vertices=verts, faces=faces)
    mesh.metadata = {"surface_face_meta": face_meta, "surface_face_meta_version": 1}
    return mesh


def _meta_value(meta, key, default=None):
    if isinstance(meta, dict):
        return meta.get(key, default)
    return getattr(meta, key, default)


def test_resolve_target_face_mask_unions_selectors():
    mesh = _mesh_with_meta(
        [
            {"face_key": "f0", "building_id": 1, "surface_kind": "wall", "orientation": "S"},
            {"face_key": "f1", "building_id": 1, "surface_kind": "roof"},
            {"face_key": "f2", "building_id": 2, "surface_kind": "wall", "orientation": "N"},
        ]
    )
    selectors = [
        {"building_id": 1, "mode": "wall_orientation", "orientation": "S"},
        {"building_id": 2, "mode": "whole"},
    ]

    mask = resolve_target_face_mask(mesh, selectors)

    np.testing.assert_array_equal(mask, [True, False, True])


def test_resolve_target_face_mask_attaches_meta_if_missing():
    """If the mesh has no surface_face_meta yet, the helper classifies it first."""
    verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
    faces = np.array([[0, 1, 2]], dtype=np.int32)
    mesh = _Mesh(vertices=verts, faces=faces)
    mesh.metadata = {"building_id": np.array([7], dtype=int)}

    mask = resolve_target_face_mask(mesh, [{"building_id": 7, "mode": "whole"}])

    assert mask.shape == (1,)
    attached_meta = _meta_value(mesh.metadata, "surface_face_meta")
    assert attached_meta is not None
    assert len(attached_meta) == 1
    assert int(_meta_value(attached_meta[0], "building_id")) == 7
    np.testing.assert_array_equal(mask, [True])


def _tiny_voxcity_two_buildings():
    classes = np.zeros((7, 4, 3), dtype=np.int32)
    classes[1:3, 1:3, 0:2] = -3
    classes[4:6, 1:3, 0:2] = -3

    building_ids = np.zeros(classes.shape[:2], dtype=np.int32)
    building_ids[1:3, 1:3] = 1
    building_ids[4:6, 1:3] = 2

    return SimpleNamespace(
        voxels=SimpleNamespace(
            classes=classes,
            meta=SimpleNamespace(meshsize=1.0),
        ),
        buildings=SimpleNamespace(ids=building_ids),
    )


def _building_2_whole_selector():
    return [{"building_id": 2, "mode": "whole"}]


@pytest.mark.gpu
def test_view_restriction_matches_full_on_target_faces():
    voxcity = _tiny_voxcity_two_buildings()
    kwargs = {"N_azimuth": 12, "N_elevation": 4, "ray_sampling": "grid"}

    full_mesh = visibility_integration.get_surface_view_factor(voxcity, mode="sky", **kwargs)
    restricted_mesh = visibility_integration.get_surface_view_factor(
        voxcity,
        mode="sky",
        target_selectors=_building_2_whole_selector(),
        **kwargs,
    )

    full_values = full_mesh.metadata["view_factor_values"]
    restricted_values = restricted_mesh.metadata["view_factor_values"]
    target_mask = resolve_target_face_mask(restricted_mesh, _building_2_whole_selector())

    assert target_mask.any()
    assert restricted_values.shape == full_values.shape
    np.testing.assert_allclose(restricted_values[target_mask], full_values[target_mask])
    assert np.all(np.isnan(restricted_values[~target_mask]))


@pytest.mark.gpu
def test_view_target_selectors_none_is_unchanged():
    voxcity = _tiny_voxcity_two_buildings()
    kwargs = {"N_azimuth": 12, "N_elevation": 4, "ray_sampling": "grid"}

    default_mesh = visibility_integration.get_surface_view_factor(
        voxcity,
        mode="sky",
        **kwargs,
    )
    none_mesh = visibility_integration.get_surface_view_factor(
        voxcity,
        mode="sky",
        target_selectors=None,
        **kwargs,
    )

    default_values = default_mesh.metadata["view_factor_values"]
    none_values = none_mesh.metadata["view_factor_values"]
    target_mask = resolve_target_face_mask(none_mesh, _building_2_whole_selector())

    assert none_values.shape == default_values.shape
    np.testing.assert_allclose(none_values, default_values, equal_nan=True)
    assert target_mask.any()
    assert np.all(np.isfinite(none_values[target_mask]))


def test_view_empty_target_returns_all_nan(monkeypatch):
    voxcity = _tiny_voxcity_two_buildings()

    class _FakeSurfaceViewFactorCalculator:
        def __init__(self, *args, **kwargs):
            pass

        def compute_surface_view_factor(self, *, face_centers, **kwargs):
            return np.full(len(face_centers), 0.5, dtype=np.float32)

    monkeypatch.setattr(visibility_integration, "_get_or_create_domain", lambda *args, **kwargs: object())
    monkeypatch.setattr(
        visibility_integration,
        "SurfaceViewFactorCalculator",
        _FakeSurfaceViewFactorCalculator,
    )

    mesh = visibility_integration.get_surface_view_factor(
        voxcity,
        mode="sky",
        target_selectors=[{"building_id": 999, "mode": "whole"}],
    )

    values = mesh.metadata["view_factor_values"]
    assert values.shape == (len(mesh.faces),)
    assert np.all(np.isnan(values))


class _ArrayField:
    def __init__(self, values):
        self._values = np.asarray(values)

    def to_numpy(self):
        return self._values.copy()


class _ScalarField:
    def __setitem__(self, key, value):
        self.value = value


class _SolarMesh(_Mesh):
    @property
    def bounds(self):
        return np.array([self.vertices.min(axis=0), self.vertices.max(axis=0)], dtype=np.float64)

    @property
    def triangles_center(self):
        return np.array([[0.25, 0.25, 1.0], [3.25, 0.25, 1.0]], dtype=np.float64)

    @property
    def face_normals(self):
        return np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]], dtype=np.float64)


def _solar_mesh_two_buildings():
    vertices = np.array(
        [
            [0, 0, 1], [1, 0, 1], [0, 1, 1],
            [3, 0, 1], [4, 0, 1], [3, 1, 1],
        ],
        dtype=np.float64,
    )
    faces = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int32)
    mesh = _SolarMesh(vertices=vertices, faces=faces)
    mesh.metadata = {
        "building_id": np.array([1, 2], dtype=np.int32),
        "provided_face_normals": np.array([[0, 0, 1], [0, 0, 1]], dtype=np.float64),
    }
    return mesh


class _FakeSolarModel:
    def __init__(self):
        self.solar_calc = SimpleNamespace(
            sun_direction=_ScalarField(),
            cos_zenith=_ScalarField(),
            sun_up=_ScalarField(),
        )
        self.surfaces = SimpleNamespace(
            count=2,
            sw_in_direct=_ArrayField([10.0, 20.0]),
            sw_in_diffuse=_ArrayField([1.0, 2.0]),
            center=_ArrayField([[0.25, 0.25, 1.0], [0.25, 3.25, 1.0]]),
            normal=_ArrayField([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]]),
        )
        self.compute_calls = 0

    def compute_shortwave_radiation(self, **kwargs):
        self.compute_calls += 1


def _patch_solar_dependencies(monkeypatch, captured=None):
    model = _FakeSolarModel()
    mesh = _solar_mesh_two_buildings()

    def fake_get_or_create(*args, **kwargs):
        if captured is not None:
            captured["n_reflection_steps"] = kwargs.get("n_reflection_steps")
        return model, np.array([True, True], dtype=bool)

    monkeypatch.setattr(building_integration, "get_or_create_building_radiation_model", fake_get_or_create)
    monkeypatch.setattr(building_integration, "get_building_radiation_model_cache", lambda: None)
    monkeypatch.setattr(building_integration, "compute_boundary_vertical_mask", lambda *args, **kwargs: np.zeros(2, dtype=bool))
    monkeypatch.setattr(building_integration, "_map_mesh_faces_to_surfaces", lambda *args, **kwargs: np.array([0, 1], dtype=np.int64))

    import voxcity.geoprocessor.mesh as mesh_mod

    monkeypatch.setattr(mesh_mod, "create_voxel_mesh", lambda *args, **kwargs: mesh)
    return model, mesh


def test_solar_restriction_nan_pads_non_target_faces(monkeypatch):
    voxcity = _tiny_voxcity_two_buildings()
    model, _ = _patch_solar_dependencies(monkeypatch)

    mesh = building_integration.get_building_solar_irradiance(
        voxcity,
        azimuth_degrees_ori=180.0,
        elevation_degrees=45.0,
        direct_normal_irradiance=100.0,
        diffuse_irradiance=10.0,
        target_selectors=_building_2_whole_selector(),
    )

    assert model.compute_calls == 1
    np.testing.assert_allclose(mesh.metadata["direct"], [np.nan, 20.0], equal_nan=True)
    np.testing.assert_allclose(mesh.metadata["diffuse"], [np.nan, 2.0], equal_nan=True)
    np.testing.assert_allclose(mesh.metadata["global"], [np.nan, 22.0], equal_nan=True)


def test_solar_target_selectors_force_reflections_off(monkeypatch):
    voxcity = _tiny_voxcity_two_buildings()
    captured = {}
    warnings = []
    _patch_solar_dependencies(monkeypatch, captured)
    monkeypatch.setattr(building_integration.logger, "warning", lambda message: warnings.append(message))

    building_integration.get_building_solar_irradiance(
        voxcity,
        azimuth_degrees_ori=180.0,
        elevation_degrees=45.0,
        direct_normal_irradiance=100.0,
        diffuse_irradiance=10.0,
        target_selectors=_building_2_whole_selector(),
        with_reflections=True,
        n_reflection_steps=2,
    )

    assert captured["n_reflection_steps"] == 0
    assert len(warnings) == 1
    assert "target_selectors" in warnings[0]
    assert "reflections" in warnings[0]


def _one_step_weather(dni=100.0, dhi=10.0):
    return pd.DataFrame(
        {"DNI": [dni], "DHI": [dhi]},
        index=pd.date_range("2020-01-01 12:00:00", periods=1, freq="h", tz="UTC"),
    )


def _patch_cumulative_time_dependencies(monkeypatch):
    monkeypatch.setattr(building_integration, "filter_df_to_period", lambda weather_df, *args, **kwargs: weather_df)

    def fake_solar_positions(index, lon, lat):
        return pd.DataFrame(
            {"azimuth": np.full(len(index), 180.0), "elevation": np.full(len(index), 45.0)},
            index=index,
        )

    monkeypatch.setattr(building_integration, "get_solar_positions_astral", fake_solar_positions)
    monkeypatch.setattr(building_integration, "compute_boundary_vertical_mask", lambda *args, **kwargs: np.zeros(2, dtype=bool))


def _fake_irradiance_mesh(*, target_selectors=None):
    if target_selectors is None:
        direct = np.array([10.0, 20.0], dtype=np.float64)
        diffuse = np.array([1.0, 2.0], dtype=np.float64)
    else:
        direct = np.array([np.nan, 20.0], dtype=np.float64)
        diffuse = np.array([np.nan, 2.0], dtype=np.float64)
    return SimpleNamespace(
        metadata={
            "direct": direct,
            "diffuse": diffuse,
            "global": direct + diffuse,
        }
    )


def test_solar_cumulative_restriction_nan_pads_non_target_faces(monkeypatch):
    voxcity = _tiny_voxcity_two_buildings()
    mesh = _solar_mesh_two_buildings()
    _patch_cumulative_time_dependencies(monkeypatch)
    monkeypatch.setattr(
        building_integration,
        "get_building_solar_irradiance",
        lambda *args, **kwargs: _fake_irradiance_mesh(target_selectors=kwargs.get("target_selectors")),
    )

    result = building_integration.get_cumulative_building_solar_irradiance(
        voxcity,
        mesh,
        _one_step_weather(),
        lon=139.0,
        lat=35.0,
        tz=0.0,
        time_step_hours=2.0,
        target_selectors=_building_2_whole_selector(),
    )

    np.testing.assert_allclose(result.metadata["cumulative_direct"], [np.nan, 40.0], equal_nan=True)
    np.testing.assert_allclose(result.metadata["cumulative_diffuse"], [np.nan, 4.0], equal_nan=True)
    np.testing.assert_allclose(result.metadata["cumulative_global"], [np.nan, 44.0], equal_nan=True)
    np.testing.assert_allclose(result.metadata["direct"], [np.nan, 40.0], equal_nan=True)
    np.testing.assert_allclose(result.metadata["diffuse"], [np.nan, 4.0], equal_nan=True)
    np.testing.assert_allclose(result.metadata["global"], [np.nan, 44.0], equal_nan=True)


def test_solar_cumulative_target_selectors_none_is_unchanged(monkeypatch):
    voxcity = _tiny_voxcity_two_buildings()
    _patch_cumulative_time_dependencies(monkeypatch)
    monkeypatch.setattr(
        building_integration,
        "get_building_solar_irradiance",
        lambda *args, **kwargs: _fake_irradiance_mesh(target_selectors=kwargs.get("target_selectors")),
    )

    default_result = building_integration.get_cumulative_building_solar_irradiance(
        voxcity,
        _solar_mesh_two_buildings(),
        _one_step_weather(),
        lon=139.0,
        lat=35.0,
        tz=0.0,
        time_step_hours=2.0,
    )
    none_result = building_integration.get_cumulative_building_solar_irradiance(
        voxcity,
        _solar_mesh_two_buildings(),
        _one_step_weather(),
        lon=139.0,
        lat=35.0,
        tz=0.0,
        time_step_hours=2.0,
        target_selectors=None,
    )

    for key in ("cumulative_direct", "cumulative_diffuse", "cumulative_global", "direct", "diffuse", "global"):
        np.testing.assert_allclose(none_result.metadata[key], default_result.metadata[key], equal_nan=True)
        assert np.all(np.isfinite(none_result.metadata[key]))


def test_solar_cumulative_sky_patch_svf_diffuse_respects_target_selectors(monkeypatch):
    voxcity = _tiny_voxcity_two_buildings()
    mesh = _solar_mesh_two_buildings()
    mesh.metadata["svf"] = np.array([0.5, 0.25], dtype=np.float64)
    _patch_cumulative_time_dependencies(monkeypatch)

    result = building_integration.get_cumulative_building_solar_irradiance(
        voxcity,
        mesh,
        _one_step_weather(dni=0.0, dhi=10.0),
        lon=139.0,
        lat=35.0,
        tz=0.0,
        use_sky_patches=True,
        target_selectors=_building_2_whole_selector(),
    )

    np.testing.assert_allclose(result.metadata["cumulative_direct"], [np.nan, 0.0], equal_nan=True)
    np.testing.assert_allclose(result.metadata["cumulative_diffuse"], [np.nan, 2.5], equal_nan=True)
    np.testing.assert_allclose(result.metadata["cumulative_global"], [np.nan, 2.5], equal_nan=True)
