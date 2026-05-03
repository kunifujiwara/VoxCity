from types import SimpleNamespace

import numpy as np
import pytest


def _fake_voxcity_with_building():
    voxel_data = np.zeros((6, 6, 4), dtype=np.int32)
    voxel_data[3, 3, 1:3] = -3
    building_ids = np.zeros((6, 6), dtype=np.int32)
    building_ids[3, 3] = 1
    return SimpleNamespace(
        voxels=SimpleNamespace(
            classes=voxel_data,
            meta=SimpleNamespace(meshsize=1.0),
        ),
        buildings=SimpleNamespace(ids=building_ids),
        extras={"rotation_angle": 0},
        dem=None,
    )


def test_surface_view_factor_sky_diffuse_uses_global_sky_hemisphere():
    from voxcity.simulator.visibility import get_surface_view_factor

    mesh = get_surface_view_factor(
        _fake_voxcity_with_building(),
        value_name="svf",
        sky_diffuse=True,
        N_azimuth=72,
        N_elevation=18,
        fast_path=True,
    )

    normals = mesh.face_normals
    factors = mesh.metadata["svf"]
    roof_faces = normals[:, 2] > 0.9
    vertical_faces = np.abs(normals[:, 2]) < 0.1
    bottom_faces = normals[:, 2] < -0.9

    assert np.nanmax(factors[roof_faces]) == pytest.approx(1.0, abs=0.1)
    assert np.nanmax(factors[vertical_faces]) == pytest.approx(0.5, abs=0.15)
    assert np.nanmax(np.nan_to_num(factors[bottom_faces], nan=0.0)) == pytest.approx(0.0)
    assert mesh.metadata.get("svf_kind") == "sky_diffuse_factor"
