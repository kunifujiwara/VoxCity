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


def test_cpu_direct_solar_north_azimuth_uses_positive_u_and_preserves_uv_layout():
    from voxcity.simulator.solar.radiation import get_direct_solar_irradiance_map

    voxel_data = _ground_grid()
    voxel_data[3, 0, 1:3] = -3
    voxcity = _fake_voxcity(voxel_data)

    result = get_direct_solar_irradiance_map(
        voxcity,
        azimuth_degrees_ori=0,
        elevation_degrees=45,
        direct_normal_irradiance=1000,
        view_point_height=0,
    )

    assert result.shape == voxel_data.shape[:2]
    assert np.isnan(result[3, 0])
    assert result[2, 0] == 0
    assert result[1, 0] > 0


def test_cpu_visibility_maps_preserve_uv_layout():
    from voxcity.simulator.common.raytracing import (
        _compute_vi_map_generic_fast,
        _prepare_masks_for_vi,
        compute_vi_map_generic,
    )

    voxel_data = _ground_grid()
    voxel_data[3, 0, 1:3] = -3
    ray_directions = np.array([[0.0, 0.0, 1.0]], dtype=np.float64)
    hit_values = (0,)
    meshsize = 1.0
    tree_k = 0.6
    tree_lad = 1.0
    inclusion_mode = False

    slow_result = compute_vi_map_generic(
        voxel_data,
        ray_directions,
        0,
        hit_values,
        meshsize,
        tree_k,
        tree_lad,
        inclusion_mode,
    )

    is_tree, is_target, is_allowed, is_blocker_inc = _prepare_masks_for_vi(
        voxel_data, hit_values, inclusion_mode
    )
    fast_result = _compute_vi_map_generic_fast(
        voxel_data,
        ray_directions,
        0,
        meshsize,
        tree_k,
        tree_lad,
        is_tree,
        np.zeros(1, dtype=np.bool_) if is_target is None else is_target,
        np.zeros(1, dtype=np.bool_) if is_allowed is None else is_allowed,
        np.zeros(1, dtype=np.bool_) if is_blocker_inc is None else is_blocker_inc,
        inclusion_mode,
        False,
    )

    for result in (slow_result, fast_result):
        assert result.shape == voxel_data.shape[:2]
        assert np.isnan(result[3, 0])
        assert result[2, 0] == 1.0
        assert result[0, 0] == 1.0


def test_cpu_building_solar_interprets_scene_mesh_as_uv_layout():
    from voxcity.geoprocessor.mesh import create_voxel_mesh
    from voxcity.simulator.solar.radiation import get_building_solar_irradiance

    voxel_data = np.zeros((5, 4, 4), dtype=np.int32)
    voxel_data[2, 2, 1:3] = -3
    building_ids = np.zeros((5, 4), dtype=np.int32)
    building_ids[2, 2] = 1
    voxcity = _fake_voxcity(voxel_data)
    mesh = create_voxel_mesh(
        voxel_data,
        -3,
        1.0,
        building_id_grid=building_ids,
        mesh_type="open_air",
    )
    mesh.metadata["svf"] = np.ones(len(mesh.faces), dtype=np.float64)

    north = get_building_solar_irradiance(voxcity, mesh, 0, 45, 1000, 0)
    east = get_building_solar_irradiance(voxcity, mesh, 90, 45, 1000, 0)

    normals = mesh.face_normals
    north_faces = normals[:, 1] > 0.9
    east_faces = normals[:, 0] > 0.9
    finite_north = north_faces & np.isfinite(north.metadata["direct"])
    finite_east = east_faces & np.isfinite(east.metadata["direct"])

    assert np.nanmax(north.metadata["direct"][finite_north]) > 0
    assert np.nanmax(north.metadata["direct"][finite_east]) == 0
    assert np.nanmax(east.metadata["direct"][finite_east]) > 0


def test_cpu_building_solar_diffuse_uses_sky_hemisphere_factor():
    from voxcity.simulator.solar.radiation import get_building_solar_irradiance
    from voxcity.simulator.visibility import get_surface_view_factor

    voxel_data = np.zeros((6, 6, 4), dtype=np.int32)
    voxel_data[3, 3, 1:3] = -3
    building_ids = np.zeros((6, 6), dtype=np.int32)
    building_ids[3, 3] = 1
    voxcity = _fake_voxcity_with_buildings(voxel_data, building_ids)

    svf_mesh = get_surface_view_factor(
        voxcity,
        value_name="svf",
        sky_diffuse=True,
        N_azimuth=72,
        N_elevation=18,
        fast_path=True,
    )
    result = get_building_solar_irradiance(
        voxcity,
        svf_mesh,
        azimuth_degrees=0,
        elevation_degrees=5,
        direct_normal_irradiance=0,
        diffuse_irradiance=100,
    )

    normals = svf_mesh.face_normals
    roof_faces = normals[:, 2] > 0.9
    vertical_faces = np.abs(normals[:, 2]) < 0.1
    bottom_faces = normals[:, 2] < -0.9

    assert np.nanmax(result.metadata["diffuse"][roof_faces]) == pytest.approx(100.0, abs=10.0)
    assert np.nanmax(result.metadata["diffuse"][vertical_faces]) == pytest.approx(50.0, abs=15.0)
    assert np.nanmax(np.nan_to_num(result.metadata["diffuse"][bottom_faces], nan=0.0)) == pytest.approx(0.0)


def test_cpu_landmark_visibility_imshow_preserves_uv_layout(monkeypatch):
    """`compute_landmark_visibility` plots its result via `plt.imshow`. The
    array passed to `imshow` must keep uv layout (axis 0 = u = north) so that
    `origin='lower'` puts north at the top of the figure. A stale `flipud`
    here mirrors the displayed map north-south.
    """
    import matplotlib.pyplot as plt
    from voxcity.simulator.visibility.landmark import compute_landmark_visibility

    # Layout: landmark at the north end (u=7), tall blocker at u=4 spanning all v.
    # Observers south of the blocker (u in [0, 3]) cannot see the landmark; observers
    # north of it (u in [5, 6]) can. So the visibility map (uv layout) holds:
    #   visibility[0..3, :] = 0   (south, blocked)
    #   visibility[5..6, :] = 1   (north of blocker, visible)
    voxel_data = np.zeros((8, 4, 4), dtype=np.int32)
    voxel_data[:, :, 0] = 1                  # ground
    voxel_data[4, :, 1:4] = -3               # tall blocker spanning all v
    voxel_data[7, 1:3, 1:3] = -30            # landmark at the far north

    captured = {}

    def _capture_imshow(arr, *args, **kwargs):
        captured["arr"] = np.asarray(arr).copy()

        class _AxesImage:
            def set_clim(self, *_a, **_kw):
                pass

            def set_cmap(self, *_a, **_kw):
                pass

        return _AxesImage()

    monkeypatch.setattr(plt, "imshow", _capture_imshow)
    monkeypatch.setattr(plt, "show", lambda: None)
    monkeypatch.setattr(plt, "figure", lambda *a, **kw: None)
    monkeypatch.setattr(plt, "legend", lambda *a, **kw: None)
    monkeypatch.setattr(plt, "axis", lambda *a, **kw: None)

    compute_landmark_visibility(voxel_data, target_value=-30, view_height_voxel=0)

    arr = captured["arr"]
    assert arr.shape == voxel_data.shape[:2]

    # North-of-blocker observers should be visible (1.0); south-of-blocker should
    # be blocked (0.0). With origin='lower', high axis-0 indices render at the
    # top of the figure, so north-up means arr[6, 0] == 1 and arr[0, 0] == 0.
    # A stale flipud would invert this.
    assert arr[6, 0] == 1.0
    assert arr[0, 0] == 0.0
