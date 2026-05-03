import numpy as np
import pytest

from voxcity.visualizer.renderer import visualize_voxcity_plotly


def _finite_bounds(trace):
    xs = np.asarray([v for v in trace.x if v is not None], dtype=float)
    ys = np.asarray([v for v in trace.y if v is not None], dtype=float)
    return (float(xs.min()), float(xs.max())), (float(ys.min()), float(ys.max()))


def _trace_bounds_by_color(fig, color):
    bounds = []
    for trace in fig.data:
        if getattr(trace, "color", None) == color and getattr(trace, "x", None) is not None:
            bounds.append(_finite_bounds(trace))
    return bounds


def test_plotly_maps_uv_grid_to_scene_x_east_y_north():
    voxels = np.zeros((2, 3, 1), dtype=np.int16)
    voxels[1, 0, 0] = 10  # north / u axis
    voxels[0, 2, 0] = 20  # east / v axis

    fig = visualize_voxcity_plotly(
        voxels,
        1.0,
        voxel_color_map={10: [255, 0, 0], 20: [0, 0, 255]},
        show=False,
        return_fig=True,
    )

    north_bounds = _trace_bounds_by_color(fig, "rgb(255,0,0)")
    east_bounds = _trace_bounds_by_color(fig, "rgb(0,0,255)")

    assert any(
        xb == pytest.approx((0.0, 1.0)) and yb == pytest.approx((1.0, 2.0))
        for xb, yb in north_bounds
    )
    assert any(
        xb == pytest.approx((2.0, 3.0)) and yb == pytest.approx((0.0, 1.0))
        for xb, yb in east_bounds
    )


def test_plotly_ground_overlay_preserves_uv_cell_zero():
    voxels = np.zeros((2, 2, 1), dtype=np.int16)
    voxels[:, :, 0] = 1
    ground = np.array([[5.0, np.nan], [np.nan, np.nan]])

    fig = visualize_voxcity_plotly(
        voxels,
        1.0,
        voxel_color_map={1: [200, 200, 200]},
        ground_sim_grid=ground,
        ground_dem_grid=np.zeros((2, 2)),
        ground_z_offset=0.0,
        show=False,
        return_fig=True,
    )

    sim_traces = [t for t in fig.data if getattr(t, "name", None) == "sim_surface"]
    assert len(sim_traces) == 1
    x_bounds, y_bounds = _finite_bounds(sim_traces[0])
    assert x_bounds == pytest.approx((0.0, 1.0))
    assert y_bounds == pytest.approx((0.0, 1.0))


def test_plotly_voxel_light_uses_z_extent():
    voxels = np.ones((1, 1, 2), dtype=np.int16)

    fig = visualize_voxcity_plotly(
        voxels,
        1.0,
        voxel_color_map={1: [200, 200, 200]},
        show=False,
        return_fig=True,
    )

    assert fig.data[0].lightposition.z == pytest.approx(3.8)


def test_plotly_default_camera_views_from_southwest_above():
    voxels = np.ones((1, 1, 1), dtype=np.int16)

    fig = visualize_voxcity_plotly(
        voxels,
        1.0,
        voxel_color_map={1: [200, 200, 200]},
        show=False,
        return_fig=True,
    )

    eye = fig.layout.scene.camera.eye
    assert eye.x == pytest.approx(-1.6)
    assert eye.y == pytest.approx(-1.6)
    assert eye.z == pytest.approx(1.0)
