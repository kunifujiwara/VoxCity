import importlib.util
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "make_readme_demo_gif.py"

def load_module():
    spec = importlib.util.spec_from_file_location("make_readme_demo_gif", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["make_readme_demo_gif"] = mod
    spec.loader.exec_module(mod)
    return mod

import numpy as np


def test_config_defaults():
    m = load_module()
    cfg = m.Config()
    assert cfg.width == 820
    assert cfg.height == 512
    assert cfg.fps == 15
    assert cfg.overlay == "solar"
    assert cfg.out.name == "demo.gif"
    assert m.MAX_BYTES_DEFAULT == 8 * 1024 * 1024


def test_mask_classes_cumulative():
    m = load_module()
    # one voxel of each kind stacked in z
    c = np.array([[[-1, 1, -3, -2, 0]]], dtype=np.int8)  # ground, landcover, building, tree, air
    terrain = m.mask_classes(c, "terrain")
    assert set(np.unique(terrain)) == {0, -1}
    landcover = m.mask_classes(c, "landcover")
    assert set(np.unique(landcover)) == {0, -1, 1}
    buildings = m.mask_classes(c, "buildings")
    assert set(np.unique(buildings)) == {0, -1, 1, -3}
    trees = m.mask_classes(c, "trees")
    assert set(np.unique(trees)) == {0, -1, 1, -3, -2}
    # original untouched
    assert c[0, 0, 2] == -3


def test_isometric_camera_geometry():
    m = load_module()
    pos, look = m.isometric_camera((200, 200, 58), 5.0)
    cx, cy = 100 * 5.0 / 2, 100 * 5.0 / 2  # not exact; just sanity on look-at centering
    # look-at is centered over the horizontal extent
    assert abs(look[0] - (200 * 5.0) / 2) < 1e-6
    assert abs(look[1] - (200 * 5.0) / 2) < 1e-6
    # camera is above and outside the box
    assert pos[2] > look[2]
    assert pos != look


def test_raster_to_rgb_and_fit_canvas():
    m = load_module()
    arr = np.array([[0.0, 1.0], [2.0, np.nan]], dtype=float)
    rgb = m.raster_to_rgb(arr, cmap="viridis")
    assert rgb.shape == (2, 2, 3)
    assert rgb.dtype == np.uint8
    # NaN cell is light gray (all channels high and roughly equal)
    r, g, b = rgb[1, 1]
    assert min(int(r), int(g), int(b)) > 200 and max(abs(int(r)-int(g)), abs(int(g)-int(b))) < 20
    fitted = m.fit_canvas(rgb, (820, 512))
    assert fitted.shape == (512, 820, 3)
    assert fitted.dtype == np.uint8


def test_compose_preserves_shape_and_draws():
    m = load_module()
    cfg = m.Config()
    frame = np.full((cfg.height, cfg.width, 3), 128, dtype=np.uint8)
    out = m.compose(frame, stage_index=1, caption="2 · Download data", cfg=cfg)
    assert out.shape == frame.shape
    assert out.dtype == np.uint8
    # something was drawn (pixels changed vs. flat input)
    assert not np.array_equal(out, frame)
    assert len(m.STAGES) == 6


def test_crossfade_and_stitch():
    m = load_module()
    a = np.zeros((4, 4, 3), dtype=np.uint8)
    b = np.full((4, 4, 3), 255, dtype=np.uint8)
    mid = m.crossfade(a, b, 3)
    assert len(mid) == 3
    # monotonic increase in brightness
    means = [float(f.mean()) for f in mid]
    assert means[0] < means[1] < means[2]
    stitched = m.stitch([[a, a], [b, b]], fade=2)
    assert len(stitched) == 2 + 2 + 2  # stage A + fade + stage B
    assert stitched[0].shape == (4, 4, 3)


def test_encode_gif_writes_and_reports(tmp_path):
    m = load_module()
    frames = [np.random.randint(0, 255, (64, 100, 3), dtype=np.uint8) for _ in range(6)]
    out = tmp_path / "x.gif"
    size = m.encode_gif(frames, out, fps=10, max_bytes=50 * 1024 * 1024)
    assert out.exists()
    assert size == out.stat().st_size > 0


def test_encode_gif_ladder_shrinks(tmp_path):
    m = load_module()
    # noisy frames are hard to compress; force the ladder with a tiny budget
    frames = [np.random.randint(0, 255, (240, 400, 3), dtype=np.uint8) for _ in range(20)]
    out = tmp_path / "y.gif"
    size = m.encode_gif(frames, out, fps=15, max_bytes=400 * 1024)
    assert out.exists()
    assert size <= 400 * 1024


import pytest


def _cached_present(m):
    cfg = m.Config()
    return cfg.voxcity_h5.exists() and cfg.results_h5.exists()


def test_load_inputs_and_render_voxel():
    m = load_module()
    if not _cached_present(m):
        pytest.skip("cached demo h5 not present")
    if not m.gpu_available():
        pytest.skip("no Taichi GPU backend")
    cfg = m.Config(quick=True)
    city, results = m.load_inputs(cfg)
    assert city.voxels.classes.ndim == 3
    assert "ground" in results
    cam_pos, cam_look = m.isometric_camera(city.voxels.classes.shape, city.voxels.meta.meshsize)
    frame = m.render_voxel(city, cfg, keep="terrain", camera=(cam_pos, cam_look))
    assert frame.shape == (cfg.height, cfg.width, 3)
    assert frame.dtype == np.uint8


def test_smoke_quick_build(tmp_path):
    m = load_module()
    cfg = m.Config(quick=True)
    if not (cfg.voxcity_h5.exists() and cfg.results_h5.exists() and m.gpu_available()):
        pytest.skip("cached h5 or GPU unavailable")
    cfg = m.Config(quick=True, out=tmp_path / "demo.gif", width=320, height=200)
    size = m.run(cfg)
    assert cfg.out.exists()
    assert size == cfg.out.stat().st_size > 0


def test_parse_args_overrides():
    m = load_module()
    cfg = m.parse_args(["--width", "640", "--fps", "12", "--quick", "--out", "/tmp/z.gif"])
    assert cfg.width == 640 and cfg.fps == 12 and cfg.quick is True
    assert str(cfg.out) == "/tmp/z.gif"
