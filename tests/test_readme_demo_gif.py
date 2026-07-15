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


def test_config_defaults_webp():
    m = load_module()
    cfg = m.Config()
    assert cfg.width == 960 and cfg.height == 540
    assert cfg.fps == 24
    assert abs(cfg.seconds - 20.0) < 1e-9
    assert cfg.out.name == "demo.webp"
    assert abs(m.frame_duration_ms(24) - (1000.0 / 24)) < 1e-9
    assert m.frame_duration_ms(0) == 1000.0  # guards div-by-zero


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


def test_encode_gif_frame_duration_nonzero(tmp_path):
    m = load_module()
    frames = [np.full((16, 16, 3), i * 20, dtype=np.uint8) for i in range(6)]
    out = tmp_path / "d.gif"
    m.encode_gif(frames, out, fps=15, max_bytes=50 * 1024 * 1024)
    from PIL import Image
    im = Image.open(out)
    durs = []
    try:
        while True:
            durs.append(im.info.get("duration"))
            im.seek(im.tell() + 1)
    except EOFError:
        pass
    # every stored per-frame delay must be > 0 (the 1.0/fps bug stored 0ms => too-fast playback)
    assert all(d and d > 0 for d in durs), f"zero/None frame durations: {durs}"


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


def test_layer_cmap_defaults():
    m = load_module()
    assert m.LAYER_CMAP["terrain"] == "terrain"
    assert m.LAYER_CMAP["buildings"] == "viridis"
    assert m.LAYER_CMAP["trees"] == "Greens"


def test_land_cover_rgb_uses_lut():
    m = load_module()
    from voxcity.utils.lc import get_land_cover_classes
    lut = get_land_cover_classes("Standard")  # {(r,g,b): name}
    names = list(lut.values())
    grid = np.zeros((1, len(names)), dtype=int)
    for j, _ in enumerate(names):
        grid[0, j] = j  # class-index encoding matches land_cover_rgb's mapping
    rgb = m.land_cover_rgb(grid, source="Standard")
    assert rgb.shape == (1, len(names), 3)
    assert rgb.dtype == np.uint8
    # first class color equals the first LUT RGB key
    first_rgb = list(lut.keys())[0]
    assert tuple(int(c) for c in rgb[0, 0]) == tuple(int(c) for c in first_rgb)


def test_layer_mask_is_partition():
    m = load_module()
    c = np.array([[[-1, 1, -3, -2, 0]]], dtype=np.int8)
    assert set(np.unique(np.where(m.layer_mask(c, "terrain"), c, 0))) == {0, -1}
    assert set(np.unique(np.where(m.layer_mask(c, "landcover"), c, 0))) == {0, 1}
    assert set(np.unique(np.where(m.layer_mask(c, "buildings"), c, 0))) == {0, -3}
    assert set(np.unique(np.where(m.layer_mask(c, "trees"), c, 0))) == {0, -2}


def test_z_shift_moves_voxels_up():
    m = load_module()
    c = np.zeros((1, 1, 4), dtype=np.int8)
    c[0, 0, 0] = -1
    up = m.z_shift_classes(c, 2)
    assert up[0, 0, 2] == -1 and up[0, 0, 0] == 0
    assert up.shape == c.shape


def test_explode_city_omits_absent_layers():
    m = load_module()
    if not _cached_present(m):
        pytest.skip("cached demo h5 not present")
    city, _ = m.load_inputs(m.Config(quick=True))
    only_terrain = m.explode_city(city, offsets={"terrain": 0})
    vals = set(np.unique(only_terrain.voxels.classes))
    assert vals.issubset({0, -1})  # only terrain voxels survive
