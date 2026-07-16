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




def test_encode_webp_animates_with_nonzero_duration(tmp_path):
    m = load_module()
    frames = [np.full((32, 48, 3), i * 30, dtype=np.uint8) for i in range(8)]
    out = tmp_path / "d.webp"
    size = m.encode_webp(frames, out, fps=24, quality=80)
    assert out.exists() and size == out.stat().st_size > 0
    from PIL import Image
    im = Image.open(out)
    assert getattr(im, "n_frames", 1) == 8
    durs = []
    try:
        while True:
            im.seek(im.tell())
            im.load()
            durs.append(im.info.get("duration"))
            im.seek(im.tell() + 1)
    except EOFError:
        pass
    assert all(d and d > 0 for d in durs), f"zero/None durations: {durs}"
    assert abs(durs[0] - (1000.0 / 24)) < 5  # ~41ms


import pytest


def _cached_present(m):
    cfg = m.Config()
    return cfg.voxcity_h5.exists() and cfg.results_h5.exists()


def test_render_still_has_symbol():
    m = load_module()
    assert hasattr(m, "render_still")


def test_render_still_shape():
    m = load_module()
    if not _cached_present(m) or not m.gpu_available():
        pytest.skip("cached h5 or GPU unavailable")
    cfg = m.Config(quick=True, width=320, height=200)
    city, _ = m.load_inputs(cfg)
    cam = m.isometric_camera(city.voxels.classes.shape, city.voxels.meta.meshsize)
    img = m.render_still(city, cfg, cam)
    assert img.shape == (200, 320, 3) and img.dtype == np.uint8



def test_parse_args_overrides():
    m = load_module()
    cfg = m.parse_args(["--width", "640", "--fps", "12", "--quick", "--out", "/tmp/z.gif"])
    assert cfg.width == 640 and cfg.fps == 12 and cfg.quick is True
    assert str(cfg.out) == "/tmp/z.gif"


def test_parse_args_webp_overrides():
    m = load_module()
    cfg = m.parse_args(["--width", "640", "--fps", "20", "--seconds", "10",
                        "--quick", "--out", "/tmp/z.webp"])
    assert cfg.width == 640 and cfg.fps == 20 and cfg.quick is True
    assert abs(cfg.seconds - 10.0) < 1e-9 and str(cfg.out) == "/tmp/z.webp"


def test_smoke_quick_build_webp(tmp_path):
    m = load_module()
    base = m.Config()
    if not (base.voxcity_h5.exists() and base.results_h5.exists() and m.gpu_available()):
        pytest.skip("cached h5 or GPU unavailable")
    cfg = m.Config(quick=True, out=tmp_path / "demo.webp", width=320, height=200)
    size = m.run(cfg)
    assert cfg.out.exists() and size == cfg.out.stat().st_size > 0
    from PIL import Image
    assert getattr(Image.open(cfg.out), "n_frames", 1) > 1


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


def test_orbit_path_smooth_and_bounded():
    m = load_module()
    poses = m.orbit_path((100, 120, 40), 5.0, n=48, sweep_deg=90.0)
    assert len(poses) == 48
    center = poses[0][1]
    # look-at is constant (camera orbits a fixed target)
    for _, look in poses:
        assert np.allclose(look, center, atol=1e-6)
    # radius from center is ~constant (true orbit, not a zoom)
    import math
    radii = [math.hypot(p[0] - center[0], p[1] - center[1]) for p, _ in poses]
    assert max(radii) - min(radii) < 1e-3 * max(radii)
    # azimuth advances monotonically and total sweep ~90 deg
    az = [math.atan2(p[1] - center[1], p[0] - center[0]) for p, _ in poses]
    unwrapped = np.unwrap(az)
    assert abs(math.degrees(unwrapped[-1] - unwrapped[0]) - 90.0) < 5.0


def test_to_uv_layout_no_flip():
    m = load_module()
    ref = (3, 5)  # (nu, nv)
    g = np.arange(15).reshape(3, 5)
    out = m.to_uv_layout(g, ref)
    assert np.array_equal(out, g)  # already uv-layout => untouched (no flip/transpose)


def test_to_uv_layout_transpose_only():
    m = load_module()
    ref = (3, 5)
    gt = np.arange(15).reshape(5, 3)  # transposed
    out = m.to_uv_layout(gt, ref)
    assert out.shape == ref
    assert np.array_equal(out, gt.T)  # transpose to match, still no flip


def test_to_uv_layout_never_flips_marker():
    m = load_module()
    # asymmetric marker in the north-west corner (u=0,v=0) must stay there
    ref = (4, 6)
    g = np.zeros(ref); g[0, 0] = 9.0
    out = m.to_uv_layout(g, ref)
    assert out[0, 0] == 9.0


def test_build_timeline_structure():
    m = load_module()
    cfg = m.Config(seconds=20.0, fps=24)
    tl = m.build_timeline(cfg)
    total = len(tl)
    assert round(cfg.fps * cfg.seconds) * 0.8 <= total <= round(cfg.fps * cfg.seconds) * 1.2
    stages = [fs.stage for fs in tl]
    assert set(stages) == {0, 1, 2, 3, 4, 5}       # all six beats present
    assert stages == sorted(stages)                 # beats appear in order
    # camera parameter is monotonic non-decreasing across the whole loop
    ts = [fs.camera_t for fs in tl]
    assert all(b >= a - 1e-9 for a, b in zip(ts, ts[1:]))
    assert ts[0] <= 0.01 and ts[-1] >= 0.99
    # quick mode is short
    assert len(m.build_timeline(m.Config(quick=True))) <= 24


def test_draw_labels_and_chips_change_pixels():
    m = load_module()
    f = np.full((200, 320, 3), 128, dtype=np.uint8)
    out1 = m.draw_labels(f, [("Terrain", "terrain"), ("Building", "viridis")])
    assert out1.shape == f.shape and not np.array_equal(out1, f)
    out2 = m.draw_export_chips(f, 1.0)
    assert out2.shape == f.shape and not np.array_equal(out2, f)
    assert m.EXPORT_FORMATS[0] == "OBJ"
    # chips fan out: at t=0 they cluster near center, at t=1 they spread wider
    early = m.draw_export_chips(f, 0.0)
    assert not np.array_equal(early, out2)
