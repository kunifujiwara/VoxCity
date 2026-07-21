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


def test_stage_vocabulary_v3():
    m = load_module()
    assert m.STAGES == ["Download", "Voxelize", "Integrate",
                        "Voxel City", "Ground level", "Building surface"]
    assert not hasattr(m, "EXPORT_FORMATS")
    assert m.PLATE_BASE_ID["landcover"] == 160
    assert set(m.LAYER_CMAP) == {"terrain", "buildings", "trees"}


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


def test_quick_build_writes_animated_webp(tmp_path):
    import scripts.make_readme_demo_gif as m
    if not _cached_present(m) or not m.gpu_available():
        pytest.skip("cached h5 or GPU unavailable")
    out = tmp_path / "demo_quick.webp"
    m.main(["--quick", "--out", str(out)])
    from PIL import Image
    im = Image.open(out)
    assert getattr(im, "n_frames", 1) > 1
    durs = []
    for i in range(im.n_frames):
        im.seek(i); im.load()
        durs.append(im.info.get("duration"))
    assert all(d and d > 0 for d in durs)


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
    import math
    # radius/azimuth are measured relative to each pose's own look-at: a constant
    # screen-space pan shifts camera + look-at together each frame, so it cancels
    # here and the underlying orbit stays a true (non-zooming) circular sweep.
    radii = [math.hypot(p[0] - look[0], p[1] - look[1]) for p, look in poses]
    assert max(radii) - min(radii) < 1e-3 * max(radii)
    # azimuth advances monotonically and total sweep ~90 deg
    az = [math.atan2(p[1] - look[1], p[0] - look[0]) for p, look in poses]
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


def test_timeline_six_beats_in_order():
    import scripts.make_readme_demo_gif as m
    cfg = m.Config(quick=True)
    tl = m.build_timeline(cfg)
    kinds = [fs.scene_kind for fs in tl]
    assert kinds[0] == "download" and kinds[-1] == "sim_building"
    order = ["download", "voxelize", "integrate", "city",
             "sim_ground", "sim_building"]
    seen = [k for i, k in enumerate(kinds) if i == 0 or kinds[i-1] != k]
    assert seen == order
    assert all(fs.scene_kind != "export" for fs in tl)


def test_timeline_download_reveals_sequentially():
    import scripts.make_readme_demo_gif as m
    cfg = m.Config()
    tl = [fs for fs in m.build_timeline(cfg) if fs.scene_kind == "download"]
    first, last = tl[0], tl[-1]
    assert sum(first.reveal.values()) == 1          # terrain only at start
    assert sum(last.reveal.values()) == 4           # all four by end


def test_timeline_camera_monotonic():
    import scripts.make_readme_demo_gif as m
    tl = m.build_timeline(m.Config(quick=True))
    ts = [fs.camera_t for fs in tl]
    assert ts == sorted(ts) and 0.0 <= ts[0] and ts[-1] <= 1.0


def test_draw_callouts_marks_pixels():
    import scripts.make_readme_demo_gif as m
    frame = np.zeros((540, 960, 3), np.uint8)
    out = m.draw_callouts(frame, [("2D Terrain Elevation map", (0.6, 0.4))])
    assert out.shape == frame.shape and out.dtype == np.uint8
    assert out.sum() > 0                      # something was drawn
    assert m.draw_callouts(frame, []).sum() == 0   # empty is a no-op


def test_draw_labels_change_pixels():
    m = load_module()
    f = np.full((200, 320, 3), 128, dtype=np.uint8)
    out1 = m.draw_labels(f, [("Terrain", 0.62, "terrain"), ("Building", 0.30, "building")])
    assert out1.shape == f.shape and not np.array_equal(out1, f)


def test_colormap_plate_bins_and_colors():
    import scripts.make_readme_demo_gif as m
    vals = np.linspace(0.0, 1.0, 100).reshape(10, 10)
    ids, cd = m.colormap_plate(vals, "viridis", nbins=16, base_id=100)
    assert ids.shape == (10, 10)
    assert ids.min() >= 100 and ids.max() <= 115
    assert len(cd) == 16
    assert all(len(v) == 3 and all(0 <= c <= 255 for c in v) for v in cd.values())


def test_landcover_plate_uses_lut_ids():
    import scripts.make_readme_demo_gif as m
    lc = np.array([[0, 1], [2, 3]])
    ids, cd = m.landcover_plate(lc, base_id=160)
    assert ids.min() >= 160
    assert set(np.unique(ids)).issubset(set(cd.keys()))


def test_build_download_scene_reveal(monkeypatch):
    import scripts.make_readme_demo_gif as m
    # tiny fake city
    base = np.zeros((4, 4, 6), dtype=np.int8)
    city = type("C", (), {})()
    city.voxels = type("Vx", (), {})()
    city.voxels.classes = base
    maps = {"terrain": np.ones((4, 4)), "landcover": np.zeros((4, 4), int),
            "buildings": np.ones((4, 4)), "trees": np.ones((4, 4))}
    scene, cmap = m.build_download_scene(city, maps,
        reveal={"terrain": 1, "landcover": 0, "buildings": 0, "trees": 0})
    g = np.asarray(scene.voxels.classes)
    assert g.dtype == np.int32
    # exactly the terrain z-slice populated
    assert (g[:, :, m._EXPLODE["terrain"]] >= 100).all()
    assert (g[:, :, m._EXPLODE["landcover"]] == 0).all()


def test_load_building_mesh_shapes():
    import scripts.make_readme_demo_gif as m
    if not _cached_present(m):
        pytest.skip("cached demo h5 not present")
    cfg = m.Config()
    mesh = m.load_building_mesh(cfg)
    assert mesh.vertices.shape == (106366, 3)
    assert mesh.faces.shape == (207926, 3)
    vals = mesh.metadata["view_factor_values"]
    assert len(vals) == len(mesh.faces)


def test_building_mesh_aligns_with_voxel_footprint():
    # The cached GVI mesh is stored transposed vs the voxel model; load_building_mesh
    # swaps x/y so the surface footprint lands on the voxel-city buildings.
    import scripts.make_readme_demo_gif as m
    if not _cached_present(m):
        pytest.skip("cached demo h5 not present")
    cfg = m.Config()
    city, _ = m.load_inputs(cfg)
    cls = np.asarray(city.voxels.classes)
    ms = city.voxels.meta.meshsize
    nx, ny = cls.shape[0], cls.shape[1]
    bvox = (cls <= -3).any(axis=2)                      # (u, v) footprint
    v = np.asarray(m.load_building_mesh(cfg, y_extent=nx * ms).vertices)  # swapped + y-flipped
    vc = np.clip((v[:, 0] / ms).astype(int), 0, ny - 1)
    uc = np.clip((v[:, 1] / ms).astype(int), 0, nx - 1)
    bmesh = np.zeros((nx, ny), bool)
    bmesh[uc, vc] = True
    inter = (bvox & bmesh).sum()
    iou = inter / max(1, (bvox | bmesh).sum())
    assert iou > 0.5     # transposed mesh overlaps the voxel buildings


def test_download_maps_align_with_voxel_footprint():
    # load_download_maps must flip the south-first h5 grids to north-first so the
    # flat plates land on the same cells their voxelized counterparts occupy.
    import scripts.make_readme_demo_gif as m
    cfg = m.Config()
    if not cfg.voxcity_h5.exists():
        import pytest
        pytest.skip("cached voxcity.h5 not available")
    city, _ = m.load_inputs(cfg)
    cls = np.asarray(city.voxels.classes)
    maps = m.load_download_maps(cfg)
    bvox = (cls <= -3).any(axis=2)                 # building voxel footprint
    bmap = np.nan_to_num(np.asarray(maps["buildings"])) > 0
    iou = (bvox & bmap).sum() / max(1, (bvox | bmap).sum())
    assert iou > 0.9     # plate map now aligned (flipud) with the voxel model


def test_grayscale_voxel_color_map_is_neutral():
    import scripts.make_readme_demo_gif as m
    gm = m.grayscale_voxel_color_map()
    assert gm and all(isinstance(k, int) for k in gm)
    for r, g, b in gm.values():
        assert r == g == b and 0 <= r <= 255


def test_render_still_accepts_voxel_color_map(monkeypatch):
    import scripts.make_readme_demo_gif as m
    captured = {}
    cfg = m.Config(quick=True)

    def fake_vis(city, **kw):
        captured.update(kw)
        return np.zeros((cfg.height, cfg.width, 3), np.uint8)
    monkeypatch.setattr(
        "voxcity.visualizer.renderer_gpu.visualize_voxcity_gpu", fake_vis)
    monkeypatch.setattr(m, "gpu_available", lambda: False)
    city = type("C", (), {})()
    city.voxels = type("Vx", (), {})()
    city.voxels.classes = np.zeros((4, 4, 6), np.int32)
    m.render_still(city, cfg, ((0, 0, 0), (0, 0, 0)),
                   voxel_color_map={100: [1, 2, 3]})
    assert captured["voxel_color_map"] == {100: [1, 2, 3]}


def test_render_timeline_dispatch(monkeypatch):
    import numpy as np, scripts.make_readme_demo_gif as m
    cfg = m.Config(quick=True)
    calls = []
    def fake_render_still(city, cfg, camera, **kw):
        calls.append(kw)
        return np.zeros((cfg.height, cfg.width, 3), np.uint8)
    monkeypatch.setattr(m, "render_still", fake_render_still)
    monkeypatch.setattr(m, "load_download_maps",
        lambda cfg: {k: np.ones((4, 4)) for k in m.LAYERS})
    monkeypatch.setattr(m, "load_building_mesh", lambda cfg, **kw: object())
    monkeypatch.setattr(m, "build_download_scene",
        lambda *a, **k: (a[0], {100: [1, 2, 3]}))
    city = type("C", (), {})()
    city.voxels = type("Vx", (), {})()
    city.voxels.classes = np.zeros((4, 4, 6), np.int8)
    city.voxels.meta = type("Mt", (), {"meshsize": 5.0})()
    city.dem = type("D", (), {"elevation": np.zeros((4, 4))})()
    results = {"ground": {"solar_irradiance_instantaneous": np.ones((4, 4))}}
    frames = m.render_timeline(city, results, cfg)
    assert len(frames) == len(m.build_timeline(cfg))
    assert any("voxel_color_map" in c for c in calls)   # download beat
    assert any("ground_grid" in c for c in calls)       # sim_ground
    assert any("building_mesh" in c for c in calls)     # sim_building


def test_ground_overlay_flips_to_align_with_footprint():
    # The stored solar grid is south-first; ground_overlay flips it north-south
    # so zero/masked cells coincide with the voxel building footprint.
    import scripts.make_readme_demo_gif as m
    if not _cached_present(m):
        pytest.skip("cached demo h5 not present")
    cfg = m.Config()
    city, results = m.load_inputs(cfg)
    cls = np.asarray(city.voxels.classes)
    bldg = (cls <= -3).any(axis=2)
    grid, _ = m.ground_overlay(city, results)
    assert grid.shape == bldg.shape
    low = ~np.isfinite(grid) | (grid <= 1e-6)
    inside = low[bldg].mean()
    outside = low[~bldg].mean()
    assert inside > 0.9 and inside > outside * 1.5   # zeros land on buildings


def test_build_voxelize_scene_mixes_plates_and_voxels():
    import scripts.make_readme_demo_gif as m
    base = np.zeros((4, 4, 6), dtype=np.int8)
    base[:, :, 0] = -1            # terrain
    base[1, 1, :3] = -3           # a building column
    city = type("C", (), {})()
    city.voxels = type("Vx", (), {})()
    city.voxels.classes = base
    maps = {"terrain": np.ones((4, 4)), "landcover": np.zeros((4, 4), int),
            "buildings": np.ones((4, 4)), "trees": np.ones((4, 4))}
    offsets = dict(m._EXPLODE)
    scene, cmap = m.build_voxelize_scene(
        city, maps, plate_layers=["landcover", "trees"],
        voxel_scales={"terrain": 1.0, "buildings": 1.0}, offsets=offsets)
    g = np.asarray(scene.voxels.classes)
    # plate layers appear as synthetic colormap ids at their z-offset
    assert (g[:, :, m._EXPLODE["landcover"]] >= 100).any()
    assert (g[:, :, m._EXPLODE["trees"]] >= 100).any()
    # voxel layers keep their real (negative) class ids
    assert (g == -1).any() and (g == -3).any()


def test_timeline_voxelize_starts_with_four_slabs():
    import scripts.make_readme_demo_gif as m
    tl = [fs for fs in m.build_timeline(m.Config()) if fs.scene_kind == "voxelize"]
    first = tl[0]
    # every layer is shown from the first voxelize frame (as slab or thin voxel)
    shown = set(first.plates or []) | set((first.scales or {}).keys())
    assert shown == set(m.LAYERS)
    assert len(first.labels) == 4
    # land cover starts as a flat plate and voxelizes (recolors) at its turn:
    # it is a plate at the start and a real voxel (scale) by the end.
    assert "landcover" in (tl[0].plates or [])
    assert "landcover" in (tl[-1].scales or {})
    # buildings & land cover, once active, are full height instantly.
    for fs in tl:
        for L in ("buildings", "landcover"):
            if L in (fs.scales or {}):
                assert fs.scales[L] == 1.0


def test_timeline_voxelize_landcover_recolors_not_thickens():
    # land cover's visible step is a recolor (2D grid colormap plate -> 3D voxel
    # palette), NOT a thickness change. No frame should carry plate_thick.
    import scripts.make_readme_demo_gif as m
    tl = [fs for fs in m.build_timeline(m.Config()) if fs.scene_kind == "voxelize"]
    assert all(getattr(fs, "plate_thick", None) in (None, {}) for fs in tl)
    # land cover transitions from plate (early) to voxel scale (late)
    early = tl[0]
    late = tl[-1]
    assert "landcover" in (early.plates or [])
    assert "landcover" in (late.scales or {})


def test_build_voxelize_scene_plate_thickness_extrudes():
    import scripts.make_readme_demo_gif as m
    base = np.zeros((4, 4, 6), dtype=np.int8)
    city = type("C", (), {})()
    city.voxels = type("Vx", (), {})()
    city.voxels.classes = base
    maps = {"terrain": np.ones((4, 4)), "landcover": np.zeros((4, 4), int),
            "buildings": np.ones((4, 4)), "trees": np.ones((4, 4))}
    offsets = dict(m._EXPLODE)
    scene, _ = m.build_voxelize_scene(
        city, maps, plate_layers=["landcover"], voxel_scales={}, offsets=offsets,
        plate_thick={"landcover": 3})
    g = np.asarray(scene.voxels.classes)
    z = m._EXPLODE["landcover"]
    # three stacked z-layers populated (extruded slab), not just one
    assert (g[:, :, z] >= 100).all()
    assert (g[:, :, z - 1] >= 100).all()
    assert (g[:, :, z - 2] >= 100).all()
    assert (g[:, :, z - 3] == 0).all()


