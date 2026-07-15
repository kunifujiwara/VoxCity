"""Generate the VoxCity pipeline demo GIF for the README hero.

Build-time asset generator. Renders a looping, step-by-step animation of the
VoxCity workflow from cached artifacts (no Google Earth Engine, no live compute).

Run:
    export LC_ALL=C.UTF-8 LANG=C.UTF-8
    venv/bin/python scripts/make_readme_demo_gif.py --out images/demo.gif
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import imageio.v2 as imageio

import matplotlib
matplotlib.use("Agg")
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parents[1]
CANVAS_DEFAULT = (960, 540)          # (width, height)
FPS_DEFAULT = 24
MAX_BYTES_DEFAULT = 8 * 1024 * 1024  # safety ceiling; WebP stays well under

# Cumulative keep-sets for the voxelization build-up. Each level adds voxels.
_BUILD_LEVELS = ("terrain", "landcover", "buildings", "trees")


def _keep_mask(classes: np.ndarray, level: str) -> np.ndarray:
    """Boolean mask of voxels visible at a cumulative build-up *level*."""
    if level not in _BUILD_LEVELS:
        raise ValueError(f"unknown level {level!r}; expected one of {_BUILD_LEVELS}")
    idx = _BUILD_LEVELS.index(level)
    mask = classes == -1                       # terrain / subsurface
    if idx >= 1:
        mask |= classes >= 1                   # land-cover surface
    if idx >= 2:
        mask |= classes <= -3                  # buildings (all building material codes)
    if idx >= 3:
        mask |= classes == -2                  # tree canopy
    return mask


def mask_classes(classes: np.ndarray, keep: str) -> np.ndarray:
    """Return a copy of *classes* with voxels above cumulative *keep* set to air (0)."""
    mask = _keep_mask(classes, keep)
    return np.where(mask, classes, 0).astype(classes.dtype)


def isometric_camera(shape, meshsize, distance_factor: float = 1.6, height_factor: float = 0.9):
    """Compute a fixed isometric camera (position, look_at) for a voxel grid.

    Reused for every build-up beat and the sim overlay so the model does not
    jump between frames.
    """
    nx, ny, nz = shape
    ex, ey, ez = nx * meshsize, ny * meshsize, nz * meshsize
    center = (ex / 2.0, ey / 2.0, ez * 0.25)
    diag = float(np.hypot(ex, ey))
    # canonical isometric direction (1, 1, ~0.7), normalized, pushed out by diag
    d = np.array([1.0, 1.0, 0.7])
    d = d / np.linalg.norm(d)
    dist = diag * distance_factor
    pos = (
        center[0] + d[0] * dist,
        center[1] + d[1] * dist,
        center[2] + d[2] * dist * height_factor + ez,
    )
    return pos, center


@dataclass
class Config:
    width: int = CANVAS_DEFAULT[0]
    height: int = CANVAS_DEFAULT[1]
    fps: int = FPS_DEFAULT
    seconds: float = 20.0
    out: Path = field(default_factory=lambda: REPO_ROOT / "images" / "demo.webp")
    quick: bool = False
    voxcity_h5: Path = field(default_factory=lambda: REPO_ROOT / "demo" / "output" / "voxcity.h5")
    results_h5: Path = field(default_factory=lambda: REPO_ROOT / "demo" / "output" / "simulation_results.h5")


def raster_to_rgb(arr, cmap: str = "viridis", vmin=None, vmax=None):
    """Convert a 2D raster array to RGB using a colormap.
    
    NaNs are rendered as light gray.
    
    Returns (H, W, 3) uint8 array.
    """
    a = np.asarray(arr, dtype=float)
    finite = np.isfinite(a)
    if vmin is None:
        vmin = float(np.nanmin(a)) if finite.any() else 0.0
    if vmax is None:
        vmax = float(np.nanmax(a)) if finite.any() else 1.0
    if vmax <= vmin:
        vmax = vmin + 1.0
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    mapper = cm.ScalarMappable(norm=norm, cmap=cmap)
    rgba = mapper.to_rgba(np.nan_to_num(a, nan=vmin), bytes=True)
    rgb = rgba[..., :3].copy()
    rgb[~finite] = (230, 230, 230)  # light gray for NaN
    return rgb.astype(np.uint8)


def fit_canvas(rgb, size, pad_rgb=(245, 245, 245)):
    """Letterbox/resize an RGB frame to fit a canvas size.
    
    Returns (height, width, 3) uint8 array.
    """
    width, height = size
    img = Image.fromarray(rgb)
    img.thumbnail((width, height), Image.LANCZOS)
    canvas = Image.new("RGB", (width, height), pad_rgb)
    canvas.paste(img, ((width - img.width) // 2, (height - img.height) // 2))
    return np.asarray(canvas, dtype=np.uint8)


LAYER_CMAP = {"terrain": "terrain", "buildings": "viridis", "trees": "Greens"}


def land_cover_rgb(classes, source: str = "Standard"):
    """Color a 2D land-cover class-index grid with VoxCity's own class->RGB LUT.

    Uses voxcity.utils.lc.get_land_cover_classes(source), an ordered dict of
    {(r,g,b): name}. Grid values index into that ordered list of colors.
    """
    from voxcity.utils.lc import get_land_cover_classes
    lut = get_land_cover_classes(source)
    colors = np.array(list(lut.keys()), dtype=np.uint8)  # (K, 3)
    idx = np.asarray(classes)
    idx = np.clip(idx.astype(int), 0, len(colors) - 1)
    return colors[idx].astype(np.uint8)


def frame_duration_ms(fps: int) -> float:
    """Per-frame delay in milliseconds (WebP/GIF units)."""
    return 1000.0 / max(1, fps)


STAGES = ["Target area", "Download", "Voxelize", "Integrate", "Simulate", "Export"]


def _load_font(size: int):
    """Load a TrueType font, falling back to default if DejaVu is unavailable."""
    for name in ("DejaVuSans-Bold.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def compose(frame, stage_index: int, caption: str, cfg) -> np.ndarray:
    """Burn a bottom caption bar and top progress strip onto a frame.
    
    Args:
        frame: Input (H, W, 3) uint8 RGB array.
        stage_index: Current stage index (0-5), used to highlight the progress strip.
        caption: Caption text for the bottom bar.
        cfg: Config object with width and height.
    
    Returns:
        (H, W, 3) uint8 RGB array with overlays drawn.
    """
    img = Image.fromarray(frame).convert("RGB")
    draw = ImageDraw.Draw(img, "RGBA")
    w, h = img.size

    # top progress strip
    n = len(STAGES)
    seg = w / n
    strip_h = max(6, h // 60)
    for i in range(n):
        active = i <= stage_index
        color = (45, 110, 235, 255) if active else (200, 200, 200, 255)
        draw.rectangle([i * seg + 2, 0, (i + 1) * seg - 2, strip_h], fill=color)

    # bottom caption bar
    bar_h = max(24, h // 12)
    draw.rectangle([0, h - bar_h, w, h], fill=(20, 20, 24, 190))
    font_size = max(14, bar_h // 2)
    font = _load_font(font_size)
    font_size = getattr(font, "size", font_size)
    draw.text((16, h - bar_h + (bar_h - font_size) // 2), caption, fill=(255, 255, 255, 255), font=font)
    return np.asarray(img, dtype=np.uint8)


def crossfade(a, b, n: int):
    """Generate n intermediate frames blending a → b (exclusive of endpoints).
    
    Args:
        a: Source frame (H, W, C) uint8 array.
        b: Target frame (H, W, C) uint8 array.
        n: Number of intermediate frames to generate.
    
    Returns:
        List of n intermediate uint8 frames.
    """
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    out = []
    for k in range(1, n + 1):
        t = k / (n + 1)
        out.append(np.clip(a * (1 - t) + b * t, 0, 255).astype(np.uint8))
    return out


def stitch(stages, fade: int):
    """Concatenate per-stage frame lists with fade dissolve frames between stages.
    
    Args:
        stages: List of per-stage frame lists, each a list of uint8 arrays.
        fade: Number of dissolve frames between consecutive stages.
    
    Returns:
        List of uint8 frames with stages concatenated and dissolves inserted.
    """
    frames: list = []
    for i, stage in enumerate(stages):
        if i > 0 and fade > 0 and frames and stage:
            frames.extend(crossfade(frames[-1], stage[0], fade))
        frames.extend(stage)
    return frames


def _downscale(frames, factor: float):
    if factor >= 0.999:
        return frames
    out = []
    for f in frames:
        h, w = f.shape[:2]
        img = Image.fromarray(f).resize((max(1, int(w * factor)), max(1, int(h * factor))), Image.LANCZOS)
        out.append(np.asarray(img, dtype=np.uint8))
    return out


def _write_gif(frames, out: Path, fps: int) -> int:
    # imageio/Pillow GIF duration is in milliseconds; GIF rounds to centiseconds.
    duration = frame_duration_ms(fps)
    imageio.mimsave(out, frames, format="GIF", duration=duration, loop=0)
    return out.stat().st_size


def encode_gif(frames, out: Path, fps: int, max_bytes: int) -> int:
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    # (scale, fps, frame_stride) ladder — progressively cheaper
    ladder = [(1.0, fps, 1), (0.78, fps, 1), (0.78, max(10, fps - 3), 1),
              (0.62, max(10, fps - 3), 1), (0.62, max(8, fps - 5), 2),
              (0.5, max(8, fps - 7), 2), (0.4, max(6, fps - 9), 2)]
    size = 0
    for scale, f_fps, stride in ladder:
        fr = _downscale(frames, scale)[::stride]
        size = _write_gif(fr, out, f_fps)
        if size <= max_bytes:
            return size
    raise RuntimeError(f"GIF still {size} bytes > budget {max_bytes} after fallback ladder")


_GPU_AVAILABLE = None


def gpu_available() -> bool:
    """Check whether Taichi can initialize a CUDA-backed GPU device.

    Memoized at module scope so ti.init() only runs once per process.
    """
    global _GPU_AVAILABLE
    if _GPU_AVAILABLE is None:
        try:
            import taichi as ti
            ti.init(arch=ti.cuda)
            _GPU_AVAILABLE = True
        except Exception:
            _GPU_AVAILABLE = False
    return _GPU_AVAILABLE


def load_inputs(cfg):
    """Load the cached VoxCity model and simulation results referenced by *cfg*."""
    from voxcity.io import load_voxcity, load_results_h5
    city = load_voxcity(str(cfg.voxcity_h5))
    results = load_results_h5(str(cfg.results_h5))
    return city, results


def _masked_city(city, keep: str):
    """Return a shallow copy of *city* with its voxel classes masked to *keep*."""
    c2 = copy.copy(city)
    c2.voxels = copy.copy(city.voxels)
    c2.voxels.classes = mask_classes(city.voxels.classes, keep)
    return c2


def render_voxel(city, cfg, keep=None, camera=None, arch=None) -> np.ndarray:
    """Render a single GPU still of the (optionally masked) city, fit to the canvas."""
    from voxcity.visualizer.renderer_gpu import visualize_voxcity_gpu
    target = _masked_city(city, keep) if keep else city
    if camera is None:
        camera = isometric_camera(city.voxels.classes.shape, city.voxels.meta.meshsize)
    cam_pos, cam_look = camera
    spp = 8 if cfg.quick else 48
    img = visualize_voxcity_gpu(
        target, width=cfg.width, height=cfg.height,
        samples_per_pixel=spp, camera_position=cam_pos, camera_look_at=cam_look,
        arch=(arch or ("gpu" if gpu_available() else "cpu")),
        output_path=None, show_progress=False,
    )
    img = np.asarray(img)
    if img.dtype != np.uint8:
        img = np.clip(img * (255 if img.max() <= 1.0 else 1), 0, 255).astype(np.uint8)
    return fit_canvas(img[..., :3], (cfg.width, cfg.height))


def _beats(cfg, n_full: int) -> int:
    return 2 if cfg.quick else n_full


def stage_settings(city, cfg):
    """Stage 1 — establishing shot of the target area's terrain elevation."""
    base = fit_canvas(raster_to_rgb(city.dem.elevation, cmap="terrain"), (cfg.width, cfg.height))
    caption = "1 · Target area — data sources · mesh size"
    return [compose(base, 0, caption, cfg) for _ in range(_beats(cfg, 12))]


def stage_download(city, cfg):
    """Stage 2 — cycle through the raw downloaded data layers."""
    dem = city.dem.elevation
    lc = city.land_cover.classes
    bh = city.buildings.heights
    ch = city.tree_canopy.top
    layers = [("Terrain elevation", dem, "terrain"),
              ("Land cover", lc, "tab20"),
              ("Building height", bh, "magma"),
              ("Canopy height", ch, "Greens")]
    frames = []
    per = _beats(cfg, 12)
    for name, arr, cmap in layers:
        base = fit_canvas(raster_to_rgb(arr, cmap=cmap), (cfg.width, cfg.height))
        frames += [compose(base, 1, f"2 · Download — {name}", cfg) for _ in range(per)]
    return frames


def stage_voxelize(city, cfg, camera):
    """Stage 3 — cumulative voxel build-up: terrain → land cover → buildings → trees."""
    order = [("terrain", "Terrain"), ("landcover", "Land cover"),
             ("buildings", "Buildings"), ("trees", "Trees")]
    frames = []
    per = _beats(cfg, 12)
    for keep, name in order:
        img = render_voxel(city, cfg, keep=keep, camera=camera)
        frames += [compose(img, 2, f"3 · Voxelize — + {name}", cfg) for _ in range(per)]
    return frames


def stage_integrate(city, cfg):
    """Stage 4 — a rotating view of the fully integrated voxel city model."""
    from voxcity.visualizer.renderer_gpu import visualize_voxcity_gpu
    import tempfile
    nframes = 4 if cfg.quick else 48
    with tempfile.TemporaryDirectory() as td:
        paths = visualize_voxcity_gpu(
            city, width=cfg.width, height=cfg.height,
            samples_per_pixel=8 if cfg.quick else 32,
            arch=("gpu" if gpu_available() else "cpu"),
            rotation=True, rotation_frames=nframes, output_directory=td,
            show_progress=False,
        )
        frames = []
        for p in paths:
            path = p[0] if isinstance(p, tuple) else p
            img = fit_canvas(np.asarray(Image.open(path).convert("RGB")), (cfg.width, cfg.height))
            frames.append(compose(img, 3, "4 · Integrate — voxel city model", cfg))
    return frames


def stage_simulate(city, results, cfg, camera):
    """Stage 5 — fixed-camera still with the solar irradiance ground overlay."""
    from voxcity.visualizer.renderer_gpu import visualize_voxcity_gpu
    ground = results.get("ground", {})
    grid = np.asarray(ground["solar_irradiance_instantaneous"]) if "solar_irradiance_instantaneous" in ground else None
    dem = city.dem.elevation
    cam_pos, cam_look = camera
    img = visualize_voxcity_gpu(
        city, width=cfg.width, height=cfg.height,
        samples_per_pixel=8 if cfg.quick else 40,
        camera_position=cam_pos, camera_look_at=cam_look,
        arch=("gpu" if gpu_available() else "cpu"),
        ground_sim_grid=grid, ground_dem_grid=dem, ground_colormap="inferno",
        show_progress=False, output_path=None,
    )
    img = np.asarray(img)
    if img.dtype != np.uint8:
        img = np.clip(img * (255 if img.max() <= 1.0 else 1), 0, 255).astype(np.uint8)
    img = fit_canvas(img[..., :3], (cfg.width, cfg.height))
    return [compose(img, 4, "5 · Simulate — solar irradiance", cfg) for _ in range(_beats(cfg, 16))]


def stage_export(city, cfg):
    """Stage 6 — closing card naming the export/visualization formats."""
    base = np.full((cfg.height, cfg.width, 3), 245, dtype=np.uint8)
    labels = "OBJ · ENVI-met · MagicaVoxel · NetCDF · NumPy"
    canvas = Image.fromarray(base)
    draw = ImageDraw.Draw(canvas)
    font = _load_font(max(18, cfg.height // 18))
    tw = draw.textlength(labels, font=font)
    draw.text(((cfg.width - tw) / 2, cfg.height / 2 - font.size), labels, fill=(40, 40, 60), font=font)
    frame = compose(np.asarray(canvas, dtype=np.uint8), 5, "6 · Export & visualize", cfg)
    return [frame for _ in range(_beats(cfg, 12))]


def build_frames(cfg):
    """Run all six pipeline stages and stitch their frames into one sequence."""
    city, results = load_inputs(cfg)
    camera = isometric_camera(city.voxels.classes.shape, city.voxels.meta.meshsize)
    stages = [
        stage_settings(city, cfg),
        stage_download(city, cfg),
        stage_voxelize(city, cfg, camera),
        stage_integrate(city, cfg),
        stage_simulate(city, results, cfg, camera),
        stage_export(city, cfg),
    ]
    fade = 2 if cfg.quick else max(2, cfg.fps // 3)
    return stitch(stages, fade=fade)


def run(cfg) -> int:
    """Build frames, encode the GIF (and optional MP4), returning the GIF byte size."""
    frames = build_frames(cfg)
    size = encode_gif(frames, cfg.out, cfg.fps, MAX_BYTES_DEFAULT)
    return size


def parse_args(argv=None) -> Config:
    """Parse CLI arguments into a Config."""
    p = argparse.ArgumentParser(description="Generate the VoxCity README demo GIF.")
    p.add_argument("--out", type=Path, default=Config().out)
    p.add_argument("--width", type=int, default=Config().width)
    p.add_argument("--height", type=int, default=Config().height)
    p.add_argument("--fps", type=int, default=Config().fps)
    p.add_argument("--quick", action="store_true")
    a = p.parse_args(argv)
    return Config(width=a.width, height=a.height, fps=a.fps, out=a.out, quick=a.quick)


def main(argv=None) -> int:
    cfg = parse_args(argv)
    size = run(cfg)
    print(f"wrote {cfg.out} ({size/1024/1024:.2f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
