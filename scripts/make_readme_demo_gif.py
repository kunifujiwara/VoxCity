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
from dataclasses import dataclass as _dataclass
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

# Per-layer scene builder; non-cumulative layers for exploded assembly.
LAYERS = ("terrain", "landcover", "buildings", "trees")


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


def layer_mask(classes, layer):
    """Boolean mask of voxels belonging to exactly one pipeline layer."""
    c = np.asarray(classes)
    if layer == "terrain":
        return c == -1
    if layer == "landcover":
        return c >= 1
    if layer == "buildings":
        return c <= -3
    if layer == "trees":
        return c == -2
    raise ValueError(f"unknown layer {layer!r}")


def z_shift_classes(classes, dz: int):
    """Translate a class grid up by dz voxels along axis 2 (zero-filled)."""
    c = np.asarray(classes)
    out = np.zeros_like(c)
    if dz <= 0:
        if dz == 0:
            return c.copy()
        src = slice(-dz, None); dst = slice(0, c.shape[2] + dz)
    else:
        src = slice(0, c.shape[2] - dz); dst = slice(dz, None)
    out[:, :, dst] = c[:, :, src]
    return out


def _scale_layer_height(layer_classes, scale: float):
    """Flatten a single-layer voxel grid toward the ground by keeping only the
    lowest `scale` fraction of each column's occupied height (scale in (0,1])."""
    if scale >= 0.999:
        return layer_classes
    c = layer_classes.copy()
    nz = c.shape[2]
    keep_z = max(1, int(round(nz * scale)))
    c[:, :, keep_z:] = 0
    return c


def explode_city(city, offsets, scales=None):
    """Build one synthetic VoxCity combining the requested layers, each z-shifted
    by offsets[layer] voxels and optionally height-scaled by scales[layer]."""
    scales = scales or {}
    base = np.asarray(city.voxels.classes)
    combined = np.zeros_like(base)
    for layer, dz in offsets.items():
        layer_c = np.where(layer_mask(base, layer), base, 0).astype(base.dtype)
        layer_c = _scale_layer_height(layer_c, scales.get(layer, 1.0))
        layer_c = z_shift_classes(layer_c, int(dz))
        combined = np.where(layer_c != 0, layer_c, combined)
    c2 = copy.copy(city)
    c2.voxels = copy.copy(city.voxels)
    c2.voxels.classes = combined
    return c2


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


def _ease_in_out(t):
    """Cosine ease-in-out interpolation: 0 at t=0, 1 at t=1."""
    return 0.5 - 0.5 * np.cos(np.pi * t)


def orbit_path(shape, meshsize, n, sweep_deg=90.0, elev_factor=0.9,
               dist_factor=1.7, start_deg=45.0):
    """N eased camera poses orbiting the scene center at fixed elevation."""
    nx, ny, nz = shape
    ex, ey, ez = nx * meshsize, ny * meshsize, nz * meshsize
    center = (ex / 2.0, ey / 2.0, ez * 0.25)
    diag = float(np.hypot(ex, ey))
    radius = diag * dist_factor
    height = center[2] + diag * 0.7 * elev_factor + ez
    poses = []
    for i in range(n):
        t = 0.0 if n == 1 else i / (n - 1)
        ang = np.deg2rad(start_deg + sweep_deg * _ease_in_out(t))
        pos = (center[0] + radius * np.cos(ang),
               center[1] + radius * np.sin(ang),
               height)
        poses.append((pos, center))
    return poses


@_dataclass
class FrameSpec:
    stage: int
    caption: str
    labels: list
    offsets: dict | None
    scales: dict | None
    overlay: str | None
    chips: bool
    camera_t: float


_LAYER_LABEL = {"terrain": ("Terrain", "terrain"), "landcover": ("Land cover", "landcover"),
                "buildings": ("Building", "viridis"), "trees": ("Tree", "Greens")}
_EXPLODE = {"terrain": 0, "landcover": 10, "buildings": 20, "trees": 30}


def _beat_lengths(cfg):
    if cfg.quick:
        return [2, 4, 4, 4, 4, 2]           # 20 frames
    total = round(cfg.fps * cfg.seconds)
    weights = [0.12, 0.22, 0.22, 0.16, 0.18, 0.10]
    lens = [max(1, round(total * w)) for w in weights]
    return lens


def build_timeline(cfg):
    lens = _beat_lengths(cfg)
    total = sum(lens)
    specs = []
    done = 0
    def cam_t():
        return 0.0 if total <= 1 else min(1.0, done / (total - 1))

    b = lens[0]  # Beat 1: terrain tile rises
    for i in range(b):
        t = i / max(1, b - 1)
        specs.append(FrameSpec(0, "1 · Set target area",
            [("Terrain", "terrain")], {"terrain": int(round((1 - t) * 4))},
            {"terrain": 0.15}, None, False, cam_t())); done += 1

    b = lens[1]  # Beat 2: download slabs fan out (thin), exploded
    for i in range(b):
        t = i / max(1, b - 1)
        offs = {k: int(round(v * _ease_in_out(t))) for k, v in _EXPLODE.items()}
        specs.append(FrameSpec(1, "2 · Download data",
            [_LAYER_LABEL[k] for k in LAYERS], offs,
            {k: 0.15 for k in LAYERS}, None, False, cam_t())); done += 1

    b = lens[2]  # Beat 3: morph thin -> full voxels (still exploded)
    for i in range(b):
        t = i / max(1, b - 1)
        scales = {k: 0.15 + 0.85 * _ease_in_out(t) for k in LAYERS}
        specs.append(FrameSpec(2, "3 · Voxelize layers",
            [_LAYER_LABEL[k] for k in LAYERS], dict(_EXPLODE),
            scales, None, False, cam_t())); done += 1

    b = lens[3]  # Beat 4: ease down + assemble
    for i in range(b):
        t = i / max(1, b - 1)
        offs = {k: int(round(v * (1 - _ease_in_out(t)))) for k, v in _EXPLODE.items()}
        specs.append(FrameSpec(3, "4 · Integrate → voxel city",
            [], offs, None, None, False, cam_t())); done += 1

    b = lens[4]  # Beat 5: ground then building overlay
    half = b // 2
    for i in range(b):
        overlay = "ground" if i < half else "building"
        cap = "5 · Simulate — Ground-level" if i < half else "5 · Simulate — Building surface"
        specs.append(FrameSpec(4, cap, [], {k: 0 for k in LAYERS}, None,
            overlay, False, cam_t())); done += 1

    b = lens[5]  # Beat 6: export chips fan out
    for i in range(b):
        specs.append(FrameSpec(5, "6 · Export", [], {k: 0 for k in LAYERS},
            None, None, True, cam_t())); done += 1

    return specs


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


def render_still(city, cfg, camera, *, ground_grid=None, ground_dem=None,
                 ground_cmap="magma", building_mesh=None, building_value=None,
                 building_cmap="viridis"):
    """Render one GPU still of `city` at `camera`, fit to the canvas."""
    from voxcity.visualizer.renderer_gpu import visualize_voxcity_gpu
    cam_pos, cam_look = camera
    spp = 8 if cfg.quick else 40
    kwargs = dict(
        width=cfg.width, height=cfg.height, samples_per_pixel=spp,
        camera_position=cam_pos, camera_look_at=cam_look,
        arch=("gpu" if gpu_available() else "cpu"),
        output_path=None, show_progress=False,
    )
    if ground_grid is not None:
        nu, nv = city.voxels.classes.shape[0], city.voxels.classes.shape[1]
        kwargs.update(ground_sim_grid=to_uv_layout(ground_grid, (nu, nv)),
                      ground_dem_grid=ground_dem, ground_colormap=ground_cmap)
    if building_mesh is not None:
        kwargs.update(building_sim_mesh=building_mesh,
                      building_value_name=building_value,
                      building_colormap=building_cmap)
    img = np.asarray(visualize_voxcity_gpu(city, **kwargs))
    if img.dtype != np.uint8:
        img = np.clip(img * (255 if img.max() <= 1.0 else 1), 0, 255).astype(np.uint8)
    return fit_canvas(img[..., :3], (cfg.width, cfg.height))


def to_uv_layout(grid, ref_shape):
    """Return `grid` in the (u,v) order of ref_shape=(nu,nv) WITHOUT flipping.

    VoxCity contract (src/voxcity/geoprocessor/mesh.py header): sim arrays are
    uv-layout (axis0=u/north, axis1=v/east); the renderer remaps axes to scene
    (x=v, y=u, z). So we must NOT flipud/fliplr — only transpose if the caller
    handed us a (v,u) grid.
    """
    g = np.asarray(grid)
    nu, nv = ref_shape
    if g.shape == (nu, nv):
        return g
    if g.shape == (nv, nu):
        return g.T
    raise ValueError(f"sim grid {g.shape} incompatible with ref {ref_shape}")


def build_frames(cfg):
    """Run the pipeline and stitch frames into one sequence."""
    raise NotImplementedError("render pipeline rewired in Task 10 (render_timeline)")



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
