"""Generate the VoxCity pipeline demo GIF for the README hero.

Build-time asset generator. Renders a looping, step-by-step animation of the
VoxCity workflow from cached artifacts (no Google Earth Engine, no live compute).

Run:
    export LC_ALL=C.UTF-8 LANG=C.UTF-8
    venv/bin/python scripts/make_readme_demo_gif.py --out images/demo.gif
"""
from __future__ import annotations

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
CANVAS_DEFAULT = (820, 512)          # (width, height)
FPS_DEFAULT = 15
MAX_BYTES_DEFAULT = 8 * 1024 * 1024  # 8 MB

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
    out: Path = field(default_factory=lambda: REPO_ROOT / "images" / "demo.gif")
    mp4: bool = False
    quick: bool = False
    overlay: str = "solar"
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


STAGES = ["Settings", "Download", "Voxelize", "Integrate", "Simulate", "Export"]


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
    font = _load_font(max(14, bar_h // 2))
    draw.text((16, h - bar_h + (bar_h - font.size) // 2), caption, fill=(255, 255, 255, 255), font=font)
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
    duration = 1.0 / max(1, fps)
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
