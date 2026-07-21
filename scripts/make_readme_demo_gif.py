"""Generate the VoxCity pipeline demo GIF for the README hero.

Build-time asset generator. Renders a looping, step-by-step animation of the
VoxCity workflow from cached artifacts (no Google Earth Engine, no live compute).

Run:
    export LC_ALL=C.UTF-8 LANG=C.UTF-8
    venv/bin/python scripts/make_readme_demo_gif.py --out images/demo.webp
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parents[1]
CANVAS_DEFAULT = (960, 540)          # (width, height)
FPS_DEFAULT = 24

# Per-layer scene builder; non-cumulative layers for exploded assembly.
LAYERS = ("terrain", "landcover", "buildings", "trees")


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
    nx, ny, nz_base = base.shape
    # Pad the z-axis so a shift by the largest offset never truncates voxels
    # that started at the top of the grid (matches build_download_scene, which
    # grows nz the same way for tiny/quick cities).
    max_dz = max([int(dz) for dz in offsets.values()] + [0])
    if max_dz > 0:
        padded = np.zeros((nx, ny, nz_base + max_dz), dtype=base.dtype)
        padded[:, :, :nz_base] = base
        base = padded
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


def _ease_out_tail(t, hold: float = 0.7):
    """Constant speed for the first `hold` fraction of the sweep, then a smooth
    cosine deceleration to a full stop at t=1.

    Velocity is continuous at the junction (no visible kink) and reaches exactly
    0 at t=1, so the rotation reads as steady with a gentle settle at the end
    (unlike _ease_in_out, which also slows the start). Returns progress in [0, 1].
    """
    t = float(np.clip(t, 0.0, 1.0))
    tau = 1.0 - hold
    if tau <= 0:
        return t
    v = 1.0 / (hold + 2.0 * tau / np.pi)          # constant-phase speed
    if t <= hold:
        return v * t
    return v * hold + v * (2.0 * tau / np.pi) * np.sin(0.5 * np.pi * (t - hold) / tau)


# Horizontal framing: shift the model right by this fraction of the scene
# diagonal so the left caption column and the right empty space stay balanced.
_FRAME_PAN_FRAC = 0.12


def orbit_path(shape, meshsize, n, sweep_deg=90.0, elev_factor=0.9,
               dist_factor=1.7, start_deg=45.0, pan_frac=None):
    """N eased camera poses orbiting the scene center at fixed elevation.

    `pan_frac` shifts each pose horizontally in screen space (camera and look-at
    translated together along screen-left) so the model sits further right in the
    frame — balancing the left-hand caption column against the right-hand empty
    space. The pan cancels in look-relative geometry, so the orbit radius and
    azimuth are unchanged. Defaults to _FRAME_PAN_FRAC.
    """
    if pan_frac is None:
        pan_frac = _FRAME_PAN_FRAC
    nx, ny, nz = shape
    ex, ey, ez = nx * meshsize, ny * meshsize, nz * meshsize
    center = np.array([ex / 2.0, ey / 2.0, ez * 0.25])
    diag = float(np.hypot(ex, ey))
    radius = diag * dist_factor
    height = center[2] + diag * 0.7 * elev_factor + ez
    up = np.array([0.0, 0.0, 1.0])
    poses = []
    for i in range(n):
        t = 0.0 if n == 1 else i / (n - 1)
        ang = np.deg2rad(start_deg + sweep_deg * _ease_out_tail(t))
        pos = np.array([center[0] + radius * np.cos(ang),
                        center[1] + radius * np.sin(ang),
                        height])
        look = center.copy()
        if pan_frac:
            r = np.cross(look - pos, up)          # screen-right (horizontal)
            nrm = np.linalg.norm(r)
            if nrm > 1e-9:
                pan = (-pan_frac * diag) * (r / nrm)   # screen-left => model right
                pos = pos + pan
                look = look + pan
        poses.append((tuple(pos), tuple(look)))
    return poses


@dataclass
class FrameSpec:
    stage: int
    scene_kind: str
    caption: str
    labels: list
    callouts: list
    reveal: dict
    offsets: dict | None
    scales: dict | None
    camera_t: float
    plates: list | None = None
    plate_thick: dict | None = None


_LAYER_LABEL = {"terrain": ("Terrain", "terrain"), "landcover": ("Land Cover", "landcover"),
                "buildings": ("Building", "viridis"), "trees": ("Tree", "Greens")}
# Layer caption -> Lucide icon asset (scripts/icons/<key>.png), monochrome white.
_LAYER_ICON = {"Tree": "tree", "Building": "building",
               "Land Cover": "landcover", "Terrain": "terrain"}
_CALLOUT_ANCHOR = (0.72, 0.32)
_EXPLODE = {"terrain": 0, "landcover": 30, "buildings": 60, "trees": 90}

# Caption vertical placement. Calibrated so a fully-exploded slab (offset 90 =
# Tree) sits near the top and the ground slab (offset 0 = Terrain) near the
# bottom, with in-between layers linearly interpolated. Reused during integrate
# so captions ride down with their slabs and converge onto Terrain's row.
_LABEL_Y_TOP = 0.10   # height fraction for a fully raised slab (offset = max)
_LABEL_Y_BOT = 0.62   # height fraction for the ground slab (offset = 0)
_SIDE_MARGIN_FRAC = 0.04   # left inset for all on-frame captions (fraction of width)


def _offset_to_yfrac(offset):
    """Map a layer's z explode-offset to a caption y (fraction of frame height)."""
    maxoff = _EXPLODE["trees"] or 1
    return _LABEL_Y_BOT - (offset / maxoff) * (_LABEL_Y_BOT - _LABEL_Y_TOP)


def _layer_caption(layer, offset):
    """(text, y_fraction, icon_key) caption tuple for a layer at a z-offset."""
    name = _LAYER_LABEL[layer][0]
    return (name, _offset_to_yfrac(offset), _LAYER_ICON.get(name, ""))


def _beat_lengths(cfg):
    if cfg.quick:
        return [2, 2, 2, 2, 2, 2]
    total = round(cfg.fps * cfg.seconds)
    # Download and voxelize are pinned to fixed absolute durations so each of
    # their four layers gets a steady interval (e.g. 5.0s / 4 = 1.25s/layer).
    dl = max(1, round(cfg.fps * cfg.download_seconds))
    vx = max(1, round(cfg.fps * cfg.voxelize_seconds))
    # Remaining beats (integrate, city, sim_ground, sim_building) split the
    # leftover time in their original proportions.
    rest_w = [0.15, 0.12, 0.115, 0.115]
    rest_total = max(len(rest_w), total - dl - vx)
    wsum = sum(rest_w)
    rest = [max(1, round(rest_total * w / wsum)) for w in rest_w]
    return [dl, vx] + rest


def build_timeline(cfg):
    lens = _beat_lengths(cfg)
    total = sum(lens)
    specs = []
    done = 0
    def cam_t():
        return 0.0 if total <= 1 else min(1.0, done / (total - 1))

    # Beat 1: download — reveal layers one-by-one
    b = lens[0]
    for i in range(b):
        t = i / max(1, b - 1)
        n_rev = int(np.clip(int(t * len(LAYERS)) + 1, 1, len(LAYERS)))
        reveal = {L: (1 if j < n_rev else 0) for j, L in enumerate(LAYERS)}
        labels = [_layer_caption(L, _EXPLODE[L]) for L in LAYERS if reveal[L]]
        specs.append(FrameSpec(0, "download", "Download geospatial data",
            labels, [], reveal, None, None, cam_t())); done += 1

    # Beat 2: voxelize — start with all four flat colormap slabs, then convert
    # them into voxels one layer at a time (terrain -> land cover -> buildings
    # -> trees). Layers awaiting their turn stay as flat 2D-colormap slabs.
    # When a layer voxelizes it switches to the real voxel palette; land cover's
    # visible step IS this recolor (2D grid colormap -> 3D voxel colormap), with
    # no change in thickness. Only terrain grows from thin to full height;
    # land cover, buildings and trees appear instantly.
    b = lens[1]
    order = list(LAYERS)
    for i in range(b):
        t = i / max(1, b - 1)
        pos = t * len(order)
        active = min(len(order) - 1, int(pos))
        local_t = min(1.0, pos - active)
        offsets = {L: _EXPLODE[L] for L in order}
        plates, scales = [], {}
        for j, L in enumerate(order):
            if j < active:
                scales[L] = 1.0                        # already voxelized
            elif j == active:
                if L in ("buildings", "landcover", "trees"):
                    scales[L] = 1.0                    # instant voxelize
                else:
                    scales[L] = max(0.06, _ease_in_out(local_t))
            else:
                plates.append(L)                       # awaiting its turn (flat slab)
        labels = [_layer_caption(L, _EXPLODE[L]) for L in order]   # all four
        specs.append(FrameSpec(1, "voxelize", "Voxelize urban elements",
            labels, [], {}, offsets, scales, cam_t(),
            plates=plates)); done += 1

    # Beat 3: integrate — ease offsets to zero. Captions ride down with their
    # slabs (terrain, offset 0, stays put); once merged they become the single
    # "Voxel City Model" caption that carries into the city beat.
    b = lens[2]
    for i in range(b):
        t = i / max(1, b - 1)
        merge = _ease_in_out(t)
        offs_f = {k: v * (1 - merge) for k, v in _EXPLODE.items()}
        offs = {k: int(round(v)) for k, v in offs_f.items()}
        if merge >= 0.9:
            labels = [("Voxel City Model", _LABEL_Y_BOT, "")]
        else:
            labels = [_layer_caption(L, offs_f[L]) for L in LAYERS]
        specs.append(FrameSpec(2, "integrate", "Integrate voxelized elements",
            labels, [], {}, offs, None, cam_t())); done += 1

    # Beat 4: voxel city — keep the merged caption parked at the convergence row.
    b = lens[3]
    for i in range(b):
        specs.append(FrameSpec(3, "city", "Integrate voxelized elements",
            [("Voxel City Model", _LABEL_Y_BOT, "")], [], {}, None, None,
            cam_t())); done += 1

    # Beat 5: simulate ground — caption parked at the same lower-center slot.
    b = lens[4]
    for i in range(b):
        specs.append(FrameSpec(4, "sim_ground", "Simulate urban environment",
            [("e.g., Solar Irradiance", _LABEL_Y_BOT, "")], [], {}, None, None,
            cam_t())); done += 1

    # Beat 6: simulate building surface — caption at the same lower-center slot.
    b = lens[5]
    for i in range(b):
        specs.append(FrameSpec(5, "sim_building", "Simulate urban environment",
            [("e.g., Green View Index", _LABEL_Y_BOT, "")], [], {}, None, None,
            cam_t())); done += 1

    return specs


@dataclass
class Config:
    width: int = CANVAS_DEFAULT[0]
    height: int = CANVAS_DEFAULT[1]
    fps: int = FPS_DEFAULT
    seconds: float = 20.0
    # Fixed absolute durations (seconds) for the two "reveal" beats. The other
    # four beats share whatever time is left (seconds - download - voxelize).
    download_seconds: float = 4.0
    voxelize_seconds: float = 4.0
    out: Path = field(default_factory=lambda: REPO_ROOT / "images" / "demo.webp")
    quick: bool = False
    quality: int = 80
    # Ray-tracing quality. TEMPORARILY lowered for fast flow iteration; restore
    # spp=40, max_depth=8 for the final high-quality render.
    spp: int = 16
    max_depth: int = 5
    # Softer lighting: reduced directional intensity + raised ambient fill so
    # shadowed faces stay visible (ambient is applied by renderer_gpu's tracer).
    direct: tuple = (0.55, 0.55, 0.52)
    ambient: tuple = (0.40, 0.40, 0.43)
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
PLATE_BASE_ID = {"terrain": 100, "buildings": 120, "trees": 140, "landcover": 160}


def colormap_plate(values2d, cmap, nbins=16, base_id=100):
    """Bin a continuous 2D map into nbins synthetic voxel class ids plus a
    class_id -> [r,g,b] color dict sampled from `cmap`."""
    a = np.asarray(values2d, dtype=float)
    finite = np.isfinite(a)
    vmin = float(np.nanmin(a[finite])) if finite.any() else 0.0
    vmax = float(np.nanmax(a[finite])) if finite.any() else 1.0
    if vmax <= vmin:
        vmax = vmin + 1.0
    norm = (np.clip(np.nan_to_num(a, nan=vmin), vmin, vmax) - vmin) / (vmax - vmin)
    bins = np.clip(np.round(norm * (nbins - 1)).astype(int), 0, nbins - 1)
    ids = (base_id + bins).astype(np.int32)
    mapper = matplotlib.colormaps[cmap]
    cmap_dict = {}
    for b in range(nbins):
        r, g, bl, _ = mapper(b / max(1, nbins - 1))
        cmap_dict[base_id + b] = [int(r * 255), int(g * 255), int(bl * 255)]
    return ids, cmap_dict


def landcover_plate(lc2d, base_id=160, source="Standard"):
    """Map land-cover class indices to synthetic ids + VoxCity LUT colors."""
    from voxcity.utils.lc import get_land_cover_classes
    lut = list(get_land_cover_classes(source).keys())  # list of (r,g,b)
    idx = np.clip(np.asarray(lc2d).astype(int), 0, len(lut) - 1)
    ids = (base_id + idx).astype(np.int32)
    cmap_dict = {base_id + i: list(map(int, lut[i])) for i in range(len(lut))}
    return ids, cmap_dict


def load_download_maps(cfg):
    """Read the four 2D source maps directly from voxcity.h5.

    Returned in their native (correct, north-up) orientation. The voxel model
    is flipped to match these maps at load time (see load_inputs), so both the
    flat plates and the voxelized model share this single correct orientation.
    """
    import h5py
    with h5py.File(str(cfg.voxcity_h5), "r") as f:
        return {
            "terrain": np.asarray(f["/voxcity/dem"]),
            "landcover": np.asarray(f["/voxcity/land_cover"]),
            "buildings": np.asarray(f["/voxcity/building_height"]),
            "trees": np.asarray(f["/voxcity/canopy/top"]),
        }


def build_download_scene(city, maps, reveal, z_by_layer=None):
    """One flat plate per revealed layer at its explode z-offset."""
    from voxcity.visualizer.palette import get_voxel_color_map
    z_by_layer = z_by_layer or _EXPLODE
    base = np.asarray(city.voxels.classes)
    nx, ny, nz_base = base.shape
    # Grid must be tall enough to hold every configured explode z-offset, not
    # just the source city's height (the offsets can exceed a tiny/quick city).
    nz = max(nz_base, max(z_by_layer.values()) + 1)
    grid = np.zeros((nx, ny, nz), dtype=np.int32)
    color_map = {int(k): list(v) for k, v in get_voxel_color_map("default").items()}
    for layer in LAYERS:
        if int(reveal.get(layer, 0)) <= 0:
            continue
        if layer == "landcover":
            ids2d, cd = landcover_plate(maps["landcover"], PLATE_BASE_ID["landcover"])
        else:
            ids2d, cd = colormap_plate(maps[layer], LAYER_CMAP[layer],
                                       base_id=PLATE_BASE_ID[layer])
        color_map.update(cd)
        z = int(np.clip(z_by_layer[layer], 0, nz - 1))
        grid[:, :, z] = ids2d
    c2 = copy.copy(city)
    c2.voxels = copy.copy(city.voxels)
    c2.voxels.classes = grid
    return c2, color_map


def build_voxelize_scene(city, maps, plate_layers, voxel_scales, offsets, plate_thick=None):
    """Hybrid scene for the voxelize beat: some layers as flat colormap slabs
    and others as real (height-scaled) voxels, each at its explode z-offset.

    plate_layers   : layers rendered as a colormap plate (flat map).
    voxel_scales   : {layer: height_scale} for layers rendered as real voxels.
    offsets        : {layer: z-offset} for every shown layer.
    plate_thick    : {layer: thickness} extruding a plate downward into a thin
                     voxel slab (default 1 = single flat layer). Used to give
                     land cover a visible "voxelize" step while staying flat.

    Real voxel layers use the default palette; plate layers contribute their own
    synthetic colormap ids, mirroring build_download_scene so the transition from
    the download beat and into the integrate beat stays visually continuous.
    """
    from voxcity.visualizer.palette import get_voxel_color_map
    plate_thick = plate_thick or {}
    base = np.asarray(city.voxels.classes)
    nx, ny, nz_base = base.shape
    max_off = max([int(o) for o in offsets.values()] + [0])
    nz = nz_base + max_off
    grid = np.zeros((nx, ny, nz), dtype=np.int32)
    color_map = {int(k): list(v) for k, v in get_voxel_color_map("default").items()}
    # Real voxel layers (terrain / buildings / trees as they voxelize).
    for layer, scale in voxel_scales.items():
        layer_c = np.where(layer_mask(base, layer), base, 0).astype(base.dtype)
        layer_c = _scale_layer_height(layer_c, scale)
        padded = np.zeros((nx, ny, nz), dtype=np.int32)
        padded[:, :, :nz_base] = layer_c
        shifted = z_shift_classes(padded, int(offsets[layer]))
        grid = np.where(shifted != 0, shifted, grid)
    # Flat colormap plates (land cover + layers awaiting their turn).
    for layer in plate_layers:
        if layer == "landcover":
            ids2d, cd = landcover_plate(maps["landcover"], PLATE_BASE_ID["landcover"])
        else:
            ids2d, cd = colormap_plate(maps[layer], LAYER_CMAP[layer],
                                       base_id=PLATE_BASE_ID[layer])
        color_map.update(cd)
        z_top = int(np.clip(offsets[layer], 0, nz - 1))
        thick = max(1, int(plate_thick.get(layer, 1)))
        for z in range(max(0, z_top - thick + 1), z_top + 1):
            cell = grid[:, :, z]
            grid[:, :, z] = np.where(cell == 0, ids2d, cell)
    c2 = copy.copy(city)
    c2.voxels = copy.copy(city.voxels)
    c2.voxels.classes = grid
    return c2, color_map


CALLOUT = {
    "dl_terrain": "2D Terrain Elevation map",
    "dl_landcover": "2D Land Cover map",
    "dl_buildings": "2D building height map",
    "dl_trees": "2D Tree height map",
    "vx_terrain": "Terrain voxel",
    "vx_landcover": "Land Cover voxel",
    "vx_buildings": "building voxel",
    "vx_trees": "Tree voxel",
    "city": "Voxel City model",
    "sim_ground": "Simulation overlay (Ground level)",
    "sim_building": "Simulation overlay (Building surface)",
}


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


STAGES = ["Download", "Voxelize", "Integrate", "Voxel City", "Ground level", "Building surface"]


MONTSERRAT_PATH = REPO_ROOT / "scripts" / "fonts" / "Montserrat.ttf"


def _load_font(size: int, weight: str = "Bold"):
    """Primary font for all text: Montserrat (a free, geometric sans in the
    Gotham vein) at the given size and named weight. Falls back to DejaVu Bold
    then PIL's default if the bundled font is unavailable."""
    try:
        f = ImageFont.truetype(str(MONTSERRAT_PATH), size)
        try:
            f.set_variation_by_name(weight)
        except Exception:
            pass
        return f
    except OSError:
        pass
    for name in ("DejaVuSans-Bold.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


ICON_DIR = REPO_ROOT / "scripts" / "icons"


@lru_cache(maxsize=64)
def _load_icon(name: str, size: int):
    """Return a white, monochrome Lucide icon (RGBA) at `size` px, or None.

    Icons are pre-rasterized from lucide-react's SVGs (scripts/icons/<name>.png)
    so the render only needs Pillow. Cached per (name, size).
    """
    if not name:
        return None
    path = ICON_DIR / f"{name}.png"
    if not path.exists():
        return None
    return Image.open(path).convert("RGBA").resize((size, size), Image.LANCZOS)


def draw_labels(frame, labels):
    """Draw left-aligned captions.

    `labels` is a list of (text, y_fraction, icon_key) tuples. Each caption is
    left-aligned at a fixed side margin at its own y_fraction; the optional Lucide
    icon (a white PNG in scripts/icons/) is composited just left of the text.
    """
    img = Image.fromarray(frame).convert("RGB")
    draw = ImageDraw.Draw(img, "RGBA")
    w, h = img.size
    size = max(12, h // 30)
    font = _load_font(size)
    gap = int(round(draw.textlength(" ", font=font)))
    icon_px = int(round(size * 1.2))
    white = (255, 255, 255, 255)
    x0 = int(w * _SIDE_MARGIN_FRAC)
    for text, yfrac, icon in labels:
        y = int(h * yfrac)
        x = x0
        ic = _load_icon(icon, icon_px)
        if ic is not None:
            iy = y + int(round((size - icon_px) / 2))   # center icon on text line
            img.paste(ic, (x, iy), ic)
            x += icon_px + gap
        draw.text((x, y), text, fill=white, font=font)
    return np.asarray(img, dtype=np.uint8)


def draw_callouts(frame, callouts):
    if not callouts:
        return np.asarray(frame, dtype=np.uint8)
    img = Image.fromarray(frame).convert("RGB")
    draw = ImageDraw.Draw(img, "RGBA")
    w, h = img.size
    font = _load_font(max(13, h // 32))
    fs = getattr(font, "size", 13)
    for text, (xf, yf) in callouts:
        ax, ay = int(w * xf), int(h * yf)
        tw = draw.textlength(text, font=font)
        bx = int(w * 0.98 - tw)
        by = ay - fs
        draw.ellipse([ax - 4, ay - 4, ax + 4, ay + 4], fill=(255, 220, 80, 255))
        draw.line([(ax, ay), (bx - 10, ay), (bx - 10, by + fs // 2)],
                  fill=(255, 220, 80, 230), width=2)
        draw.rounded_rectangle([bx - 8, by - 4, bx + tw + 8, by + fs + 4],
                               radius=6, fill=(20, 20, 24, 200))
        draw.text((bx, by), text, fill=(255, 255, 255, 255), font=font)
    return np.asarray(img, dtype=np.uint8)


def compose(frame, stage_index: int, caption: str, cfg) -> np.ndarray:
    """Burn a bottom caption bar onto a frame.

    Args:
        frame: Input (H, W, 3) uint8 RGB array.
        stage_index: Current stage index (0-5); accepted for call-site
            compatibility (the top progress strip was removed).
        caption: Caption text for the bottom bar.
        cfg: Config object with width and height.

    Returns:
        (H, W, 3) uint8 RGB array with overlays drawn.
    """
    img = Image.fromarray(frame).convert("RGB")
    draw = ImageDraw.Draw(img, "RGBA")
    w, h = img.size

    # bottom caption bar (left-aligned at the side margin)
    bar_h = max(24, h // 12)
    draw.rectangle([0, h - bar_h, w, h], fill=(20, 20, 24, 190))
    font_size = max(14, bar_h // 2)
    font = _load_font(font_size)
    font_size = getattr(font, "size", font_size)
    x0 = int(w * _SIDE_MARGIN_FRAC)
    draw.text((x0, h - bar_h + (bar_h - font_size) // 2), caption,
              fill=(255, 255, 255, 255), font=font)
    return np.asarray(img, dtype=np.uint8)


def encode_webp(frames, out: Path, fps: int, quality: int = 80) -> int:
    """Write frames as an animated, looping WebP with millisecond frame delays."""
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    duration = frame_duration_ms(fps)
    imgs = [Image.fromarray(np.asarray(f, dtype=np.uint8)).convert("RGB") for f in frames]
    imgs[0].save(out, format="WEBP", save_all=True, append_images=imgs[1:],
                 duration=int(round(duration)), loop=0, quality=quality, method=6)
    return out.stat().st_size



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
    """Load the cached VoxCity model and simulation results referenced by *cfg*.

    The cached voxel classes are stored flipped north-south relative to the 2D
    source maps and to the correct geographic orientation (verified: the raw
    building map has IoU 1.0 with the building voxel footprint only after
    np.flipud). Flip the voxel model along axis 0 so the whole demo — flat
    plates (from the native maps), the voxelized model, and the sim overlays —
    renders in one correct, upright orientation.
    """
    from voxcity.io import load_voxcity, load_results_h5
    city = load_voxcity(str(cfg.voxcity_h5))
    city.voxels.classes = np.flipud(np.asarray(city.voxels.classes))
    results = load_results_h5(str(cfg.results_h5))
    return city, results


def load_building_mesh(cfg, y_extent=None):
    """Load the building GVI surface mesh + per-face view factors from H5.

    The cached simulation mesh stores vertices in (x=u, y=v) order, transposed
    relative to VoxCity's voxel meshes (which map grid (u,v,z) -> scene
    (x=v, y=u, z); see geoprocessor/mesh.py). Swap the x/y columns so the GVI
    surface lands on the voxel-city buildings.

    Because the voxel model is flipped along axis 0 (north-south) at load time
    (see load_inputs), the mesh's scene-y (=u) axis must be flipped to match:
    pass ``y_extent = nx * meshsize`` so ``y -> y_extent - y``.
    """
    import h5py, trimesh
    with h5py.File(str(cfg.results_h5), "r") as f:
        v = np.asarray(f["/building/green_view_index/mesh/vertices"])
        fc = np.asarray(f["/building/green_view_index/mesh/faces"])
        vals = np.asarray(f["/building/green_view_index/view_factor_values"])
    v = v.copy()
    v[:, [0, 1]] = v[:, [1, 0]]
    if y_extent is not None:
        v[:, 1] = float(y_extent) - v[:, 1]
    mesh = trimesh.Trimesh(vertices=v, faces=fc, process=False)
    mesh.metadata["view_factor_values"] = vals
    return mesh


def grayscale_voxel_color_map():
    """Default voxel palette mapped to neutral grays (per-class luminance).

    Used for the background city during the simulation beats so the magma
    ground overlay and viridis building-surface overlay stand out against a
    monochrome model.
    """
    from voxcity.visualizer.palette import get_voxel_color_map
    out = {}
    for k, val in get_voxel_color_map("default").items():
        r, g, b = (float(c) for c in val[:3])
        y = int(round(0.299 * r + 0.587 * g + 0.114 * b))
        out[int(k)] = [y, y, y]
    return out


def ground_overlay(city, results):
    """Ground solar irradiance grid + DEM for the ground-level sim beat.

    The voxel model is flipped to the native map orientation at load time (see
    load_inputs), which is the same orientation the cached ground grid and DEM
    are stored in. So no extra flip is needed here \u2014 the overlay already lands
    on the correct cells (verified against the voxel building footprint).
    """
    ground = results.get("ground", {}) if isinstance(results, dict) else {}
    grid = np.asarray(ground["solar_irradiance_instantaneous"])
    dem = np.asarray(city.dem.elevation)
    return grid, dem


def render_still(city, cfg, camera, *, ground_grid=None, ground_dem=None,
                 ground_cmap="magma", building_mesh=None,
                 building_value="view_factor_values", building_cmap="viridis",
                 building_vmin=None, building_vmax=None,
                 voxel_color_map="default"):
    """Render one GPU still of `city` at `camera`, fit to the canvas."""
    from voxcity.visualizer.renderer_gpu import visualize_voxcity_gpu
    cam_pos, cam_look = camera
    spp = 4 if cfg.quick else cfg.spp
    kwargs = dict(
        width=cfg.width, height=cfg.height, samples_per_pixel=spp,
        max_depth=cfg.max_depth,
        camera_position=cam_pos, camera_look_at=cam_look,
        arch=("gpu" if gpu_available() else "cpu"),
        output_path=None, show_progress=False,
        voxel_color_map=voxel_color_map,
        direct=cfg.direct, ambient=cfg.ambient,
    )
    if ground_grid is not None:
        nu, nv = city.voxels.classes.shape[0], city.voxels.classes.shape[1]
        kwargs.update(ground_sim_grid=to_uv_layout(ground_grid, (nu, nv)),
                      ground_dem_grid=(to_uv_layout(ground_dem, (nu, nv))
                                       if ground_dem is not None else None),
                      ground_colormap=ground_cmap)
    if building_mesh is not None:
        kwargs.update(building_sim_mesh=building_mesh,
                      building_value_name=building_value,
                      building_colormap=building_cmap)
        if building_vmin is not None:
            kwargs["building_vmin"] = building_vmin
        if building_vmax is not None:
            kwargs["building_vmax"] = building_vmax
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


def render_timeline(city, results, cfg):
    """Render every beat of `build_timeline(cfg)` into a list of uint8 frames."""
    shape = city.voxels.classes.shape
    meshsize = city.voxels.meta.meshsize
    tl = build_timeline(cfg)
    poses = orbit_path(shape, meshsize, n=len(tl), sweep_deg=180.0, start_deg=67.5)
    maps = load_download_maps(cfg)
    building_mesh = None
    gray_cmap = grayscale_voxel_color_map()
    frames = []
    for i, fs in enumerate(tl):
        pose = poses[i]
        if fs.scene_kind == "download":
            scene, cmap = build_download_scene(city, maps, fs.reveal)
            img = render_still(scene, cfg, pose, voxel_color_map=cmap)
        elif fs.scene_kind == "voxelize":
            scene, cmap = build_voxelize_scene(city, maps, fs.plates or [],
                                               fs.scales or {}, fs.offsets,
                                               plate_thick=fs.plate_thick)
            img = render_still(scene, cfg, pose, voxel_color_map=cmap)
        elif fs.scene_kind == "integrate":
            scene = explode_city(city, fs.offsets, fs.scales)
            img = render_still(scene, cfg, pose)
        elif fs.scene_kind == "city":
            img = render_still(city, cfg, pose)
        elif fs.scene_kind == "sim_ground":
            gg, gd = ground_overlay(city, results)
            img = render_still(city, cfg, pose, ground_grid=gg, ground_dem=gd,
                               ground_cmap="magma", voxel_color_map=gray_cmap)
        elif fs.scene_kind == "sim_building":
            if building_mesh is None:
                building_mesh = load_building_mesh(cfg, y_extent=shape[0] * meshsize)
            img = render_still(city, cfg, pose, building_mesh=building_mesh,
                               building_value="view_factor_values",
                               building_cmap="viridis", building_vmax=0.2,
                               voxel_color_map=gray_cmap)
        else:
            raise ValueError(f"unknown scene_kind {fs.scene_kind!r}")
        img = compose(img, fs.stage, fs.caption, cfg)
        if fs.labels:
            img = draw_labels(img, fs.labels)
        frames.append(img)
    return frames


def build_frames(cfg):
    """Run the pipeline and stitch frames into one sequence."""
    city, results = load_inputs(cfg)
    return render_timeline(city, results, cfg)


def run(cfg) -> int:
    """Build frames, encode the WebP, returning the byte size."""
    frames = build_frames(cfg)
    return encode_webp(frames, cfg.out, cfg.fps, quality=cfg.quality)


def parse_args(argv=None) -> Config:
    """Parse CLI arguments into a Config."""
    p = argparse.ArgumentParser(description="Generate the VoxCity README demo reel (WebP).")
    p.add_argument("--out", type=Path, default=Config().out)
    p.add_argument("--width", type=int, default=Config().width)
    p.add_argument("--height", type=int, default=Config().height)
    p.add_argument("--fps", type=int, default=Config().fps)
    p.add_argument("--seconds", type=float, default=Config().seconds)
    p.add_argument("--quality", type=int, default=80)
    p.add_argument("--spp", type=int, default=Config().spp,
                   help="ray-tracing samples per pixel (higher = slower/cleaner)")
    p.add_argument("--max-depth", type=int, default=Config().max_depth,
                   help="ray bounce depth")
    p.add_argument("--quick", action="store_true")
    a = p.parse_args(argv)
    cfg = Config(width=a.width, height=a.height, fps=a.fps, seconds=a.seconds,
                 out=a.out, quick=a.quick)
    cfg.quality = a.quality
    cfg.spp = a.spp
    cfg.max_depth = a.max_depth
    return cfg


def main(argv=None) -> int:
    cfg = parse_args(argv)
    size = run(cfg)
    print(f"wrote {cfg.out} ({size/1024/1024:.2f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
