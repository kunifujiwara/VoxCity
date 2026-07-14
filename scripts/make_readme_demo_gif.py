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
