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

REPO_ROOT = Path(__file__).resolve().parents[1]
CANVAS_DEFAULT = (820, 512)          # (width, height)
FPS_DEFAULT = 15
MAX_BYTES_DEFAULT = 8 * 1024 * 1024  # 8 MB


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
