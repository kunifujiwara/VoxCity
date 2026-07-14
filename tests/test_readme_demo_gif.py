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
