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

def test_config_defaults():
    m = load_module()
    cfg = m.Config()
    assert cfg.width == 820
    assert cfg.height == 512
    assert cfg.fps == 15
    assert cfg.overlay == "solar"
    assert cfg.out.name == "demo.gif"
    assert m.MAX_BYTES_DEFAULT == 8 * 1024 * 1024
