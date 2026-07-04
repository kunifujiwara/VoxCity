import importlib
import os
import subprocess
import sys
import tempfile


def _fresh_config(monkeypatch, env):
    """Reimport backend.config with a controlled environment."""
    for key in [
        "VOXCITY_DATA_DIR", "VOXCITY_OUTPUT_DIR",
        "DEFAULT_TOKYO_EPW", "NDSM_COG_PATH", "CITYGML_PATH",
        "VOXCITY_FRONTEND_DIST",
    ]:
        monkeypatch.delenv(key, raising=False)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    sys.modules.pop("backend.config", None)
    return importlib.import_module("backend.config")


def test_data_dir_defaults_to_bundled_app_data(monkeypatch):
    cfg = _fresh_config(monkeypatch, {})
    assert cfg.DATA_DIR.replace("\\", "/").endswith("app/data")


def test_data_dir_env_override_wins(monkeypatch):
    cfg = _fresh_config(monkeypatch, {"VOXCITY_DATA_DIR": "/work-voxcity/data"})
    assert cfg.DATA_DIR == "/work-voxcity/data"


def test_derived_paths_follow_data_dir(monkeypatch):
    cfg = _fresh_config(monkeypatch, {"VOXCITY_DATA_DIR": "/d"})
    assert cfg.DEFAULT_TOKYO_EPW == os.path.join(
        "/d", "temp", "epw_tokyo", "JPN_TK_Tokyo-Chiyoda.476620_TMYx.epw"
    )
    assert cfg.NDSM_COG_PATH == os.path.join("/d", "temp", "ndsm_cog.tif")
    assert cfg.CITYGML_CACHE_DIR == os.path.join("/d", "temp", "citygml_cache")


def test_explicit_epw_override_wins(monkeypatch):
    cfg = _fresh_config(monkeypatch, {"DEFAULT_TOKYO_EPW": "/custom/x.epw"})
    assert cfg.DEFAULT_TOKYO_EPW == "/custom/x.epw"


def test_output_dir_default_under_tempdir(monkeypatch):
    cfg = _fresh_config(monkeypatch, {})
    assert cfg.OUTPUT_DIR == os.path.join(tempfile.gettempdir(), "voxcity_output")


def test_citygml_path_none_when_absent(monkeypatch):
    cfg = _fresh_config(monkeypatch, {"VOXCITY_DATA_DIR": "/nonexistent-xyz"})
    assert cfg.CITYGML_PATH is None


def test_citygml_path_env_override_wins(monkeypatch):
    cfg = _fresh_config(monkeypatch, {"CITYGML_PATH": "/data/plateau/tokyo"})
    assert cfg.CITYGML_PATH == "/data/plateau/tokyo"


def test_frontend_dist_none_when_absent(monkeypatch):
    cfg = _fresh_config(monkeypatch, {"VOXCITY_DATA_DIR": "/nonexistent-xyz"})
    assert cfg.FRONTEND_DIST is None


def test_frontend_dist_env_override_wins(monkeypatch):
    cfg = _fresh_config(monkeypatch, {"VOXCITY_FRONTEND_DIST": "/srv/dist"})
    assert cfg.FRONTEND_DIST == "/srv/dist"


def test_config_import_is_subprocess_cheap():
    """backend.config must not pull in fastapi/voxcity/numpy/geopandas."""
    code = (
        "import sys; import backend.config as c; "
        "heavy = {'fastapi', 'voxcity', 'numpy', 'geopandas', 'taichi'}; "
        "loaded = heavy & set(sys.modules); "
        "print('LOADED:' + ','.join(sorted(loaded)))"
    )
    app_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # app/
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, cwd=app_dir
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().endswith("LOADED:")
