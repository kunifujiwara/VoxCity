"""Guards against heavy optional dependencies loading at import time.

Each test runs ``python -c`` in a subprocess so the measurement is not
polluted by modules imported earlier in the pytest process. Taichi prints a
banner to stdout when it initializes, so tests parse only the LAST stdout
line (the JSON payload).
"""

import json
import subprocess
import sys

# Heavy optional stacks that must NOT load when importing the generator.
FORBIDDEN_ON_GENERATOR_IMPORT = {
    "taichi",       # GPU runtime; creates a CUDA context on import
    "ee",           # earthengine-api (~1.4 s)
    "geemap",       # (~1.0 s)
    "ipyleaflet",   # notebook widgets, via geoprocessor.draw
    "osmnx",        # via geoprocessor.network
    "pyvista",      # via visualizer.renderer
    "plotly",       # via visualizer.renderer
}


def _loaded_top_level_modules(import_stmt: str) -> set:
    code = (
        f"import sys, json; {import_stmt}; "
        "print(json.dumps(sorted({m.split('.')[0] for m in sys.modules})))"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    return set(json.loads(proc.stdout.strip().splitlines()[-1]))


def test_generator_import_avoids_heavy_optional_deps():
    loaded = _loaded_top_level_modules("import voxcity.generator")
    leaked = loaded & FORBIDDEN_ON_GENERATOR_IMPORT
    assert not leaked, f"heavy modules leaked into generator import: {sorted(leaked)}"


def test_geoprocessor_import_avoids_heavy_optional_deps():
    loaded = _loaded_top_level_modules("import voxcity.geoprocessor")
    leaked = loaded & FORBIDDEN_ON_GENERATOR_IMPORT
    assert not leaked, f"heavy modules leaked into geoprocessor import: {sorted(leaked)}"


def test_geoprocessor_lazy_attrs_still_resolve():
    # Lazy __getattr__ must still serve submodules and re-exported functions.
    code = (
        "import json; from voxcity.geoprocessor import filter_buildings, utils; "
        "import voxcity.geoprocessor as gp; "
        "print(json.dumps([callable(filter_buildings), gp.io.__name__]))"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    ok, io_name = json.loads(proc.stdout.strip().splitlines()[-1])
    assert ok is True
    assert io_name == "voxcity.geoprocessor.io"


def test_visualizer_lazy_attrs_still_resolve():
    code = (
        "import json; from voxcity.visualizer import get_voxel_color_map; "
        "import voxcity.visualizer as v; "
        "print(json.dumps([callable(get_voxel_color_map), hasattr(v, 'GPURenderer'), hasattr(v, '_HAS_GPU_RENDERER')]))"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    vals = json.loads(proc.stdout.strip().splitlines()[-1])
    assert vals == [True, True, True]


def test_simulator_gpu_import_does_not_mutate_numba_cache_env():
    code = (
        "import os, json; before = 'NUMBA_CACHE_DIR' in os.environ; "
        "import voxcity.simulator_gpu; "
        "print(json.dumps(('NUMBA_CACHE_DIR' in os.environ) == before))"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert json.loads(proc.stdout.strip().splitlines()[-1]) is True


def test_reverse_geocoder_uses_single_process_mode(monkeypatch):
    """rg.search must pass mode=1: the default mode 2 spawns a cpu_count()
    multiprocessing pool (~600 MB per worker) and can re-execute unguarded
    user scripts on Windows."""
    import types

    calls = {}

    def fake_search(coords, mode=2):
        calls["mode"] = mode
        return [{"cc": "JP", "name": "Tokyo"}]

    monkeypatch.setitem(
        sys.modules, "reverse_geocoder", types.SimpleNamespace(search=fake_search)
    )
    from voxcity.geoprocessor import utils

    utils._country_name_cache.clear()
    name = utils.get_country_name(139.6503, 35.6762)
    assert calls["mode"] == 1
    assert name == "Japan"
