"""simulator_gpu: GPU-accelerated simulation modules using Taichi.

Compatibility goal:
    Allow the common VoxCity pattern to work without code changes beyond the
    import alias:

        import simulator_gpu as simulator

    by flattening a VoxCity-like public namespace (view/visibility/solar/utils).
"""

import os

# Disable Numba caching to prevent stale cache issues when module paths change.
# This avoids "ModuleNotFoundError: No module named 'simulator_gpu'" errors
# that can occur when Numba tries to load cached functions with old module paths.
os.environ.setdefault("NUMBA_CACHE_DIR", "")  # Disable disk caching
os.environ.setdefault("NUMBA_DISABLE_JIT", "0")  # Keep JIT enabled for performance

# Import Taichi initialization utilities first
from .init_taichi import (  # noqa: F401
    init_taichi,
    ensure_initialized,
    is_initialized,
)

# Check if Taichi is available
try:
    import taichi as ti
    _TAICHI_AVAILABLE = True
except ImportError:
    _TAICHI_AVAILABLE = False

# VoxCity-style flattening
from .view import *  # noqa: F401,F403
from .solar import *  # noqa: F401,F403
from .utils import *  # noqa: F401,F403

# Export submodules for explicit access
from . import solar  # noqa: F401
from . import visibility  # noqa: F401
from . import view  # noqa: F401
from . import utils  # noqa: F401
from . import common  # noqa: F401

# VoxCity-flattened module names that some code expects to exist on the toplevel
from . import sky  # noqa: F401
from . import kernels  # noqa: F401
from . import radiation  # noqa: F401
from . import temporal  # noqa: F401
from . import integration  # noqa: F401

# Commonly re-exported VoxCity solar helpers
from .kernels import compute_direct_solar_irradiance_map_binary  # noqa: F401
from .radiation import compute_solar_irradiance_for_all_faces  # noqa: F401

# Backward compatibility: some code treats `simulator.view` as `simulator.visibility`
# (VoxCity provides `view.py` wrapper; we also provide that module).

# Export shared modules (kept; extra symbols are fine)
from .core import (  # noqa: F401
    Vector3, Point3,
    PI, TWO_PI, DEG_TO_RAD, RAD_TO_DEG,
    SOLAR_CONSTANT, EXT_COEF,
)
from .domain import Domain, IUP, IDOWN, INORTH, ISOUTH, IEAST, IWEST  # noqa: F401


def clear_numba_cache():
    """Clear Numba's compiled function cache to resolve stale cache issues.
    
    Call this function if you encounter errors like:
        ModuleNotFoundError: No module named 'simulator_gpu'
    
    After calling this function, restart your Python kernel/interpreter.
    """
    import shutil
    import glob
    from pathlib import Path
    
    cleared = []
    
    # Clear .nbc and .nbi files in the package directory
    package_dir = Path(__file__).parent
    for pattern in ["**/*.nbc", "**/*.nbi"]:
        for cache_file in package_dir.glob(pattern):
            try:
                cache_file.unlink()
                cleared.append(str(cache_file))
            except Exception:
                pass
    
    # Clear __pycache__ directories
    for pycache in package_dir.glob("**/__pycache__"):
        try:
            shutil.rmtree(pycache)
            cleared.append(str(pycache))
        except Exception:
            pass
    
    # Try to clear user's .numba_cache if it exists
    home = Path.home()
    numba_cache = home / ".numba_cache"
    if numba_cache.exists():
        try:
            shutil.rmtree(numba_cache)
            cleared.append(str(numba_cache))
        except Exception:
            pass
    
    print(f"Cleared {len(cleared)} cache items. Please restart your Python kernel.")
    return cleared


__version__ = "0.1.0"
