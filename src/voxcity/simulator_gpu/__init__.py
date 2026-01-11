"""simulator_gpu: GPU-accelerated simulation modules using Taichi.

Compatibility goal:
    Allow the common VoxCity pattern to work without code changes beyond the
    import alias:

        import simulator_gpu as simulator

    by flattening a VoxCity-like public namespace (view/visibility/solar/utils).
"""

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

__version__ = "0.1.0"
