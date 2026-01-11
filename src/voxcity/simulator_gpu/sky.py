"""VoxCity-style `sky` module (toplevel) for compatibility.

VoxCity exposes sky patch utilities under `voxcity.simulator.solar.sky`, but the
flattened `voxcity.simulator` namespace often ends up with a `sky` attribute.

This module forwards to `simulator_gpu.solar.sky`.
"""

from .solar.sky import *  # noqa: F401,F403
