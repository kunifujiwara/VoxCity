"""
Solar Irradiance Simulation Package

Public API exports for the refactored solar simulator. The implementation
is decomposed into focused stages:
1) geometry.py   - Visibility and geometric analysis
2) radiation.py  - Physics: convert geometry to irradiance
3) temporal.py   - Time-series integration and scheduling
4) integration.py- High-level workflows and I/O
"""

# Stage 1: Geometry
from .geometry import (  # noqa: F401
    compute_direct_solar_irradiance_map_binary,
    get_solar_positions_astral,
)

# Stage 2: Radiation
from .radiation import (  # noqa: F401
    get_direct_solar_irradiance_map,
    get_diffuse_solar_irradiance_map,
    get_global_solar_irradiance_map,
    compute_solar_irradiance_for_all_faces,
    get_building_solar_irradiance,
)

# Stage 3: Temporal
from .temporal import (  # noqa: F401
    get_cumulative_global_solar_irradiance,
    get_cumulative_building_solar_irradiance,
    _configure_num_threads,
    _auto_time_batch_size,
)

# Stage 4: Integration
from .integration import (  # noqa: F401
    get_global_solar_irradiance_using_epw,
    get_building_global_solar_irradiance_using_epw,
    save_irradiance_mesh,
    load_irradiance_mesh,
)
