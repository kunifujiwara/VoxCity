"""
Solar Irradiance Simulation Package

Public API exports for the refactored solar simulator. The implementation
is decomposed into focused stages:
1) kernels.py    - Low-level kernels for visibility/irradiance
2) radiation.py  - Physics: convert geometry to irradiance
3) temporal.py   - Time-series integration and solar position
4) integration.py- High-level workflows and I/O
5) sky.py        - Sky hemisphere discretization methods
"""

# Stage 1: Kernels / Solar position
from .kernels import (  # noqa: F401
    compute_direct_solar_irradiance_map_binary,
)
from .temporal import (  # noqa: F401
    get_solar_positions_astral,
)

# Sky discretization methods
from .sky import (  # noqa: F401
    # Tregenza (145 patches)
    generate_tregenza_patches,
    get_tregenza_patch_index,
    get_tregenza_patch_index_fast,
    TREGENZA_BANDS,
    TREGENZA_BAND_BOUNDARIES,
    # Reinhart (subdivided Tregenza)
    generate_reinhart_patches,
    # Uniform grid
    generate_uniform_grid_patches,
    # Fibonacci spiral
    generate_fibonacci_patches,
    # Sun position binning
    bin_sun_positions_to_patches,
    bin_sun_positions_to_tregenza_fast,
    # Utilities
    get_patch_info,
    visualize_sky_patches,
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
