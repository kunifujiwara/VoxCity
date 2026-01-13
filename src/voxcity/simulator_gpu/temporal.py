"""VoxCity-style `temporal` module (toplevel) for compatibility."""

from .solar.integration import (
    get_solar_positions_astral,
    get_cumulative_global_solar_irradiance,
    get_cumulative_building_solar_irradiance,
)

__all__ = [
    "get_solar_positions_astral",
    "get_cumulative_global_solar_irradiance",
    "get_cumulative_building_solar_irradiance",
]
