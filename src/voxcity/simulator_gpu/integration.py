"""VoxCity-style `integration` module (toplevel) for compatibility."""

from .solar.voxcity import (
    get_global_solar_irradiance_using_epw,
    get_building_global_solar_irradiance_using_epw,
    save_irradiance_mesh,
    load_irradiance_mesh,
)

__all__ = [
    "get_global_solar_irradiance_using_epw",
    "get_building_global_solar_irradiance_using_epw",
    "save_irradiance_mesh",
    "load_irradiance_mesh",
]
