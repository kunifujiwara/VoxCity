"""VoxCity-style `radiation` module (toplevel) for compatibility."""

from .solar.integration import (
    get_direct_solar_irradiance_map,
    get_diffuse_solar_irradiance_map,
    get_global_solar_irradiance_map,
)


def compute_solar_irradiance_for_all_faces(*args, **kwargs):
    """Compatibility stub.

    VoxCity's public API sometimes re-exports this symbol; if your workflow
    depends on it, we can map it to a GPU equivalent (or implement it) once the
    expected inputs/outputs are confirmed.
    """
    raise NotImplementedError(
        "compute_solar_irradiance_for_all_faces is not implemented in simulator_gpu yet. "
        "Use get_*_solar_irradiance_map functions or open an issue with the expected signature."
    )


__all__ = [
    "get_direct_solar_irradiance_map",
    "get_diffuse_solar_irradiance_map",
    "get_global_solar_irradiance_map",
    "compute_solar_irradiance_for_all_faces",
]
