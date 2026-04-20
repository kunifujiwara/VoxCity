from typing import Protocol, runtime_checkable

from .envimet import *
from .magicavoxel import *
from .obj import *
from .cityles import *
from .netcdf import *


@runtime_checkable
class Exporter(Protocol):
    def export(self, obj, output_directory: str, base_filename: str):  # pragma: no cover - protocol
        ...


__all__ = [
    # Protocol
    "Exporter",
    # envimet
    "EnvimetExporter",
    "export_inx",
    "generate_edb_file",
    "generate_lad_profile",
    # magicavoxel
    "MagicaVoxelExporter",
    "export_magicavoxel_vox",
    "export_large_voxel_model",
    # obj
    "OBJExporter",
    "export_obj",
    "grid_to_obj",
    "export_netcdf_to_obj",
    "convert_colormap_indices",
    "mesh_faces",
    "create_face_vertices",
    # cityles
    "CityLesExporter",
    "export_cityles",
    # netcdf (netcdf.py already defines __all__, forwarded via import *)
    "NetCDFExporter",
    "voxel_to_xarray_dataset",
    "save_voxel_netcdf",
]