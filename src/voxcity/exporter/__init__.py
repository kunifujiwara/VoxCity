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