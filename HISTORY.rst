=======
History
=======

1.4.0 (2026-07-05)
------------------

* ``rectangle_vertices`` are now validated and canonicalized to
  ``[SW, NW, NE, SE]`` order at all public entry points (``get_voxcity``,
  ``get_voxcity_CityGML`` and the four grid functions). Non-canonical
  input is reordered with a warning; closed 5-point rings are accepted;
  out-of-range lon/lat raises ``ValueError`` with a (lat, lon) hint.
* Grid orientation conversions (GeoTIFF, MagicaVoxel, OBJ, rasterio
  interop, coastline masks) now go through named helpers in
  ``voxcity.utils.orientation``. No behavior change. The ENVI-met exporter
  intentionally remains an exception: it keeps SOUTH_UP internally and
  writes north-first rows only at the file-format boundary.
* Fixed contradictory 3D orientation statements in the documented grid
  contract.

0.1.0 (2024-08-01)
------------------

* First release on PyPI.
