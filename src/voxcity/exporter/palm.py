"""
PALM static driver export module for VoxCity.

Writes a PIDS-conformant NetCDF static driver (``<domain_name>_static``) for
the PALM model system (https://palm.muk.uni-hannover.de/).

Notes:
- Expects raw land cover grids as produced per-source by VoxCity (same
  contract as the CityLES exporter). Supported sources: 'OpenStreetMap'
  (alias 'Standard'), 'Urbanwatch', 'OpenEarthMapJapan', 'ESA WorldCover',
  'ESRI 10m Annual Land Cover', 'Dynamic World V1'.
- Grids are written with no vertical flip: VoxCity's axis contract
  (axis 0 = north, row 0 = south edge; axis 1 = east) equals PALM's (y, x).
- Automated validation covers the file format plus PALM's documented runtime
  consistency rules; the exporter has additionally been validated end-to-end
  against real PALM v25.04 runs (see the design doc's validation record).
- Building footprints carry no ground surface class, matching PALM's own
  reference drivers. Consequence for the _p3d namelist: a domain with
  buildings needs the urban surface model enabled (with the land surface
  model and a radiation scheme); LSM alone makes PALM reject the driver
  with DRV0021 at the footprint cells, while a dynamics-only namelist
  accepts it as plain topography.
  Design: docs/superpowers/specs/2026-08-20-palm-exporter-design.md
"""

from __future__ import annotations

import datetime
import os
from collections import namedtuple
from pathlib import Path

import numpy as np
from netCDF4 import Dataset
from pyproj import Transformer

from ..geoprocessor.utils import compute_rotation_angle, normalize_rectangle_vertices
from ..models import VoxCity
from ..utils.lc import get_land_cover_classes
from ..utils.logging import get_logger

_logger = get_logger(__name__)

__all__ = ["PalmExporter", "export_palm"]

# ---------------------------------------------------------------------------
# PIDS fill values
FILL_FLOAT = -9999.0
FILL_INT = -9999
FILL_BYTE = -127

# Voxel class codes, mirroring generator/voxelizer.py's GROUND_CODE/
# TREE_CODE/BUILDING_CODE (re-declared here, not imported, the same way
# simulator_gpu's solar/visibility integration modules do -- voxelizer.py
# is a heavier generator-side import this exporter module otherwise has no
# reason to pull in). >=1 (not enumerated below) is the positive
# land-cover class stamped on the single ground-surface voxel.
_VOXEL_GROUND_CODE = -1
_VOXEL_TREE_CODE = -2
_VOXEL_BUILDING_CODE = -3

# surface_fraction index order per PIDS
IDX_VEGETATION = 0
IDX_PAVEMENT = 1
IDX_WATER = 2

# Valid PIDS class ranges (used by the consistency validator)
BYTE_RANGES = {
    "vegetation_type": (1, 18),
    "pavement_type": (1, 15),
    "water_type": (1, 5),
    "soil_type": (1, 6),
    "building_type": (1, 6),
}

# ---------------------------------------------------------------------------
# Land-cover class name -> (category, PALM code) per source.
# Categories: 'vegetation' -> vegetation_type, 'pavement' -> pavement_type,
# 'water' -> water_type, 'building' -> handled by the building mask (code None).
#
# PALM codes used: vegetation_type 1 bare soil, 2 crops, 3 short grass,
# 7 deciduous broadleaf trees, 13 ice caps, 14 bogs and marshes,
# 16 deciduous shrubs; pavement_type 2 asphalt, 3 concrete; water_type
# 1 lake, 3 ocean. pavement_type per PALM's own documented table
# (https://palm.muk.uni-hannover.de/trac/wiki/doc/app/land_surface_parameters):
# 1 is "asphalt/concrete mix", 2 is "asphalt (asphalt concrete)", 3 is
# "concrete (Portland concrete)" -- this module never uses 1.

OSM_CLASS_TO_PALM = {
    'Bareland': ('vegetation', 1),
    'Rangeland': ('vegetation', 3),
    'Shrub': ('vegetation', 16),
    'Moss and lichen': ('vegetation', 1),
    'Agriculture land': ('vegetation', 2),
    'Tree': ('vegetation', 7),
    'Wet land': ('vegetation', 14),
    'Mangroves': ('vegetation', 7),
    'Water': ('water', 1),
    'Snow and ice': ('vegetation', 13),
    'Developed space': ('pavement', 3),
    'Road': ('pavement', 2),
    'Building': ('building', None),
    'No Data': ('vegetation', 3),
}

URBANWATCH_CLASS_TO_PALM = {
    'Building': ('building', None),
    'Road': ('pavement', 2),
    'Parking Lot': ('pavement', 2),
    'Tree Canopy': ('vegetation', 7),
    'Grass/Shrub': ('vegetation', 3),
    'Agriculture': ('vegetation', 2),
    'Water': ('water', 1),
    'Barren': ('vegetation', 1),
    'Unknown': ('vegetation', 3),
    'Sea': ('water', 3),
}

OEMJ_CLASS_TO_PALM = {
    'Bareland': ('vegetation', 1),
    'Rangeland': ('vegetation', 3),
    'Developed space': ('pavement', 3),
    'Road': ('pavement', 2),
    'Tree': ('vegetation', 7),
    'Water': ('water', 1),
    'Agriculture land': ('vegetation', 2),
    'Building': ('building', None),
    'No Data': ('vegetation', 3),
}

ESA_CLASS_TO_PALM = {
    'Trees': ('vegetation', 7),
    'Shrubland': ('vegetation', 16),
    'Grassland': ('vegetation', 3),
    'Cropland': ('vegetation', 2),
    'Built-up': ('pavement', 3),
    'Barren / sparse vegetation': ('vegetation', 1),
    'Snow and ice': ('vegetation', 13),
    'Open water': ('water', 1),
    'Herbaceous wetland': ('vegetation', 14),
    'Mangroves': ('vegetation', 7),
    'Moss and lichen': ('vegetation', 1),
}

ESRI_CLASS_TO_PALM = {
    'No Data': ('vegetation', 3),
    'Water': ('water', 1),
    'Trees': ('vegetation', 7),
    'Grass': ('vegetation', 3),
    'Flooded Vegetation': ('vegetation', 14),
    'Crops': ('vegetation', 2),
    'Scrub/Shrub': ('vegetation', 16),
    'Built Area': ('pavement', 3),
    'Bare Ground': ('vegetation', 1),
    'Snow/Ice': ('vegetation', 13),
    'Clouds': ('vegetation', 3),
}

DYNAMIC_WORLD_CLASS_TO_PALM = {
    'Water': ('water', 1),
    'Trees': ('vegetation', 7),
    'Grass': ('vegetation', 3),
    'Flooded Vegetation': ('vegetation', 14),
    'Crops': ('vegetation', 2),
    'Shrub and Scrub': ('vegetation', 16),
    'Built': ('pavement', 3),
    'Bare': ('vegetation', 1),
    'Snow and Ice': ('vegetation', 13),
}

# Fallback assignment when a raw index has no name entry
DEFAULT_ASSIGNMENT = ('vegetation', 3)


def _get_source_name_mapping(land_cover_source):
    """Return the class-name -> (category, PALM code) table for a source."""
    if land_cover_source in ('OpenStreetMap', 'Standard'):
        return OSM_CLASS_TO_PALM
    if land_cover_source == 'Urbanwatch':
        return URBANWATCH_CLASS_TO_PALM
    if land_cover_source == 'OpenEarthMapJapan':
        return OEMJ_CLASS_TO_PALM
    if land_cover_source == 'ESA WorldCover':
        return ESA_CLASS_TO_PALM
    if land_cover_source == 'ESRI 10m Annual Land Cover':
        return ESRI_CLASS_TO_PALM
    if land_cover_source == 'Dynamic World V1':
        return DYNAMIC_WORLD_CLASS_TO_PALM
    _logger.warning(
        f"Unknown land cover source {land_cover_source!r}; "
        "falling back to the OpenStreetMap/Standard mapping"
    )
    return OSM_CLASS_TO_PALM


def _build_index_to_palm_map(land_cover_source):
    """Map raw per-source 0-based index -> (category, PALM code).

    Uses the source's class-name order from get_land_cover_classes, exactly
    like the CityLES exporter. Unknown sources use the OSM name list so raw
    indices still resolve deterministically.
    """
    try:
        class_names = list(get_land_cover_classes(land_cover_source).values())
    except Exception:
        # get_land_cover_classes has no else branch: an unrecognised source
        # raises rather than returning a default (UnboundLocalError today).
        # Fall back to the OSM class order so raw indices still resolve.
        class_names = list(get_land_cover_classes('OpenStreetMap').values())

    name_to_assignment = _get_source_name_mapping(land_cover_source)
    index_to_assignment = {
        idx: name_to_assignment.get(name, DEFAULT_ASSIGNMENT)
        for idx, name in enumerate(class_names)
    }
    return index_to_assignment, class_names


def _build_georeference(rectangle_vertices):
    """Georeferencing attributes from the AOI rectangle.

    Returns dict with origin_lon/origin_lat (SW corner), origin_x/origin_y
    (that corner in the auto-detected UTM zone), rotation_angle (degrees
    clockwise; VoxCity and PIDS conventions agree), and the EPSG code used.
    """
    rect = normalize_rectangle_vertices(rectangle_vertices, warn=False)
    origin_lon = float(rect[0][0])
    origin_lat = float(rect[0][1])
    # PALM hard-rejects rotation_angle outside [0, 360] with DRV0041 at
    # parameter checking. compute_rotation_angle returns a signed angle, and
    # even an axis-aligned rectangle can come back as a tiny NEGATIVE value
    # from floating-point noise (observed: -0.000802 deg killing a real run),
    # so every un-normalized export is one FP sign away from aborting.
    rotation_angle = float(compute_rotation_angle(rect)) % 360.0

    zone = min(max(int((origin_lon + 180.0) // 6.0) + 1, 1), 60)
    epsg = (32600 if origin_lat >= 0.0 else 32700) + zone
    transformer = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
    origin_x, origin_y = transformer.transform(origin_lon, origin_lat)

    return {
        "origin_lon": origin_lon,
        "origin_lat": origin_lat,
        "origin_x": float(origin_x),
        "origin_y": float(origin_y),
        "rotation_angle": rotation_angle,
        "epsg": epsg,
    }


def _ground_level(voxel_classes):
    """Per-column ground datum: 1 + the highest k whose voxel is ground
    (_VOXEL_GROUND_CODE) or land cover (>= 1); -1 where no such voxel.

    This is THE datum rule -- _build_zt (terrain quantization) and
    _reconcile_buildings_with_voxels (LOD2 segment synthesis) both route
    through it so the exported terrain frame and the building/tree levels
    can never disagree (spec 2026-08-24-palm-exporter-alignment §Design 1).

    It unifies the DATUM RULE, not the NO-DATUM POLICY: what to do with a
    column that returns -1 is deliberately each caller's own decision
    (_build_zt substitutes the domain minimum so the terrain field stays
    complete; _reconcile_buildings_with_voxels skips the column and warns
    rather than inventing a building on a datum it does not have). Do not
    unify those two fallbacks -- they answer different questions.
    """
    classes = np.asarray(voxel_classes)
    nz = classes.shape[2]
    if nz == 0:
        return np.full(classes.shape[:2], -1, dtype=np.int64)
    is_ground = (classes == _VOXEL_GROUND_CODE) | (classes >= 1)
    has = is_ground.any(axis=2)
    top = nz - 1 - np.argmax(is_ground[:, :, ::-1], axis=2)
    return np.where(has, top + 1, -1).astype(np.int64)


def _build_zt(voxel_classes, dem_grid, meshsize):
    """Terrain height for the static driver, QUANTIZED to the voxel grid.

    Each column's terrain top is ``_ground_level(voxel_classes) * meshsize``
    (min-shifted so min(zt) == 0), so every value is an exact multiple of
    ``meshsize``. PALM's own terrain rounding (empirically closest to ceil,
    but modified further by its topography filter) then becomes the
    IDENTITY regardless of which rule it applies -- on exact multiples of
    dz, ceil == floor == nint. This is what keeps PALM's discretized world
    aligned cell-for-cell with the voxcity voxel model (spec
    2026-08-24-palm-exporter-alignment-design.md).

    The quantization is exact in the float32 REPRESENTATION, not to any
    metric tolerance. A non-dyadic ``meshsize`` (2.2, 1/3, ...) is not
    exactly representable in binary, so ``k * meshsize`` carries a
    rounding error that grows with k -- at meshsize 2.2 and ~1000 cells of
    relief the residue off an exact metric multiple reaches ~4e-4 m, and
    no fixed tolerance in metres holds for all inputs. The invariant that
    DOES hold for every meshsize is the round-trip::

        k = np.round(zt.astype(np.float64) / meshsize)   # exact level
        np.float32(k * meshsize) == zt                   # bit-exact

    i.e. the integer level is always recoverable, which is what PALM's
    re-discretization actually depends on. Assert that, never ``abs(zt %
    meshsize) < eps``.

    ``origin_z`` (PIDS global attr, absolute elevation of domain z = 0)
    stays the minimum finite DEM value, exactly as before quantization;
    the zero plane sits within one cell of it, an accepted georeferencing
    imprecision. ``dem_grid`` no longer influences any cell's height in
    any branch -- it survives only as ``origin_z``.

    Two no-datum rules cover columns where ``_ground_level`` returns -1
    (no ground or land-cover voxel anywhere in the column): a column with
    no datum takes the domain's MINIMUM ground level, so the terrain field
    stays complete and gains no spurious relief; and if NO column has a
    datum the terrain is flat zero. Both warn.

    Falls back to the legacy continuous-DEM behavior (with a warning) when
    no voxel grid is available -- alignment is impossible then anyway.
    """
    dem = np.asarray(dem_grid, dtype=np.float64)
    finite = np.isfinite(dem)
    origin_z = float(dem[finite].min()) if finite.any() else 0.0
    if voxel_classes is None:
        _logger.warning(
            "PALM zt: no voxel grid on the model; falling back to "
            "continuous DEM heights (PALM will re-round them and the "
            "domain will NOT be cell-aligned with the voxel model)."
        )
        if not finite.any():
            # origin_z is 0.0 here by construction; return the computed
            # value rather than a second literal, so the one expression
            # at the top stays the only definition of origin_z.
            return np.zeros(dem.shape, dtype=np.float32), origin_z
        zt = dem.copy()
        zt[~finite] = origin_z
        zt -= origin_z
        return zt.astype(np.float32), origin_z

    gl = _ground_level(voxel_classes).astype(np.float64)
    ok = gl >= 0
    if not ok.any():
        _logger.warning(
            "PALM zt: no column has a ground datum (no ground/land-cover "
            "voxel anywhere), so the exported terrain is entirely flat. "
            "This usually means the voxel grid is empty or all-air."
        )
        return np.zeros(gl.shape, dtype=np.float32), origin_z
    if (~ok).any():
        _logger.warning(
            f"PALM zt: {int((~ok).sum())} column(s) have no ground datum "
            "(no ground/land-cover voxel); using the domain minimum "
            "ground level for them."
        )
        gl = np.where(ok, gl, gl[ok].min())
    zt = (gl - gl.min()) * float(meshsize)
    return zt.astype(np.float32), origin_z


def _clean_heights(heights):
    """Heights with NaN/inf treated as 'no building'."""
    h = np.asarray(heights, dtype=np.float64)
    return np.where(np.isfinite(h), h, 0.0)


def _to_level(value, meshsize):
    """Convert a height in meters to a voxel level index.

    Uses the same rounding rule as the voxelizer
    (voxelizer.py:_flatten_building_segments uses
    ``int(seg[0] * inv_vs + 0.5)``): ``int(value / meshsize + 0.5)``. Every
    height-to-level conversion in this module routes through here so
    nz-sizing and slice bounds are derived from one rounding rule.
    """
    return int(float(value) / meshsize + 0.5)


def _zu_levels(n_centres, meshsize):
    """PALM's scalar-grid heights ``zu(0:n_centres)``: a surface level at
    0.0, then cell centres ``(k - 0.5) * dz``.

    PALM checks the static driver's ``z`` coordinate (topography_mod,
    error PAC0337) and ``zlad`` coordinate (plant_canopy_model_mod)
    against its own zu grid within 0.001 * dz on read and aborts on any
    mismatch, so both coordinates must come from this one constructor.
    Verified against PALM v25.04 and its urban_environment reference
    static driver (z = [0, 1, 3, 5, ...] at dz = 2).
    """
    return np.concatenate(
        ([0.0], (np.arange(1, n_centres + 1) - 0.5) * meshsize)
    ).astype(np.float32)


def _clean_segment_bound(value):
    """A single LOD2 segment bound with NaN/inf treated as ground level (0.0).

    Segment bounds are otherwise the one input in this module not routed
    through finite-cleaning (heights via _clean_heights, ids via
    nan_to_num in _build_buildings): without this, _segment_top_m would
    silently drop a non-finite bound to 0 while _build_buildings_3d's
    _to_level call on the same raw value raised ValueError/OverflowError --
    the two builders disagreeing about whether the segment exists.
    """
    v = float(value)
    return v if np.isfinite(v) else 0.0


def _segment_top_m(min_heights, shape):
    """Per-cell tallest LOD2 segment top, in meters above ground.

    Each cell's value is ``max(0.0, seg[1])`` over its segments (0.0 if the
    cell has no segments or all segments are entirely below ground).
    """
    top = np.zeros(shape, dtype=np.float64)
    if min_heights is None:
        return top
    mh = np.asarray(min_heights, dtype=object)
    ny, nx = shape
    for i in range(ny):
        for j in range(nx):
            segs = mh[i, j]
            if not segs:
                continue
            cell_top = 0.0
            for seg in segs:
                # The max(0.0, ...) is redundant with the cell_top = 0.0
                # initialization above (t > cell_top already excludes any
                # t <= 0, so cell_top can never go negative either way);
                # kept so the floor survives a refactor of this loop.
                t = max(0.0, _clean_segment_bound(seg[1]))
                if t > cell_top:
                    cell_top = t
            top[i, j] = cell_top
    return top


def _build_building_mask(heights, min_heights):
    """Shared building-presence mask for the LOD1 and LOD2 builders.

    Returns ``(mask, segment_top_m)``, both (y, x): ``mask`` is
    ``(heights > 0) | (segment_top_m > 0)`` and ``segment_top_m`` is each
    cell's tallest LOD2 segment top (see _segment_top_m). Pass both to
    _build_buildings and _build_buildings_3d so the two outputs agree on
    which cells are buildings.
    """
    h = _clean_heights(heights)
    segment_top_m = _segment_top_m(min_heights, h.shape)
    mask = (h > 0.0) | (segment_top_m > 0.0)
    return mask, segment_top_m


def _reconcile_buildings_with_voxels(voxel_classes, heights, min_heights, meshsize):
    """Add LOD2 segments for building columns the 2-D grids miss but the
    voxel grid does not.

    Root cause (see the PALM-driver design doc / the 149-column field
    report this fix addresses): VoxCity's LOD2 pipeline voxelizes detailed
    building SHELLS -- partial-coverage edge cells, complex geometry, small
    structures -- while the 2-D heights/min_heights grids rasterize
    footprints. A column can end up with a real ``building_id`` (> 0) and
    real -3 voxels in ``city.voxels.classes`` while ``heights`` stays 0.0
    and ``min_heights`` stays an empty list -- invisible to
    _build_building_mask. Every other simulator in this ecosystem (wind
    LBM, solar, view) runs on the voxel grid, so PALM must not silently run
    on a different city.

    This is additive reconciliation, not a second presence rule: it
    *synthesizes* min_heights segments for the voxel-only columns (from
    contiguous runs of -3 in that column) and hands the augmented
    min_heights back to the caller, which must feed it into the ONE
    existing call to _build_building_mask before building_mask reaches
    _build_buildings/_build_buildings_3d (see that function's docstring:
    "so the two outputs agree on which cells are buildings"). A voxel-only
    column becomes, from that point on, an ordinary orphan-segment cell
    (heights == 0, min_heights non-empty) -- the exact case
    _build_buildings' existing repair path already handles (it raises
    buildings_2d to the segment top and logs a WARNING), and
    _build_buildings_3d already turns arbitrary min_heights segments into
    exact zu-level occupancy, so the reconciled 3-D mask comes from the
    real voxel column shape, not a blind ground-up extrusion.

    Height/segment datum matches generator/voxelizer.py and
    importer/integrate.py exactly (same rounding convention _to_level
    documents elsewhere in this module): for a column, let
    ``ground_level`` = 1 + the highest k whose voxel is ground/land-cover
    solid (``_VOXEL_GROUND_CODE`` or a positive land-cover code -- trees
    ``_VOXEL_TREE_CODE`` and buildings ``_VOXEL_BUILDING_CODE`` are never
    terrain). A contiguous run of building voxels from k=a to k=b
    (inclusive) becomes the segment
    ``[(a - ground_level) * meshsize, (b + 1 - ground_level) * meshsize]``;
    the top of that segment simplifies to
    ``(b - (ground_level - 1)) * meshsize`` -- i.e. exactly
    ``(top_k_building - terrain_top_k) * meshsize``, height above local
    terrain in meters, the same datum _build_buildings already uses.

    Building presence is decided purely by ``_VOXEL_BUILDING_CODE`` (-3):
    tree voxels (-2) are never reconciled into buildings.

    Returns ``(min_heights, n_reconciled_columns)``. ``min_heights`` is
    returned unchanged (same object) when there is nothing to reconcile,
    including when ``voxel_classes`` is ``None`` (pure-2D models / no voxel
    grid available) -- callers must treat that as "behave exactly as
    before this fix". A column with -3 voxels but no ground/land-cover
    voxel beneath them at all (should not happen given the voxelizer's own
    invariants -- every column always has a ground datum) is skipped with
    a warning rather than guessed at, and does not count toward
    n_reconciled_columns.
    """
    if voxel_classes is None:
        return min_heights, 0

    existing_mask, _ = _build_building_mask(heights, min_heights)
    has_building_voxel = np.any(voxel_classes == _VOXEL_BUILDING_CODE, axis=2)
    to_reconcile = has_building_voxel & ~existing_mask
    if not to_reconcile.any():
        return min_heights, 0

    is_ground = (voxel_classes == _VOXEL_GROUND_CODE) | (voxel_classes >= 1)
    min_heights = np.asarray(min_heights, dtype=object).copy()
    n_reconciled = 0
    for i, j in zip(*np.nonzero(to_reconcile)):
        ground_ks = np.nonzero(is_ground[i, j, :])[0]
        if ground_ks.size == 0:
            _logger.warning(
                f"PALM voxel reconciliation: column ({i}, {j}) has a "
                "building voxel (-3) but no ground/land-cover voxel "
                "beneath it in city.voxels.classes; skipping (cannot "
                "derive a height datum for this column)."
            )
            continue

        ground_level = int(ground_ks.max()) + 1
        building_ks = np.nonzero(
            voxel_classes[i, j, :] == _VOXEL_BUILDING_CODE
        )[0]
        # Contiguous runs of building_ks, e.g. [4, 5, 6, 9, 10] -> [(4, 6),
        # (9, 10)]: a voxel-only column can hold multiple disjoint runs
        # (e.g. a shell with a gap), each becoming its own LOD2 segment.
        run_start = prev = int(building_ks[0])
        runs = []
        for k in building_ks[1:]:
            k = int(k)
            if k != prev + 1:
                runs.append((run_start, prev))
                run_start = k
            prev = k
        runs.append((run_start, prev))

        cell = min_heights[i, j]
        if not isinstance(cell, list):
            cell = []
        for a, b in runs:
            lo = (a - ground_level) * meshsize
            hi = (b + 1 - ground_level) * meshsize
            cell.append([lo, hi])
        min_heights[i, j] = cell
        n_reconciled += 1

    if n_reconciled:
        _logger.info(
            f"PALM driver: {n_reconciled} building column(s) reconciled "
            "from the voxel grid (present as -3 in city.voxels.classes but "
            "missing from city.buildings.heights/min_heights)."
        )
    return min_heights, n_reconciled


def _build_buildings(heights, ids, building_type, segment_top_m, building_mask):
    """LOD1 building fields.

    Returns (buildings_2d float32, building_id int32, building_type int8),
    all (y, x) with PIDS fill values off ``building_mask`` (see
    _build_building_mask). ``buildings_2d`` is
    ``max(heights, segment_top_m)`` on the mask, so a cell with LOD2
    geometry but no recorded LOD1 height still gets a height. Cells with no
    positive id receive generated ids above the existing max.
    """
    h = _clean_heights(heights)
    effective_h = np.maximum(h, segment_top_m)

    buildings_2d = np.full(h.shape, FILL_FLOAT, dtype=np.float32)
    buildings_2d[building_mask] = effective_h[building_mask].astype(np.float32)

    repaired = building_mask & (segment_top_m > h)
    n_repaired = int(repaired.sum())
    if n_repaired:
        max_discrepancy = float((segment_top_m[repaired] - h[repaired]).max())
        _logger.warning(
            f"{n_repaired} cell(s) had LOD2 segment geometry taller than "
            "the recorded LOD1 height; buildings_2d was raised to match "
            f"(orphan-segment repair) (max discrepancy {max_discrepancy:.1f} m)."
        )

    if ids is None:
        ids_arr = np.zeros(h.shape, dtype=np.int64)
    else:
        # ids assumed small positive integers (VoxCity pipeline convention);
        # not validated here.
        ids_arr = np.nan_to_num(
            np.asarray(ids, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0
        ).astype(np.int64)

    building_id = np.full(h.shape, FILL_INT, dtype=np.int32)
    has_id = building_mask & (ids_arr > 0)
    building_id[has_id] = ids_arr[has_id].astype(np.int32)
    missing = building_mask & (ids_arr <= 0)
    if missing.any():
        next_id = int(ids_arr.max()) + 1
        building_id[missing] = np.arange(
            next_id, next_id + int(missing.sum()), dtype=np.int32
        )

    building_type_arr = np.full(h.shape, FILL_BYTE, dtype=np.int8)
    building_type_arr[building_mask] = np.int8(building_type)
    return buildings_2d, building_id, building_type_arr


def _has_elevated_segments(min_heights, meshsize):
    """True when any building segment starts above ground level after
    _to_level rounding — geometry buildings_2d cannot express."""
    if min_heights is None:
        return False
    for cell in np.asarray(min_heights, dtype=object).ravel():
        if not cell:
            continue
        for seg in cell:
            if _to_level(_clean_segment_bound(seg[0]), meshsize) > 0:
                return True
    return False


def _build_buildings_3d(heights, min_heights, meshsize, segment_top_m, building_mask):
    """LOD2 byte mask (z, y, x) from per-cell [min, max] segments.

    Levels follow PALM's zu grid (see _zu_levels): level 0 is the surface,
    level k >= 1 sits at (k - 0.5) * dz. A level is building when its
    height lies inside a segment's [min, max] -- the same "level inside
    geometry" rule _build_lad uses for crowns. Cells without segments fall
    back to ground extrusion from ``heights``. Aligned to
    ``building_mask`` (see _build_building_mask): every masked cell gets
    at least one filled level.
    """
    h = _clean_heights(heights)
    ny, nx = h.shape
    mh = None if min_heights is None else np.asarray(min_heights, dtype=object)

    height_top = _to_level(float(h.max()), meshsize) if h.size else 0
    # segment_top_m already holds each cell's max segment top in meters, and
    # _to_level is monotonic on non-negative inputs, so converting the
    # array-wide max once is equivalent to converting every segment
    # individually and taking the max of the resulting levels -- no second
    # scan of min_heights needed here. _to_level also equals the count of
    # zu centres at or below a height ((k - 0.5) * dz <= top iff
    # k <= top / dz + 0.5), so it doubles as the centre count here.
    segment_top_level = (
        _to_level(float(segment_top_m.max()), meshsize) if segment_top_m.size else 0
    )
    n_centres = max(height_top, segment_top_level, 1)
    zl = _zu_levels(n_centres, meshsize).astype(np.float64)
    nz = zl.size

    b3d = np.zeros((nz, ny, nx), dtype=np.int8)
    for i in range(ny):
        for j in range(nx):
            segs = mh[i, j] if mh is not None else None
            filled = False
            if segs:
                for seg in segs:
                    # Boolean level selection, not index slicing: a
                    # fully-below-ground segment selects no level and a
                    # straddling one starts at the surface, with no
                    # negative-index arithmetic to clamp (the wraparound
                    # bug class the old slice-based fill needed guards
                    # for). A segment sitting entirely between two levels
                    # (e.g. [2.5, 2.6] at dz=2) selects nothing and falls
                    # through to the guarantees below.
                    lo = _clean_segment_bound(seg[0])
                    hi = _clean_segment_bound(seg[1])
                    if hi <= 0.0:
                        # Mirrors _segment_top_m's contribution rule: a
                        # segment topping out at or below ground is not
                        # geometry, and letting it select the surface
                        # level (zl[0] = 0.0 lies inside e.g. [-5, 0])
                        # would fill a 3D column on a cell the shared
                        # mask says is not a building -- breaking the
                        # presence invariant the validator enforces.
                        continue
                    sel = (zl >= lo) & (zl <= hi)
                    if sel.any():
                        b3d[sel, i, j] = 1
                        filled = True
            if not filled and h[i, j] > 0.0:
                # Segments present but none of them filled (e.g. all
                # entirely below ground) must still fall back to extrusion
                # from heights, the same as a cell with no segments at all --
                # otherwise a recorded LOD1 height is silently dropped in
                # favor of the forced single-level fill below. zl[0] is 0.0,
                # so any positive height marks at least the surface level.
                b3d[zl <= h[i, j], i, j] = 1
                filled = True
            if not filled and building_mask[i, j]:
                # Every masked cell must end up with >=1 level: PALM treats
                # buildings_3d as authoritative when present, so a cell the
                # mask counts as a building whose geometry selects no zu
                # level (only reachable via a degenerate aloft segment
                # sitting entirely between two levels, since any positive
                # height already marks the surface level above) must not
                # vanish from the LOD2 mask while still holding a
                # building_id in buildings_2d -- that would silently delete
                # a building the user asked to export. Placing it at the
                # surface misstates position by less than one dz (the
                # vertical resolution limit anyway): a bounded, explainable
                # error beats unbounded silent data loss.
                b3d[0, i, j] = 1
    return b3d


def _build_surface_types(land_cover_grid, land_cover_source, canopy_mask,
                         under_tree_vegetation_type, soil_type_code, building_mask):
    """Mutually exclusive surface classification per PIDS.

    Precedence per cell: building (all fields stay fill) > canopy-over-
    non-water (ground under trees becomes ``under_tree_vegetation_type``)
    > land-cover mapping. Canopy over a cell the land-cover mapping
    resolves to water does NOT override it: overhanging/riparian canopy
    keeps its water surface (water differs from short grass in albedo,
    heat capacity, and evaporation by margins that dominate a microclimate
    result), and the LAD field already represents the trees independently
    of the surface type below them. A 'building' land-cover class without
    building height becomes pavement 3 (concrete): the surface is sealed
    but there is no obstacle.

    Returns dict with vegetation_type/pavement_type/water_type/soil_type
    (int8 (y, x)) and surface_fraction (float32 (3, y, x)).
    """
    index_to_assignment, class_names = _build_index_to_palm_map(land_cover_source)
    ny, nx = land_cover_grid.shape
    if building_mask.shape != (ny, nx) or canopy_mask.shape != (ny, nx):
        # A shape mismatch (e.g. a stale mask from a different grid) is the
        # one place in this module that would otherwise produce a wrong
        # answer quietly: NumPy broadcasting/fancy-indexing on a
        # differently-shaped boolean mask silently ignores the extra rows
        # or raises somewhere unrelated, rather than failing at the point
        # that actually has the wrong input.
        raise ValueError(
            "_build_surface_types: building_mask "
            f"{building_mask.shape} and canopy_mask {canopy_mask.shape} "
            f"must both match land_cover_grid {(ny, nx)}"
        )

    vegetation_type = np.full((ny, nx), FILL_BYTE, dtype=np.int8)
    pavement_type = np.full((ny, nx), FILL_BYTE, dtype=np.int8)
    water_type = np.full((ny, nx), FILL_BYTE, dtype=np.int8)
    soil_type = np.full((ny, nx), FILL_BYTE, dtype=np.int8)
    surface_fraction = np.full((3, ny, nx), FILL_FLOAT, dtype=np.float32)

    category_index = {
        'vegetation': IDX_VEGETATION,
        'pavement': IDX_PAVEMENT,
        'water': IDX_WATER,
    }
    mapping_stats = {}

    for i in range(ny):
        for j in range(nx):
            # Tier 1: building footprint -- the cell stays all-fill (the
            # building surface takes over); nothing below this applies.
            if building_mask[i, j]:
                continue
            # Tier 3 (resolved first so tier 2 can check whether it landed
            # on water): the land-cover mapping. A raw 'building' class
            # without recorded height has no obstacle, so it becomes
            # pavement (concrete) rather than staying an unmapped category.
            raw_idx = int(land_cover_grid[i, j])
            category, code = index_to_assignment.get(raw_idx, DEFAULT_ASSIGNMENT)
            if category == 'building':
                category, code = 'pavement', 3
            # Tier 2: canopy over non-water -- ground under trees becomes
            # vegetation. Skipped when tier 3 resolved to water: overhanging
            # canopy keeps its water surface (see the docstring).
            if canopy_mask[i, j] and category != 'water':
                category, code = 'vegetation', int(under_tree_vegetation_type)

            if category == 'vegetation':
                vegetation_type[i, j] = np.int8(code)
                soil_type[i, j] = np.int8(soil_type_code)
            elif category == 'pavement':
                pavement_type[i, j] = np.int8(code)
                soil_type[i, j] = np.int8(soil_type_code)
            else:  # water
                water_type[i, j] = np.int8(code)

            surface_fraction[:, i, j] = 0.0
            surface_fraction[category_index[category], i, j] = 1.0

            key = (raw_idx, category, code)
            mapping_stats[key] = mapping_stats.get(key, 0) + 1

    _logger.info("Land cover -> PALM surface classification summary:")
    total = ny * nx
    n_building = int(building_mask.sum())
    if n_building:
        # mapping_stats only covers non-building cells, so building_mask's
        # share is logged as its own line -- otherwise the itemized
        # percentages below sum to ~80% on a 20%-built domain with nothing
        # explaining the remainder.
        _logger.info(
            f"  {n_building} cell(s) ({n_building / total * 100:.1f}%) are "
            "buildings (classified separately; not counted below)."
        )
    for (raw_idx, category, code), count in sorted(mapping_stats.items()):
        name = class_names[raw_idx] if 0 <= raw_idx < len(class_names) else 'Unknown'
        _logger.info(
            f"  {raw_idx}: {name} -> {category}_type {code}: "
            f"{count} cells ({count / total * 100:.1f}%)"
        )

    return {
        "vegetation_type": vegetation_type,
        "pavement_type": pavement_type,
        "water_type": water_type,
        "soil_type": soil_type,
        "surface_fraction": surface_fraction,
    }


def _build_lad(canopy_top, canopy_bottom, meshsize, lad_value, building_mask):
    """Leaf area density field lad(zlad, y, x).

    zlad levels are [0, (k - 0.5) * dz ...] up to the highest canopy top
    (palm_csd convention: surface level plus cell centres); zlad[0] is
    always exactly 0.0 by construction. Vegetated columns carry 0.0 below
    the crown, ``lad_value`` inside [bottom, top] (both boundaries
    inclusive), and fill above the top; non-vegetated columns are all
    fill. Canopy over buildings is cleared. ``bottom`` needs no clamping:
    a negative bottom already satisfies ``level >= bottom`` for every zlad
    level, since every level is >= 0 (see the empty-crown fallback below
    for the other edge case this invariant covers).

    Returns (lad, zlad) or (None, None) when no canopy remains.
    """
    # NaN/inf -> 0.0 ("no canopy here"), matching _clean_heights and the
    # ids cleaning in _build_buildings. Without posinf/neginf, nan_to_num's
    # default substitutes +-inf with the largest finite float64 rather than
    # literal infinity, so n_centres below ends up astronomically large but
    # finite, and np.arange(1, n_centres + 1) raises ValueError: Maximum
    # allowed size exceeded -- not the OverflowError int() would raise on
    # literal infinity. On the bottom side, an unsanitized +inf would leave
    # bottom far above every zlad level, forcing the single-level fallback
    # instead of filling from the ground.
    top = np.nan_to_num(
        np.asarray(canopy_top, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0
    )
    top = np.where(np.asarray(building_mask, dtype=bool), 0.0, top)
    bottom = np.nan_to_num(
        np.asarray(canopy_bottom, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0
    )

    max_top = float(top.max()) if top.size else 0.0
    if max_top <= 0.0:
        return None, None

    # zu-grid convention (see _zu_levels), not the edge convention _to_level
    # implements: level k sits at height (k - 0.5) * dz, so the centre
    # count is ceil(max_top / dz - 0.5), not the int(x / dz + 0.5)
    # nearest-edge rounding used for slab sizing elsewhere. This is a
    # genuinely different rule, not an inline duplicate of _to_level's --
    # do not "fix" it to route through _to_level.
    n_centres = max(int(np.ceil(max_top / meshsize - 0.5)), 1)
    zlad = _zu_levels(n_centres, meshsize)

    ny, nx = top.shape
    lad = np.full((zlad.size, ny, nx), FILL_FLOAT, dtype=np.float32)
    for i, j in zip(*np.nonzero(top > 0.0)):
        t, b = top[i, j], bottom[i, j]
        col = np.zeros(zlad.size, dtype=np.float32)
        col[zlad > t] = FILL_FLOAT
        in_crown = (zlad >= b) & (zlad <= t)
        if not in_crown.any():
            # Empty crown: either too thin to catch a level (bottom and top
            # both between two levels) or inverted (bottom > top). Both
            # fall back to the single topmost level at/below top. `[-1]` is
            # safe because zlad[0] == 0.0 and t > 0 here (top > 0.0 is the
            # loop condition above), so `zlad <= t` always has >=1 match.
            in_crown[np.nonzero(zlad <= t)[0][-1]] = True
        col[in_crown] = np.float32(lad_value)
        lad[:, i, j] = col
    return lad, zlad


# Single source of truth for every static-driver field: the validator
# asserts dtype from this table, the writer takes dims/fill/units/
# long_name/lod from it too, so the two can no longer independently
# drift. nc_type is a netCDF4 createVariable type code, and numpy
# understands the same codes (np.dtype("f4") == float32, "i4" == int32,
# "b" == int8), so it also serves as the expected in-memory dtype.
_FieldSpec = namedtuple("_FieldSpec", "dims nc_type fill units long_name lod optional")

_FIELD_SPECS = {
    "zt": _FieldSpec(("y", "x"), "f4", FILL_FLOAT, "m", "terrain height", None, False),
    "buildings_2d": _FieldSpec(("y", "x"), "f4", FILL_FLOAT, "m", "building height", 1, False),
    "building_id": _FieldSpec(("y", "x"), "i4", FILL_INT, "1", "building id numbers", None, False),
    "building_type": _FieldSpec(("y", "x"), "b", FILL_BYTE, "1", "building type classification", None, False),
    "vegetation_type": _FieldSpec(("y", "x"), "b", FILL_BYTE, "1", "vegetation type classification", None, False),
    "pavement_type": _FieldSpec(("y", "x"), "b", FILL_BYTE, "1", "pavement type classification", None, False),
    "water_type": _FieldSpec(("y", "x"), "b", FILL_BYTE, "1", "water type classification", None, False),
    "soil_type": _FieldSpec(("y", "x"), "b", FILL_BYTE, "1", "soil type classification", 1, False),
    "surface_fraction": _FieldSpec(
        ("nsurface_fraction", "y", "x"), "f4", FILL_FLOAT, "1", "surface fraction", None, False),
    "buildings_3d": _FieldSpec(("z", "y", "x"), "b", FILL_BYTE, "1", "building structure in 3d", 2, True),
    "lad": _FieldSpec(("zlad", "y", "x"), "f4", FILL_FLOAT, "m2 m-3", "leaf area density", None, True),
}


def _validate_static_fields(fields):
    """Enforce PALM's documented runtime consistency rules before writing.

    A violation is an exporter bug (the builders should never produce one),
    so this raises RuntimeError rather than silently correcting.

    Enforced:
    - Every field's in-memory dtype matches its _FIELD_SPECS declaration
      (the same table _write_static_driver writes from).
    - zt is finite, and its minimum is (within 1e-4 m) exactly 0.
    - buildings_2d is finite: a NaN/inf height is otherwise invisible to
      every rule below it -- `bld = buildings_2d != FILL_FLOAT` is True
      for NaN (NaN != anything is True), so a non-finite height silently
      counts as "this cell is a building" and would go on to satisfy
      every downstream presence rule rather than getting caught.
    - Every non-building cell has exactly one of vegetation_type/
      pavement_type/water_type set; every building cell has none.
    - soil_type is set exactly where vegetation_type or pavement_type is,
      in both directions.
    - surface_fraction has exactly 3 leading slices (vegetation/pavement/
      water) and sums to 1 on every classified cell.
    - Every building cell has a building_id and a building_type, and,
      unconditionally (not only when buildings_3d is present -- the
      buildings_3d="auto" default means most exports never build one),
      neither is set on a non-building cell.
    - Byte-coded fields (BYTE_RANGES) fall within their documented PIDS
      class range, both bounds.
    - When buildings_3d is present, its per-cell column presence agrees
      with buildings_2d/building_id/building_type in both directions.
      This deliberately never compares magnitudes: a buildings_2d height
      may legitimately exceed a shorter 3D column's top (e.g. 20 m next
      to a single [0, 4] segment is valid LOD2 data).
    - lad values (where not fill) are finite and non-negative.

    Deliberately not enforced here, not silently dropped:
    - "LAD values only within [bottom, top] layers": this function
      receives only the final lad array, not the canopy top/bottom inputs
      _build_lad placed it from, so it has no independent reference to
      re-derive the expected layer range against -- that placement is
      _build_lad's own contract, not something a downstream consumer of
      its output can re-verify. (The design spec's validation section
      previously listed this as enforced; it does not agree with this
      docstring or with _build_lad's own empty-crown fallback, which
      deliberately fills one level outside [bottom, top] -- the spec has
      been corrected to match the code.)
    """
    problems = []

    for name, spec in _FIELD_SPECS.items():
        if spec.optional:
            arr = fields.get(name)
            if arr is None:
                continue
        else:
            arr = fields[name]
        expected_dtype = np.dtype(spec.nc_type)
        if arr.dtype != expected_dtype:
            problems.append(
                f"{name} has dtype {arr.dtype}, expected {expected_dtype}"
            )

    zt = fields["zt"]
    if not np.isfinite(zt).all():
        problems.append("zt contains non-finite values")
    elif abs(float(zt.min())) > 1e-4:
        problems.append(f"zt minimum is {float(zt.min())}, expected 0")

    b2d_raw = fields["buildings_2d"]
    if not np.isfinite(b2d_raw).all():
        problems.append("buildings_2d contains non-finite values")

    veg = fields["vegetation_type"] != FILL_BYTE
    pav = fields["pavement_type"] != FILL_BYTE
    wat = fields["water_type"] != FILL_BYTE
    bld = b2d_raw != np.float32(FILL_FLOAT)
    has_id = fields["building_id"] != FILL_INT
    has_type = fields["building_type"] != FILL_BYTE

    n_types = veg.astype(int) + pav.astype(int) + wat.astype(int)
    if (n_types[~bld] != 1).any():
        problems.append("non-building cells must have exactly one surface type")
    if (n_types[bld] != 0).any():
        problems.append("building cells must have no surface type")

    soil = fields["soil_type"] != FILL_BYTE
    if ((veg | pav) != soil).any():
        problems.append("soil_type must be set exactly where vegetation or pavement is")

    sf = fields["surface_fraction"]
    if sf.shape[0] != 3:
        problems.append(
            f"surface_fraction leading dimension must be 3, got {sf.shape[0]}"
        )
    classified = veg | pav | wat
    if classified.any() and not np.allclose(sf.sum(axis=0)[classified], 1.0):
        problems.append("surface_fraction must sum to 1 on classified cells")

    # Both id/type presence directions are checked unconditionally (not
    # only inside the buildings_3d block below): a stray building_id or
    # building_type on a non-building cell is just as invalid whether or
    # not a 3D mask happens to be present, and buildings_3d="auto" means
    # most exports never build one -- LOD1-only is the common path, not
    # an edge case, so this cannot be deferred to the buildings_3d branch.
    if (bld & ~has_id).any():
        problems.append("building cells missing building_id")
    if (bld & ~has_type).any():
        problems.append("building cells missing building_type")
    if (~bld & has_id).any():
        problems.append("building_id present on a non-building cell")
    if (~bld & has_type).any():
        problems.append("building_type present on a non-building cell")

    for name, (lo, hi) in BYTE_RANGES.items():
        arr = fields[name]
        set_mask = arr != FILL_BYTE
        if set_mask.any():
            vals = arr[set_mask]
            if (vals < lo).any() or (vals > hi).any():
                problems.append(f"{name} values outside valid range [{lo}, {hi}]")

    # LOD1/LOD2 presence invariant is bidirectional: a cell is a building in
    # every building output or in none of them (see _build_building_mask).
    # buildings_2d may legitimately exceed a shorter 3D column's top (a 20 m
    # buildings_2d height next to a single [0, 4] segment is valid LOD2
    # data) -- this deliberately never compares magnitudes, only presence.
    b3d = fields.get("buildings_3d")
    if b3d is not None:
        col = b3d.astype(bool).any(axis=0)
        presence_checks = (
            (col & ~bld, "buildings_3d column present without a buildings_2d height"),
            (bld & ~col, "buildings_2d height present without a buildings_3d column"),
            (col & ~has_id, "buildings_3d column present without a building_id"),
            (has_id & ~col, "building_id present without a buildings_3d column"),
            (col & ~has_type, "buildings_3d column present without a building_type"),
            (has_type & ~col, "building_type present without a buildings_3d column"),
        )
        for mask, message in presence_checks:
            if mask.any():
                problems.append(message)

    lad = fields.get("lad")
    if lad is not None:
        data = lad[lad != np.float32(FILL_FLOAT)]
        if data.size and (not np.isfinite(data).all() or (data < 0).any()):
            problems.append("lad contains negative or non-finite values")

    if problems:
        raise RuntimeError(
            "PALM static driver consistency check failed: " + "; ".join(problems)
        )


def _write_static_driver(path, fields, coords, attrs):
    """Write the PIDS static driver NetCDF file (NETCDF4 format).

    Truncates and overwrites an existing file at ``path`` (netCDF4's
    ``"w"`` mode), matching how every other exporter in this package
    writes its output file.
    """
    with Dataset(str(path), "w", format="NETCDF4") as nc:
        for key, value in attrs.items():
            if value is not None:
                setattr(nc, key, value)

        def coord(name, values, long_name):
            nc.createDimension(name, values.size)
            var = nc.createVariable(name, "f4", (name,))
            var[:] = values
            var.units = "m"
            var.long_name = long_name

        coord("x", coords["x"], "distance to origin in x-direction")
        coord("y", coords["y"], "distance to origin in y-direction")
        if "z" in coords:
            coord("z", coords["z"], "height above origin")
        if "zlad" in coords:
            coord("zlad", coords["zlad"], "height above ground")

        nc.createDimension("nsurface_fraction", 3)
        nsf = nc.createVariable("nsurface_fraction", "i4", ("nsurface_fraction",))
        nsf[:] = [IDX_VEGETATION, IDX_PAVEMENT, IDX_WATER]

        def write(name, data, dims, dtype, fill, **var_attrs):
            var = nc.createVariable(name, dtype, dims, fill_value=fill, zlib=True)
            var[:] = data
            for k, v in var_attrs.items():
                setattr(var, k, v)

        # Declared from the same _FIELD_SPECS table the validator's dtype
        # check reads (defined above _validate_static_fields).
        for name, spec in _FIELD_SPECS.items():
            data = fields.get(name) if spec.optional else fields[name]
            if data is None:
                continue
            var_attrs = {"units": spec.units, "long_name": spec.long_name}
            if spec.lod is not None:
                var_attrs["lod"] = np.int32(spec.lod)
            fill = np.dtype(spec.nc_type).type(spec.fill)
            write(name, data, spec.dims, spec.nc_type, fill, **var_attrs)


_TRUNK_RATIO_SENTINEL = object()  # internal sentinel -- never pass this

# Applied when no explicit ratio, canopy_bottom_height_grid, or
# city.tree_canopy.bottom is available (see _check_export_inputs'
# sibling resolution logic in export_palm). A named constant, not an
# inline literal, so its value is directly assertable from a test
# (tests/test_exporter_palm.py checks this equals cityles.py's own
# default 0.3) rather than only inferable from crown geometry.
_DEFAULT_TRUNK_HEIGHT_RATIO = 0.3

# (name, accessor) pairs for every grid export_palm must shape-check
# against building heights before any builder runs. A single source of
# truth for _check_export_inputs' named_grids loop below, so a newly
# added grid is covered by construction rather than by remembering to
# add an `if x is not None: named_grids[...] = x` line -- the same
# "flat table pinned only where someone remembered" gap _FIELD_SPECS
# closes for the writer/validator and BYTE_RANGES closes for class
# ranges. accessor takes (city, canopy_bottom_height_grid) and
# returns the grid or None (grid absent -- nothing to check).
# tests/test_exporter_palm.py::TestExportPalmShapePreChecksParametrized
# iterates this same tuple.
_SHAPE_CHECKED_GRID_ACCESSORS = (
    ("land_cover", lambda city, _cbhg: city.land_cover.classes),
    ("dem", lambda city, _cbhg: city.dem.elevation),
    ("canopy_top", lambda city, _cbhg: (
        city.tree_canopy.top if city.tree_canopy is not None else None)),
    ("buildings.ids", lambda city, _cbhg: city.buildings.ids),
    ("buildings.min_heights", lambda city, _cbhg: city.buildings.min_heights),
    ("canopy_bottom_height_grid", lambda _city, cbhg: cbhg),
)


def _check_export_inputs(city, meshsize, building_type, soil_type,
                          under_tree_vegetation_type, lad, buildings_3d,
                          canopy_bottom_height_grid, trunk_height_ratio):
    """Validate export_palm's user-supplied inputs before any builder runs.

    Every failure here is a ValueError naming the offending input: bad user
    data, not an exporter bug (contrast _validate_static_fields, whose
    failures are always exporter bugs -- see its own docstring). This
    function has no side effects (no I/O, no mutation of its arguments):
    it only raises or returns, so it is independently testable without
    exercising the rest of export_palm.

    Left unchecked, each of these fails somewhere confusing instead of
    naming the offending input -- confirmed directly: a mismatched
    buildings.ids raises a raw numpy broadcast ValueError inside
    _build_buildings; a mismatched buildings.min_heights or
    canopy_bottom_height_grid each raise IndexError (not even ValueError)
    deep inside _build_buildings_3d/_build_lad's per-cell loops; an
    out-of-range building_type/soil_type/under_tree_vegetation_type either
    raises OverflowError (np.int8 cast) or reaches
    _validate_static_fields, producing a RuntimeError that tells the user
    the exporter is broken when they typed the bad value; meshsize=0
    raises ZeroDivisionError inside _to_level, and meshsize<0 exports
    "successfully" with negative coordinates and a negative dx. An explicit
    trunk_height_ratio of nan or a large negative number silently fills the
    entire canopy column with LAD (both sanitize to "fill from the ground"
    somewhere downstream); a ratio above 1 silently collapses the crown to
    a single level via _build_lad's empty-crown fallback; and
    trunk_height_ratio=None reaches a bare float(None), raising a raw
    TypeError instead of this module's documented ValueError -- all
    confirmed directly, and this is also the parameter whose misuse is
    least visible in the output (no crash, no obviously-wrong shape, just
    a physically implausible canopy). A mismatched zt specifically would
    otherwise pass _validate_static_fields silently, since that function
    only checks zt for finiteness/minimum, never against the other fields'
    shapes.

    Returns (rect, extras, heights, shape, ids, min_heights, canopy_top).
    """
    extras = city.extras or {}
    rect = extras.get("rectangle_vertices")
    if rect is None:
        raise ValueError(
            "city.extras['rectangle_vertices'] is required for PALM export "
            "(georeferencing); refusing to write an unreferenced static driver"
        )

    heights = city.buildings.heights
    shape = heights.shape
    ids = city.buildings.ids
    min_heights = city.buildings.min_heights
    canopy_top = city.tree_canopy.top if city.tree_canopy is not None else None

    named_grids = {}
    for name, accessor in _SHAPE_CHECKED_GRID_ACCESSORS:
        grid = accessor(city, canopy_bottom_height_grid)
        if grid is not None:
            named_grids[name] = grid
    for name, grid in named_grids.items():
        if grid.shape != shape:
            raise ValueError(
                f"{name} grid shape {grid.shape} does not match building "
                f"heights shape {shape}"
            )
    if city.voxels.classes.shape[:2] != shape:
        raise ValueError(
            f"voxel grid horizontal shape {city.voxels.classes.shape[:2]} "
            f"does not match building heights shape {shape}"
        )

    if not (np.isfinite(meshsize) and meshsize > 0):
        raise ValueError(f"meshsize must be a positive, finite number, got {meshsize}")

    for name, value in (
        ("building_type", building_type),
        ("soil_type", soil_type),
        ("under_tree_vegetation_type", under_tree_vegetation_type),
    ):
        # under_tree_vegetation_type is a vegetation_type value by PIDS
        # definition (see _build_surface_types), so it shares that range.
        lo, hi = BYTE_RANGES["vegetation_type" if name == "under_tree_vegetation_type"
                              else name]
        if not (lo <= value <= hi):
            raise ValueError(
                f"{name}={value} is outside the valid PIDS class range [{lo}, {hi}]"
            )

    if not (np.isfinite(lad) and lad >= 0):
        raise ValueError(f"lad must be a finite, non-negative number, got {lad}")

    # isinstance(..., bool), not `buildings_3d in (True, False, "auto")`:
    # `in` compares with `==`, and bool is an int subtype, so 1 == True and
    # 0 == False -- an `in`-based check would silently accept
    # buildings_3d=1 (confirmed: it did, before this was caught).
    if not (isinstance(buildings_3d, bool) or buildings_3d == "auto"):
        raise ValueError(
            f"buildings_3d must be True, False, or 'auto', got {buildings_3d!r}"
        )

    # trunk_height_ratio: a fraction of canopy height (0 = crown reaches
    # the ground, 1 = crown is a single point at the top), so both ends of
    # [0, 1] are physically meaningful and inclusive; only checked when the
    # caller actually passed one (the sentinel means "use the default").
    # float(...) inside a try, not a bare comparison, because
    # trunk_height_ratio=None reaches a bare float(None) downstream and
    # raises TypeError -- this module documents ValueError for every bad
    # user input, not a mix of the two.
    if trunk_height_ratio is not _TRUNK_RATIO_SENTINEL:
        try:
            ratio_value = float(trunk_height_ratio)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "trunk_height_ratio must be a finite number in [0, 1], got "
                f"{trunk_height_ratio!r}"
            ) from exc
        if not (np.isfinite(ratio_value) and 0.0 <= ratio_value <= 1.0):
            raise ValueError(
                "trunk_height_ratio must be a finite number in [0, 1], got "
                f"{trunk_height_ratio!r}"
            )

    return rect, extras, heights, shape, ids, min_heights, canopy_top


def export_palm(city: VoxCity,
                output_directory: str = "output/palm",
                domain_name: str = "voxcity",
                origin_time: str = "2000-01-01 00:00:00 +00",
                lad: float = 1.0,
                trunk_height_ratio: float = _TRUNK_RATIO_SENTINEL,
                canopy_bottom_height_grid=None,
                building_type: int = 3,
                buildings_3d="auto",
                under_tree_vegetation_type: int = 3,
                soil_type: int = 3,
                land_cover_source: str | None = None,
                author: str | None = None,
                comment: str | None = None) -> str:
    """Export a VoxCity model to a PALM (PIDS) static driver NetCDF file.

    Assembles every field via this module's pure builders, validates the
    result against PALM's documented runtime consistency rules
    (_validate_static_fields), and writes it to
    ``<output_directory>/<domain_name>_static``.

    Parameters
    ----------
    city : VoxCity
        Model instance. ``city.extras['rectangle_vertices']`` is required
        for georeferencing.
    output_directory : str
        Directory for the output file; created (including parents) if it
        does not already exist.
    domain_name : str
        Output file is ``<domain_name>_static``.
    origin_time : str
        PIDS ``origin_time`` global attribute.
    lad : float
        Constant leaf area density (m2/m3) inside tree crowns.
    trunk_height_ratio : float, optional
        Fraction of canopy height below which there is no leaf area;
        must be in ``[0, 1]`` when given (0: crown reaches the ground; 1:
        crown is a single point at the top). When explicitly provided,
        canopy bottoms are always recomputed as ``top * ratio``, even when
        the model already carries a per-cell bottom grid (same sentinel
        semantics as ``export_cityles``). When omitted: an explicit
        ``canopy_bottom_height_grid`` wins; otherwise
        ``city.tree_canopy.bottom`` when present; otherwise the default
        ratio 0.3.
    canopy_bottom_height_grid : numpy.ndarray, optional
        Explicit canopy bottom heights; ignored when *trunk_height_ratio*
        is explicitly provided.
    building_type : int
        PALM building type class assigned to every building (default 3).
    buildings_3d : bool or "auto"
        ``True`` always writes the LOD2 mask, ``False`` never, ``"auto"``
        (default) writes it only when the model contains LOD2 segments
        starting above ground (see _has_elevated_segments) -- geometry a
        LOD1 height alone cannot express.
    under_tree_vegetation_type : int
        PALM vegetation type for ground beneath tree canopy (default 3).
    soil_type : int
        PALM soil type wherever vegetation or pavement is set (default 3).
    land_cover_source : str, optional
        Auto-detected from ``city.extras['land_cover_source']`` when
        omitted (falls back to ``'Standard'``).
    author, comment : str, optional
        Extra PIDS global attributes.

    Returns
    -------
    str
        Path to the written ``<domain_name>_static`` file.

    Raises
    ------
    ValueError
        For bad user input, checked before anything is written to disk
        (see _check_export_inputs): a missing
        ``city.extras['rectangle_vertices']``; a 2D grid (land cover, DEM,
        canopy top, buildings.ids, buildings.min_heights,
        canopy_bottom_height_grid) or the voxel grid's horizontal shape
        disagreeing with the building heights grid's shape; a non-positive
        or non-finite *meshsize*; *building_type*/*soil_type*/
        *under_tree_vegetation_type* outside their PIDS class range
        (``BYTE_RANGES``); a negative or non-finite *lad*; an explicit
        *trunk_height_ratio* outside ``[0, 1]``; or a *buildings_3d* value
        other than ``True``/``False``/``"auto"``.
    RuntimeError
        If the assembled fields fail PALM's documented consistency rules
        (see _validate_static_fields) -- an exporter bug rather than a
        user-data problem (every user-input violation above is a
        ValueError instead), also raised before anything is written.
    """
    meshsize = float(city.voxels.meta.meshsize)
    rect, extras, heights, shape, ids, min_heights, canopy_top = _check_export_inputs(
        city, meshsize, building_type, soil_type, under_tree_vegetation_type,
        lad, buildings_3d, canopy_bottom_height_grid, trunk_height_ratio,
    )
    land_cover_source = land_cover_source or extras.get("land_cover_source", "Standard")
    _logger.info(f"Exporting PALM static driver (source: {land_cover_source})")

    # ── canopy bottom resolution (export_cityles sentinel semantics) ──
    user_specified_ratio = trunk_height_ratio is not _TRUNK_RATIO_SENTINEL
    ratio = float(trunk_height_ratio) if user_specified_ratio else _DEFAULT_TRUNK_HEIGHT_RATIO
    if canopy_top is None:
        canopy_top = np.zeros(shape, dtype=np.float64)
    if user_specified_ratio:
        canopy_bottom = canopy_top * ratio
    elif canopy_bottom_height_grid is not None:
        canopy_bottom = canopy_bottom_height_grid
    elif city.tree_canopy is not None and city.tree_canopy.bottom is not None:
        canopy_bottom = city.tree_canopy.bottom
    else:
        canopy_bottom = canopy_top * ratio

    # ── build all fields ──
    geo = _build_georeference(rect)
    voxel_classes = city.voxels.classes if city.voxels is not None else None
    # zt is derived from the VOXEL GRID (not the raw DEM) so every terrain
    # height is an exact multiple of meshsize and PALM's own re-rounding is
    # the identity -- see _build_zt.
    zt, origin_z = _build_zt(voxel_classes, city.dem.elevation, meshsize)
    # Reconcile against the voxel grid BEFORE the one _build_building_mask
    # call below: VoxCity's LOD2 shell voxelization can mark a column a
    # building (-3 voxels, real building_id) that the 2-D heights/
    # min_heights rasterization missed entirely (see
    # _reconcile_buildings_with_voxels' docstring for the root cause).
    # Synthesizing min_heights segments here -- rather than a second,
    # independent presence rule downstream -- keeps the single presence
    # predicate intact (see the comment below).
    min_heights, _n_voxel_reconciled = _reconcile_buildings_with_voxels(
        voxel_classes, heights, min_heights, meshsize,
    )
    # One presence predicate feeds both LOD1 and LOD2 so the two cannot
    # disagree (see _build_building_mask): a cell is a building if it has a
    # positive height or LOD2 geometry rising above ground.
    building_mask, segment_top_m = _build_building_mask(heights, min_heights)
    buildings_2d, building_id, building_type_arr = _build_buildings(
        heights, ids, building_type=building_type,
        segment_top_m=segment_top_m, building_mask=building_mask,
    )

    b3d = None
    want_3d = buildings_3d is True or (
        buildings_3d == "auto" and _has_elevated_segments(min_heights, meshsize)
    )
    if want_3d and building_mask.any():
        # A domain with no buildings at all has nothing for a LOD2 mask to
        # represent; skip it even when buildings_3d=True was forced, rather
        # than writing an all-zero z-dimensioned variable.
        b3d = _build_buildings_3d(
            heights, min_heights, meshsize,
            segment_top_m=segment_top_m, building_mask=building_mask,
        )

    lad_arr, zlad = _build_lad(
        canopy_top, canopy_bottom, meshsize, lad_value=lad, building_mask=building_mask,
    )

    # Canopy presence for surface classification, sanitized exactly like
    # _build_lad treats its own `top` input: NaN/+-inf must read as "no
    # canopy" here too, or this mask could disagree with what _build_lad
    # actually resolved for the same column (e.g. an unsanitized +inf would
    # mark a column canopy-covered here while _build_lad treats it as bare).
    canopy_present = np.nan_to_num(
        np.asarray(canopy_top, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0
    ) > 0.0
    # `& ~building_mask` is redundant-but-kept: _build_surface_types' own
    # tier 1 already excludes every building_mask cell before it ever looks
    # at canopy_mask (see its docstring), so this term has no effect on
    # today's output. Kept anyway as a second, independent guard against a
    # future refactor of that precedence -- the same reasoning as
    # _segment_top_m's redundant max(0.0, ...) floor.
    canopy_mask = canopy_present & ~building_mask

    surfaces = _build_surface_types(
        city.land_cover.classes, land_cover_source, canopy_mask,
        under_tree_vegetation_type=under_tree_vegetation_type,
        soil_type_code=soil_type, building_mask=building_mask,
    )

    fields = {
        "zt": zt,
        "buildings_2d": buildings_2d,
        "building_id": building_id,
        "building_type": building_type_arr,
        "buildings_3d": b3d,
        "lad": lad_arr,
        **surfaces,
    }
    _validate_static_fields(fields)

    ny, nx = shape
    coords = {
        "x": ((np.arange(nx) + 0.5) * meshsize).astype(np.float32),
        "y": ((np.arange(ny) + 0.5) * meshsize).astype(np.float32),
    }
    # z/zlad are only ever added when the corresponding data field is
    # present: a dangling coordinate nothing references breaks nothing at
    # write time but is dead weight in the file, and the reverse (a data
    # field present without its coordinate) makes netCDF4 raise a raw
    # "cannot find dimension" error instead of this module's own checks.
    if b3d is not None:
        # b3d's leading dimension is n_centres + 1 zu levels (see
        # _build_buildings_3d); the coordinate must be the matching zu
        # heights or PALM aborts with PAC0337 on read.
        coords["z"] = _zu_levels(b3d.shape[0] - 1, meshsize)
    if zlad is not None:
        coords["zlad"] = zlad

    attrs = {
        "Conventions": "CF-1.7",
        "title": f"VoxCity PALM static driver: {domain_name}",
        "source": "VoxCity",
        "origin_time": origin_time,
        "creation_time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S +00"),
        "origin_lon": geo["origin_lon"],
        "origin_lat": geo["origin_lat"],
        "origin_x": geo["origin_x"],
        "origin_y": geo["origin_y"],
        "origin_z": float(origin_z),
        "rotation_angle": geo["rotation_angle"],
        "comment": (
            f"origin_x/origin_y in EPSG:{geo['epsg']}"
            + (f"; {comment}" if comment else "")
        ),
        "author": author,
    }

    out_path = Path(output_directory) / f"{domain_name}_static"
    # exist_ok=True, keyed on out_path.parent (not output_directory) so a
    # domain_name containing a path separator (e.g. "a/b") also gets its
    # nested parent created -- otherwise it would hit the same confusing
    # netCDF4 "PermissionError [Errno 13]" on Windows this call exists to
    # prevent, matching cityles.py's adapter, which makedirs the output
    # location before writing for the same reason.
    os.makedirs(out_path.parent, exist_ok=True)

    # Write to a temp sibling and rename into place on success, rather than
    # writing out_path directly: netCDF4's Dataset(path, "w") truncates the
    # target immediately, so a failure partway through a re-export would
    # otherwise destroy a previously valid, stable <domain_name>_static
    # file that users re-export to repeatedly -- replacing good data with
    # a file that opens cleanly as NetCDF4 but has zero variables.
    # os.replace only ever swaps in a completely-written file, and is
    # atomic on both POSIX and Windows for a same-directory rename.
    tmp_out_path = out_path.with_name(out_path.name + ".tmp")
    try:
        _write_static_driver(tmp_out_path, fields, coords, attrs)
        os.replace(tmp_out_path, out_path)
    except Exception as exc:
        # os.replace is INSIDE this try, not after it: on Windows,
        # os.replace(tmp, out_path) raises PermissionError [WinError 5] if
        # out_path is open elsewhere -- including the ordinary flow of
        # inspecting a driver with netCDF4.Dataset(...) and then
        # re-exporting into the same path -- and if that call were outside
        # the handler, the .tmp file would survive every such attempt and
        # accumulate (confirmed directly: it does). The target file itself
        # stays intact either way (os.replace never partially completes),
        # so re-raising a clear, named error here is about not leaking a
        # confusing raw PermissionError naming a .tmp path the caller never
        # asked about -- the same reasoning as the makedirs call above.
        if tmp_out_path.exists():
            tmp_out_path.unlink()
        raise RuntimeError(
            f"failed to write PALM static driver to {out_path} "
            f"({exc.__class__.__name__}: {exc}); any previous file at that "
            "path was left untouched"
        ) from exc

    _logger.info(
        f"PALM static driver written: {out_path} "
        f"({nx}x{ny} cells, dx={meshsize} m, "
        f"buildings_3d={'yes' if b3d is not None else 'no'}, "
        f"lad={'yes' if lad_arr is not None else 'no'})"
    )
    return str(out_path)


class PalmExporter:
    """Exporter adapter to write a VoxCity model to a PALM static driver.

    ``**kwargs`` here forward to export_palm's own optional parameters
    (e.g. ``building_type=``, ``buildings_3d=``), the same shape every
    other exporter adapter in this package follows by convention (see
    e.g. CityLesExporter). Note the Exporter protocol itself
    (voxcity.exporter.Exporter) does NOT declare ``**kwargs`` -- this is a
    convention shared by the concrete adapters, not an enforced part of
    the protocol. export_palm itself deliberately does NOT take
    ``**kwargs``: with 14 parameters, silently swallowing an unrecognised
    one (e.g. a typo'd ``buildings3d=True``) would drop it with no error
    rather than failing loudly, so an unknown keyword passed through this
    adapter still raises TypeError from export_palm's own signature check.
    """

    def export(self, obj, output_directory: str, base_filename: str, **kwargs):
        if not isinstance(obj, VoxCity):
            raise TypeError("PalmExporter expects a VoxCity instance")
        return export_palm(
            obj,
            output_directory=output_directory,
            domain_name=base_filename,
            **kwargs,
        )
