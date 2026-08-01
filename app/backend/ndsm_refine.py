"""Evidence-based nDSM canopy refinement.

Replaces the coincidence heuristics that lived in main.py: the nDSM raster
genuinely contains roofs (Tokyo LAS has no vegetation/building classes), so
tree/roof separation here uses physical evidence -- multi-return fraction and
surface roughness pooled from the COG's count/sum bands -- with building
footprints and height coincidence only as fallback for ambiguous cells.

Frames: every grid returned to callers is anchored at ``rectangle_vertices[0]``
with axis 0 running along ``side_1`` (v0 -> v1) and axis 1 along ``side_2``
(v0 -> v3) -- the same frame every other voxcity grid built from the same
geometry uses. Under the app's ``[SW, NW, NE, SE]`` vertex convention that makes
row 0 the *south* edge, which is the display frame; but that is a consequence of
the convention, not a promise of this module. ``/api/rectangle-from-dimensions``
can orient the v0->v1 edge to an arbitrary azimuth, and at, say, 137 degrees
``side_1`` has negative delta-lat and row 0 is compass-north. What holds
unconditionally is the anchoring, which is what matters: the grids stay mutually
consistent. The COG is north-up raster space; the conversion happens exactly
once, in load_ndsm_evidence(), by assigning raster pixels to cells through the
grid geometry (coordinate-based, orientation-free).
"""
from __future__ import annotations

import math
import os
import warnings
from dataclasses import dataclass
from typing import Dict, Mapping, Optional

import numpy as np

__all__ = [
    "groupwise_percentile",
    "pool_evidence",
    "local_tree_median",
    "load_ndsm_evidence",
    "RefineParams",
    "classify_and_refine",
    "refine_from_evidence",
    "format_counts",
    "PARTITION_KEYS",
    "VERDICT_NONE",
    "VERDICT_CANOPY",
    "VERDICT_ROOF",
    "VERDICT_AMBIGUOUS_KEEP",
    "VERDICT_AMBIGUOUS_REPLACE",
    "VERDICT_NO_DATA",
    "VERDICT_NAMES",
]

# Band schema of the evidence COG. Band 1 is the nDSM height; bands 2-6 are the
# additive per-pixel LiDAR statistics that pool_evidence() turns into per-cell
# evidence. A COG with fewer than EVIDENCE_BANDS bands is read in degraded
# mode: height only, no evidence.
EVIDENCE_BANDS = 6


def _as_group_labels(groups: np.ndarray) -> np.ndarray:
    """Coerce *groups* to a flat array of integer cell labels.

    Integer dtype is required rather than cast into: callers derive labels from
    pixel coordinates, and a float array there means a division that should have
    been a floor -- ``astype(intp)`` would silently truncate 2.7 to cell 2 and
    the error would surface as a subtly misplaced canopy, not an exception.
    """
    arr = np.asarray(groups)
    if not np.issubdtype(arr.dtype, np.integer):
        raise TypeError(
            "groups must be an integer array of cell labels, got dtype "
            f"{arr.dtype!r}; floor the coordinate arithmetic before calling"
        )
    return arr.ravel().astype(np.intp, copy=False)


def _valid_group_selection(groups: np.ndarray, n_groups: int) -> np.ndarray:
    """Boolean mask of group labels that address a real cell.

    Pixels that fall outside the model grid carry a negative sentinel, and
    clipping can in principle produce a label at or past ``n_groups``. Both are
    dropped rather than raising: the raster covers more ground than the grid by
    construction, so out-of-range labels are normal, not an error.
    """
    return (groups >= 0) & (groups < n_groups)


def groupwise_percentile(
    values: np.ndarray,
    groups: np.ndarray,
    n_groups: int,
    q: float,
) -> np.ndarray:
    """Exact percentile of *values* within each group label.

    Non-finite values (NaN, +/-inf) are ignored; a group with no finite values
    yields NaN. Matches ``np.percentile``'s default 'linear' method.

    Vectorized: one ``np.lexsort`` puts every group's values in a contiguous
    ascending run, after which the interpolated rank of each group is read out
    with fancy indexing. There is no Python loop over cells or pixels.

    Returns a ``(n_groups,)`` float array. Raises ``ValueError`` if *q* is
    outside [0, 100] -- the rank arithmetic below would otherwise index out of
    the run for q > 100 and, worse, return a plausible-looking extrapolated
    number for q < 0.
    """
    q = float(q)
    if not 0.0 <= q <= 100.0:
        raise ValueError(f"q must be in [0, 100], got {q!r}")

    n_groups = int(n_groups)
    out = np.full(n_groups, np.nan, dtype=np.float64)
    if n_groups <= 0:
        return out

    values = np.asarray(values, dtype=np.float64).ravel()
    groups = _as_group_labels(groups)
    if values.size != groups.size:
        raise ValueError(
            f"values and groups must be the same length, got {values.size} "
            f"and {groups.size}"
        )

    keep = np.isfinite(values) & _valid_group_selection(groups, n_groups)
    if not keep.any():
        return out
    values = values[keep]
    groups = groups[keep]

    # Sort by group, then by value: each group is now an ascending run.
    order = np.lexsort((values, groups))
    sorted_values = values[order]
    sorted_groups = groups[order]

    counts = np.bincount(sorted_groups, minlength=n_groups)
    starts = np.concatenate(([0], np.cumsum(counts)[:-1]))

    present = np.flatnonzero(counts)
    # Interpolated rank within each run, exactly as np.percentile computes it.
    pos = (counts[present] - 1) * (q / 100.0)
    lo = np.floor(pos).astype(np.intp)
    hi = np.ceil(pos).astype(np.intp)
    frac = pos - lo
    base = starts[present]
    out[present] = (
        sorted_values[base + lo] * (1.0 - frac)
        + sorted_values[base + hi] * frac
    )
    return out


def pool_evidence(
    groups: np.ndarray,
    n_groups: int,
    n_all: np.ndarray,
    n_multi: np.ndarray,
    n_nonground: np.ndarray,
    sum_z: np.ndarray,
    sum_z2: np.ndarray,
) -> Dict[str, np.ndarray]:
    """Pool per-pixel LiDAR counts and sums into per-cell evidence.

    The raster carries *counts and sums*, never pre-computed ratios, because
    counts and sums are additive: summing them over whichever pixels happen to
    fall in a cell reproduces the statistic that would have been computed from
    the underlying points directly. Any runtime cell size therefore pools
    exactly, with no resampling error. Pre-computed per-pixel ratios would only
    be averageable, which is wrong whenever pixel point counts differ. The
    whole design rests on this property.

    Returns ``{"mrf", "roughness", "n_all", "n_nonground"}``, each a
    ``(n_groups,)`` float array:

    * ``mrf``       -- multi-return fraction, ``sum(n_multi) / sum(n_all)``;
                      NaN where the cell received no returns. Canopy scatters a
                      pulse into several echoes; a roof plane echoes once.
    * ``roughness`` -- population standard deviation of non-ground return
                      heights, ``sqrt(E[z^2] - E[z]^2)`` over ``n_nonground``
                      points. Canopy is rough at metre scale, roofs are planar.
                      NaN where ``n_nonground < 2``: std is *defined* at n=1 but
                      carries no information about dispersion, and 0.0 is not
                      "unknown" -- it is the maximally roof-like value on this
                      axis, so a sparse cell where one pulse clipped a thin tree
                      top would be classified as roof and suppressed. That is
                      precisely the false cap this module exists to remove.
    * ``n_all``       -- pooled return count, the confidence weight for ``mrf``.
    * ``n_nonground`` -- pooled non-ground count, the confidence weight for
                      ``roughness``; the classifier needs it to tell a
                      well-sampled smooth cell from a barely-sampled one.
    """
    n_groups = int(n_groups)
    groups = _as_group_labels(groups)

    def _pool(weights: np.ndarray) -> np.ndarray:
        weights = np.asarray(weights, dtype=np.float64).ravel()
        if weights.size != groups.size:
            raise ValueError(
                "every per-pixel array must be the same length as groups, got "
                f"{weights.size} and {groups.size}"
            )
        if groups.size == 0:
            return np.zeros(n_groups, dtype=np.float64)
        keep = _valid_group_selection(groups, n_groups)
        # No trailing slice needed: keep guarantees every label is < n_groups,
        # so bincount(minlength=n_groups) is exactly n_groups long.
        return np.bincount(
            groups[keep], weights=weights[keep], minlength=n_groups
        ).astype(np.float64, copy=False)

    tot_all = _pool(n_all)
    tot_multi = _pool(n_multi)
    tot_ng = _pool(n_nonground)
    tot_z = _pool(sum_z)
    tot_z2 = _pool(sum_z2)

    mrf = np.full(n_groups, np.nan, dtype=np.float64)
    has_returns = tot_all > 0
    mrf[has_returns] = tot_multi[has_returns] / tot_all[has_returns]

    roughness = np.full(n_groups, np.nan, dtype=np.float64)
    # Two non-ground points minimum: see the docstring -- at n=1 the statistic
    # exists but means nothing, and its value (0.0) reads as "definitely a roof".
    has_ng = tot_ng >= 2
    if has_ng.any():
        n = tot_ng[has_ng]
        mean = tot_z[has_ng] / n
        # Clamp before the sqrt: the sums arrive pre-accumulated, so
        # E[z^2] can land a hair below mean^2 through cancellation.
        var = np.maximum(tot_z2[has_ng] / n - mean * mean, 0.0)
        roughness[has_ng] = np.sqrt(var)

    return {
        "mrf": mrf,
        "roughness": roughness,
        "n_all": tot_all,
        "n_nonground": tot_ng,
    }


def _cell_labels(
    lon: np.ndarray,
    lat: np.ndarray,
    origin: np.ndarray,
    u_full: np.ndarray,
    v_full: np.ndarray,
    n_rows: int,
    n_cols: int,
) -> np.ndarray:
    """Cell label ``row * n_cols + col`` for each (lon, lat).

    The model grid is the parallelogram ``origin + a*u_full + b*v_full`` with
    ``a, b`` in [0, 1); *u_full* spans the row axis (v0 -> v1, side_1) and
    *v_full* the column axis (v0 -> v3, side_2). Inverting that 2x2 system by
    Cramer's rule gives each point's fractional position along the two grid
    axes, which floors directly to a row and a column.

    This is the *only* place the raster's north-up layout meets the model grid,
    and it never touches an array index: a pixel's cell is decided by where the
    pixel *is*, so rotation, mirroring and pixel order are all handled by
    construction. Any point outside the grid gets label -1, which pool_evidence
    and groupwise_percentile drop.
    """
    det = u_full[0] * v_full[1] - u_full[1] * v_full[0]
    if not np.isfinite(det) or det == 0.0:
        raise ValueError("degenerate grid geometry: side vectors are collinear")

    d_lon = np.asarray(lon, dtype=np.float64) - origin[0]
    d_lat = np.asarray(lat, dtype=np.float64) - origin[1]
    a = (d_lon * v_full[1] - d_lat * v_full[0]) / det       # fraction along u
    b = (d_lat * u_full[0] - d_lon * u_full[1]) / det       # fraction along v

    # Floor explicitly: ndsm_refine's group helpers reject float labels because
    # an implicit astype() here would truncate towards zero and put a pixel at
    # a = -0.3 into row 0 instead of outside the grid.
    row = np.floor(a * n_rows).astype(np.intp)
    col = np.floor(b * n_cols).astype(np.intp)
    inside = (row >= 0) & (row < n_rows) & (col >= 0) & (col < n_cols)
    return np.where(inside, row * n_cols + col, -1)


def _pixel_lonlat(transform, height: int, width: int, src_crs):
    """(lon, lat) of every pixel *centre* of a read window, raveled row-major.

    Applies the window's affine by broadcasting a row vector against a column
    vector, rather than materializing two full index grids with ``meshgrid``:
    identical to ``rasterio.transform.xy`` (asserted in the tests) at half the
    peak allocation, which matters because these arrays are one per raster pixel
    in the window, not one per model cell.
    """
    from pyproj import Transformer

    rows = np.arange(height, dtype=np.float64)[:, None] + 0.5
    cols = np.arange(width, dtype=np.float64)[None, :] + 0.5
    x = (transform.c + cols * transform.a + rows * transform.b).ravel()
    y = (transform.f + cols * transform.d + rows * transform.e).ravel()
    lon, lat = Transformer.from_crs(src_crs, "EPSG:4326", always_xy=True).transform(x, y)
    return np.asarray(lon), np.asarray(lat)


def load_ndsm_evidence(
    rectangle_vertices,
    meshsize: float,
    cog_path: str,
    height_q: float = 90,
) -> Optional[Dict[str, object]]:
    """Read the nDSM evidence COG once, aggregated onto the model grid.

    One windowed read replaces the old per-cell ``src.sample()``: *every* raster
    pixel under a cell contributes, instead of the single pixel that happened to
    sit under the cell centre. That point sampling is the spike mechanism -- one
    roof-edge pixel under a 2 m cell centre became a 30 m "tree" -- so the height
    returned here is the ``height_q`` percentile over the cell's pixels, which a
    lone contaminated pixel cannot move.

    Everything returned is anchored at ``rectangle_vertices[0]`` with axis 0
    along ``side_1`` -- row 0 = south under the app's ``[SW, NW, NE, SE]``
    convention; see the module docstring. ``src.read(window=...)`` yields
    north-up raster layout; the conversion happens in :func:`_cell_labels`, by
    coordinate, never by index arithmetic.

    Heights are returned exactly as the raster stores them, negatives included:
    clamping is the canopy builder's job (``_build_canopy_from_ndsm``), and a
    reader that silently clamped would hide a botched ground surface.

    Precondition on the COG writer: nodata must co-occur across all six bands.
    Where band 1 is nodata but the count bands are not, this returns a NaN height
    alongside confident-looking evidence. The evidence is not discarded there on
    purpose -- returns exist independently of whether a ground reference did, so
    zeroing them would paper over a preprocessing bug rather than fix it.

    Memory: the per-pixel intermediates are sized by the *raster window*, not the
    model grid. For a 2 km target at 0.5 m (16 M pixels) the concurrently-live
    arrays peak near 2 GB -- ~730 MB for the six-band float64 stack, ~240 MB each
    for the projected and geographic coordinates, and ~490 MB of transient
    products inside :func:`_cell_labels`. Coordinate arrays are released as soon
    as they are consumed. Targets much beyond that want the window read in row
    blocks, pooling each block's labels into the same accumulators.

    Args:
        rectangle_vertices: target rectangle, ``[SW, NW, NE, SE]`` in (lon, lat).
        meshsize: model cell size in metres.
        cog_path: path to the nDSM COG.
        height_q: percentile of band 1 within each cell, in [0, 100].

    Returns:
        ``None`` when the COG is missing or does not overlap the rectangle,
        otherwise a dict with

        * ``height``       -- (rows, cols) float, per-cell percentile of band 1;
                              NaN where no valid pixel fell in the cell.
        * ``mrf``, ``roughness``, ``n_all``, ``n_nonground`` -- per-cell evidence
                              from :func:`pool_evidence`, or ``None`` each in
                              degraded mode.
        * ``degraded``     -- True when the COG carries fewer than six bands, as
                              the single-band raster on disk does until the
                              rebuild ships. Callers must stay functional then.
        * ``shape``        -- ``(rows, cols)``, the model grid size.
    """
    import rasterio
    from rasterio.windows import Window, from_bounds, intersect, intersection
    from pyproj import Transformer
    from voxcity.geoprocessor.raster.core import compute_grid_geometry

    if not cog_path or not os.path.exists(cog_path):
        return None

    geom = compute_grid_geometry(rectangle_vertices, meshsize)
    if geom is None:
        return None

    # grid_size is (along side_1, along side_2) -- see calculate_grid_size in
    # voxcity/geoprocessor/raster/core.py, and compute_cell_center_coords which
    # builds its (nx, ny) arrays with index 0 along side_1.
    n_rows, n_cols = (int(n) for n in geom["grid_size"])
    n_cells = n_rows * n_cols
    origin = np.asarray(geom["origin"], dtype=np.float64)
    d_row, d_col = geom["adj_mesh"]
    # Full-extent side vectors rebuilt from the *cell* step, so the parallelogram
    # inverted below is the one compute_cell_center_coords lays cells on. (It
    # differs from side_1/side_2 by ~1e-10 degrees, since u_vec round-trips
    # through a geodesic distance -- but rebuilding from the step is the
    # definition the cell centres actually use.)
    u_full = n_rows * float(d_row) * np.asarray(geom["u_vec"], dtype=np.float64)
    v_full = n_cols * float(d_col) * np.asarray(geom["v_vec"], dtype=np.float64)

    with rasterio.open(cog_path) as src:
        if src.crs is None:
            raise ValueError("nDSM COG has no CRS")
        to_src = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
        corners = [origin, origin + u_full, origin + u_full + v_full, origin + v_full]
        corner_x, corner_y = to_src.transform(
            [float(c[0]) for c in corners], [float(c[1]) for c in corners]
        )
        exact = from_bounds(
            min(corner_x), min(corner_y), max(corner_x), max(corner_y),
            transform=src.transform,
        )
        # Grow to whole pixels on BOTH edges. round_offsets().round_lengths()
        # floors twice in rasterio 1.5, which drops the fractional far edge --
        # measured at 0.87 px, enough to starve the whole boundary row of cells
        # of a quarter of their returns while leaving the height percentile
        # looking fine. n_all and n_nonground are the classifier's confidence
        # weights, so a systematic deficit there silently demotes edge cells.
        col_off = math.floor(exact.col_off)
        row_off = math.floor(exact.row_off)
        window = Window(
            col_off,
            row_off,
            math.ceil(exact.col_off + exact.width) - col_off,
            math.ceil(exact.row_off + exact.height) - row_off,
        )

        # Clip rather than read boundless. A non-boundless read of a window that
        # overhangs the raster silently returns the *clipped* array while
        # window_transform() still describes the unclipped window, which would
        # shift every pixel coordinate; and boundless=True with fill_value=nan
        # on an integer band (the counts are uint16) fills 0 without warning,
        # inventing returns where there is no data. Neither failure raises.
        full = Window(0, 0, src.width, src.height)
        if not intersect(window, full):
            return None
        window = intersection(window, full)
        if window.width <= 0 or window.height <= 0:
            return None

        n_bands = int(src.count)
        degraded = n_bands < EVIDENCE_BANDS
        indexes = [1] if degraded else list(range(1, EVIDENCE_BANDS + 1))
        data = src.read(indexes=indexes, window=window).astype(np.float64)
        nodata = src.nodata
        window_transform = src.window_transform(window)
        src_crs = src.crs

    if nodata is not None:
        # Height must become NaN so the percentile ignores it. The count/sum
        # bands are additive, so their nodata is "no returns here" -- zero.
        data[0] = np.where(data[0] == float(nodata), np.nan, data[0])
        if not degraded:
            data[1:] = np.where(data[1:] == float(nodata), 0.0, data[1:])

    lon, lat = _pixel_lonlat(window_transform, data.shape[1], data.shape[2], src_crs)
    labels = _cell_labels(lon, lat, origin, u_full, v_full, n_rows, n_cols)
    del lon, lat          # one float64 per raster pixel each; see Memory above
    if not (labels >= 0).any():
        return None

    height = groupwise_percentile(
        data[0].ravel(), labels, n_cells, height_q
    ).reshape(n_rows, n_cols)

    out: Dict[str, object] = {
        "height": height,
        "degraded": bool(degraded),
        "shape": (n_rows, n_cols),
    }
    if degraded:
        out.update(mrf=None, roughness=None, n_all=None, n_nonground=None)
        return out

    flat = data.reshape(data.shape[0], -1)
    evidence = pool_evidence(
        labels, n_cells,
        n_all=flat[1], n_multi=flat[2], n_nonground=flat[3],
        sum_z=flat[4], sum_z2=flat[5],
    )
    for key, values in evidence.items():
        out[key] = values.reshape(n_rows, n_cols)
    return out


def local_tree_median(
    canopy: np.ndarray,
    tree_mask: np.ndarray,
    radius: int = 3,
) -> np.ndarray:
    """Median of valid tree heights in the ``(2r+1)^2`` window around each cell.

    A height counts as valid where *tree_mask* is set and *canopy* is finite.
    Cells whose window contains no valid tree yield NaN.

    Fully vectorized: the padded grid is stacked into ``(window, H, W)`` shifted
    views and reduced with a single ``np.nanmedian``. The only Python loop runs
    over the window offsets (a few dozen), never over cells.

    Memory: the stack is ``(2r+1)^2 * H * W`` float64 -- ~192 MB at 700x700 with
    r=3, ~390 MB at 1000x1000 -- and ``np.nanmedian`` copies internally, so peak
    usage is roughly double that. The superseded per-cell loop in main.py used
    O(1) extra memory; this trades that for a ~3 orders-of-magnitude speedup.
    Grids beyond ~1000x1000 will want row-block chunking over the reduction.
    """
    can = np.asarray(canopy, dtype=np.float64)
    mask = np.asarray(tree_mask, dtype=bool)
    radius = int(radius)

    # NaN marks "not a valid tree height" so nanmedian ignores it.
    valid = np.where(mask & np.isfinite(can), can, np.nan)
    if radius <= 0:
        return valid

    height, width = valid.shape
    size = 2 * radius + 1
    padded = np.pad(valid, radius, mode="constant", constant_values=np.nan)

    stack = np.empty((size * size, height, width), dtype=np.float64)
    k = 0
    for d_row in range(size):
        for d_col in range(size):
            stack[k] = padded[d_row:d_row + height, d_col:d_col + width]
            k += 1

    with warnings.catch_warnings():
        # Windows with no tree at all are expected and return NaN by design.
        warnings.filterwarnings(
            "ignore",
            message="All-NaN slice encountered",
            category=RuntimeWarning,
        )
        return np.nanmedian(stack, axis=0)


# ---------------------------------------------------------------------------
# Evidence classifier
# ---------------------------------------------------------------------------
# Verdict codes written into the returned ``verdict`` grid. They partition every
# land-cover Tree cell: exactly one applies, so the five counts they index sum to
# the tree-cell total. Non-tree cells carry VERDICT_NONE.
VERDICT_NONE = 0
VERDICT_CANOPY = 1                  # evidence says canopy -> nDSM height kept
VERDICT_ROOF = 2                    # evidence says roof, footprint adjacent
VERDICT_AMBIGUOUS_KEEP = 3          # no verdict, nothing to suspect -> kept
VERDICT_AMBIGUOUS_REPLACE = 4       # no verdict, adjacent and height coincides
VERDICT_NO_DATA = 5                 # land cover says tree, nDSM has no height

VERDICT_NAMES = {
    VERDICT_NONE: "none",
    VERDICT_CANOPY: "canopy",
    VERDICT_ROOF: "roof",
    VERDICT_AMBIGUOUS_KEEP: "ambiguous_keep",
    VERDICT_AMBIGUOUS_REPLACE: "ambiguous_replace",
    VERDICT_NO_DATA: "no_data",
}

#: The count keys that partition the tree cells. ``sum(counts[k] for k in
#: PARTITION_KEYS) == counts["tree"]`` always holds. Every other key in the
#: counts dict (``guard_low``, ``guard_high``, ``replaced``, ``distrusted``) is a
#: diagnostic that *overlaps* the partition and must never be summed into it.
PARTITION_KEYS = (
    "canopy",
    "roof",
    "ambiguous_keep",
    "ambiguous_replace",
    "no_data",
)


@dataclass(frozen=True)
class RefineParams:
    """Thresholds for the evidence classifier.

    These are **seeds**, not calibrated values: they come from the distribution
    of one Chuo-ku tile (among tall cells, 5.3% have mrf <= 0.1, 66.6% >= 0.4,
    28.1% in between), and the roughness cuts are physical guesses rather than
    measurements. Calibration against labelled cells is a separate step. The
    classifier is built so that mis-set thresholds degrade towards *ambiguous*,
    which is the conservative bucket, rather than towards a confident mistake.

    * ``mrf_hi`` / ``rough_hi_m`` -- both must be met for the canopy verdict.
    * ``mrf_lo`` / ``rough_lo_m`` -- both must be met (plus an adjacent
      footprint) for the roof verdict.
    * ``min_points`` -- pooled return count below which ``mrf`` is not trusted.
    * ``min_nonground`` -- pooled non-ground count below which ``roughness`` is
      not trusted. ``pool_evidence`` already returns NaN below two points; this
      is the confidence band above that hard floor. It gates *both* verdicts,
      not just the roof one: a thinly-sampled cell reads spuriously planar as
      easily as spuriously rough, and a spurious "planar" is the false cap this
      module exists to remove.
    * ``coincidence_tol_m`` -- the old height-coincidence tolerance, now used
      only for ambiguous cells that are also adjacent to a footprint.
    * ``min_tree_height_m`` / ``max_tree_height_m`` -- sanity guards, applied
      after the verdict and no longer load-bearing.
    * ``median_radius`` -- window half-width for the replacement median.
    * ``adjacency_radius`` -- footprint adjacency half-width. The window
      *includes the cell itself*, so a tree cell sitting on a footprint counts
      as adjacent, which is the strongest leakage case there is.
    """

    mrf_hi: float = 0.35
    mrf_lo: float = 0.15
    rough_hi_m: float = 1.5
    rough_lo_m: float = 1.0
    min_points: int = 8
    min_nonground: int = 4
    coincidence_tol_m: float = 5.0
    min_tree_height_m: float = 2.0
    max_tree_height_m: float = 45.0
    median_radius: int = 3
    adjacency_radius: int = 1

    def __post_init__(self) -> None:
        if not self.mrf_lo <= self.mrf_hi:
            raise ValueError(
                f"mrf_lo ({self.mrf_lo}) must not exceed mrf_hi ({self.mrf_hi})"
            )
        if not self.rough_lo_m <= self.rough_hi_m:
            raise ValueError(
                f"rough_lo_m ({self.rough_lo_m}) must not exceed rough_hi_m "
                f"({self.rough_hi_m})"
            )
        if not self.min_tree_height_m <= self.max_tree_height_m:
            raise ValueError(
                f"min_tree_height_m ({self.min_tree_height_m}) must not exceed "
                f"max_tree_height_m ({self.max_tree_height_m})"
            )
        if self.min_points < 1:
            raise ValueError(f"min_points must be >= 1, got {self.min_points}")
        if self.min_nonground < 2:
            raise ValueError(
                "min_nonground must be >= 2; pool_evidence already returns NaN "
                f"roughness below two points, got {self.min_nonground}"
            )
        if self.median_radius < 0 or self.adjacency_radius < 0:
            raise ValueError("radii must be non-negative")
        if self.coincidence_tol_m < 0:
            raise ValueError("coincidence_tol_m must be non-negative")


DEFAULT_PARAMS = RefineParams()


def _as_2d(values, shape, name: str, dtype=float) -> np.ndarray:
    arr = np.asarray(values, dtype=dtype)
    if arr.shape != shape:
        raise ValueError(
            f"{name} has shape {arr.shape}, expected the height grid's {shape}"
        )
    return arr


def _max_in_window(values: np.ndarray, radius: int) -> np.ndarray:
    """Max over the ``(2r+1)^2`` window centred on each cell, itself included.

    Outside the grid counts as zero, i.e. no building -- which is what
    ``mode="constant", cval=0.0`` gives, and why the input must have had its
    NaNs replaced by 0.0 first.
    """
    radius = int(radius)
    arr = np.asarray(values, dtype=np.float64)
    if radius <= 0:
        return arr.copy()
    from scipy.ndimage import maximum_filter

    return maximum_filter(arr, size=2 * radius + 1, mode="constant", cval=0.0)


def classify_and_refine(
    height,
    tree_mask,
    building_heights,
    static_tree_height: float,
    *,
    mrf=None,
    roughness=None,
    n_all=None,
    n_nonground=None,
    degraded: Optional[bool] = None,
    params: RefineParams = DEFAULT_PARAMS,
) -> Dict[str, object]:
    """Refine nDSM tree-canopy heights from physical evidence.

    The nDSM genuinely contains roofs -- Tokyo LAS carries no vegetation or
    building classification, so nothing upstream separates a crown from a roof
    plane. The superseded heuristic inferred contamination from *height
    coincidence*: any tree cell within one cell of a building whose height
    matched that building's within a few metres was declared leakage and
    flattened to a constant. A genuine 25 m tree beside a 24 m building is
    indistinguishable from leakage under that test, and flattening it is the
    user-visible bug this function exists to remove.

    The decision is made on evidence instead. A LiDAR pulse entering a canopy
    echoes several times and the surface it samples is rough at metre scale; a
    roof returns one echo off a plane. Per tree cell with a finite height:

    ==================  ==============================================  ================
    verdict             condition                                       action
    ==================  ==============================================  ================
    canopy              ``mrf >= mrf_hi`` and ``roughness >= rough_hi``  keep the height
    roof                ``mrf <= mrf_lo`` and ``roughness <= rough_lo``  local median
                        and a footprint is adjacent
    ambiguous           anything else                                   replace only if
                                                                        adjacent *and*
                                                                        the height
                                                                        coincides
    ==================  ==============================================  ================

    **The canopy verdict overrides the coincidence test.** That is the point of
    the whole design: it is what lets a real tall tree beside a matching-height
    building survive, and it is checked directly by
    ``test_tall_tree_beside_matching_building_survives``.

    Evidence is *distrusted* -- the cell is forced to ambiguous -- when it is
    NaN (``pool_evidence`` NaNs roughness below two non-ground points, and mrf
    where no return landed), when ``n_all < min_points``, or when
    ``n_nonground < min_nonground``. NaN satisfies neither verdict by explicit
    finiteness test rather than by relying on comparison semantics: a NaN
    roughness read as 0.0 would be *maximally roof-like* and would recreate the
    false cap from a sparse cell.

    In degraded mode -- ``mrf``/``roughness``/``n_all`` all ``None``, which is
    the live path until the evidence COG is rebuilt -- every cell is ambiguous.
    That is still strictly better than the superseded heuristic on both failure
    modes: cells away from footprints are never touched, and the ones that are
    get a neighbourhood median rather than a flat constant.

    Replacement is always the local median of *credible* tree heights, never a
    flat constant. Credible means kept by the classifier and at least
    ``min_tree_height_m`` tall, so a run of contaminated cells along a roof edge
    cannot vouch for itself, and heights are clamped into the median so one
    implausible cell cannot drag a neighbourhood upward. ``static_tree_height``
    is used only when the window holds no credible neighbour at all.

    Args:
        height: (rows, cols) nDSM height per cell; NaN where the raster had no
            valid pixel. Typically ``load_ndsm_evidence(...)["height"]``.
        tree_mask: (rows, cols) bool, land cover == tree.
        building_heights: (rows, cols) float; ``> 0`` marks a footprint. NaN and
            negative values are treated as "no building".
        static_tree_height: fallback height, used only where a replacement is
            needed and the window holds no credible tree.
        mrf, roughness, n_all, n_nonground: per-cell evidence from
            :func:`pool_evidence`, or ``None`` for degraded mode.
        degraded: force degraded mode. ``None`` infers it from the evidence
            arrays; ``False`` with missing evidence is an error rather than a
            silent downgrade.
        params: :class:`RefineParams`.

    Returns:
        ``{"canopy", "verdict", "counts", "degraded"}``:

        * ``canopy``  -- (rows, cols) float, refined heights; 0.0 at non-tree
                         cells, finite everywhere.
        * ``verdict`` -- (rows, cols) int8 of ``VERDICT_*`` codes.
        * ``counts``  -- dict; the :data:`PARTITION_KEYS` entries sum to
                         ``counts["tree"]``, the rest are overlapping
                         diagnostics.
        * ``degraded``-- bool, whether evidence was available.
    """
    p = params
    h = np.asarray(height, dtype=np.float64)
    if h.ndim != 2:
        raise ValueError(f"height must be a 2-D grid, got {h.ndim} dimensions")
    shape = h.shape

    is_tree = _as_2d(tree_mask, shape, "tree_mask", dtype=bool)
    finite_h = np.isfinite(h)
    classified = is_tree & finite_h
    no_data = is_tree & ~finite_h

    # A NaN or negative building height is not a footprint. Substituting 0.0
    # before the max filter keeps NaN out of the comparison chain entirely.
    bld = _as_2d(building_heights, shape, "building_heights")
    bld = np.where(np.isfinite(bld), bld, 0.0)
    near_bld_h = _max_in_window(bld, p.adjacency_radius)
    near_bld = near_bld_h > 0.0

    if degraded is None:
        degraded = mrf is None or roughness is None or n_all is None
    else:
        degraded = bool(degraded)
        if not degraded and (mrf is None or roughness is None or n_all is None):
            raise ValueError(
                "degraded=False requires mrf, roughness and n_all; pass "
                "degraded=None to infer, or degraded=True to ignore evidence"
            )

    n_distrusted = 0
    if degraded:
        canopy_ev = np.zeros(shape, dtype=bool)
        roof_ev = np.zeros(shape, dtype=bool)
    else:
        m = _as_2d(mrf, shape, "mrf")
        r = _as_2d(roughness, shape, "roughness")
        na = _as_2d(n_all, shape, "n_all")
        if n_nonground is None:
            # Roughness confidence then rests only on pool_evidence's NaN
            # contract (>= 2 non-ground points). Callers that have the count
            # should pass it; the reader always returns it.
            ng_ok = np.ones(shape, dtype=bool)
        else:
            ng = _as_2d(n_nonground, shape, "n_nonground")
            ng_ok = np.isfinite(ng) & (ng >= p.min_nonground)

        # NaN satisfies NEITHER verdict, stated as an explicit finiteness test
        # rather than left to the comparison operators.
        trusted = (
            np.isfinite(m)
            & np.isfinite(r)
            & np.isfinite(na)
            & (na >= p.min_points)
            & ng_ok
        )
        canopy_ev = trusted & (m >= p.mrf_hi) & (r >= p.rough_hi_m)
        roof_ev = trusted & (m <= p.mrf_lo) & (r <= p.rough_lo_m)
        n_distrusted = int((classified & ~trusted).sum())

    # THE line: canopy evidence wins over everything downstream, including the
    # coincidence test that used to cap real trees beside matching buildings.
    v_canopy = classified & canopy_ev
    v_roof = classified & ~canopy_ev & roof_ev & near_bld
    ambiguous = classified & ~v_canopy & ~v_roof
    # Tightened coincidence: adjacency is required, not merely implied.
    coincident = near_bld & (np.abs(h - near_bld_h) <= p.coincidence_tol_m)
    v_amb_replace = ambiguous & coincident
    v_amb_keep = ambiguous & ~coincident

    keep = v_canopy | v_amb_keep
    replace = v_roof | v_amb_replace

    # Median source: kept cells only, and only at plausible heights, clamped so
    # one 60 m cell cannot lift its whole neighbourhood. Because credibility
    # requires h >= min_tree_height_m, a finite median is always >= that floor.
    credible = keep & (h >= p.min_tree_height_m)
    source = np.where(credible, np.minimum(h, p.max_tree_height_m), np.nan)
    median = local_tree_median(source, credible, radius=p.median_radius)
    fill = np.where(np.isfinite(median), median, float(static_tree_height))

    # Starts at zero, which is the value non-tree cells keep: every write below
    # is gated on a subset of is_tree, and there is deliberately no trailing
    # ``canopy[~is_tree] = 0.0``. Such a line would mask an ungated guard --
    # every non-tree cell sits at 0.0, i.e. below min_tree_height_m -- and
    # test_non_tree_cells_are_zero is what detects that.
    canopy = np.zeros(shape, dtype=np.float64)
    canopy[keep] = h[keep]
    needs_fill = replace | no_data
    canopy[needs_fill] = fill[needs_fill]

    # Guards, applied to the composed result so a replacement value is sanity
    # checked too. They are orthogonal to the verdict: a guard can fire on a
    # cell in any bucket, which is why their counts sit outside the partition.
    too_low = is_tree & (canopy < p.min_tree_height_m)
    canopy = np.where(too_low, fill, canopy)
    too_high = is_tree & (canopy > p.max_tree_height_m)
    canopy = np.where(too_high, p.max_tree_height_m, canopy)

    verdict = np.zeros(shape, dtype=np.int8)
    verdict[v_canopy] = VERDICT_CANOPY
    verdict[v_roof] = VERDICT_ROOF
    verdict[v_amb_keep] = VERDICT_AMBIGUOUS_KEEP
    verdict[v_amb_replace] = VERDICT_AMBIGUOUS_REPLACE
    verdict[no_data] = VERDICT_NO_DATA

    counts = {
        "tree": int(is_tree.sum()),
        # --- partition (see PARTITION_KEYS) ---
        "canopy": int(v_canopy.sum()),
        "roof": int(v_roof.sum()),
        "ambiguous_keep": int(v_amb_keep.sum()),
        "ambiguous_replace": int(v_amb_replace.sum()),
        "no_data": int(no_data.sum()),
        # --- overlapping diagnostics, never summed into the partition ---
        "replaced": int(replace.sum()),
        "distrusted": n_distrusted,
        "guard_low": int(too_low.sum()),
        "guard_high": int(too_high.sum()),
    }
    return {
        "canopy": canopy,
        "verdict": verdict,
        "counts": counts,
        "degraded": bool(degraded),
    }


def refine_from_evidence(
    evidence: Mapping[str, object],
    tree_mask,
    building_heights,
    static_tree_height: float,
    params: RefineParams = DEFAULT_PARAMS,
) -> Dict[str, object]:
    """Run :func:`classify_and_refine` on a :func:`load_ndsm_evidence` dict.

    The single seam between the reader and the classifier, so callers never
    unpack the evidence keys themselves and cannot pair a height grid with
    another grid's evidence. The reader's ``degraded`` flag is forwarded
    verbatim rather than re-inferred.

    A ``None`` from :func:`load_ndsm_evidence` -- missing COG, or no overlap --
    is rejected here rather than fed through: there is no nDSM to refine, and
    the caller's response is to keep static tree heights, not to run the
    classifier on nothing.
    """
    if evidence is None:
        raise ValueError(
            "load_ndsm_evidence returned None (missing COG or no overlap with "
            "the target); fall back to static tree heights instead of refining"
        )
    return classify_and_refine(
        evidence["height"],
        tree_mask,
        building_heights,
        static_tree_height,
        mrf=evidence.get("mrf"),
        roughness=evidence.get("roughness"),
        n_all=evidence.get("n_all"),
        n_nonground=evidence.get("n_nonground"),
        degraded=evidence.get("degraded"),
        params=params,
    )


def format_counts(counts: Mapping[str, int]) -> str:
    """One-line summary of a classification, for the pipeline log."""
    return (
        "{tree} tree cells: {canopy} canopy (kept), {roof} roof -> median, "
        "{ambiguous_keep} ambiguous keep, {ambiguous_replace} ambiguous "
        "replace, {no_data} no data; guards: {guard_low} below min, "
        "{guard_high} above max; {distrusted} distrusted evidence"
    ).format(**{key: counts.get(key, 0) for key in (
        "tree", *PARTITION_KEYS, "guard_low", "guard_high", "distrusted",
    )})
