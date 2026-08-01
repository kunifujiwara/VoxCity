"""Evidence-based nDSM canopy refinement.

Replaces the coincidence heuristics that lived in main.py: the nDSM raster
genuinely contains roofs (Tokyo LAS has no vegetation/building classes), so
tree/roof separation here uses physical evidence -- multi-return fraction and
surface roughness pooled from the COG's count/sum bands -- with building
footprints and height coincidence only as fallback for ambiguous cells.

Frames: everything returned to callers is in the model's display frame
(south-up). The COG is north-up raster space; the conversion happens exactly
once, in load_ndsm_evidence(), by assigning raster pixels to display-frame
cells through the grid geometry (coordinate-based, orientation-free).
"""
from __future__ import annotations

import os
import warnings
from typing import Dict, Optional

import numpy as np

__all__ = [
    "groupwise_percentile",
    "pool_evidence",
    "local_tree_median",
    "load_ndsm_evidence",
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
    """Display-frame cell label ``row * n_cols + col`` for each (lon, lat).

    The model grid is the parallelogram ``origin + a*u_full + b*v_full`` with
    ``a, b`` in [0, 1); *u_full* spans SW->NW (the row axis, growing north) and
    *v_full* spans SW->SE (the column axis, growing east). Inverting that 2x2
    system by Cramer's rule gives each point's fractional position along the two
    grid axes, which floors directly to a row and a column.

    This is the *only* place the raster's north-up layout meets the model's
    south-up display frame, and it never touches an array index: a pixel's cell
    is decided by where the pixel *is*, so rotation, mirroring and pixel order
    are all handled by construction. Any point outside the grid gets label -1,
    which pool_evidence and groupwise_percentile drop.
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

    Everything returned is in the model's **display frame** (row 0 = south).
    ``src.read(window=...)`` yields north-up raster layout; the conversion
    happens in :func:`_cell_labels`, by coordinate, never by index arithmetic.

    Heights are returned exactly as the raster stores them, negatives included:
    clamping is the canopy builder's job (``_build_canopy_from_ndsm``), and a
    reader that silently clamped would hide a botched ground surface.

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
    # builds its (nx, ny) arrays with index 0 along side_1. side_1 is SW->NW, so
    # index 0 is the row axis growing north: the display frame.
    n_rows, n_cols = (int(n) for n in geom["grid_size"])
    n_cells = n_rows * n_cols
    origin = np.asarray(geom["origin"], dtype=np.float64)
    d_row, d_col = geom["adj_mesh"]
    # Full-extent side vectors rebuilt from the *cell* step, so the parallelogram
    # inverted below is exactly the one compute_cell_center_coords lays cells on.
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
        window = from_bounds(
            min(corner_x), min(corner_y), max(corner_x), max(corner_y),
            transform=src.transform,
        ).round_offsets().round_lengths()

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

    rows, cols = np.meshgrid(
        np.arange(data.shape[1]), np.arange(data.shape[2]), indexing="ij"
    )
    pixel_x, pixel_y = rasterio.transform.xy(
        window_transform, rows.ravel(), cols.ravel()
    )
    from_src = Transformer.from_crs(src_crs, "EPSG:4326", always_xy=True)
    lon, lat = from_src.transform(np.asarray(pixel_x), np.asarray(pixel_y))

    labels = _cell_labels(lon, lat, origin, u_full, v_full, n_rows, n_cols)
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
