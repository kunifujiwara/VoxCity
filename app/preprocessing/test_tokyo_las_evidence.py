# -*- coding: utf-8 -*-
"""Evidence bands through the Tokyo LAS rasterize/merge/nDSM path.

The nDSM COG grew from one band (height) to six: the five extra bands are the
additive per-pixel LiDAR statistics that ``app/backend/ndsm_refine.py``'s
``load_ndsm_evidence`` pools into per-cell multi-return fraction and roughness.
These tests pin the schema, the physics the schema is supposed to carry, and the
two preconditions the runtime reader depends on -- band 1's meaning and nodata
co-occurrence across all six bands.

Fixture discipline: the synthetic tile is deliberately **asymmetric**. The roof
sits in the south half and the canopy in the north half, the raster is 11 x 9
(never square), the ground plane slopes in x only, and the two halves differ in
*every* band, so a flipped, transposed or half-swapped implementation cannot
pass. Where a value is hand-computable it is asserted exactly rather than
approximately.
"""
from __future__ import annotations

import os
import sys

os.environ.setdefault("MPLBACKEND", "Agg")   # before tokyo_las imports pyplot

import numpy as np
import pytest

laspy = pytest.importorskip("laspy")

import rasterio                                              # noqa: E402
from rasterio.transform import from_origin                   # noqa: E402
from rasterio.windows import Window                          # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tokyo_las                                             # noqa: E402


# ---------------------------------------------------------------------------
# Synthetic tile
# ---------------------------------------------------------------------------
# A plausible EPSG:6677 (JGD2011 / Japan Plane Rectangular CRS IX) location in
# central Tokyo. Nothing depends on the exact value; it exists so the fixture is
# a believable Tokyo sheet rather than a raster at the origin.
X0 = -8000.0
Y0 = -35500.0
RES = 0.5

# Point lattice. i runs west -> east, j runs south -> north; the LAS carries no
# grid of its own, so these only decide where each point lands.
N_I = 10
N_J = 12

# j bands, chosen so that after ``process_las_to_raster`` derives its grid from
# the point extents (row = floor((y_max - y) / res), clipped) each band occupies
# known raster rows -- see ROW_OF_J below.
J_ROOF = (0, 1, 2, 3)         # south half: opaque planar roof, single returns
J_SUNKEN = 4                  # surface *below* its own ground -> negative nDSM
J_GAP = 5                     # no points at all -> nodata in every band
J_GROUND_ONLY = 6             # class 2 only -> no DSM -> band 1 nodata
J_NO_GROUND = 7               # class 1 only -> no DTM -> band 1 nodata
J_CANOPY = (8, 9, 10, 11)     # north half: penetrating pulses, spread z

ROOF_CLASS = 1
GROUND_CLASS = 2
NODATA = -9999.0

#: Depth of the sunken band below its own ground, in metres.
#:
#: The fixture would otherwise be entirely non-negative, and a writer that
#: clamped band 1 at zero -- which ndsm_refine explicitly forbids, because a
#: clamp hides a botched ground surface instead of surfacing it -- would produce
#: byte-identical output and no test could see it. Measured: without this band,
#: mutating the six-band writer to ``max(nd, 0.0)`` survives the whole suite.
SUNKEN_DEPTH_M = 1.5


def ground_z(i: int) -> float:
    """Terrain: a plane sloping in x only.

    Deliberately *not* flat and deliberately independent of j. Not flat, so an
    implementation that filled a DTM gap with a global statistic (mean 0.725 at
    this slope) or with zero gives a different answer than the nearest-neighbour
    fill the spec requires. Independent of j, so the nearest-fill at the
    no-ground row has no tie to break: its two equidistant candidates (the rows
    above and below) carry the same value, and the assertion does not depend on
    ``distance_transform_edt``'s tie-breaking.
    """
    return 0.5 + 0.05 * i


def roof_z(i: int) -> float:
    """Roof: one plane, tilted in x. Planar, but not degenerate -- a perfectly
    constant roof would make the roughness exactly 0.0 and "small" would then be
    unfalsifiable."""
    return 10.0 + 0.02 * i


def canopy_z(i: int, j: int):
    """Two echoes per canopy pixel, several metres apart and varying with both
    i and j so no two pixels agree."""
    k = j - J_CANOPY[0]
    return (12.0 + 0.1 * i + 0.3 * k, 18.0 + 0.2 * i - 0.4 * k)


def _x(i: int) -> float:
    return X0 + i * RES


def _y(j: int) -> float:
    return Y0 + j * RES


# Raster geometry implied by the point extents, i.e. what process_las_to_raster
# builds. Restated here rather than read back from the output so the tests know
# independently which row is which.
X_MIN = _x(0)
Y_MAX = _y(N_J - 1)
WIDTH = max(1, int(np.ceil((_x(N_I - 1) - X_MIN) / RES)))       # 9
HEIGHT = max(1, int(np.ceil((Y_MAX - _y(0)) / RES)))            # 11


def _row_of_j(j: int) -> int:
    return int(min(HEIGHT - 1, np.floor((Y_MAX - _y(j)) / RES)))


def _col_of_i(i: int) -> int:
    return int(min(WIDTH - 1, np.floor((_x(i) - X_MIN) / RES)))


ROW_CANOPY = tuple(_row_of_j(j) for j in J_CANOPY)              # 3, 2, 1, 0
ROW_NO_GROUND = _row_of_j(J_NO_GROUND)                          # 4
ROW_GROUND_ONLY = _row_of_j(J_GROUND_ONLY)                      # 5
ROW_GAP = _row_of_j(J_GAP)                                      # 6
ROW_SUNKEN = _row_of_j(J_SUNKEN)                                # 7
ROW_ROOF = tuple(sorted({_row_of_j(j) for j in J_ROOF}))        # 8, 9, 10


def _make_points():
    """(x, y, z, classification, return_number, number_of_returns) arrays.

    The multi-return pattern is the physics under test. A pulse that reaches the
    ground *through* a crown echoes several times, so the ground return under
    canopy is itself a multi-return point; a pulse landing beside a roof echoes
    once. That is why the pooled multi-return fraction is 1.0 over the canopy and
    0.0 over the roof, ground returns included in both denominators.
    """
    xs, ys, zs, cls, rnum, nret = [], [], [], [], [], []

    def add(i, j, z, c, r, n):
        xs.append(_x(i)); ys.append(_y(j)); zs.append(z)
        cls.append(c); rnum.append(r); nret.append(n)

    for j in J_ROOF:
        for i in range(N_I):
            add(i, j, ground_z(i), GROUND_CLASS, 1, 1)     # single-return ground
            add(i, j, roof_z(i), ROOF_CLASS, 1, 1)         # single-return roof
    for j in J_CANOPY:
        for i in range(N_I):
            z1, z2 = canopy_z(i, j)
            add(i, j, z2, ROOF_CLASS, 1, 3)                # first echo, top
            add(i, j, z1, ROOF_CLASS, 2, 3)                # second echo, inside
            add(i, j, ground_z(i), GROUND_CLASS, 3, 3)     # last echo, ground
    for i in range(N_I):
        add(i, J_SUNKEN, ground_z(i), GROUND_CLASS, 1, 1)
        add(i, J_SUNKEN, ground_z(i) - SUNKEN_DEPTH_M, ROOF_CLASS, 1, 1)
    for i in range(N_I):
        add(i, J_GROUND_ONLY, ground_z(i), GROUND_CLASS, 1, 1)
    for i in range(N_I):
        add(i, J_NO_GROUND, roof_z(i), ROOF_CLASS, 1, 1)

    return (
        np.array(xs, dtype=np.float64),
        np.array(ys, dtype=np.float64),
        np.array(zs, dtype=np.float64),
        np.array(cls, dtype=np.uint8),
        np.array(rnum, dtype=np.uint8),
        np.array(nret, dtype=np.uint8),
    )


@pytest.fixture(scope="module")
def las_path(tmp_path_factory):
    x, y, z, cls, rnum, nret = _make_points()
    header = laspy.LasHeader(point_format=3, version="1.2")
    header.offsets = np.array([X0, Y0, 0.0])
    header.scales = np.array([0.001, 0.001, 0.001])
    try:
        from pyproj import CRS
        header.add_crs(CRS.from_epsg(6677))
    except Exception:                                   # pragma: no cover
        pass
    las = laspy.LasData(header)
    las.x = x
    las.y = y
    las.z = z
    las.classification = cls
    las.return_number = rnum
    las.number_of_returns = nret
    out = tmp_path_factory.mktemp("las") / "synthetic_6677.las"
    las.write(str(out))
    return str(out)


def _write_las(path, rows):
    """Minimal LAS writer for the auxiliary fixtures.

    *rows* are ``(x, y, z, classification, return_number, number_of_returns)``.
    """
    header = laspy.LasHeader(point_format=3, version="1.2")
    header.offsets = np.array([X0, Y0, 0.0])
    header.scales = np.array([0.001, 0.001, 0.001])
    las = laspy.LasData(header)
    cols = list(zip(*rows))
    las.x = np.array(cols[0], dtype=np.float64)
    las.y = np.array(cols[1], dtype=np.float64)
    las.z = np.array(cols[2], dtype=np.float64)
    las.classification = np.array(cols[3], dtype=np.uint8)
    las.return_number = np.array(cols[4], dtype=np.uint8)
    las.number_of_returns = np.array(cols[5], dtype=np.uint8)
    las.write(str(path))
    return str(path)


@pytest.fixture(scope="module")
def rasters(las_path):
    """(dsm, dtm, evidence) from the evidence-enabled rasterizer."""
    return tokyo_las.process_las_to_raster(las_path, resolution=RES, with_evidence=True)


def _band(evidence, name):
    idx = tokyo_las.EVIDENCE_BAND_NAMES.index(name)
    return evidence["array"][idx]


# ---------------------------------------------------------------------------
# The fixture itself must be able to fail
# ---------------------------------------------------------------------------
def test_fixture_geometry_is_asymmetric_and_non_square(rasters):
    """Guard the guards. Every later test reads specific rows; if the derived
    grid is not what this module assumes, those tests would silently move to the
    wrong place."""
    dsm, dtm, ev = rasters
    assert dsm["array"].shape == (HEIGHT, WIDTH) == (11, 9)
    assert HEIGHT != WIDTH, "a square tile would hide a row/col transpose"
    assert set(ROW_ROOF).isdisjoint(ROW_CANOPY)
    # Roof strictly south (higher row index, north-up raster) of canopy.
    assert min(ROW_ROOF) > max(ROW_CANOPY)
    special = {ROW_GAP, ROW_SUNKEN, ROW_NO_GROUND, ROW_GROUND_ONLY}
    assert len(special) == 4
    assert special.isdisjoint(ROW_ROOF) and special.isdisjoint(ROW_CANOPY)


# ---------------------------------------------------------------------------
# 4. DSM/DTM regression guard
# ---------------------------------------------------------------------------
def _legacy_process_las_to_raster(las_path, resolution=0.5,
                                  dsm_classes=(1, 3), dtm_classes=(2,)):
    """Verbatim copy of ``process_las_to_raster`` as it stood at d42292e.

    Frozen on purpose: it is the oracle for "the evidence work changed nothing
    about the DSM/DTM the 4.8 GB production COG was built from". It must never
    be refactored to track the live function -- that would make the comparison
    vacuous.
    """
    las = laspy.read(las_path)
    x = np.asarray(las.x, dtype=np.float64)
    y = np.asarray(las.y, dtype=np.float64)
    z = np.asarray(las.z, dtype=np.float32)
    cls = np.asarray(las.classification)

    x_min, x_max = float(np.min(x)), float(np.max(x))
    y_min, y_max = float(np.min(y)), float(np.max(y))

    width = max(1, int(np.ceil((x_max - x_min) / resolution)))
    height = max(1, int(np.ceil((y_max - y_min) / resolution)))
    transform = from_origin(x_min, y_max, resolution, resolution)

    cols = np.clip(((x - x_min) / resolution).astype(np.int64), 0, width - 1)
    rows = np.clip(((y_max - y) / resolution).astype(np.int64), 0, height - 1)
    idx_flat = rows * width + cols

    dsm_arr = np.full((height, width), -np.inf, dtype=np.float32)
    dsm_mask = np.isin(cls, dsm_classes)
    if dsm_mask.any():
        np.maximum.at(dsm_arr.ravel(), idx_flat[dsm_mask], z[dsm_mask])
    dsm_arr[np.isneginf(dsm_arr)] = np.nan

    dtm_arr = np.full((height, width), np.inf, dtype=np.float32)
    dtm_mask = np.isin(cls, dtm_classes)
    if dtm_mask.any():
        np.minimum.at(dtm_arr.ravel(), idx_flat[dtm_mask], z[dtm_mask])
    dtm_arr[np.isposinf(dtm_arr)] = np.nan

    base = {'transform': transform,
            'bounds': (x_min, y_min, x_max, y_max),
            'nodata': -9999.0}
    return dict(array=dsm_arr.copy(), **base), dict(array=dtm_arr.copy(), **base)


def _assert_bit_identical(new, old, what):
    a, b = new["array"], old["array"]
    assert a.dtype == b.dtype == np.float32, what
    assert a.shape == b.shape, what
    # tobytes(), not allclose(): NaN patterns and the exact float32 bit pattern
    # both have to survive, and == would call every NaN a mismatch.
    assert a.tobytes() == b.tobytes(), f"{what} is not bit-identical"
    assert new["transform"] == old["transform"], what
    assert new["bounds"] == old["bounds"], what
    assert new["nodata"] == old["nodata"], what


def test_dsm_dtm_bit_identical_to_legacy_path(las_path, rasters):
    dsm, dtm, _ = rasters
    old_dsm, old_dtm = _legacy_process_las_to_raster(las_path, resolution=RES)
    # The oracle must actually contain data and holes, or "identical" is cheap.
    assert np.isfinite(old_dsm["array"]).any() and np.isnan(old_dsm["array"]).any()
    assert np.isfinite(old_dtm["array"]).any() and np.isnan(old_dtm["array"]).any()
    _assert_bit_identical(dsm, old_dsm, "DSM")
    _assert_bit_identical(dtm, old_dtm, "DTM")


def test_default_call_still_returns_two_rasters(las_path):
    """Old callers unpack two values. The evidence is opt-in."""
    result = tokyo_las.process_las_to_raster(las_path, resolution=RES)
    assert len(result) == 2
    dsm, dtm = result
    old_dsm, old_dtm = _legacy_process_las_to_raster(las_path, resolution=RES)
    _assert_bit_identical(dsm, old_dsm, "DSM (default call)")
    _assert_bit_identical(dtm, old_dtm, "DTM (default call)")


# ---------------------------------------------------------------------------
# Band schema
# ---------------------------------------------------------------------------
def test_evidence_band_order_matches_the_reader_contract(rasters):
    assert tokyo_las.EVIDENCE_BAND_NAMES == (
        "n_all", "n_multi", "n_nonground", "sum_z", "sum_z2")
    assert tokyo_las.NDSM_BAND_COUNT == 6
    _, _, ev = rasters
    assert ev["array"].shape == (5, HEIGHT, WIDTH)


# ---------------------------------------------------------------------------
# Exact per-pixel values
# ---------------------------------------------------------------------------
def test_roof_pixel_counts_and_sums_are_exact(rasters):
    """One ground echo + one roof echo, both single-return."""
    _, _, ev = rasters
    r, c = _row_of_j(3), _col_of_i(0)
    assert (r, c) == (8, 0)
    assert _band(ev, "n_all")[r, c] == 2
    assert _band(ev, "n_multi")[r, c] == 0
    assert _band(ev, "n_nonground")[r, c] == 1
    h = roof_z(0) - ground_z(0)                    # 9.5 m above local ground
    assert _band(ev, "sum_z")[r, c] == pytest.approx(h, abs=1e-4)
    assert _band(ev, "sum_z2")[r, c] == pytest.approx(h * h, abs=1e-3)


def test_canopy_pixel_counts_and_sums_are_exact(rasters):
    """Two crown echoes + one ground echo, all part of 3-return pulses.

    ``n_nonground == 2``, not 3, is the load-bearing number: it is what proves
    the class-2 ground echo is excluded from the height sums. The sums alone
    could not prove it -- the ground point's height above ground is 0.0, so
    including it would leave sum_z and sum_z2 untouched.
    """
    _, _, ev = rasters
    r, c = _row_of_j(8), _col_of_i(0)
    assert (r, c) == (3, 0)
    z1, z2 = canopy_z(0, 8)
    h1, h2 = z1 - ground_z(0), z2 - ground_z(0)
    assert _band(ev, "n_all")[r, c] == 3
    assert _band(ev, "n_multi")[r, c] == 3
    assert _band(ev, "n_nonground")[r, c] == 2
    assert _band(ev, "sum_z")[r, c] == pytest.approx(h1 + h2, abs=1e-4)
    assert _band(ev, "sum_z2")[r, c] == pytest.approx(h1 * h1 + h2 * h2, abs=1e-3)


def test_empty_pixels_are_nodata_in_every_evidence_band(rasters):
    _, _, ev = rasters
    arr = ev["array"]
    assert (arr[:, ROW_GAP, :] == NODATA).all()
    # ... and nowhere a point landed.
    assert (arr[0, ROW_ROOF[0], :] != NODATA).all()


def test_evidence_heights_use_the_nearest_filled_ground(rasters):
    """The no-ground row has no DTM of its own.

    Its height must come from the *nearest* ground pixel. At this fixture's
    slope the nearest value is 0.5 at column 0, whereas the tile mean is 0.725,
    the max 0.95 and an unfilled DTM would give NaN -- so the single exact value
    below rules out all four alternatives at once.
    """
    _, dtm, ev = rasters
    r, c = ROW_NO_GROUND, _col_of_i(0)
    assert np.isnan(dtm["array"][r, c]), "fixture must have a DTM gap here"
    assert _band(ev, "n_nonground")[r, c] == 1
    assert _band(ev, "sum_z")[r, c] == pytest.approx(roof_z(0) - ground_z(0), abs=1e-4)


def test_nonground_is_the_complement_of_ground_not_the_dsm_set(tmp_path):
    """Pins a deliberate divergence between two class sets.

    The DSM is built from ``dsm_classes=(1, 3)``; the evidence's "non-ground" is
    ``~isin(cls, dtm_classes)``. On Tokyo LAS the two coincide -- only classes
    1, 2 and 3 occur -- so nothing in the main fixture can tell them apart, and
    a future edit could quietly swap one for the other. Here a class-5 point
    (high vegetation, which the DSM ignores) makes the difference observable:
    it must count as non-ground and contribute its height, while the DSM must
    still ignore it.
    """
    # Two clusters 2 m apart so they cannot share a 0.5 m pixel.
    path = _write_las(tmp_path / "cls5.las", [
        (_x(0), _y(0), 0.0, GROUND_CLASS, 1, 1),
        (_x(0), _y(0), 4.0, 5, 1, 1),            # high vegetation: not in (1, 3)
        (_x(4), _y(4), 0.0, GROUND_CLASS, 1, 1),
        (_x(4), _y(4), 2.0, ROOF_CLASS, 1, 1),
    ])
    dsm, dtm, ev = tokyo_las.process_las_to_raster(path, resolution=RES,
                                                  with_evidence=True)

    def at(x, y):
        col, row = ~dsm["transform"] * (x, y)
        h, w = dsm["array"].shape
        return min(int(row), h - 1), min(int(col), w - 1)

    veg_px = at(_x(0), _y(0))
    bld_px = at(_x(4), _y(4))
    assert veg_px != bld_px, "fixture must separate the two clusters"

    # The class-5 point counts as non-ground and contributes its 4 m height ...
    assert _band(ev, "n_nonground")[veg_px] == 1
    assert _band(ev, "n_all")[veg_px] == 2
    assert _band(ev, "sum_z")[veg_px] == pytest.approx(4.0, abs=1e-4)
    # ... but it never reaches the DSM, which only takes classes (1, 3).
    assert np.isnan(dsm["array"][veg_px])
    assert dsm["array"][bld_px] == pytest.approx(2.0, abs=1e-4)


def test_evidence_failure_never_costs_the_dsm_or_dtm(las_path, monkeypatch):
    """Adding bands must not be able to lose a sheet's height data.

    A single shared exception handler would return ``(None, None, None)`` on any
    evidence failure and drop the tile from the DSM and DTM merges too --
    coverage the pre-evidence pipeline produced. The failure has to surface as a
    short evidence list, which the drivers treat as fatal, not as a missing
    sheet nobody counted.
    """
    def boom(*args, **kwargs):
        raise RuntimeError("synthetic evidence failure")

    monkeypatch.setattr(tokyo_las, "_evidence_arrays", boom)
    dsm, dtm, ev = tokyo_las.process_las_to_raster(las_path, resolution=RES,
                                                   with_evidence=True)
    assert ev is None
    old_dsm, old_dtm = _legacy_process_las_to_raster(las_path, resolution=RES)
    _assert_bit_identical(dsm, old_dsm, "DSM after an evidence failure")
    _assert_bit_identical(dtm, old_dtm, "DTM after an evidence failure")


def test_process_las_files_reports_a_short_evidence_list(las_path, tmp_path,
                                                         monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError("synthetic evidence failure")

    monkeypatch.setattr(tokyo_las, "_evidence_arrays", boom)
    with pytest.warns(UserWarning, match="no evidence raster"):
        dsm_files, dtm_files, ev_files = tokyo_las.process_las_files(
            [las_path], str(tmp_path / "d"), str(tmp_path / "g"), RES,
            "EPSG:6677", evidence_output_dir=str(tmp_path / "e"))
    assert len(dsm_files) == 1 and len(ev_files) == 0


def test_incomplete_evidence_is_fatal_for_a_rebuild():
    """The guard both drivers share. 'Some evidence merged' is not enough."""
    tokyo_las.require_complete_evidence(["a", "b"], ["a", "b"])      # no raise
    with pytest.raises(RuntimeError, match="nodata holes in band 1"):
        tokyo_las.require_complete_evidence(["a", "b"], ["a"])
    with pytest.raises(RuntimeError):
        tokyo_las.require_complete_evidence(["a"], [])


# ---------------------------------------------------------------------------
# 1. The physics: roof vs canopy separate
# ---------------------------------------------------------------------------
def _pool(ev, rows):
    """Pool the evidence bands over whole raster rows, exactly as the runtime
    pools them over a model cell (see ndsm_refine.pool_evidence)."""
    arr = ev["array"][:, list(rows), :].reshape(5, -1)
    valid = arr[0] != NODATA
    arr = arr[:, valid]
    n_all, n_multi, n_ng, s_z, s_z2 = (a.astype(np.float64).sum() for a in arr)
    mrf = n_multi / n_all
    var = max(s_z2 / n_ng - (s_z / n_ng) ** 2, 0.0)
    return mrf, float(np.sqrt(var)), n_all, n_ng


def test_multi_return_fraction_separates_roof_from_canopy(rasters):
    _, _, ev = rasters
    roof_mrf, _, roof_n, _ = _pool(ev, ROW_ROOF)
    can_mrf, _, can_n, _ = _pool(ev, ROW_CANOPY)
    assert roof_n >= 8 and can_n >= 8, "too few returns to mean anything"
    assert roof_mrf == pytest.approx(0.0, abs=1e-9)
    assert can_mrf == pytest.approx(1.0, abs=1e-9)


def test_roughness_separates_roof_from_canopy(rasters):
    _, _, ev = rasters
    _, roof_rough, _, roof_ng = _pool(ev, ROW_ROOF)
    _, can_rough, _, can_ng = _pool(ev, ROW_CANOPY)
    assert roof_ng >= 2 and can_ng >= 2
    # A tilted plane over 4.5 m of tile: tens of millimetres.
    assert roof_rough < 0.2
    # Two echoes 6 m apart: metres.
    assert can_rough > 2.0
    assert can_rough > 10 * roof_rough


def test_evidence_is_not_mirrored_north_south(rasters):
    """The one assertion a north/south flip cannot survive.

    Everything above is stated per named row, so a flipped writer would move the
    canopy onto the roof rows and the roof onto the canopy rows and every test
    would still pass while describing the wrong ground. This one names the
    physical direction: the canopy is north, i.e. at the *low* row indices of a
    north-up raster.
    """
    _, _, ev = rasters
    north_mrf, _, _, _ = _pool(ev, ROW_CANOPY)
    south_mrf, _, _, _ = _pool(ev, ROW_ROOF)
    assert north_mrf > south_mrf + 0.5
    assert max(ROW_CANOPY) < min(ROW_ROOF)


# ---------------------------------------------------------------------------
# The merge/nDSM path
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def built(las_path, tmp_path_factory):
    """Run the tile -> save -> merge -> nDSM path, with and without evidence."""
    out = tmp_path_factory.mktemp("build")
    dsm_dir, dtm_dir, ev_dir = out / "dsm", out / "dtm", out / "ev"
    dsm_files, dtm_files, ev_files = tokyo_las.process_las_files(
        [las_path], str(dsm_dir), str(dtm_dir), RES, "EPSG:6677",
        evidence_output_dir=str(ev_dir),
    )
    assert len(dsm_files) == len(dtm_files) == len(ev_files) == 1

    merged = {}
    for name, files in (("dsm", dsm_files), ("dtm", dtm_files), ("ev", ev_files)):
        merged[name] = tokyo_las.merge_geotiffs(
            files, str(out / f"merged_{name}.tif"), "EPSG:6677")

    six = str(out / "ndsm6.tif")
    one = str(out / "ndsm1.tif")
    tokyo_las.build_ndsm(merged["dsm"], merged["dtm"], six,
                         evidence_path=merged["ev"])
    tokyo_las.build_ndsm(merged["dsm"], merged["dtm"], one)
    return {"six": six, "one": one, **merged}


def test_merged_evidence_keeps_five_bands(built):
    with rasterio.open(built["ev"]) as src:
        assert src.count == 5


def test_ndsm_with_evidence_has_six_bands(built):
    with rasterio.open(built["six"]) as src:
        assert src.count == tokyo_las.NDSM_BAND_COUNT == 6
        assert src.nodata == NODATA


def test_old_single_band_writer_still_available(built):
    """Requirement 6: the pre-change writer stays reachable for comparison runs."""
    with rasterio.open(built["one"]) as src:
        assert src.count == 1


def test_band_one_semantics_are_unchanged(built):
    """Band 1 of the six-band nDSM is byte-for-byte the old single-band nDSM."""
    with rasterio.open(built["six"]) as six, rasterio.open(built["one"]) as one:
        a = six.read(1)
        b = one.read(1)
        assert six.transform == one.transform and six.crs == one.crs
        assert a.dtype == b.dtype == np.float32
        assert a.tobytes() == b.tobytes()
        # And it is a real nDSM, not an empty raster. Located by coordinate, not
        # by index, so a merge that shifted the grid shows up here rather than
        # being absorbed by an index that moved with it.
        probes = {
            "roof": (six.index(_x(0), _y(J_ROOF[-1])),
                     roof_z(0) - ground_z(0)),
            "canopy": (six.index(_x(0), _y(J_CANOPY[0])),
                       max(canopy_z(0, J_CANOPY[0])) - ground_z(0)),
            # Negative, and it must stay negative: ndsm_refine returns heights
            # "exactly as the raster stores them, negatives included", because a
            # clamp here would hide a botched ground surface rather than report
            # it. Without this probe a writer clamping band 1 at zero passes.
            "sunken": (six.index(_x(0), _y(J_SUNKEN)), -SUNKEN_DEPTH_M),
        }
    for name, ((r, c), expected) in probes.items():
        assert a[r, c] == pytest.approx(expected, abs=1e-4), name
    assert (a[a != NODATA] < 0).any(), "fixture must contain negative nDSM values"


def test_an_old_single_band_reader_still_reads_band_one(built):
    """``src.read()[0]`` -- what a pre-change reader does -- is still the height."""
    with rasterio.open(built["six"]) as src:
        first = src.read()[0]
    with rasterio.open(built["one"]) as src:
        assert first.tobytes() == src.read()[0].tobytes()


def test_nodata_co_occurs_across_all_six_bands(built):
    """The precondition ``load_ndsm_evidence`` documents and cannot check.

    Where they disagree the reader hands the classifier a NaN height beside a
    confident ``n_all``/``roughness`` -- evidence for a cell that has no height.
    """
    with rasterio.open(built["six"]) as src:
        data = src.read()
    height_nodata = data[0] == NODATA
    for band in range(1, 6):
        assert np.array_equal(data[band] == NODATA, height_nodata), (
            f"band {band + 1} nodata does not co-occur with band 1")


def test_pixels_with_returns_but_no_height_are_nodata_everywhere(built):
    """The measured failure this rule exists for, made non-vacuous.

    The no-ground row carries returns (so the tile evidence is valid there) but
    no ground reference (so band 1 is not). The first assertion proves the
    fixture really contains that case; without it the co-occurrence test above
    would pass on a tile where the case never arises.
    """
    with rasterio.open(built["ev"]) as src:
        tile_ev = src.read()
    for row in (ROW_NO_GROUND, ROW_GROUND_ONLY):
        assert (tile_ev[0, row, :] != NODATA).any(), (
            f"fixture row {row} was meant to carry returns without a height")

    with rasterio.open(built["six"]) as src:
        data = src.read()
    for row in (ROW_NO_GROUND, ROW_GROUND_ONLY, ROW_GAP):
        assert (data[:, row, :] == NODATA).all(), (
            f"row {row} has no nDSM height, so no band may claim data there")


def test_a_height_without_evidence_is_dropped_from_every_band(tmp_path):
    """The other direction of the co-occurrence rule, which the LAS fixture
    cannot reach.

    ``build_ndsm`` masks both ways: a pixel loses its evidence if it has no
    height, *and* it loses its height if it has no evidence. Only the first
    direction ever arises from a real point cloud -- a pixel with a DSM and a DTM
    return necessarily has returns to count -- so on the synthetic LAS tile the
    second mask is dead code, and deleting it changes nothing that the suite
    could see. Measured: mutating ``valid &= ~getmaskarray(ev).any(axis=0)`` to a
    no-op left all tests passing.

    That mask is not decoration. The three rasters come from three independent
    merges, and a sheet that produced a DSM/DTM pair but whose evidence bands
    failed (:func:`process_las_to_raster` returns them separately, precisely so a
    failure there cannot cost the tile its heights) leaves exactly this shape:
    height everywhere, evidence missing. Without the mask the runtime reader maps
    that to ``n_all=0, roughness=0`` beside a real height -- the maximally
    roof-like evidence value, on a cell nothing was measured about.

    So the case is built by hand rather than derived from points.
    """
    transform = from_origin(0.0, 4.0, 1.0, 1.0)
    common = {"driver": "GTiff", "height": 4, "width": 4, "dtype": "float32",
              "crs": "EPSG:6677", "transform": transform, "nodata": NODATA}

    dsm = np.full((4, 4), 12.0, dtype=np.float32)
    dtm = np.full((4, 4), 2.0, dtype=np.float32)
    ev = np.stack([np.full((4, 4), float(10 * (b + 1)), dtype=np.float32)
                   for b in range(5)])
    ev[:, 1, 2] = NODATA          # returns missing here, height present

    paths = {}
    for name, arr, count in (("dsm", dsm[None], 1), ("dtm", dtm[None], 1),
                             ("ev", ev, 5)):
        paths[name] = str(tmp_path / f"{name}.tif")
        with rasterio.open(paths[name], "w", count=count, **common) as dst:
            dst.write(arr)

    out = str(tmp_path / "ndsm6.tif")
    tokyo_las.build_ndsm(paths["dsm"], paths["dtm"], out, evidence_path=paths["ev"])

    with rasterio.open(out) as src:
        data = src.read()
    assert data.shape == (6, 4, 4)
    # The pixel with no evidence loses its height too, in every band.
    assert (data[:, 1, 2] == NODATA).all(), (
        f"pixel (1, 2) kept data {data[:, 1, 2]} although it had no evidence")
    # ... and it is the only one, so the mask is targeted rather than blanket.
    assert (data[0] != NODATA).sum() == 15
    assert data[0, 0, 0] == pytest.approx(10.0)
    assert data[1, 0, 0] == pytest.approx(10.0) and data[5, 0, 0] == pytest.approx(50.0)


def _import_ndsm_refine():
    sys.path.insert(0, os.path.abspath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")))
    try:
        import app.backend.ndsm_refine as mod
    except Exception:                                       # pragma: no cover
        pytest.skip("app.backend.ndsm_refine not importable in this env")
    return mod


def test_six_band_ndsm_pools_exactly(built):
    """The pooling arithmetic, against the consumer's own ``pool_evidence``."""
    pool_evidence = _import_ndsm_refine().pool_evidence

    with rasterio.open(built["six"]) as src:
        data = src.read().astype(np.float64)
    data[1:] = np.where(data[1:] == NODATA, 0.0, data[1:])
    n_rows = data.shape[1]
    # One group per raster row, which is how the runtime pools a model cell.
    labels = np.repeat(np.arange(n_rows, dtype=np.intp), data.shape[2])
    flat = data.reshape(data.shape[0], -1)
    ev = pool_evidence(labels, n_rows, flat[1], flat[2], flat[3], flat[4], flat[5])

    canopy_mrf = np.nanmax(ev["mrf"][list(ROW_CANOPY)])
    roof_mrf = np.nanmax(ev["mrf"][list(ROW_ROOF)])
    assert canopy_mrf == pytest.approx(1.0, abs=1e-9)
    assert roof_mrf == pytest.approx(0.0, abs=1e-9)
    assert np.nanmax(ev["roughness"][list(ROW_CANOPY)]) > 2.0
    assert np.nanmax(ev["roughness"][list(ROW_ROOF)]) < 0.2


def _stub_grid_geometry(monkeypatch, path, n_rows, n_cols):
    """Install a ``compute_grid_geometry`` covering *path*'s exact extent.

    ``load_ndsm_evidence`` imports the real one from ``voxcity`` at call time,
    and that package is not installed in the env that has laspy -- so without
    this the whole reader, the part of the contract this module has to satisfy,
    would go untested everywhere. Only the *geometry* is faked: band indexing,
    degraded detection, the nodata mappings and the pooling are all the real
    reader's. The grid is laid out with row 0 at the SOUTH edge, which is the
    reader's documented anchoring and the opposite of the raster's north-up
    row order, so a reader that indexed by array position instead of by
    coordinate would fail this.
    """
    import types
    from pyproj import Transformer

    with rasterio.open(path) as src:
        b, crs = src.bounds, src.crs
    to_wgs = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    (west, east), (south, north) = to_wgs.transform(
        [b.left, b.right], [b.bottom, b.top])

    origin = np.array([west, south], dtype=np.float64)       # SW corner
    u_full = np.array([0.0, north - south], dtype=np.float64)   # side_1: north
    v_full = np.array([east - west, 0.0], dtype=np.float64)     # side_2: east

    def compute_grid_geometry(rectangle_vertices, meshsize):
        return {
            "grid_size": (n_rows, n_cols),
            "origin": origin,
            "adj_mesh": (1.0, 1.0),
            "u_vec": u_full / n_rows,
            "v_vec": v_full / n_cols,
        }

    core = types.ModuleType("voxcity.geoprocessor.raster.core")
    core.compute_grid_geometry = compute_grid_geometry
    for name, mod in (
        ("voxcity", types.ModuleType("voxcity")),
        ("voxcity.geoprocessor", types.ModuleType("voxcity.geoprocessor")),
        ("voxcity.geoprocessor.raster", types.ModuleType("voxcity.geoprocessor.raster")),
        ("voxcity.geoprocessor.raster.core", core),
    ):
        monkeypatch.setitem(sys.modules, name, mod)


def test_load_ndsm_evidence_reads_the_six_band_raster(built, monkeypatch):
    """The real reader, end to end: band count, band order, nodata mapping.

    ``pool_evidence`` alone tests none of these -- it is handed already-unpacked
    arrays. This is the test that would catch a band written in the wrong slot,
    a nodata value the reader does not recognise, or a six-band file the reader
    still calls degraded.
    """
    refine = _import_ndsm_refine()
    _stub_grid_geometry(monkeypatch, built["six"], HEIGHT, 1)

    ev = refine.load_ndsm_evidence([(0, 0), (0, 1), (1, 1), (1, 0)], 1.0,
                                   built["six"])
    assert ev is not None
    assert ev["degraded"] is False, "six bands must not read as degraded"
    assert ev["shape"] == (HEIGHT, 1)
    for key in ("mrf", "roughness", "n_all", "n_nonground"):
        assert ev[key] is not None

    # Row 0 of the model grid is the SOUTH edge, i.e. the roof; the last row is
    # the north edge, i.e. the canopy. Getting this backwards is exactly the
    # orientation bug the coordinate-based mapping exists to prevent.
    assert ev["mrf"][0, 0] == pytest.approx(0.0, abs=1e-9)
    assert ev["mrf"][-1, 0] == pytest.approx(1.0, abs=1e-9)
    assert ev["roughness"][-1, 0] > 2.0

    # The co-occurrence precondition, seen through the reader's own two
    # different nodata mappings: band 1 -> NaN, evidence -> 0.0. They agree only
    # if a nodata pixel is nodata in every band, so a cell that pooled no
    # returns must also have no height. This is the assertion that would fire on
    # the measured bad state -- NaN height beside n_all=160.
    empty = ev["n_all"] == 0
    assert empty.any(), "fixture must contain a cell with no returns at all"
    assert np.isnan(ev["height"][empty]).all()
    assert np.isfinite(ev["height"][~empty]).any()


def test_load_ndsm_evidence_calls_the_single_band_raster_degraded(built, monkeypatch):
    """The other half of the contract: the raster in production today has one
    band, and the reader must keep working on it."""
    refine = _import_ndsm_refine()
    _stub_grid_geometry(monkeypatch, built["one"], HEIGHT, 1)

    ev = refine.load_ndsm_evidence([(0, 0), (0, 1), (1, 1), (1, 0)], 1.0,
                                   built["one"])
    assert ev is not None
    assert ev["degraded"] is True
    assert ev["mrf"] is None and ev["roughness"] is None
    assert ev["spread"] is not None          # derived from band 1, always there
    assert np.isfinite(ev["height"]).any()


# ---------------------------------------------------------------------------
# 5. Merge overlap check
# ---------------------------------------------------------------------------
#: Tile side, in pixels, for the merge-overlap fixtures. Large enough that a
#: one-pixel sliver is a realistic *fraction* (0.5%) rather than a quarter of
#: the tile -- see test_merge_tolerates_a_one_pixel_sliver, which is the test
#: that can tell a calibrated threshold from a zero one.
TILE_PX = 200


def _write_tile(path, west, south, size=TILE_PX, res=1.0, value=1.0, count=1):
    transform = from_origin(west, south + size * res, res, res)
    profile = {"driver": "GTiff", "height": size, "width": size, "count": count,
               "dtype": "float32", "crs": "EPSG:6677", "transform": transform,
               "nodata": NODATA}
    with rasterio.open(str(path), "w", **profile) as dst:
        dst.write(np.full((count, size, size), value, dtype=np.float32))
    return str(path)


def _overlap_warnings(recwarn):
    return [w for w in recwarn.list if "overlap" in str(w.message)]


def test_merge_warns_when_survey_sheets_overlap(tmp_path):
    a = _write_tile(tmp_path / "a.tif", 0.0, 0.0)
    b = _write_tile(tmp_path / "b.tif", TILE_PX / 2, 0.0, value=2.0)   # 50% overlap
    with pytest.warns(UserWarning, match="overlap"):
        tokyo_las.merge_geotiffs([a, b], str(tmp_path / "m.tif"), "EPSG:6677")


def test_merge_does_not_warn_for_adjacent_sheets(tmp_path, recwarn):
    """Tokyo sheets abut. A check that fires on every real merge is noise, and
    noise is how a real violation gets ignored."""
    a = _write_tile(tmp_path / "a.tif", 0.0, 0.0)
    b = _write_tile(tmp_path / "b.tif", float(TILE_PX), 0.0, value=2.0)  # shares an edge
    tokyo_las.merge_geotiffs([a, b], str(tmp_path / "m.tif"), "EPSG:6677")
    assert not _overlap_warnings(recwarn)


def test_merge_tolerates_a_one_pixel_sliver(tmp_path, recwarn):
    """The realistic near-miss, and the one that pins the threshold.

    A tile's raster extent is its point extent rounded *up* to a whole pixel, so
    neighbouring sheets routinely share one pixel column. Two sheets that merely
    touch have zero intersection *area*, so they cannot distinguish a calibrated
    threshold from ``0.0`` -- this pair can: it overlaps by 1/200 of a side,
    i.e. 0.5%, which is under OVERLAP_WARN_FRAC and over nothing.
    """
    assert 0.0 < 1.0 / TILE_PX < tokyo_las.OVERLAP_WARN_FRAC
    a = _write_tile(tmp_path / "a.tif", 0.0, 0.0)
    b = _write_tile(tmp_path / "b.tif", float(TILE_PX - 1), 0.0, value=2.0)
    tokyo_las.merge_geotiffs([a, b], str(tmp_path / "m.tif"), "EPSG:6677")
    assert not _overlap_warnings(recwarn)


def test_merge_warns_on_a_small_but_real_overlap(tmp_path):
    """Pins OVERLAP_WARN_FRAC from above.

    The 50% case below would still pass at a threshold of 0.4, and the sliver
    case only pins it from below, so between them any value in (0.005, 0.5]
    survives. A 5% overlap -- an order of magnitude past the sliver and an order
    below a duplicated sheet -- is what makes the interval two-sided.
    """
    overlap_px = TILE_PX // 20                       # 5% of the side
    assert tokyo_las.OVERLAP_WARN_FRAC < overlap_px / TILE_PX < 0.5
    a = _write_tile(tmp_path / "a.tif", 0.0, 0.0)
    b = _write_tile(tmp_path / "b.tif", float(TILE_PX - overlap_px), 0.0, value=2.0)
    with pytest.warns(UserWarning, match="overlap"):
        tokyo_las.merge_geotiffs([a, b], str(tmp_path / "m.tif"), "EPSG:6677")


def test_batched_merge_checks_the_whole_list_exactly_once(tmp_path, recwarn):
    """The batched merge is the production path, and it is the *only* place the
    check has to run over the whole list rather than per chunk.

    Exactly one warning, not at least one. Per-chunk checking would report the
    same sheets again once per batch -- 23 times over a real rebuild's 5,669
    tiles at batch_size 250 -- and would still miss every pair split across a
    chunk boundary. One whole-list pass is both cheaper and strictly more
    complete, and the count is what pins it.
    """
    files = [_write_tile(tmp_path / f"t{i}.tif", i * TILE_PX / 2, 0.0, value=i + 1.0)
             for i in range(4)]                       # each overlaps its neighbour 50%
    tokyo_las.merge_geotiffs_batched(
        files, str(tmp_path / "m.tif"), "EPSG:6677", batch_size=2,
        tmp_dir=str(tmp_path / "tmp"))
    hits = _overlap_warnings(recwarn)
    assert len(hits) == 1, [str(w.message) for w in hits]
    assert "3 pair(s)" in str(hits[0].message)


def test_batched_merge_ignores_interleaving_intermediates(tmp_path, recwarn):
    """Abutting sheets, ordered so the chunks interleave.

    A chunk is an arbitrary slice of the file list, so chunk bounding boxes
    overlap by construction even when no two *sheets* do -- here chunk 0 spans
    x=[0,600) and chunk 1 spans x=[200,800), a 67% box overlap over data that
    never collides. Checking the intermediates would warn on every real rebuild;
    this is the test that says so.
    """
    xs = [0.0, 2.0 * TILE_PX, 1.0 * TILE_PX, 3.0 * TILE_PX]     # interleaved order
    files = [_write_tile(tmp_path / f"t{i}.tif", x, 0.0, value=i + 1.0)
             for i, x in enumerate(xs)]
    tokyo_las.merge_geotiffs_batched(
        files, str(tmp_path / "m.tif"), "EPSG:6677", batch_size=2,
        tmp_dir=str(tmp_path / "tmp"))
    assert not _overlap_warnings(recwarn)


def test_touching_sheets_are_not_an_overlap_at_any_threshold(tmp_path):
    """``min_overlap_frac`` is a public argument, and at 0.0 the fraction test
    stops distinguishing anything -- two sheets sharing only an edge have zero
    shared *area*, and merge() never has to choose between them. Without this,
    dropping the strict-positive-area guard is invisible: at the 1% default the
    fraction test hides it.
    """
    a = _write_tile(tmp_path / "a.tif", 0.0, 0.0)
    touching = _write_tile(tmp_path / "b.tif", float(TILE_PX), 0.0, value=2.0)
    assert tokyo_las.warn_if_inputs_overlap(
        [a, touching], "EPSG:6677", min_overlap_frac=0.0) == []
    # ... and the same call does report a real, arbitrarily small overlap, so
    # the assertion above is not passing merely because nothing is ever found.
    sliver = _write_tile(tmp_path / "c.tif", float(TILE_PX - 1), 0.0, value=3.0)
    with pytest.warns(UserWarning, match="overlap"):
        found = tokyo_las.warn_if_inputs_overlap(
            [a, sliver], "EPSG:6677", min_overlap_frac=0.0)
    assert len(found) == 1


def test_merge_overlap_check_is_opt_outable(tmp_path, recwarn):
    a = _write_tile(tmp_path / "a.tif", 0.0, 0.0)
    b = _write_tile(tmp_path / "b.tif", TILE_PX / 2, 0.0, value=2.0)
    tokyo_las.merge_geotiffs([a, b], str(tmp_path / "m.tif"), "EPSG:6677",
                             check_overlap=False)
    assert not _overlap_warnings(recwarn)


def test_merge_overlap_check_handles_multiband_evidence(tmp_path):
    a = _write_tile(tmp_path / "a.tif", 0.0, 0.0, count=5)
    b = _write_tile(tmp_path / "b.tif", TILE_PX / 2, 0.0, value=2.0, count=5)
    with pytest.warns(UserWarning, match="overlap"):
        out = tokyo_las.merge_geotiffs([a, b], str(tmp_path / "m.tif"), "EPSG:6677")
    with rasterio.open(out) as src:
        assert src.count == 5


# ---------------------------------------------------------------------------
# The merge has to fit in RAM, and on disk
# ---------------------------------------------------------------------------
# A city-scale rebuild merges 5,669 survey sheets into a 65,900 x 64,464 grid.
# One float32 band of that is 17.0 GB; the five evidence bands are 85.0 GB, and
# the rebuild machine has 63.7 GB. The pre-change merge built the whole merged
# array in memory before writing it, so the evidence merge could not run at all
# -- and the failure mode is a MemoryError hours into a multi-hour job, or
# thrashing that never finishes. These tests pin the two properties that make it
# possible: the merge never materialises the full extent, and every raster this
# module writes is compressed and tiled.


def _legacy_merge_geotiffs(input_files, output_path, target_crs,
                           nodata_value=-9999.0, res=None):
    """The pre-Task-7b merge, verbatim, as the bit-identity oracle.

    Materialises the entire merged array and writes it uncompressed. This is the
    implementation that produced the live ``ndsm_cog.tif``, so "the streaming
    merge did not change any value" means "equal to this".

    Note that this oracle also pins the *rasterio* merge algorithm: it calls
    ``rasterio.merge.merge`` directly, and rasterio 1.3 and 1.4+ composite
    differently (1.4 added its own chunked path, which clips each source to the
    chunk boundary). If this suite is ever run against a different rasterio than
    the 1.3.11 that ``voxcityapp2`` pins, these tests will say so rather than
    quietly comparing against a reference that moved.
    """
    from rasterio.merge import merge as _rio_merge

    warped, opened = [], []
    try:
        first_src = rasterio.open(input_files[0])
        opened.append(first_src)
        if res is None:
            res = first_src.res
        tcrs = tokyo_las.normalize_crs(target_crs) or target_crs
        for fp in input_files:
            src = rasterio.open(fp)
            opened.append(src)
            kwargs = dict(crs=tcrs, src_nodata=src.nodata, dst_nodata=nodata_value,
                          resampling=tokyo_las.Resampling.nearest, resolution=res)
            if src.crs is None:
                kwargs["src_crs"] = tcrs
            warped.append(tokyo_las.WarpedVRT(src, **kwargs))

        merged, transform = _rio_merge(warped, nodata=nodata_value, method="first")
        meta = {"driver": "GTiff", "height": merged.shape[1], "width": merged.shape[2],
                "count": merged.shape[0], "dtype": merged.dtype, "crs": tcrs,
                "transform": transform, "nodata": nodata_value}
        with rasterio.open(output_path, "w", **meta) as dst:
            dst.write(merged)
        return output_path
    finally:
        for v in warped:
            try:
                v.close()
            except Exception:
                pass
        for s in opened:
            try:
                s.close()
            except Exception:
                pass


def _assert_rasters_bit_identical(new_path, old_path, what):
    """Compare *arrays*, not files.

    Deliberately: the point of Task 7b is that the new writer compresses and
    tiles, so the bytes on disk differ by construction while every pixel value
    must not. ``tobytes()`` on the decoded array is the strongest statement that
    can be true here -- it catches a NaN pattern change and a single-ulp change
    that ``allclose`` would wave through.
    """
    with rasterio.open(new_path) as a, rasterio.open(old_path) as b:
        assert (a.count, a.width, a.height) == (b.count, b.width, b.height), what
        assert a.transform == b.transform, f"{what}: transform moved"
        assert a.bounds == b.bounds, f"{what}: bounds moved"
        assert a.crs == b.crs, what
        assert a.nodata == b.nodata, what
        na, nb = a.read(), b.read()
        assert na.dtype == nb.dtype == np.float32, what
        assert na.tobytes() == nb.tobytes(), f"{what} is not bit-identical"
        # ... and the comparison is not cheap: the oracle has to hold real data,
        # real holes, and more than one distinct value.
        real = nb[nb != NODATA]
        assert real.size > 0 and (nb == NODATA).any(), f"{what}: trivial oracle"
        assert np.unique(real).size > 1, f"{what}: oracle is a constant"


#: A value that is *not* nodata but is within ``np.isclose``'s default tolerance
#: of it (rtol 1e-5 on 9999 is ~0.1, so anything in [-9999.1, -9998.9] counts).
#:
#: ``rasterio.merge`` asks "has this destination pixel been written yet?" with
#: ``np.isclose(region, nodata)``, not ``==``. On a pixel holding this value the
#: two answers differ, and a later overlapping sheet therefore overwrites under
#: one rule and not under the other. Without such a pixel the streamed merge
#: could swap ``isclose`` for ``==`` and no fixture could tell.
NEAR_NODATA = np.float32(-9998.95)


def _write_varied_tile(path, west, south, height, width, res, base, count=5,
                       hole=None, near_nodata_at=None):
    """A tile whose every pixel differs from every other pixel.

    A constant-valued tile cannot see a paste that lands one row off, is
    transposed, or is mirrored -- the overlap fixtures above are constant because
    they only test the overlap *warning*. Here the value encodes (band, row, col)
    plus a per-tile base, so any misplacement changes a value.
    """
    b, r, c = np.meshgrid(np.arange(count), np.arange(height), np.arange(width),
                          indexing="ij")
    arr = (base + 1000.0 * b + r + 0.001 * c).astype(np.float32)
    if hole is not None:
        hr, hc = hole
        arr[:, hr, hc] = NODATA          # an interior nodata hole, all bands
    if near_nodata_at is not None:
        nr, nc = near_nodata_at
        arr[:, nr, nc] = NEAR_NODATA
    transform = from_origin(west, south + height * res, res, res)
    profile = {"driver": "GTiff", "height": height, "width": width, "count": count,
               "dtype": "float32", "crs": "EPSG:6677", "transform": transform,
               "nodata": NODATA}
    with rasterio.open(str(path), "w", **profile) as dst:
        dst.write(arr)
    return str(path)


@pytest.fixture
def varied_sheets(tmp_path):
    """Sheets chosen to make the merge arithmetic hard.

    - Different shapes (non-square, and not multiples of each other), so a
      transposed or size-assuming paste cannot pass.
    - Origins that are *not* on a common multiple of the tile size, so the
      destination windows have fractional offsets and ``round_offsets`` is
      actually exercised rather than being a no-op.
    - One overlapping pair, so ``method="first"`` has to make a choice and the
      choice has to be the same one; without it the merge is a disjoint paste
      and the compositing logic is untested.
    - Interior nodata holes, so the "already written?" mask is non-trivial.
    - A :data:`NEAR_NODATA` pixel inside the contested strip, so the mask is
      built with ``isclose`` semantics rather than equality semantics.
    - One sheet on a half-pixel origin, so the merged grid's width and height in
      pixels are *not* whole numbers before rounding. Every other sheet sits on
      integer coordinates, and with only those a merge that truncated the pixel
      count instead of rounding it would be indistinguishable. Real survey sheets
      are the awkward case: each one's extent is its own point cloud's bounding
      box rounded up, so their origins share no common grid.
    """
    res = 1.0
    specs = [
        # (west, south, height, width, base, hole, near_nodata_at)
        (0.0,    0.0,   37, 53, 10_000.0, (5, 7), None),
        (53.0,   0.0,   37, 41, 20_000.0, (11, 3), None),
        (94.0,  17.0,   29, 47, 30_000.0, (2, 2), None),
        # near-nodata at local (5, 45) -> x=75.5, y=54.5, inside the strip below
        (30.0,  37.0,   23, 61, 40_000.0, (9, 40), (5, 45)),
        # overlaps the sheet above by half its width -> method="first" decides
        (60.0,  37.0,   23, 61, 50_000.0, (1, 1), None),
        # half-pixel origin: pushes the union bounds off the pixel grid
        (-0.5,  -0.5,   19, 29, 60_000.0, (3, 3), None),
    ]
    return [_write_varied_tile(tmp_path / f"s{i}.tif", w, s, h, wd, res, base,
                               hole=hole, near_nodata_at=near)
            for i, (w, s, h, wd, base, hole, near) in enumerate(specs)]


@pytest.mark.parametrize("stripe_rows", [1, 3, 8, 64, 100_000])
def test_streamed_merge_is_bit_identical_to_the_dense_merge(
        varied_sheets, tmp_path, monkeypatch, stripe_rows):
    """The invariant the whole rebuild rests on.

    Swept across stripe heights that do not divide any sheet height (1, 3, 8),
    one that divides neither the sheet heights nor the output height (64), and
    one larger than the whole raster (the degenerate single-stripe case). A seam
    is a bug that appears at one particular alignment, so a single stripe height
    would be testing one alignment out of many.
    """
    monkeypatch.setattr(tokyo_las, "MERGE_STRIPE_ROWS", stripe_rows)
    new = tokyo_las.merge_geotiffs(varied_sheets, str(tmp_path / "new.tif"),
                                   "EPSG:6677", check_overlap=False)
    old = _legacy_merge_geotiffs(varied_sheets, str(tmp_path / "old.tif"),
                                 "EPSG:6677")
    _assert_rasters_bit_identical(new, old, f"merge (stripe_rows={stripe_rows})")


def test_streamed_merge_keeps_first_wins_at_an_overlap(varied_sheets, tmp_path):
    """The overlapping pair really does contest pixels.

    Without this the fixture could be disjoint by accident -- a merge that
    ignored ``method="first"`` entirely would still be bit-identical to an oracle
    that never had to choose. Here the earlier sheet's values must win on the
    shared ground.
    """
    out = tokyo_las.merge_geotiffs(varied_sheets, str(tmp_path / "m.tif"),
                                   "EPSG:6677", check_overlap=False)
    with rasterio.open(out) as src:
        # sheet 3 spans x=[30,91), sheet 4 spans x=[60,121); both y=[37,60)
        row, col = src.index(70.5, 45.5)        # inside the contested strip
        got = src.read(1, window=Window(col, row, 1, 1))[0, 0]
    assert 40_000.0 <= got < 50_000.0, (
        f"contested pixel took the later sheet ({got}); method='first' broke")


def test_a_near_nodata_pixel_counts_as_unwritten(varied_sheets, tmp_path):
    """``isclose``, not ``==``, decides whether a destination pixel is free.

    This is the one place the two rules disagree, and the merge has to disagree
    the same way the library does or the streamed merge is not a faithful
    replacement. Kept as its own test because the bit-identity comparison would
    report it as "some byte differs" without saying which rule moved.

    Asserted by absence rather than at a fixed coordinate: the half-pixel sheet
    puts the merged grid half a pixel off every other sheet's, so which merged
    cell the value lands in is a rounding decision, and a test that hard-coded
    the cell would be testing the rounding instead.
    """
    with rasterio.open(varied_sheets[3]) as sheet3:
        r, c = sheet3.index(75.5, 54.5)
        assert sheet3.read(1, window=Window(c, r, 1, 1))[0, 0] == NEAR_NODATA, (
            "fixture lost its near-nodata pixel")
        assert (sheet3.read() == NEAR_NODATA).sum() == 5, "one pixel, all 5 bands"

    out = tokyo_las.merge_geotiffs(varied_sheets, str(tmp_path / "m.tif"),
                                   "EPSG:6677", check_overlap=False)
    with rasterio.open(out) as src:
        merged = src.read()
    assert (merged == NEAR_NODATA).sum() == 0, (
        "the near-nodata pixel survived the merge, so the destination mask used "
        "equality; rasterio.merge uses np.isclose and the outputs diverge")


def test_batched_merge_is_bit_identical_to_the_dense_merge(varied_sheets, tmp_path):
    """The production path. Its final merge is the full-extent one."""
    new = tokyo_las.merge_geotiffs_batched(
        varied_sheets, str(tmp_path / "new.tif"), "EPSG:6677", batch_size=2,
        tmp_dir=str(tmp_path / "tmp"), check_overlap=False)
    old = _legacy_merge_geotiffs(varied_sheets, str(tmp_path / "old.tif"),
                                 "EPSG:6677")
    _assert_rasters_bit_identical(new, old, "batched merge")


# ---------------------------------------------------------------------------
# The memory ceiling
# ---------------------------------------------------------------------------
#: Bands, pixel size and layout of the memory-ceiling fixture.
#:
#: Twenty-four five-band 128 x 128 tiles laid on a diagonal, spaced far enough
#: apart that the *merged* grid is 1,600 x 11,904 -- 381 MB dense, 1,164 times
#: the 327 KB of any single tile. That ratio is the whole point: a merge that
#: allocates per tile is fine and a merge that allocates per merged extent is
#: not, and only a fixture whose combined extent dwarfs its inputs can tell them
#: apart.
CEIL_BANDS = 5
CEIL_TILE = 128
CEIL_N = 24
CEIL_STRIDE_X = 64.0
CEIL_STRIDE_Y = 512.0

#: Traced-allocation ceiling for that merge, in bytes.
#:
#: The streaming merge holds one stripe of the merged grid --
#: MERGE_STRIPE_ROWS x 1,600 x 5 bands x 4 B = 16.4 MB -- plus the rows of one
#: source that the stripe touches. Measured on this exact fixture: 17.7 MB,
#: repeatable to 0.1 MB over three runs. 32 MB is 1.8x that and 11.9x below the
#: 380.9 MB dense array -- loose enough not to be flaky, tight enough that
#: reintroducing the dense allocation cannot slip under it.
#:
#: It is deliberately *not* set just above 17.7 MB. A ceiling that tight would
#: also catch a merge that merely doubled its stripe, which is a speed and
#: memory regression but not a wrong answer, at the cost of a test that fails on
#: an unrelated numpy allocation. ``test_stripe_writes_are_contiguous`` pins that
#: narrower property directly instead.
CEIL_BYTES = 32 * 1024 * 1024


def test_merge_never_materialises_the_full_extent(tmp_path):
    """A future change that rebuilds the whole merged array in memory fails here.

    **Instrument: ``tracemalloc``, not process RSS.** Three reasons, each
    measured rather than assumed:

    1. ``psutil`` is not installed in ``voxcityapp2`` -- the only environment
       with ``laspy``, so the only one a rebuild can run in -- and an RSS probe
       would have to be hand-rolled per platform.
    2. numpy's data buffers *are* traced: a 100 MB ``np.zeros`` shows up as
       100.0 MB of traced memory in this environment, verified before this test
       was written. What is being guarded against is precisely a numpy
       allocation -- ``rasterio.merge.merge`` builds ``np.zeros((count, H, W))``.
    3. Process RSS on Windows also counts GDAL's block cache and the page cache
       for the GeoTIFF being written: large, reclaimable, and not under this
       module's control. A threshold tight enough to catch a 381 MB array would
       be dominated by that noise.

    The trade is that GDAL-side C++ allocations are invisible here. That is why
    the test also asserts the *output*: a merge that allocated nothing because it
    produced nothing would otherwise pass.
    """
    import tracemalloc

    files = [
        _write_varied_tile(
            tmp_path / f"c{i}.tif", i * CEIL_STRIDE_X, i * CEIL_STRIDE_Y,
            CEIL_TILE, CEIL_TILE, 1.0, 100_000.0 * (i + 1), count=CEIL_BANDS)
        for i in range(CEIL_N)
    ]

    width = int((CEIL_N - 1) * CEIL_STRIDE_X) + CEIL_TILE
    height = int((CEIL_N - 1) * CEIL_STRIDE_Y) + CEIL_TILE
    dense_bytes = CEIL_BANDS * width * height * 4
    tile_bytes = CEIL_BANDS * CEIL_TILE * CEIL_TILE * 4
    # The fixture has to be *able* to fail: a merged extent that fits under the
    # ceiling anyway would make this test vacuous.
    assert dense_bytes > 5 * CEIL_BYTES, dense_bytes
    assert dense_bytes > 100 * tile_bytes, "combined extent must dwarf any tile"

    out = str(tmp_path / "merged.tif")
    tracemalloc.start()
    try:
        tracemalloc.reset_peak()
        tokyo_las.merge_geotiffs(files, out, "EPSG:6677", check_overlap=False)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    assert peak < CEIL_BYTES, (
        f"merge peaked at {peak / 1e6:.0f} MB of traced allocations; the merged "
        f"grid is {width} x {height} x {CEIL_BANDS} = {dense_bytes / 1e6:.0f} MB "
        f"dense, so this looks like a full materialisation again")

    # The merge actually happened, and landed where it should.
    with rasterio.open(out) as src:
        assert (src.width, src.height, src.count) == (width, height, CEIL_BANDS)
        for i in (0, CEIL_N // 2, CEIL_N - 1):
            r, c = src.index(i * CEIL_STRIDE_X + 0.5,
                             i * CEIL_STRIDE_Y + CEIL_TILE - 0.5)
            got = src.read(1, window=Window(c, r, 1, 1))[0, 0]
            assert got == pytest.approx(100_000.0 * (i + 1)), f"tile {i} misplaced"
        # ... and the ground between tiles really is empty, so the check above is
        # not passing on a raster uniformly filled with one tile's value.
        r, c = src.index(CEIL_STRIDE_X * 3 + 0.5, CEIL_STRIDE_Y * 10 + 0.5)
        assert src.read(1, window=Window(c, r, 1, 1))[0, 0] == NODATA


def test_stripe_writes_are_contiguous(varied_sheets, tmp_path, monkeypatch):
    """Every array handed to ``dst.write`` is C-contiguous.

    The stripe buffer is allocated once and reused. The obvious way to write that
    -- a 3-D buffer sliced ``[:, :rows, :]`` for the short final stripe -- yields
    a non-contiguous view, and ``dst.write`` then copies it silently, doubling
    the merge's peak for the last stripe. Measured on real sheets: 158 MB against
    an 80 MB stripe.

    Pinned here rather than by tightening the memory ceiling, because the ceiling
    would have to sit within ~25% of the measured peak to see it, and a threshold
    that tight is a flaky test rather than a strict one. This asserts the actual
    property, and it cannot be satisfied vacuously: the stripe height is chosen
    so that the final stripe *is* short, and the test says so.
    """
    stripe = 7
    monkeypatch.setattr(tokyo_las, "MERGE_STRIPE_ROWS", stripe)

    seen = []

    class _SpyWriter:
        def write(self, arr, window=None):
            seen.append((bool(arr.flags["C_CONTIGUOUS"]), arr.shape[1]))

    srcs = [rasterio.open(f) for f in varied_sheets]
    try:
        xs, ys = [], []
        for s in srcs:
            left, bottom, right, top = s.bounds
            xs.extend([left, right])
            ys.extend([bottom, top])
        res = srcs[0].res
        w, s_, e, n = min(xs), min(ys), max(xs), max(ys)
        width = int(round((e - w) / res[0]))
        height = int(round((n - s_) / res[1]))
        tokyo_las._merge_streaming(
            srcs, _SpyWriter(), from_origin(w, n, res[0], res[1]), height, width,
            (w, s_, e, n), NODATA, tokyo_las.Resampling.nearest, stripe)
    finally:
        for s in srcs:
            s.close()

    assert seen, "no stripes were written"
    assert height % stripe != 0 and any(rows != stripe for _, rows in seen), (
        "the fixture must produce a short final stripe or this test is vacuous")
    bad = [rows for ok, rows in seen if not ok]
    assert not bad, f"non-contiguous stripe writes of heights {bad}"


# ---------------------------------------------------------------------------
# Compression and tiling
# ---------------------------------------------------------------------------
def _smooth_array(count, height, width):
    """Spatially correlated float32 -- what a DSM or a return count looks like.

    Random noise is incompressible, so a ``np.random.rand`` fixture would make a
    "compression shrinks the file" assertion fail even with compression fully
    enabled, and the natural response would be to weaken the assertion. Real
    rasters are smooth, and the deflate/predictor-3 pair on this module's writers
    is chosen for exactly that.
    """
    r = np.arange(height, dtype=np.float32)[:, None]
    c = np.arange(width, dtype=np.float32)[None, :]
    base = 20.0 + 0.01 * r + 0.02 * c
    return np.stack([base + 3.0 * b for b in range(count)]).astype(np.float32)


@pytest.mark.parametrize("count", [1, 5])
def test_save_raster_writes_compressed_tiled_rasters(tmp_path, count):
    arr = _smooth_array(count, 600, 700)
    path = str(tmp_path / "t.tif")
    tokyo_las.save_raster(
        {"array": arr if count > 1 else arr[0],
         "transform": from_origin(0.0, 600.0, 1.0, 1.0), "nodata": NODATA},
        path, "EPSG:6677")
    with rasterio.open(path) as src:
        assert src.profile.get("compress") == "deflate"
        assert src.profile.get("tiled") is True
        assert src.count == count
        # Lossless: the values that come back are the values that went in.
        assert src.read().tobytes() == arr.reshape(count, 600, 700).tobytes()
    raw = count * 600 * 700 * 4
    assert os.path.getsize(path) < raw / 2, (
        "compression is declared but the file is not actually smaller; a 51 GB "
        "evidence_geotiffs/ does not fit next to a 102 GB free C:")


def test_merged_rasters_are_compressed_and_tiled(varied_sheets, tmp_path):
    out = tokyo_las.merge_geotiffs(varied_sheets, str(tmp_path / "m.tif"),
                                   "EPSG:6677", check_overlap=False)
    with rasterio.open(out) as src:
        assert src.profile.get("compress") == "deflate"
        assert src.profile.get("tiled") is True


def test_creation_options_ask_for_bigtiff_when_it_is_needed():
    """A 65,900 x 64,464 five-band float32 raster is 85 GB -- twenty times past
    the 4 GB where a classic TIFF's 32-bit directory offsets overflow. There is
    no way to observe that on a fixture-sized file, so the option itself is what
    gets pinned, as ``IF_SAFER`` rather than ``YES`` so small tiles stay classic
    TIFFs that any reader can open.
    """
    assert tokyo_las.GTIFF_CREATION_OPTS["BIGTIFF"] == "IF_SAFER"
    assert tokyo_las.GTIFF_CREATION_OPTS["compress"] == "deflate"
    assert tokyo_las.GTIFF_CREATION_OPTS["tiled"] is True


def test_an_integer_merge_still_works(tmp_path):
    """``PREDICTOR=3`` is float-only, and GDAL *refuses* it rather than warning.

    Everything this pipeline writes is float32, but ``merge_geotiffs`` takes its
    output dtype from its inputs, so pinning the float predictor unconditionally
    would turn a working integer merge into ``RasterioIOError: PREDICTOR=3 is
    only supported with Float32 or Float64`` -- a hard failure introduced by a
    change whose whole point was that nothing observable changes.
    """
    assert tokyo_las.creation_opts("float32")["predictor"] == 3
    assert tokyo_las.creation_opts("int32")["predictor"] == 2
    assert tokyo_las.creation_opts("uint16")["compress"] == "deflate"

    paths = []
    for i, west in enumerate((0.0, 40.0)):
        p = str(tmp_path / f"i{i}.tif")
        with rasterio.open(p, "w", driver="GTiff", height=40, width=40, count=1,
                           dtype="int32", crs="EPSG:6677", nodata=-9999,
                           transform=from_origin(west, 40.0, 1.0, 1.0)) as dst:
            dst.write(np.full((1, 40, 40), 7 + i, dtype=np.int32))
        paths.append(p)
    out = tokyo_las.merge_geotiffs(paths, str(tmp_path / "m.tif"), "EPSG:6677",
                                   nodata_value=-9999, check_overlap=False)
    with rasterio.open(out) as src:
        assert src.dtypes[0] == "int32"
        assert src.profile.get("compress") == "deflate"
        data = src.read(1)
    assert data[0, 0] == 7 and data[0, -1] == 8


# ---------------------------------------------------------------------------
# Copying the finished nDSM without reading it
# ---------------------------------------------------------------------------
def test_copy_raster_file_does_not_read_the_raster(tmp_path):
    """``precompute_las_cache``'s no-AOI path used to do ``data = src.read()``.

    On the six-band city raster that is a 102 GB read on a 63.7 GB machine, to
    produce a file identical to the one already on disk. Same instrument and same
    reasoning as the merge ceiling above.
    """
    import tracemalloc

    src_path = str(tmp_path / "src.tif")
    arr = _smooth_array(6, 800, 900)                      # 17.3 MB of pixels
    profile = {"driver": "GTiff", "height": 800, "width": 900, "count": 6,
               "dtype": "float32", "crs": "EPSG:6677", "nodata": NODATA,
               "transform": from_origin(0.0, 800.0, 1.0, 1.0),
               "tiled": True, "blockxsize": 256, "blockysize": 256}
    with rasterio.open(src_path, "w", **profile) as dst:
        dst.write(arr)

    dst_path = str(tmp_path / "dst.tif")
    tracemalloc.start()
    try:
        tracemalloc.reset_peak()
        tokyo_las.copy_raster_file(src_path, dst_path)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    dense = 6 * 800 * 900 * 4
    assert peak < dense / 8, (
        f"copy peaked at {peak / 1e6:.1f} MB against a {dense / 1e6:.1f} MB "
        f"raster; it is reading the whole thing again")
    with rasterio.open(src_path) as a, rasterio.open(dst_path) as b:
        assert a.read().tobytes() == b.read().tobytes()
        assert a.profile == b.profile
