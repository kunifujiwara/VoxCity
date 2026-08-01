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


def test_six_band_ndsm_survives_the_runtime_reader(built):
    """End to end against the actual consumer: ``load_ndsm_evidence`` must read
    the file as non-degraded and recover the roof/canopy contrast."""
    sys.path.insert(0, os.path.abspath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")))
    try:
        from app.backend.ndsm_refine import pool_evidence
    except Exception:                                       # pragma: no cover
        pytest.skip("app.backend.ndsm_refine not importable in this env")

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
