"""Consumers that were silently broken by the old mixed-frame LOD2 model.

A PLATEAU LOD2 model used to reach the app with **two** coordinate frames:
``voxels.classes``, ``dem.elevation``, ``tree_canopy.*``, ``buildings.*`` and
``extras['mesh_vegetation_mask']`` were north-up (row 0 = north) while
``land_cover.classes`` alone was south-up — the frame ``voxcity.utils.orientation``
declares (row 0 = the southern origin edge) and the one ``compute_grid_geometry``
draws in.  VoxCityGML ``494816a`` fixed it at the assembly seam
(``voxcitygml.pipeline._to_south_up``) and app ``407d990`` deleted the
compensating flips.

Four consumers were repaired *for free* by that seam fix and had no coverage at
all.  This module pins them down.  Nothing here needs an app-side code change:
the tests are the whole deliverable.

Fixture strategy — why it has teeth
-----------------------------------
Every model below is built by handing a synthetic ``PipelineArtifacts`` to the
**real** ``VoxCityGML.run()``, so the arrays genuinely pass through
``_to_south_up`` on their way into ``assemble_voxcity``.  The artifacts are
authored north-up (``_as_artifact`` = ``np.flipud`` of the south-up spec),
exactly as ``run_core`` produces them — except ``land_cover_grid``, which
voxcitygml already emits south-up and therefore does *not* send through the
seam.  Revert ``_to_south_up`` to the identity and every model here comes back
in the old north-up frame, which is precisely the pre-fix condition.

Mirror-detection discipline (see ``test_model_geo_overlays``): a mirror is only
observable with an **asymmetric** fixture *and* an **absolute** reference.
Flat terrain, a centred feature, a high sun and a symmetric grid are all
flip-invariant and prove nothing, and comparing two arrays that flip *together*
(``buildings.heights`` against ``voxels.classes``, say) proves nothing either.
So every feature here sits off-centre and is checked against a lon/lat, or
against content that came through the seam independently of the consumer under
test.  ``test_the_*_is_asymmetric`` guards keep it that way.

Caveat: a VoxCityGML model carries no ``extras['rotation_angle']`` and the
solar paths read ``extras.get('rotation_angle', 0)``.  For the axis-aligned AOI
used here that default is correct, so it does not affect these tests — but they
must not be extended to a rotated AOI without revisiting it.
"""
from __future__ import annotations

import io
import os
import zipfile

import numpy as np
import pytest
import trimesh
from fastapi.testclient import TestClient

from backend.main import app, import_obj_store
from backend.state import app_state
from voxcity.geoprocessor.raster import compute_grid_geometry
from voxcity.utils.orientation import check_axes
from voxcity.utils import GridProjector

pytest.importorskip("voxcitygml", reason="the seam under test lives in voxcitygml")

# --------------------------------------------------------------------------
# Geography.  A small axis-aligned AOI in Tokyo; 2 m cells over ~120 m gives a
# 60x60 grid, big enough for a 10-row band either side of a feature.
# --------------------------------------------------------------------------
MESHSIZE = 2.0
SIDE_M = 120.0
LON0, LAT0 = 139.7700, 35.6700
_DLAT = SIDE_M / 111320.0
_DLON = SIDE_M / (111320.0 * float(np.cos(np.deg2rad(LAT0))))
# VoxCity order: [SW, NW, NE, SE]
RECT = [
    (LON0, LAT0),
    (LON0, LAT0 + _DLAT),
    (LON0 + _DLON, LAT0 + _DLAT),
    (LON0 + _DLON, LAT0),
]
GRID_GEOM = compute_grid_geometry(RECT, MESHSIZE)
NX, NY = GRID_GEOM["grid_size"]
PROJ = GridProjector(GRID_GEOM)
NZ = 24  # 48 m of headroom at 2 m cells

GROUND_CODE = -1
BUILDING_CODE = -3

# A DEM that increases strictly monotonically with latitude.  Flat terrain
# cannot detect a mirror at all (the real Chuo-ku DEM correlates only +0.32
# with its own mirror image, which is why a real-data A/B was inconclusive);
# a monotonic ramp is maximally sensitive to one.
DEM_BASE = 10.0
DEM_SLOPE = 0.5  # metres per row northward -> 10.0 m at the south edge, 39.5 m at the north

# The CityGML "marker" building: rows 8..14 of 60, i.e. the southern third.
# Its mirror image is rows 46..52.
MARK_LO, MARK_HI = 8, 14
MARK_COL_LO, MARK_COL_HI = 22, 30
MARK_H = 12.0

# The solar fixture's isolated tower: rows 18..24, southern half, with room for
# a 10-row band on either side.
TOWER_LO, TOWER_HI = 18, 24
TOWER_COL_LO, TOWER_COL_HI = 26, 34
TOWER_H = 20.0
BAND = 10


def _as_artifact(grid):
    """South-up spec -> the north-up array ``run_core`` would have produced.

    Deliberately the inverse of the seam, so ``_to_south_up`` puts it back.
    Neutering the seam therefore delivers a north-up model — the old bug.
    """
    return None if grid is None else np.ascontiguousarray(np.flipud(np.asarray(grid)))


def _ramp_dem() -> np.ndarray:
    """DEM in the south-up frame: strictly increasing with latitude."""
    dem = np.zeros((NX, NY), dtype=float)
    dem += (DEM_BASE + DEM_SLOPE * np.arange(NX, dtype=float))[:, None]
    return dem


def _flat_ground_voxels() -> np.ndarray:
    vox = np.zeros((NX, NY, NZ), dtype=np.int8)
    vox[:, :, 0] = GROUND_CODE
    return vox


def _with_block(vox, row_lo, row_hi, col_lo, col_hi, height_m):
    """Stamp a solid building block into a south-up voxel grid, in place."""
    k_top = int(round(height_m / MESHSIZE))
    vox[row_lo:row_hi, col_lo:col_hi, 1:1 + k_top] = BUILDING_CODE
    return vox


def _row_lat(row: int) -> float:
    """Latitude of the centre of grid row *row* — the absolute reference."""
    return float(PROJ.cell_to_lon_lat(row, NY // 2)[1])


def _cell_lonlat(row: int, col: int) -> tuple[float, float]:
    lon, lat = PROJ.cell_to_lon_lat(row, col)
    return float(lon), float(lat)


def _seam_city(monkeypatch, *, dem=None, voxels=None, land_cover=None,
               heights=None, canopy=None):
    """Assemble a VoxCity through the real voxcitygml seam.

    All grids are given in the **south-up** frame the finished model must be
    in; they are flipped to the north-up artifact frame on the way in so that
    ``_to_south_up`` is what puts them right again.  ``land_cover`` is the
    exception the seam itself documents: voxcitygml gets it from voxcity's own
    downloader already south-up and passes it through untouched.
    """
    import voxcitygml.pipeline as vgp
    from voxcitygml.models import VoxelizerConfig

    dem = _ramp_dem() if dem is None else np.asarray(dem, dtype=float)
    voxels = _flat_ground_voxels() if voxels is None else np.asarray(voxels)
    land_cover = (np.zeros((NX, NY), dtype=np.int32) if land_cover is None
                  else np.asarray(land_cover, dtype=np.int32))
    heights = (np.zeros((NX, NY), dtype=float) if heights is None
               else np.asarray(heights, dtype=float))
    canopy = (np.zeros((NX, NY), dtype=float) if canopy is None
              else np.asarray(canopy, dtype=float))

    min_heights = np.empty((NX, NY), dtype=object)
    for i in range(NX):
        for j in range(NY):
            min_heights[i, j] = []

    art = vgp.PipelineArtifacts(
        collection=None,
        rectangle=RECT,
        buffered_rectangle=RECT,
        center_lon=LON0 + _DLON / 2,
        center_lat=LAT0 + _DLAT / 2,
        citygml_paths=[],
        land_cover_source="OpenEarthMapJapan",
        canopy_height_source="Static",
        dem_grid=_as_artifact(dem),
        land_cover_grid=land_cover,           # already south-up; not seam-converted
        building_height_grid=_as_artifact(heights),
        building_min_height_grid=_as_artifact(min_heights),
        building_id_grid=_as_artifact(np.zeros((NX, NY), dtype=np.int32)),
        canopy_top=_as_artifact(canopy),
        canopy_bottom=_as_artifact(np.zeros((NX, NY), dtype=float)),
        voxel_grid=_as_artifact(voxels),
        voxel_min_z=0.0,
        mesh_vegetation_mask=_as_artifact(np.zeros((NX, NY), dtype=bool)),
    )
    monkeypatch.setattr(vgp, "run_core", lambda cfg: art)

    cfg = VoxelizerConfig(
        citygml_path="<synthetic>",
        center_lon=LON0 + _DLON / 2,
        center_lat=LAT0 + _DLAT / 2,
        size_meters=SIDE_M,
        meshsize=MESHSIZE,
        rectangle_vertices=RECT,
        save_output=False,
    )
    city = vgp.VoxCityGML(cfg).run()
    city.extras["building_lod"] = 2
    return city


def _install(city):
    app_state.voxcity = city
    app_state.rectangle_vertices = city.extras["rectangle_vertices"]
    app_state.land_cover_source = "OpenEarthMapJapan"
    app_state.auxiliary_lines = []
    return city


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture(autouse=True)
def _clean_state():
    import_obj_store.clear()
    yield
    app_state.voxcity = None
    app_state.rectangle_vertices = None
    app_state.auxiliary_lines = []
    import_obj_store.clear()


# ==========================================================================
# Guards on the fixtures themselves
# ==========================================================================

def test_the_test_dem_is_asymmetric_and_monotonic():
    """A flat or symmetric DEM would make the anchor tests vacuous."""
    dem = _ramp_dem()
    assert np.all(np.diff(dem[:, 0]) > 0), "the DEM must increase strictly northward"
    assert not np.array_equal(dem, np.flipud(dem))


def test_the_test_features_sit_off_centre():
    """Both stamped blocks must be off-centre, and clear of their mirrors."""
    for lo, hi in ((MARK_LO, MARK_HI), (TOWER_LO, TOWER_HI)):
        assert hi < NX / 2, "feature must sit in the southern half"
        assert hi <= NX - hi, "feature must not overlap its own mirror image"
    # The solar bands must fit inside the domain on both sides of the tower.
    assert TOWER_LO - BAND >= 0 and TOWER_HI + BAND <= NX


def test_the_seam_fixture_actually_goes_through_the_seam(monkeypatch):
    """Guard the guard: the model must be south-up, or nothing below has teeth.

    ``_as_artifact`` pre-flips, so this passes only while ``_to_south_up``
    flips back.  If this test fails, every other test in this module is
    measuring the wrong thing.
    """
    city = _seam_city(monkeypatch)
    dem = np.asarray(city.dem.elevation)
    assert dem[0, 0] == pytest.approx(DEM_BASE), (
        "row 0 of the assembled DEM is not the southern (lowest) edge — the "
        "voxcitygml assembly seam is not converting to south-up")
    assert dem[-1, 0] == pytest.approx(DEM_BASE + DEM_SLOPE * (NX - 1))
    # RECT is axis-aligned, so the angle must be present *and* zero. It used to
    # be absent: assemble_voxcity — the constructor voxcitygml assembles
    # through — never set it, and the solar paths'
    # extras.get('rotation_angle', 0) default merely happened to be right here.
    # On a rotated AOI that default silently rotated every shadow by the AOI's
    # own angle, so absence is no longer an acceptable answer.
    assert city.extras["rotation_angle"] == 0, (
        "this module assumes an axis-aligned AOI; a non-zero rotation_angle "
        "means the fixture rectangle changed and the frame assertions below "
        "no longer hold")


# ==========================================================================
# Step 1 — /api/model/anchor_ground on a monotonic N-S gradient
# ==========================================================================

@pytest.mark.parametrize("row", [3, NX - 4], ids=["near-south-edge", "near-north-edge"])
def test_anchor_ground_samples_the_dem_at_the_requested_latitude(client, monkeypatch, row):
    """The DEM elevation returned must be the one the DEM holds at that lat.

    Probed at two latitudes near opposite edges so that a constant offset —
    or a coincidence at the domain centre — cannot pass.  The reference is the
    lon/lat, not another array from the same model.
    """
    _install(_seam_city(monkeypatch))
    lon, lat = _cell_lonlat(row, NY // 2)

    r = client.get("/api/model/anchor_ground", params={"lon": lon, "lat": lat})
    assert r.status_code == 200, r.text
    body = r.json()

    expected = DEM_BASE + DEM_SLOPE * row
    mirrored = DEM_BASE + DEM_SLOPE * (NX - 1 - row)
    assert body["dem_elevation"] == pytest.approx(expected), (
        f"anchor_ground at lat {lat:.6f} (row {row} from the south edge) "
        f"returned {body['dem_elevation']} m; the DEM holds {expected} m there "
        f"and {mirrored} m at the mirrored row — the DEM is stored "
        "north<->south mirrored relative to the grid geometry")
    assert body["dem_min"] == pytest.approx(DEM_BASE)
    assert body["meshsize_m"] == pytest.approx(MESHSIZE)


def test_anchor_ground_reads_higher_ground_further_north(client, monkeypatch):
    """Terrain rises northward, so the northern probe must read higher."""
    _install(_seam_city(monkeypatch))

    def _elev(row):
        lon, lat = _cell_lonlat(row, NY // 2)
        r = client.get("/api/model/anchor_ground", params={"lon": lon, "lat": lat})
        assert r.status_code == 200, r.text
        return r.json()["dem_elevation"]

    south, north = _elev(3), _elev(NX - 4)
    assert north > south, (
        f"the DEM rises {DEM_SLOPE} m/row northward, but anchor_ground reports "
        f"{north} m in the north and {south} m in the south — the grid is "
        "mirrored north<->south")


# ==========================================================================
# Step 2 — OBJ import placement
# ==========================================================================

def _marker_city(monkeypatch):
    """Flat terrain plus one CityGML block in the southern third."""
    vox = _with_block(_flat_ground_voxels(), MARK_LO, MARK_HI,
                      MARK_COL_LO, MARK_COL_HI, MARK_H)
    heights = np.zeros((NX, NY), dtype=float)
    heights[MARK_LO:MARK_HI, MARK_COL_LO:MARK_COL_HI] = MARK_H
    flat = np.full((NX, NY), DEM_BASE, dtype=float)
    return _seam_city(monkeypatch, dem=flat, voxels=vox, heights=heights)


def _box_obj_bytes(size=(6.0, 6.0, 12.0)) -> bytes:
    """A box centred on the model origin, so anchor_model_point=(0,0,0) is its centre."""
    mesh = trimesh.creation.box(extents=size)
    mesh.apply_translation((0.0, 0.0, size[2] / 2.0))  # base at z=0, centred in XY
    return mesh.export(file_type="obj").encode("utf-8")


def _building_rows(classes) -> set[int]:
    arr = np.asarray(classes)
    return set(np.flatnonzero(np.any(arr == BUILDING_CODE, axis=(1, 2))).tolist())


def _commit_box_beside_the_marker(client) -> tuple[np.ndarray, set[int]]:
    """Import a 6x6x12 m box anchored at the marker's latitude, a few cells east.

    Returns ``(new_voxel_mask, marker_rows)``, where ``marker_rows`` is read off
    the model *before* the import so it names the CityGML block alone.
    """
    marker_rows = _building_rows(app_state.voxcity.voxels.classes)
    assert marker_rows, "fixture is broken: no CityGML block in the model"
    before = np.asarray(app_state.voxcity.voxels.classes) == BUILDING_CODE

    lon, lat = _cell_lonlat((MARK_LO + MARK_HI) // 2, MARK_COL_HI + 6)
    up = client.post(
        "/api/model/import_obj/upload",
        files={"file": ("box.obj", io.BytesIO(_box_obj_bytes()), "text/plain")},
    )
    assert up.status_code == 200, up.text
    commit = client.post("/api/model/import_obj/commit", json={
        "import_id": up.json()["import_id"],
        "placement": {
            "anchor_lonlat": [lon, lat],
            "anchor_model_point": [0.0, 0.0, 0.0],
            "rotation": 0.0,
            "move": [0.0, 0.0, 0.0],
            "units": "m",
            "z_up": True,
            "swap_yz": False,
        },
        "roles": {},
        "overwrite": True,
    })
    assert commit.status_code == 200, commit.text
    assert commit.json()["n_building_voxels_added"] > 0, commit.json().get("warning")

    after = np.asarray(app_state.voxcity.voxels.classes) == BUILDING_CODE
    # The grid may have grown vertically; compare on the common depth.
    kz = min(before.shape[2], after.shape[2])
    return after[:, :, :kz] & ~before[:, :, :kz], marker_rows


def test_imported_obj_lands_on_the_rows_the_anchor_lonlat_points_at(client, monkeypatch):
    """The most severe of the repaired bugs: horizontal placement.

    The error was independent of terrain and up to the domain's full N-S
    extent.  The absolute reference has to be the CityGML marker block: it
    reached the voxel grid through the seam and therefore *moves* when the seam
    is neutered.  ``buildings.heights`` would not do — it flips together with
    ``voxels.classes``, so comparing the two proves nothing.

    Anchoring the OBJ at the marker's latitude (a few cells east of it, so the
    two never overlap) must land it on the marker's own rows.
    """
    _install(_marker_city(monkeypatch))
    new, marker_rows = _commit_box_beside_the_marker(client)

    new_rows = set(np.flatnonzero(np.any(new, axis=(1, 2))).tolist())
    assert new_rows, "no new building voxels — the import landed nowhere"
    assert new_rows <= marker_rows, (
        f"the OBJ was anchored at the latitude of the CityGML block (model "
        f"rows {sorted(marker_rows)}) but landed on rows {sorted(new_rows)} — "
        "placement and voxel content disagree by a north<->south mirror")
    assert max(new_rows) < NX / 3, (
        f"the OBJ landed on rows {sorted(new_rows)} of {NX}; the anchor lat is "
        "in the southern third of the domain")


def test_imported_obj_leaves_the_mirror_of_the_target_rows_empty(client, monkeypatch):
    """The other half — but the mirror has to be of the *block*, not the domain.

    The tempting form of this assertion, "no new voxels in the northern half of
    the grid", is mirror-invariant and worth nothing: the importer projects the
    anchor lon/lat through the grid geometry, which is the same whichever way
    up the voxel array is stored, so the stamped rows are identical in both
    frames.  What differs is where the seam put the CityGML content relative to
    them.  So the emptiness is claimed about the reflection of the marker's
    actual rows — on a mirrored model those are precisely the rows the OBJ
    lands on, and this fails.
    """
    _install(_marker_city(monkeypatch))
    new, marker_rows = _commit_box_beside_the_marker(client)

    new_rows = set(np.flatnonzero(np.any(new, axis=(1, 2))).tolist())
    mirror_of_marker = {NX - 1 - r for r in marker_rows}
    assert not (mirror_of_marker & marker_rows), (
        "fixture is too tall to distinguish the block from its own reflection")
    assert new_rows and not (new_rows & mirror_of_marker), (
        f"the OBJ landed on rows {sorted(new_rows)}, which fall in the mirror "
        f"image ({sorted(mirror_of_marker)}) of the CityGML block it was "
        f"anchored on ({sorted(marker_rows)}) — the voxel grid is stored "
        "north<->south mirrored relative to the grid geometry")


# ==========================================================================
# Step 3 — GPU solar shadows fall on the right side
# ==========================================================================
#
# Azimuth convention, confirmed rather than assumed.  ``compute_sun_direction``
# (simulator_gpu/solar/integration/utils.py:192-203) builds
# ``(cos az, sin az, sin el)`` with component 0 along array axis 0 = north —
# the same formula as ``orientation.direction_to_axis_vector``, whose docstring
# calls its azimuth a *toward* direction.  The vector is then used by
# ``trace_rays_kernel`` (integration/caching.py:1011-1013) as
# ``p = origin + sun_dir * t``: the ray marches **towards the sun**.  So the
# vector is "toward the sun" and the API's ``azimuth_degrees_ori`` is therefore
# the sun's own compass bearing (the meteorological *from*-azimuth) — no +180
# correction is needed here, unlike a call to ``direction_to_axis_vector`` with
# a wind direction.  ``test_the_sun_azimuth_is_the_suns_own_bearing`` confirms
# it empirically by flipping the azimuth and watching the shadow change sides.
#
# So: azimuth 180 = sun due south = ray direction ``(-cos el, 0, sin el)`` =
# marching south and up = shadows cast to the **north** = towards higher row
# indices, since row 0 is the southern edge.

SUN_ELEVATION = 20.0   # low: 20 m / tan(20 deg) ~= 55 m ~= 27 cells of shadow
DNI = 1000.0

_TAICHI_SKIP = pytest.mark.skipif(
    os.environ.get("VOXCITY_SKIP_GPU_TESTS") == "1",
    reason="VOXCITY_SKIP_GPU_TESTS=1",
)


def _tower_city(monkeypatch):
    """One isolated tall building in the southern half, over flat terrain."""
    vox = _with_block(_flat_ground_voxels(), TOWER_LO, TOWER_HI,
                      TOWER_COL_LO, TOWER_COL_HI, TOWER_H)
    heights = np.zeros((NX, NY), dtype=float)
    heights[TOWER_LO:TOWER_HI, TOWER_COL_LO:TOWER_COL_HI] = TOWER_H
    flat = np.full((NX, NY), DEM_BASE, dtype=float)
    return _seam_city(monkeypatch, dem=flat, voxels=vox, heights=heights)


def _direct_irradiance(city, azimuth_deg):
    """Instantaneous direct-beam ground irradiance, GPU ray-traced.

    Calls the simulator directly rather than ``POST /api/solar``: the endpoint
    derives the sun position from an EPW file and a timestamp, which cannot pin
    the azimuth to exactly 180 deg.  Both routes reach this same function with
    the same ``compute_sun_direction`` call, so nothing frame-related is
    skipped by going straight in.
    """
    from voxcity.simulator_gpu.solar import (
        clear_all_caches,
        get_direct_solar_irradiance_map,
    )

    clear_all_caches()
    return np.asarray(get_direct_solar_irradiance_map(
        city,
        azimuth_degrees_ori=azimuth_deg,
        elevation_degrees=SUN_ELEVATION,
        direct_normal_irradiance=DNI,
        show_plot=False,
        with_reflections=False,
        view_point_height=1.5,
        tree_k=0.6,
        tree_lad=1.0,
    ), dtype=float)


def _band_means(grid):
    """(south-of-tower mean, north-of-tower mean) over the tower's columns.

    The bands are fixed by the tower's *geographic* placement, so a mirrored
    model puts the tower nowhere near either of them and both bands come back
    bright — which is what makes this assertion mirror-sensitive.
    """
    cols = slice(TOWER_COL_LO, TOWER_COL_HI)
    south = grid[TOWER_LO - BAND:TOWER_LO, cols]
    north = grid[TOWER_HI:TOWER_HI + BAND, cols]
    return float(np.nanmean(south)), float(np.nanmean(north))


@_TAICHI_SKIP
def test_solar_shadow_falls_north_of_the_building_for_a_due_south_sun(monkeypatch):
    """A due-south sun must darken the cells north of the tower, not south.

    On the old north-up grid axis 0 pointed south, so the mapping in
    ``simulator_gpu/solar/integration/volumetric.py`` (and the identical one on
    the ground path) put the shadow on the wrong side of every building — a
    defect nobody would have connected to a canopy overlay.
    """
    ti = pytest.importorskip("taichi")
    city = _tower_city(monkeypatch)
    try:
        grid = _direct_irradiance(city, azimuth_deg=180.0)
    except (ti.TaichiRuntimeError, RuntimeError) as exc:   # pragma: no cover
        pytest.skip(f"GPU/Taichi unavailable: {exc}")

    south_mean, north_mean = _band_means(grid)
    assert south_mean > 0.5 * DNI * np.sin(np.deg2rad(SUN_ELEVATION)), (
        "the band south of the tower should be in full sun; got "
        f"{south_mean:.1f} W/m2 — the fixture, not the frame, is wrong")
    assert north_mean < 0.1 * south_mean, (
        f"mean irradiance is {north_mean:.1f} W/m2 over the {BAND} rows north "
        f"of the tower and {south_mean:.1f} W/m2 over the {BAND} rows south of "
        "it. With the sun due south the northern band must be the shadowed "
        "one; near-equal means say the tower is not where the grid geometry "
        "places it — the voxel grid is mirrored north<->south")


@_TAICHI_SKIP
def test_the_sun_azimuth_is_the_suns_own_bearing(monkeypatch):
    """Confirm the convention empirically instead of trusting the docstring.

    ``azimuth_degrees_ori`` is the compass bearing the sun *stands at*.  Move
    the sun from due south (180) to due north (0) and the shadow must swap
    bands.  Get this backwards and the test above would "prove" the opposite.
    """
    ti = pytest.importorskip("taichi")
    city = _tower_city(monkeypatch)
    try:
        south_sun = _direct_irradiance(city, azimuth_deg=180.0)
        north_sun = _direct_irradiance(city, azimuth_deg=0.0)
    except (ti.TaichiRuntimeError, RuntimeError) as exc:   # pragma: no cover
        pytest.skip(f"GPU/Taichi unavailable: {exc}")

    s_lit, n_shadow = _band_means(south_sun)
    s_shadow, n_lit = _band_means(north_sun)
    assert n_shadow < 0.1 * s_lit, "azimuth 180 must shadow the northern band"
    assert s_shadow < 0.1 * n_lit, "azimuth 0 must shadow the southern band"


# ==========================================================================
# Step 4 — save_voxcity round trip
# ==========================================================================

def _saved_h5(tmp_path) -> str:
    """Run the app's real session-save path and return the extracted voxcity.h5."""
    from backend.session_io import save_session_to_zip

    buf = save_session_to_zip(app_state)
    out = tmp_path / "session"
    out.mkdir()
    with zipfile.ZipFile(io.BytesIO(buf.getvalue())) as zf:
        zf.extractall(out)
    path = out / "voxcity.h5"
    assert path.is_file(), sorted(p.name for p in out.iterdir())
    return str(path)


def test_saved_session_declares_and_honours_the_axis_contract(monkeypatch, tmp_path):
    """``check_axes`` alone proves nothing — it already passed while false.

    ``session_io.save_session_to_zip`` -> ``voxcity.io.save_voxcity`` stamps
    ``axes = "north,east,up"``, which ``orientation.AXES`` defines as row 0 =
    south.  That attribute was being written onto a north-up model, so the file
    declared a frame it did not have and ``check_axes()`` passed anyway.  The
    real test is the projector lookup: map the marker's lon/lat through
    ``GridProjector.from_h5`` and the saved voxel grid must actually hold the
    marker at that cell.
    """
    import h5py

    _install(_marker_city(monkeypatch))
    path = _saved_h5(tmp_path)

    check_axes(path)  # necessary, nowhere near sufficient

    lon, lat = _cell_lonlat((MARK_LO + MARK_HI) // 2,
                            (MARK_COL_LO + MARK_COL_HI) // 2)
    i, j = GridProjector.from_h5(path).lon_lat_to_cell(lon, lat)
    with h5py.File(path, "r") as f:
        voxels = np.asarray(f["voxcity/voxel_grid"][:])

    column = voxels[i, j, :]
    mirror = voxels[voxels.shape[0] - 1 - i, j, :]
    assert BUILDING_CODE in column.tolist(), (
        f"the marker building's lon/lat maps to saved cell ({i}, {j}), which "
        f"holds no building voxel; the mirrored cell "
        f"({voxels.shape[0] - 1 - i}, {j}) "
        f"{'does' if BUILDING_CODE in mirror.tolist() else 'does not'}. The "
        "file declares axes='north,east,up' (row 0 = south) but its voxel grid "
        "is stored the other way up")
    assert BUILDING_CODE not in mirror.tolist(), (
        "the mirrored cell also holds a building — the fixture is too "
        "symmetric to detect a flip")


def test_saved_session_dem_matches_the_latitude_it_is_read_back_at(monkeypatch, tmp_path):
    """Same round trip, checked on the ramped DEM: value must track latitude."""
    import h5py

    _install(_seam_city(monkeypatch))
    path = _saved_h5(tmp_path)
    check_axes(path)

    proj = GridProjector.from_h5(path)
    with h5py.File(path, "r") as f:
        dem = np.asarray(f["voxcity/dem"][:], dtype=float)

    for row in (3, NX - 4):
        lon, lat = _cell_lonlat(row, NY // 2)
        i, j = proj.lon_lat_to_cell(lon, lat)
        expected = DEM_BASE + DEM_SLOPE * row
        assert dem[i, j] == pytest.approx(expected), (
            f"the saved DEM reads {dem[i, j]} m at lat {lat:.6f}, where the "
            f"model holds {expected} m; the file's declared axes and its data "
            "disagree by a north<->south mirror")
