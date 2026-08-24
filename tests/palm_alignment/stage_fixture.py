"""Stage the PALM terrain-alignment fixture through the REAL app exporter.

NOT A PYTEST TARGET. This directory holds manually-run integration scripts
that need a working PALM install; there is deliberately no ``__init__.py``
and nothing here is named ``test_*``. Run it by hand:

    conda run -n voxcity python tests/palm_alignment/stage_fixture.py [JOB] [CORES]

See README.md in this directory for the full loop (stage -> palmrun -> compare).

WHAT IT PROVES
--------------
``voxcity.exporter.palm._build_zt`` derives terrain from the VOXEL GRID:
``zt = (_ground_level(voxel_classes) - min) * meshsize``, i.e. every terrain
height is an exact multiple of ``meshsize``. On exact multiples of dz PALM's
own re-discretization (ceil/floor/nint all agree) becomes the identity, so
PALM's simulated world should sit cell-for-cell on the voxcity voxel model.
This script builds a fixture city that exercises every surface of that claim
and stages it exactly as the app would; ``compare_topo.py`` then checks the
claim against what PALM actually built.

THE FIXTURE
-----------
20 x 20 columns at meshsize 2.0, nz 40 (u = axis 0 = PALM y, v = axis 1 =
PALM x -- the exporter's axis contract, no flip):

* Three terraces stepping along v (2-cell steps): ground levels 3 / 5 / 7.
* A water strip along u < 5 cut into every terrace with STEPPED banks --
  bank rows u in {0, 4} one cell below the terrace, bed rows u in {1, 2, 3}
  two cells below. The bed's top voxel carries the OSM ``Water`` land-cover
  code, so it is ground for the datum but water for the surface types.
* Buildings of three very different sizes, all rising from their own
  terrace: a 1-VOXEL hut, a 4-voxel mid-rise, a 12-voxel tower.
* Two tree patches with ELEVATED crowns (canopy bottom > 0), one on the low
  terrace and one on the high one, so the lad levels are terrain-following
  and cannot be confused with a ground-up extrusion.

TWO TRAPS THIS FIXTURE IS BUILT AROUND
--------------------------------------
1. ``zt`` now comes from the VOXEL GRID, not the DEM. Relief placed in
   ``city.dem.elevation`` alone produces FLAT terrain and fake coverage, so
   every terrace/bank/bed here is cut into ``voxels.classes``.
   ``dem.elevation`` is set to ``(ground_level - 1) * meshsize`` anyway --
   the exact inverse of the voxelizer's ``int(dem/ms + 0.5) + 1`` rule -- so
   the fixture stays physically coherent and ``origin_z`` is meaningful.
2. PALM's plant-canopy module aborts (PCM0011) when the namelist enables it
   but the static driver carries no lad/zlad. Every column with a -2 voxel
   here also gets ``tree_canopy.top``/``.bottom``, and the script asserts
   that correspondence before staging.
"""
import json
import os
import re
import sys

import numpy as np

os.environ["PALM_SURFACE_NETCDF"] = "1"   # must precede the backend.config import
sys.path.insert(0, r"c:\Users\kunih\OneDrive\00_Codes\python\VoxCityApp")

from backend import config, palm_driver            # noqa: E402
from tests.fixtures import make_voxcity_with_building   # noqa: E402  (app fixture helper)
from voxcity.exporter.palm import _build_zt, _ground_level   # noqa: E402
from voxcity.utils.lc import get_land_cover_classes         # noqa: E402

GROUND = -1
TREE = -2
BUILDING = -3
RANGELAND = 1     # any positive land-cover code is "ground" for the datum

NX = NY = 20      # u, v
NZ = 40
MESHSIZE = 2.0

# (u0, u1, v0, v1, n_voxels, id) -- buildings rise from their own column's
# ground level, so a single (height in metres) covers both the voxel run and
# the 2-D grids.
BUILDINGS = [
    (8, 9, 2, 3, 1, 101),        # 1-VOXEL hut, the smallest thing PALM can resolve
    (5, 8, 8, 11, 4, 102),       # mid-rise
    (10, 13, 15, 18, 12, 103),   # tower
]
# (u0, u1, v0, v1, bottom_m, top_m) -- crowns ELEVATED above their terrain.
TREES = [
    (15, 18, 3, 6, 4.0, 10.0),
    (6, 9, 16, 19, 2.0, 8.0),
]


def water_code():
    """OSM raw land-cover index for Water (the voxelizer stores the raw
    0-based index straight into the top ground voxel, and the exporter maps
    the same raw index back through _build_index_to_palm_map)."""
    names = list(get_land_cover_classes("OpenStreetMap").values())
    return names.index("Water")


def ground_levels():
    """Intended per-column ground level (the voxcity datum: 1 + the highest
    ground/land-cover k). Terraces along v, water channel along u."""
    gl = np.zeros((NX, NY), dtype=np.int64)
    for v in range(NY):
        terrace = 0 if v < 7 else (1 if v < 14 else 2)
        base = 3 + 2 * terrace
        for u in range(NX):
            if u in (1, 2, 3):
                gl[u, v] = base - 2        # channel bed
            elif u in (0, 4):
                gl[u, v] = base - 1        # stepped bank
            else:
                gl[u, v] = base            # terrace top
    return gl


def build_city():
    city = make_voxcity_with_building(nx=NX, ny=NY, nz=NZ, meshsize=MESHSIZE, bh=4)
    gl = ground_levels()
    wc = water_code()

    classes = np.zeros((NX, NY, NZ), dtype=np.int8)
    lc = np.full((NX, NY), RANGELAND, dtype=np.int32)
    for u in range(NX):
        for v in range(NY):
            g = int(gl[u, v])
            classes[u, v, :g] = GROUND
            code = wc if u < 5 else RANGELAND
            classes[u, v, g - 1] = code      # top ground voxel carries the cover
            lc[u, v] = code

    heights = np.zeros((NX, NY), dtype=float)
    ids = np.zeros((NX, NY), dtype=np.int32)
    min_heights = np.empty((NX, NY), dtype=object)
    for u in range(NX):
        for v in range(NY):
            min_heights[u, v] = []
    for u0, u1, v0, v1, n, bid in BUILDINGS:
        h = n * MESHSIZE
        for u in range(u0, u1):
            for v in range(v0, v1):
                g = int(gl[u, v])
                classes[u, v, g:g + n] = BUILDING
                heights[u, v] = h
                ids[u, v] = bid
                min_heights[u, v] = [[0.0, h]]

    top = np.zeros((NX, NY), dtype=float)
    bottom = np.zeros((NX, NY), dtype=float)
    for u0, u1, v0, v1, b_m, t_m in TREES:
        kb, kt = int(round(b_m / MESHSIZE)), int(round(t_m / MESHSIZE))
        for u in range(u0, u1):
            for v in range(v0, v1):
                g = int(gl[u, v])
                classes[u, v, g + kb:g + kt] = TREE
                top[u, v] = t_m
                bottom[u, v] = b_m

    # dem.elevation is the exact inverse of the voxelizer's own
    # int(dem/ms + 0.5) + 1 rule, so the DEM and the voxel grid describe the
    # same terrain. It no longer drives zt (that is the whole point of the
    # alignment change) but it still sets origin_z, and a fixture whose DEM
    # contradicted its voxels would be a fixture that encodes a fiction.
    dem = (gl.astype(float) - 1.0) * MESHSIZE

    city.voxels.classes = classes
    city.land_cover.classes = lc
    city.buildings.heights = heights
    city.buildings.ids = ids
    city.buildings.min_heights = min_heights
    city.tree_canopy.top = top
    city.tree_canopy.bottom = bottom
    city.dem.elevation = dem
    return city, gl


def self_check(city, gl):
    """Fixtures must encode reality: every claim this fixture is supposed to
    make is asserted here, against the SAME helper the exporter uses."""
    classes = city.voxels.classes
    derived = _ground_level(classes)
    assert (derived >= 0).all(), (
        f"{int((derived < 0).sum())} column(s) have no ground datum")
    assert np.array_equal(derived, gl), "derived ground level != intended layout"
    assert derived.min() == 1, (
        f"the lowest column must hold exactly one ground voxel (the voxelizer's "
        f"own invariant), got {int(derived.min())}")

    # Three STRICTLY INCREASING terraces, read off a land row (u >= 5).
    terraces = [int(gl[10, 3]), int(gl[10, 10]), int(gl[10, 17])]
    assert terraces[0] < terraces[1] < terraces[2], terraces
    assert len(set(terraces)) == 3, terraces

    # Stepped banks: bed < bank < terrace on every terrace.
    for v in (3, 10, 17):
        assert gl[2, v] < gl[0, v] < gl[10, v], (v, gl[2, v], gl[0, v], gl[10, v])

    # Every tree column carries canopy geometry (PCM0011), and the crowns are
    # genuinely elevated.
    has_tree = (classes == TREE).any(axis=2)
    assert (city.tree_canopy.top[has_tree] > 0).all()
    assert (city.tree_canopy.bottom[has_tree] > 0).all(), "crowns must be ELEVATED"
    assert not (city.tree_canopy.top[~has_tree] > 0).any()

    # A 1-voxel building really is 1 voxel, and the tower really is tall.
    assert (classes[8, 2] == BUILDING).sum() == 1
    assert (classes[11, 16] == BUILDING).sum() == 12

    # And the exporter's zt is exactly the quantized datum.
    zt, origin_z = _build_zt(classes, city.dem.elevation, MESHSIZE)
    want = ((gl - gl.min()) * MESHSIZE).astype(np.float32)
    assert np.array_equal(zt, want), "zt is not the quantized ground datum"
    k = np.round(zt.astype(np.float64) / MESHSIZE)
    assert np.array_equal((k * MESHSIZE).astype(np.float32), zt), "zt round-trip"
    print(f"  self-check OK: ground levels {sorted(set(gl.ravel().tolist()))}, "
          f"zt levels {sorted(set(np.round(zt.ravel() / MESHSIZE).astype(int).tolist()))}, "
          f"origin_z {origin_z}")


# Shrink the run so PALM finishes in minutes: 5 min spinup, 4 min LES, 2 min
# averaging. Every section of the namelist (incl. surface_data_output) stays
# structurally identical to production -- same substitutions as the staging
# script for the pnc_dual job.
TIME_SHRINK = [
    (r"spinup_time = [0-9.]+", "spinup_time = 300.0"),
    (r"end_time = [0-9.]+", "end_time = 240.0"),
    (r"averaging_interval = [0-9.]+", "averaging_interval = 120.0"),
    (r"dt_data_output_av = [0-9.]+", "dt_data_output_av = 120.0"),
    (r"skip_time_data_output = [0-9.]+", "skip_time_data_output = 120.0"),
    (r"averaging_interval_surf = [0-9.]+", "averaging_interval_surf = 120.0"),
    (r"dt_dosurf_av = [0-9.]+", "dt_dosurf_av = 120.0"),
    (r"skip_time_dosurf_av = [0-9.]+", "skip_time_dosurf_av = 120.0"),
]


def main():
    job = sys.argv[1] if len(sys.argv) > 1 else "align1"
    cores = int(sys.argv[2]) if len(sys.argv) > 2 else 1

    assert config.PALM_SURFACE_NETCDF, "PALM_SURFACE_NETCDF did not take"
    job_dir = os.path.join(config.PALM_JOBS_DIR, job)
    os.makedirs(os.path.join(job_dir, "INPUT"), exist_ok=True)

    city, gl = build_city()
    self_check(city, gl)

    # The reference the comparator uses must be the grid PALM ACTUALLY got,
    # i.e. after stage_inputs' SW-anchored crop -- not the model as authored.
    plan = palm_driver.plan_domain(
        nu=NX, nv=NY, meshsize=MESHSIZE,
        max_building_m=float(np.max(city.buildings.heights)), cores=cores)
    cropped = palm_driver.crop_city(
        city, ny_points=plan.ny_points, nx_points=plan.nx_points)
    ref_classes = np.asarray(cropped.voxels.classes)
    print(f"  plan: {plan}  (crop {NX}x{NY} -> {plan.ny_points}x{plan.nx_points})")

    record = palm_driver.parse_epw_record(config.DEFAULT_TOKYO_EPW, 8, 1, 12)
    meta = palm_driver.stage_inputs(job_dir, job, city, record,
                                    cores=cores, land_cover_source="OpenStreetMap")
    print("  staged:", meta)

    ref = os.path.join(job_dir, "alignment_ref.npz")
    np.savez(ref, voxel_classes=ref_classes, meshsize=np.float64(MESHSIZE),
             ground_level=_ground_level(ref_classes),
             meta_json=np.array(json.dumps(meta)))
    print("  reference:", ref, ref_classes.shape)

    p3d = os.path.join(job_dir, "INPUT", f"{job}_p3d")
    with open(p3d, encoding="utf-8") as f:
        src = f.read()
    for pat, rep in TIME_SHRINK:
        src, n = re.subn(pat, rep, src)
        if n == 0:
            print(f"  WARNING: time-shrink {pat!r} matched nothing")
    with open(p3d, "w", encoding="utf-8") as f:
        f.write(src)
    print("  time-shrink applied to", p3d)
    print("\njob dir:", job_dir)
    print(f"run it:  wsl.exe -e bash -lc 'cd ~/palm/pnc-work && "
          f"export PATH=$HOME/palm/v25.10.1/install/bin:$PATH && "
          f"palmrun -r {job} -c default -a \"d3#\" -X {cores} -v -z'")


if __name__ == "__main__":
    main()
