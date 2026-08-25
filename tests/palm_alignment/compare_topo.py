"""Compare PALM's OWN discretized world against the voxcity voxel grid.

NOT A PYTEST TARGET. This directory holds manually-run integration scripts
that need the output of a real PALM run; there is deliberately no
``__init__.py`` and nothing here is named ``test_*``. Run it by hand:

    conda run -n voxcity python tests/palm_alignment/compare_topo.py <JOB_DIR>

See README.md in this directory for the full loop (stage -> palmrun -> compare).

Exits 0 when every mismatch is explained by PALM's own topography filter,
1 when any HARD mismatch survives.

THE ENCODING, SETTLED EMPIRICALLY
---------------------------------
``<JOB>_topo_surf.<cycle>.nc`` carries PALM's processed world. Two things
about it are undocumented in the PIDS spec and were settled from probe
columns of real runs (printed again on every invocation, see ``probe()``):

* ``topo_all`` values. The file's own ``values`` attribute spells them out:
  ``0: non topography, 1: building, 2: terrain, 3: non-classified
  topography``. Confirmed on the fixture: pure-terrain columns are 2 up to
  their ground level and 0 above; tower columns are 2 up to the terrace top
  and 1 for exactly the building run.
* ``z[0] == 0.0`` -- the zu(0) "real surface" placeholder -- DOES occupy a
  mask slot. A dead-flat reference run (zt == 0 everywhere) has
  ``topo_all[0] == 2`` for EVERY column and ``topo_all[1] == 0``: one solid
  cell, not zero. PALM's terrain rule is ``zu(k) <= zt`` (topography_mod.f90,
  "Map orography as well as buildings onto grid"), and zu(0) == 0 <= zt
  always, so raw index 0 is always solid. That makes PALM's solid run
  ``k = 0 .. zt/dz`` -- exactly ``ground_level`` cells once zt is the
  quantized datum ``(ground_level - 1) * meshsize`` -- and therefore PALM's
  RAW z index equals voxcity's own k with NO shift. No level is dropped
  before comparing.

* ``buildings_3d``, when PALM writes one, is NOT comparable cell-for-cell
  with the voxel column: it is terrain-relative and PALM copies its second
  slot into its first on output (see the ``got_b3`` comment in ``main``).
  Buildings are counted from ``topo_all == 1`` in every case.

WHY A MISMATCH CAN STILL BE BENIGN
----------------------------------
PALM runs ``filter_topography`` (topography_mod.f90) before it starts. It
fills fluid points with >= 4 solid neighbours ("holes") and horizontally
traced fluid pockets of fewer than ``num_thresh`` (<= 9) points
("cavities"). It only ever ADDS solid cells -- there is no branch that
removes one. So ``got > want`` next to a real step is a filter candidate,
while ``got < want`` is always HARD: nothing in PALM deletes topography.
"""
import glob
import json
import os
import re
import sys

import netCDF4
import numpy as np

SOLID_GROUND = -1
TREE = -2
BUILDING = -3

TOPO_AIR, TOPO_BUILDING, TOPO_TERRAIN, TOPO_UNCLASSIFIED = 0, 1, 2, 3


def newest_topo_surf(job_dir):
    """The topo_surf file with the highest CYCLE number.

    Sorted by the parsed integer suffix, never lexically: PALM's cycle
    counter is zero-padded to three digits today, so a run that reaches
    cycle 1000 would sort '1000' before '999' as a string.
    """
    pattern = os.path.join(job_dir, "OUTPUT", "*_topo_surf.*.nc")
    files = glob.glob(pattern)
    if not files:
        raise SystemExit(f"no topo_surf output under {pattern}")

    def cycle(path):
        m = re.search(r"_topo_surf\.(\d+)\.nc$", os.path.basename(path))
        return int(m.group(1)) if m else -1

    return max(files, key=cycle)


def voxel_counts(classes):
    """Per-column (solid, building, tree) counts in the voxcity grid.

    Solid = ground (-1) + land cover (>= 1) + building (-3). Trees are NOT
    solid: PALM represents them as leaf area density, not topography.
    """
    solid = ((classes == SOLID_GROUND) | (classes >= 1) | (classes == BUILDING))
    return (solid.sum(axis=2).astype(int),
            (classes == BUILDING).sum(axis=2).astype(int),
            (classes == TREE).sum(axis=2).astype(int))


def self_consistency(ds, topo, dz):
    """Cross-check topo_all against the file's OWN zt / buildings_2d.

    PALM writes all three from the same ``topo_flags`` array in the same
    routine (data_output_topo_and_surface_setup_mod.f90), so they cannot
    legitimately disagree. They CAN disagree on disk: PALM's own source
    warns that "in case of DATA_TOPO_SURF DOM occasionally reports the
    error 'Parallel operation on file opened for non-parallel access',
    although output seems to be fine", and a multi-rank run of this very
    fixture drops cells from topo_all while zt/buildings_2d stay right.
    A file that fails this gate says nothing about the exporter, so the
    caller must not read its mismatches as an alignment result.

    zt   = zw(k_terrain_top) = k_terrain_top * dz  (zw(0) == 0, uniform dz)
    b2d  = zw(k_building_top) - zw(k_terrain_top)
    """
    zt = np.ma.filled(ds["zt"][:], np.nan)
    nz = topo.shape[0]
    k = np.arange(nz)[:, None, None]
    terr_top = np.where((topo == TOPO_TERRAIN).any(axis=0),
                        np.max(np.where(topo == TOPO_TERRAIN, k, -1), axis=0), 0)
    bad = []
    zt_from_topo = terr_top * dz
    dz_off = np.argwhere(np.abs(zt_from_topo - zt) > 1e-4)
    for y, x in dz_off:
        bad.append(f"zt({y},{x}): file {zt[y, x]} vs topo_all {zt_from_topo[y, x]}")
    if "buildings_2d" in ds.variables:
        b2 = np.ma.filled(ds["buildings_2d"][:].astype(np.float64), 0.0)
        b2 = np.where(np.isfinite(b2), b2, 0.0)
        has_b = (topo == TOPO_BUILDING).any(axis=0)
        b_top = np.where(has_b, np.max(np.where(topo == TOPO_BUILDING, k, -1), axis=0),
                         terr_top)
        b2_from_topo = np.where(has_b, (b_top - terr_top) * dz, 0.0)
        for y, x in np.argwhere(np.abs(b2_from_topo - b2) > 1e-4):
            bad.append(f"buildings_2d({y},{x}): file {b2[y, x]} vs "
                       f"topo_all {b2_from_topo[y, x]}")
    return bad


def probe(ds, topo, classes, gl, dz, probes):
    print("\n=== ENCODING PROBES (settle the semantics from the data) ===")
    print("  np.unique(topo_all) =", np.unique(topo).tolist())
    print("  topo_all:values attr =",
          ds["topo_all"].getncattr("values") if "values" in ds["topo_all"].ncattrs()
          else "(none)")
    z = np.ma.filled(ds["z"][:], np.nan)
    print(f"  z[:6] = {np.round(z[:6], 3).tolist()}   dz = {dz}")
    print(f"  z[0] == 0.0 -> {bool(z[0] == 0.0)};  topo_all[0] solid everywhere -> "
          f"{bool((topo[0] != TOPO_AIR).all())}  "
          f"(=> raw index 0 IS a mask slot, no level is dropped)")
    for y, x, label in probes:
        vox = classes[y, x, :topo.shape[0]]
        print(f"\n  ({y},{x}) {label}: voxcity ground_level={gl[y, x]}, "
              f"zt={np.ma.filled(ds['zt'][:], np.nan)[y, x]}")
        print(f"      z        : {np.round(z, 1).tolist()}")
        print(f"      topo_all : {topo[:, y, x].tolist()}")
        print(f"      voxcity k: {vox.tolist()}   "
              "(-1 ground, >=1 land cover, -3 building, -2 tree, 0 air)")


def classify(y, x, want, got, gl, kind):
    """PALM-FILTER RESIDUE or HARD.

    PALM's filter only ever ADDS solid cells, so a column that LOST cells is
    always HARD. A column that gained them is a filter candidate when it sits
    next to a >= 2-cell step in the REFERENCE ground level -- the geometry
    that produces the narrow pockets filter_topography is written to fill.
    Buildings and canopy are never touched by the terrain filter, so only
    terrain/solid mismatches can be filter residue.
    """
    if got < want:
        return "HARD", "PALM lost solid cells; its filter can only add them"
    if kind != "solid":
        return "HARD", f"{kind} counts are not produced by the terrain filter"
    ny, nx = gl.shape
    step = 0
    for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        yy, xx = y + dy, x + dx
        if 0 <= yy < ny and 0 <= xx < nx:
            step = max(step, abs(int(gl[yy, xx]) - int(gl[y, x])))
    if step >= 2:
        return "PALM-FILTER RESIDUE", f"adjacent to a {step}-cell step in the reference"
    return "HARD", f"largest adjacent reference step is {step} cell(s)"


def measure_z_offset(topo, gl, meta):
    """Empirical vs. computed ``palm_results._z_index_offset``.

    The empirical number: find the RAW z index of the terrain top in a
    reference column and compare it with that column's voxcity k. voxcity's
    voxelizer gives even its lowest column a whole ground voxel
    (``ground_level = nint(dem/ms) + 1``, filling ``[:ground_level]``), so
    voxcity's z=0 sits one full cell BELOW the lowest terrain surface, while
    the quantized ``zt`` puts PALM's z=0 AT that surface.

    Nothing here changes production code -- this only reports the two
    numbers side by side.
    """
    print("\n=== BONUS: _z_index_offset, measured vs computed ===")
    ny, nx = gl.shape
    k = np.arange(topo.shape[0])[:, None, None]
    terr_top = np.max(np.where(topo == TOPO_TERRAIN, k, -1), axis=0)
    offsets = []
    for y in range(ny):
        for x in range(nx):
            if terr_top[y, x] < 0:
                continue                      # building-covered terrain top
            offsets.append(int(terr_top[y, x]) - (int(gl[y, x]) - 1))
    uniq, counts = np.unique(offsets, return_counts=True)
    print(f"  raw PALM terrain-top index minus voxcity terrain-top k, over "
          f"{len(offsets)} columns: {dict(zip(uniq.tolist(), counts.tolist()))}")
    empirical = int(uniq[np.argmax(counts)])
    print(f"  => EMPIRICAL z offset (levels to drop from the bottom of a raw "
          f"PALM 3-D volume) = {empirical}")

    offset = 1
    full_min, window_min = meta.get("dem_min_full_m"), meta.get("dem_min_window_m")
    if full_min is not None and window_min is not None:
        offset -= int(round((float(window_min) - float(full_min))
                            / float(meta["meshsize"])))
    print(f"  => palm_results._z_index_offset(meta) would return = {offset}   "
          f"(dem_min_full_m={full_min}, dem_min_window_m={window_min})")
    print(f"  => derivation j = k + 1 - ground_level.min() with "
          f"ground_level.min()={int(gl.min())} gives {1 - int(gl.min())}")
    if empirical != offset:
        print(f"  !! MISMATCH: the function is {offset - empirical} level(s) "
              "too high for this job (reported only; no production code touched)")
    else:
        print("  == the function agrees with the measurement for this job")
    return empirical, offset


def main():
    if len(sys.argv) != 2:
        raise SystemExit("usage: compare_topo.py <job_dir>")
    job_dir = sys.argv[1]

    ref_path = os.path.join(job_dir, "alignment_ref.npz")
    ref = np.load(ref_path, allow_pickle=False)
    classes = ref["voxel_classes"]
    meshsize = float(ref["meshsize"])
    gl = ref["ground_level"]
    meta = json.loads(str(ref["meta_json"]))

    path = newest_topo_surf(job_dir)
    print(f"reference : {ref_path}  classes {classes.shape}  meshsize {meshsize}")
    print(f"palm      : {path}")

    ds = netCDF4.Dataset(path)
    topo = np.ma.filled(ds["topo_all"][:], -9999)
    z = np.ma.filled(ds["z"][:], np.nan)
    dz = float(z[2] - z[1]) if z.size >= 3 else meshsize
    if abs(dz - meshsize) > 1e-3 * meshsize:
        raise SystemExit(f"PALM dz {dz} disagrees with the reference meshsize {meshsize}")

    ny, nx = topo.shape[1], topo.shape[2]
    if (ny, nx) != classes.shape[:2]:
        raise SystemExit(f"shape mismatch: PALM (y,x)=({ny},{nx}) vs reference "
                         f"{classes.shape[:2]}")

    # Probe columns chosen to span the three things the encoding question
    # turns on: bare terrace terrain, a building, and a water cell.
    probe(ds, topo, classes, gl, dz,
          [(10, 3, "pure terrain, low terrace"),
           (11, 16, "tower"),
           (2, 3, "water, channel bed")])

    bad = self_consistency(ds, topo, dz)
    print("\n=== FILE SELF-CONSISTENCY (topo_all vs the file's own zt/buildings_2d) ===")
    if bad:
        print(f"  {len(bad)} internal disagreement(s) -- this topo_surf file is NOT "
              "trustworthy; see this function's docstring.")
        for line in bad[:20]:
            print("   ", line)
        if len(bad) > 20:
            print(f"    ... and {len(bad) - 20} more")
    else:
        print("  clean: topo_all reproduces the file's own zt and buildings_2d "
              "for every column")

    want_solid, want_building, want_tree = voxel_counts(classes)
    got_solid = (topo != TOPO_AIR).sum(axis=0).astype(int)
    got_building = (topo == TOPO_BUILDING).sum(axis=0).astype(int)
    if (topo == TOPO_UNCLASSIFIED).any():
        n = int((topo == TOPO_UNCLASSIFIED).sum())
        print(f"\n  note: {n} cell(s) are topo_all == 3 (non-classified topography); "
              "counted as solid but neither terrain nor building")

    # Buildings are always counted from topo_all, never from the file's own
    # buildings_3d variable, even when PALM writes one. PALM's OUTPUT
    # buildings_3d is TERRAIN-RELATIVE and its lowest slot is a DUPLICATE:
    # it writes index k - k_ttop_ij for k = k_ttop_ij + 1 .. and then
    # "Set lowest grid point in output ... output(nzb) = output(nzb+1)"
    # (data_output_topo_and_surface_setup_mod.f90), so every ground-mounted
    # building column reads one cell taller than it is. topo_all is the
    # unambiguous field -- PALM's own comment: "in contrast to output of
    # buildings, output of entire topography is relative to absolute
    # coordinates" -- and it is what the voxel -3 count is comparable with.
    got_b3 = got_building
    if "buildings_3d" in ds.variables:
        b3 = np.ma.filled(ds["buildings_3d"][:], 0).astype(int)
        b3_note = ("topo_all == 1 -- the driver is LOD2 (buildings_3d present, "
                   f"max column {int((b3 == 1).sum(axis=0).max())} incl. PALM's "
                   "duplicated lowest output slot, not counted here)")
    else:
        b3_note = ("topo_all == 1 -- the driver is LOD1, so PALM writes no "
                   "buildings_3d")

    if "lad" in ds.variables:
        lad = np.ma.filled(ds["lad"][:], 0.0)
        got_tree = (lad > 0).sum(axis=0).astype(int)
    else:
        got_tree = np.zeros((ny, nx), dtype=int)

    checks = [("solid", want_solid, got_solid),
              ("building", want_building, got_b3),
              ("canopy", want_tree, got_tree)]
    print(f"\n=== PER-COLUMN COUNTS ({ny * nx} columns) ===")
    print(f"  building check reads {b3_note}")

    residue, hard = [], []
    for kind, want, got in checks:
        diff = np.argwhere(want != got)
        print(f"  {kind:9s}: {len(diff)} mismatching column(s) of {ny * nx}")
        for y, x in diff:
            cls, why = classify(int(y), int(x), int(want[y, x]), int(got[y, x]),
                                gl, kind)
            row = (cls, kind, int(y), int(x), int(want[y, x]), int(got[y, x]), why)
            (residue if cls == "PALM-FILTER RESIDUE" else hard).append(row)

    if residue or hard:
        print("\n=== MISMATCHES ===")
        for cls, kind, y, x, want_v, got_v, why in residue + hard:
            print(f"  [{cls}] {kind} (y={y}, x={x}) want={want_v} got={got_v}  -- {why}")
    print(f"\nPALM-FILTER RESIDUE: {len(residue)}   HARD: {len(hard)}")

    measure_z_offset(topo, gl, meta)
    ds.close()

    if hard:
        print("\nVERDICT: MISALIGNED -- hard mismatches above.")
        return 1
    if bad:
        print("\nVERDICT: INCONCLUSIVE -- the per-column comparison found no hard "
              "mismatch, but the topo_surf file failed its own self-consistency "
              "gate, so it is not evidence either way. Re-stage and re-run the "
              "job on ONE rank (cores=1, palmrun -X 1); the corruption is a "
              "multi-rank DATA_TOPO_SURF write artefact, not a model result.")
        return 1
    print("\nVERDICT: ALIGNED -- PALM's discretized world matches the voxcity voxel "
          "grid cell-for-cell in every column"
          + (f", modulo {len(residue)} filter-residue column(s)." if residue else "."))
    return 0


if __name__ == "__main__":
    sys.exit(main())
