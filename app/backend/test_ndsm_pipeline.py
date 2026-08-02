"""Tests for the app-level nDSM canopy refinement entry point.

The classifier itself is covered by ``test_ndsm_refine.py``; what is at stake
here is the wiring: which evidence reaches the classifier, which grid-writing
entrypoint the refined canopy leaves through, and -- above all -- that every
skip path leaves the model bitwise untouched. Half-applying a refinement is
worse than not applying it: the 2.5-D component grids would describe crowns the
voxel grid does not contain, for the rest of the session, reported only in a
server log.

No COG is read: ``load_ndsm_evidence`` is stubbed, so the reader's own tests own
the raster and these tests own the pipeline.
"""
import inspect
import sys
import types

import numpy as np
import pytest

import backend.ndsm_pipeline as pipeline
from backend.ndsm_refine import RefineParams

# Captured at import time, *before* any fixture stubs ``sys.modules``. An
# ``importorskip`` inside a test using the ``env`` fixture would resolve to the
# fake package and make the signature contract below vacuous.
try:
    from voxcitygml import reapply_canopy as _real_reapply_canopy
except ImportError:  # package predates canopy re-apply, or is not installed
    _real_reapply_canopy = None

# Axis-aligned rectangle near Tokyo ([SW, NW, NE, SE], [lon, lat]). Never read
# by anything here -- load_ndsm_evidence is stubbed -- but passed through so the
# call shape stays the production one.
RECT = [
    [139.770, 35.646], [139.770, 35.650],
    [139.775, 35.650], [139.775, 35.646],
]
LC_SOURCE = "OpenEarthMapJapan"
COG = "/nonexistent/ndsm_cog.tif"

# Index of "Tree" in _LC_CLASSES. Non-zero here, but the zero case has its own
# test: a land cover whose tree class *is* index 0 exists (ESA WorldCover).
_TREE_ID = 1
_LC_CLASSES = {"bareland": "Bareland", "tree": "Tree"}
# Non-square on purpose: a rows/cols mix-up cannot hide behind it.
SHAPE = (6, 4)
STATIC = 10.0
NDSM_HEIGHT = 8.0
# The trunk ratio the pipeline derives crown bases with.
TRUNK_RATIO = 11.76 / 19.98


def _asymmetric_land_cover():
    """Tree cells in the high row indices only -- so a mirror would show.

    An all-tree grid is flip-invariant, which is exactly why a mirrored-canopy
    defect survived earlier tests in this project.
    """
    lc = np.zeros(SHAPE, dtype=np.int16)
    lc[SHAPE[0] // 2:, :] = _TREE_ID
    return lc


def _expected_canopy(height=NDSM_HEIGHT):
    """The canopy for the default scene: *height* at tree cells, 0 elsewhere.

    Derived from ``_asymmetric_land_cover`` rather than restating its layout, so
    that flattening the fixture trips the non-vacuity guard in the mirror test
    instead of failing an unrelated equality first.
    """
    return np.where(_asymmetric_land_cover() == _TREE_ID, height, 0.0)


class _Model:
    """Stand-in VoxCity carrying exactly the grids the pipeline reads."""

    def __init__(self, building_lod, land_cover=None):
        self.land_cover = types.SimpleNamespace(
            classes=_asymmetric_land_cover() if land_cover is None else land_cover)
        self.buildings = types.SimpleNamespace(
            heights=np.zeros(SHAPE, dtype=float))
        self.tree_canopy = types.SimpleNamespace(
            top=np.zeros(SHAPE, dtype=float),
            bottom=np.zeros(SHAPE, dtype=float))
        self.voxels = types.SimpleNamespace(
            classes=np.zeros((*SHAPE, 3), dtype=np.int16))
        self.extras = {"building_lod": building_lod}


def _snapshot(vc):
    """Every grid in the model, copied. Compared bitwise by _assert_unchanged.

    Spot-checking one grid is how a partial write survives review; this is why
    the skip-path tests compare all of them.
    """
    return {
        "land_cover": vc.land_cover.classes.copy(),
        "buildings": vc.buildings.heights.copy(),
        "canopy_top": vc.tree_canopy.top.copy(),
        "canopy_bottom": vc.tree_canopy.bottom.copy(),
        "voxels": vc.voxels.classes.copy(),
    }


def _assert_unchanged(vc, snap):
    for key, value in _snapshot(vc).items():
        assert np.array_equal(value, snap[key]), f"{key} was modified"


def _evidence(*, height=None, degraded=False, shape=None, canopy_like=True,
              spread=3.0):
    """A load_ndsm_evidence() return value.

    ``canopy_like`` puts every cell comfortably past the canopy thresholds, so
    the default scene keeps its nDSM heights untouched and any change in the
    written-back canopy is the pipeline's doing, not the classifier's. That
    includes a crown-like ``spread``: the reader returns one whether or not the
    evidence bands exist, so it is never None here either.
    """
    if height is None:
        height = np.full(SHAPE, NDSM_HEIGHT, dtype=float)
    if degraded:
        evidence = dict(mrf=None, roughness=None, n_all=None, n_nonground=None)
    elif canopy_like:
        evidence = dict(
            mrf=np.full(SHAPE, 0.6),
            roughness=np.full(SHAPE, 2.5),
            n_all=np.full(SHAPE, 40.0),
            n_nonground=np.full(SHAPE, 20.0),
        )
    else:
        evidence = dict(
            mrf=np.full(SHAPE, 0.25),
            roughness=np.full(SHAPE, 1.2),
            n_all=np.full(SHAPE, 40.0),
            n_nonground=np.full(SHAPE, 20.0),
        )
    return dict(
        height=height,
        spread=(spread if isinstance(spread, np.ndarray)
                else np.full(height.shape, spread, dtype=float)),
        degraded=degraded,
        shape=tuple(height.shape) if shape is None else shape,
        **evidence,
    )


@pytest.fixture
def env(monkeypatch):
    """Record which grid-writing entrypoint the refinement reaches.

    Both are stubbed, so a test can assert on the one that was *not* called --
    which is the whole point: calling the wrong one is the bug. Arrays handed to
    the stubs are copied on capture, never held by reference: the real
    reapply_canopy writes through the model's grids, so a retained reference
    would report post-call state.
    """
    calls = {"evidence": _evidence()}

    monkeypatch.setattr(pipeline, "get_land_cover_classes",
                        lambda source: _LC_CLASSES)
    monkeypatch.setattr(
        pipeline, "load_ndsm_evidence",
        lambda verts, meshsize, cog_path, **kw: calls["evidence"])

    def fake_regenerate(obj, **kwargs):
        calls["regenerate"] = kwargs

    monkeypatch.setattr(pipeline, "regenerate_voxels", fake_regenerate)

    fake_pkg = types.ModuleType("voxcitygml")

    def fake_reapply(*args, **kwargs):
        def snap(value):
            return value.copy() if isinstance(value, np.ndarray) else value

        calls["reapply"] = {
            "args": tuple(snap(a) for a in args),
            "kwargs": {k: snap(v) for k, v in kwargs.items()},
        }

    fake_pkg.reapply_canopy = fake_reapply
    monkeypatch.setitem(sys.modules, "voxcitygml", fake_pkg)
    return calls


def _refine(vc, **kwargs):
    kwargs.setdefault("static_tree_height", STATIC)
    return pipeline.refine_canopy_with_ndsm(
        vc, RECT, 5.0, LC_SOURCE, COG, **kwargs)


def _overlay(calls):
    """The captured reapply_canopy call, resolved to parameter names.

    Accepts either calling convention, so a positional/keyword switch in the
    pipeline is not a spurious failure here;
    ``TestTheCrossRepoContract`` below pins the call shape against the real
    function's signature.
    """
    names = ("city", "canopy_top", "canopy_bottom", "trunk_height_ratio")
    resolved = dict(zip(names, calls["reapply"]["args"]))
    resolved.update(calls["reapply"]["kwargs"])
    return resolved


class TestWriteBackRouting:
    def test_lod1_rebuilds_the_voxel_grid(self, env):
        """LOD1 voxels *are* extruded footprints, so regenerating them from the
        revised 2.5-D grids is the correct -- and only -- way to apply canopy."""
        vc = _Model(building_lod=1)
        assert _refine(vc) is True

        assert "regenerate" in env
        assert env["regenerate"]["inplace"] is True
        assert env["regenerate"]["land_cover_source"] == LC_SOURCE
        assert "reapply" not in env
        assert np.array_equal(vc.tree_canopy.top, _expected_canopy())

    def test_lod2_overlays_instead_of_rebuilding(self, env):
        """LOD2 must take reapply_canopy, which clears and rewrites only canopy
        voxels. regenerate_voxels here rebuilds from the 2.5-D grids and would
        discard the mesh-voxelized roof/wall geometry the mode exists to
        produce. Measured in ``test_ndsm_lod2_geometry.py``, which runs both
        routes over one real Tsukiji model: the rebuild takes roof slope
        fraction 0.2868 -> 0.2039 and building voxels 117,270 -> 89,053, while
        the overlay leaves both bitwise unchanged."""
        vc = _Model(building_lod=2)
        assert _refine(vc) is True

        assert "reapply" in env
        assert "regenerate" not in env, (
            "regenerate_voxels rebuilds from the 2.5-D grids and flattens LOD2")
        overlay = _overlay(env)
        assert overlay["city"] is vc
        assert np.array_equal(overlay["canopy_top"], _expected_canopy())
        assert overlay.get("canopy_bottom") is not None, (
            "voxcitygml's own default trunk ratio currently equals ours, so "
            "passing the ratio instead would yield the same array today -- "
            "passing the derived bottom keeps that a coincidence we do not "
            "depend on")
        assert np.allclose(overlay["canopy_bottom"],
                           _expected_canopy() * TRUNK_RATIO)

    def test_lod2_leaves_the_component_grids_to_reapply_canopy(self, env):
        """On the LOD2 path reapply_canopy owns the whole update: it writes
        canopy_top/canopy_bottom into tree_canopy itself. It also *raises* on a
        missing extras['voxel_min_z'] or a mismatched mesh mask, and refinement
        is non-fatal upstream -- so pre-writing the component grids would leave
        them describing crowns the voxel grid never received."""
        vc = _Model(building_lod=2)
        assert _refine(vc) is True
        assert not vc.tree_canopy.top.any(), (
            "the pipeline wrote the canopy grids itself; reapply_canopy owns "
            "them on this path and may still raise")
        assert not vc.tree_canopy.bottom.any()

    def test_lod2_leaves_the_model_untouched_when_the_overlay_raises(self, env):
        """reapply_canopy raises ValueError on a missing extras['voxel_min_z'],
        a mismatched mesh_vegetation_mask, or a canopy shape mismatch. The
        endpoint treats refinement failures as non-fatal, so a half-written
        model would sail on with its 2.5-D grids describing crowns the voxel
        grid does not contain -- for the rest of the session, reported only in a
        server log. On this path reapply_canopy owns those grids, so the
        pipeline writes nothing of its own before calling it.
        """
        def boom(*a, **k):
            raise ValueError("extras['voxel_min_z'] is missing or None")

        sys.modules["voxcitygml"].reapply_canopy = boom
        vc = _Model(building_lod=2)
        snap = _snapshot(vc)

        with pytest.raises(ValueError):
            _refine(vc)
        _assert_unchanged(vc, snap)
        assert "regenerate" not in env, (
            "a failed overlay must not fall back to the LOD2-flattening rebuild")

    def test_lod2_hands_over_the_canopy_at_component_resolution(self, env):
        """The canopy must keep land_cover's shape, not the voxel grid's.

        reapply_canopy resamples a mismatched canopy itself -- as it already
        does for the DEM -- but it *stores* what the caller passed. Pre-
        resampling to the voxel shape would leave tree_canopy.top at voxel
        resolution while land_cover, dem and buildings stayed at component
        resolution: the 2.5-D grids desync, and a later update_voxcity raises
        "Grid shape mismatch". Resampling is the package's job.
        """
        vc = _Model(building_lod=2)
        vc.voxels = types.SimpleNamespace(
            classes=np.zeros((SHAPE[0] * 2, SHAPE[1] * 2, 3), dtype=np.int16))

        assert _refine(vc) is True
        overlay = _overlay(env)
        assert overlay["canopy_top"].shape == SHAPE
        assert overlay["canopy_bottom"].shape == SHAPE
        assert vc.land_cover.classes.shape == SHAPE

    def test_lod2_hands_over_the_canopy_unreoriented(self, env):
        """The canopy reaches reapply_canopy in the frame it was built in.

        It is built from land_cover.classes, and voxcitygml returns a model that
        shares one frame throughout -- DEM, canopy and voxel grid included -- so
        a flip here would land every crown mirrored across the target.
        """
        vc = _Model(building_lod=2)
        assert _refine(vc) is True

        got = _overlay(env)["canopy_top"]
        assert np.array_equal(got, _expected_canopy())
        assert not np.array_equal(got, np.flipud(_expected_canopy())), (
            "the canopy was flipped out of the land-cover frame on the way to "
            "reapply_canopy")


class TestSkipPathsLeaveTheModelAlone:
    def test_lod2_skips_when_reapply_canopy_is_unavailable(self, env):
        """An older voxcitygml has no reapply_canopy. Refinement is optional and
        non-fatal, so it degrades to a no-op -- but it must NOT fall back to
        regenerate_voxels, which is exactly the rebuild the missing entrypoint
        exists to avoid."""
        del sys.modules["voxcitygml"].reapply_canopy
        vc = _Model(building_lod=2)
        snap = _snapshot(vc)

        assert _refine(vc) is False
        assert "regenerate" not in env, (
            "no silent fallback: rebuilding is what LOD2 must never do")
        assert "reapply" not in env
        _assert_unchanged(vc, snap)

    def test_missing_evidence_is_a_no_op(self, env, monkeypatch):
        """load_ndsm_evidence returns None when the COG is missing or does not
        overlap the target. That is a skip, never an error."""
        monkeypatch.setattr(pipeline, "load_ndsm_evidence",
                            lambda *a, **k: None)
        vc = _Model(building_lod=1)
        snap = _snapshot(vc)

        assert _refine(vc) is False
        assert "regenerate" not in env
        assert "reapply" not in env
        _assert_unchanged(vc, snap)

    @pytest.mark.parametrize("building_lod", [1, 2])
    def test_shape_mismatch_is_a_no_op(self, env, building_lod):
        """Both grids come from the same rectangle and meshsize, so a mismatch
        means something upstream is wrong. Resampling here would paper over it
        and silently put the canopy in the wrong places."""
        env["evidence"] = _evidence(
            height=np.full((SHAPE[0] + 1, SHAPE[1]), NDSM_HEIGHT))
        vc = _Model(building_lod=building_lod)
        snap = _snapshot(vc)

        assert _refine(vc) is False
        assert "regenerate" not in env
        assert "reapply" not in env
        _assert_unchanged(vc, snap)

    def test_shape_mismatch_is_taken_from_the_declared_shape(self, env):
        """The reader reports its grid size in ``shape``; a height array that
        disagrees with it is the same upstream fault and must not slip past."""
        env["evidence"] = _evidence(shape=(SHAPE[0] + 1, SHAPE[1]))
        vc = _Model(building_lod=1)
        snap = _snapshot(vc)

        assert _refine(vc) is False
        _assert_unchanged(vc, snap)

    def test_shape_mismatch_is_also_taken_from_the_height_grid(self, env):
        """The declared ``shape`` agreeing with the model is not enough.

        Isolates the second half of the guard: today's reader builds ``height``
        by reshaping to the very tuple it reports, so the two can only disagree
        if the evidence came from somewhere else -- and it is ``height``, not
        the declared shape, that decides everything downstream
        (``refine_from_evidence`` takes its working shape from ``height.shape``,
        and the canopy it returns is what gets written into the component
        grids). Without this case the whole ``height`` clause can be deleted
        with the suite still green: the other two mismatch tests both break
        ``shape`` as well.
        """
        env["evidence"] = _evidence(
            height=np.full((SHAPE[0] + 1, SHAPE[1]), NDSM_HEIGHT),
            shape=SHAPE)
        vc = _Model(building_lod=1)
        snap = _snapshot(vc)

        assert _refine(vc) is False, (
            "a height grid that does not match the model must be an announced "
            "no-op, not an exception the caller's non-fatal handler swallows")
        assert "regenerate" not in env
        _assert_unchanged(vc, snap)


    def test_a_source_with_no_tree_class_is_a_no_op(self, env, monkeypatch,
                                                    capsys):
        """A source with no tree class must skip, not guess an index.

        The land cover here is a *treeless* source whose index 4 is "Grassland",
        and the grid's southern half is index 4 -- so the guess this replaces
        (``_TREE_ID_FALLBACK = 4``, the id of "Tree" in OpenEarthMapJapan and
        nowhere else) writes nDSM crowns straight onto grassland and trips
        ``_assert_unchanged``. That is the same shape of damage the ESA
        WorldCover case did on buildings, reached from the other direction.

        The ``is False`` and ``"regenerate" not in env`` halves are load-bearing
        on their own: ``land_cover_grid == None`` is an all-False mask, so an
        unguarded ``None`` produces a zero canopy, writes it back over the zeros
        already there, rebuilds the voxels and returns ``True`` -- a no-op that
        reports success and that a bitwise comparison alone would not catch.
        """
        monkeypatch.setattr(pipeline, "get_land_cover_classes",
                            lambda source: {"bareland": "Bareland",
                                            "water": "Water",
                                            "built_up": "Built-up",
                                            "road": "Road",
                                            "grassland": "Grassland",
                                            "snow": "Snow"})
        lc = np.zeros(SHAPE, dtype=np.int16)
        lc[SHAPE[0] // 2:, :] = 4          # "Grassland" -- the guessed id
        vc = _Model(building_lod=1, land_cover=lc)
        snap = _snapshot(vc)

        assert _refine(vc) is False
        assert "regenerate" not in env
        assert "reapply" not in env
        _assert_unchanged(vc, snap)
        assert "no tree class" in capsys.readouterr().out, (
            "an optional refinement that skips must say why; this is the only "
            "report the caller gets")


class TestCanopyValues:
    def test_tree_cells_without_a_height_get_the_static_height(self, env):
        """No nDSM height and no credible neighbour to borrow from: the tree
        must still be a tree, at the caller's static height."""
        height = np.full(SHAPE, np.nan)
        env["evidence"] = _evidence(height=height)
        vc = _Model(building_lod=1)

        assert _refine(vc) is True
        tree = vc.land_cover.classes == _TREE_ID
        assert np.all(vc.tree_canopy.top[tree] == STATIC)
        assert np.all(vc.tree_canopy.top[~tree] == 0.0)

    def test_a_tree_cell_the_classifier_left_unfilled_gets_the_static_height(
            self, env, monkeypatch):
        """The classifier fills every tree cell, so this is a backstop -- but a
        NaN reaching the component grids would propagate into the voxel grid
        with nothing to say so, which is why the backstop is there at all.
        Driven by stubbing the classifier, since no input reaches it."""
        holed = _expected_canopy()
        holed[5, 0] = np.nan

        def fake_refine(evidence, **kwargs):
            return {"canopy": holed, "verdict": np.zeros(SHAPE, np.int8),
                    "counts": {"tree": 12}, "degraded": False}

        monkeypatch.setattr(pipeline, "refine_from_evidence", fake_refine)
        vc = _Model(building_lod=1)

        assert _refine(vc) is True
        assert vc.tree_canopy.top[5, 0] == STATIC
        assert np.isfinite(vc.tree_canopy.top).all()

    def test_canopy_bottom_is_the_trunk_ratio_of_the_top(self, env):
        vc = _Model(building_lod=1)
        assert _refine(vc) is True
        assert np.allclose(vc.tree_canopy.bottom,
                           vc.tree_canopy.top * TRUNK_RATIO)
        assert np.all(vc.tree_canopy.bottom <= vc.tree_canopy.top), (
            "a crown base above its own top is a degenerate voxel column")

    def test_the_bottom_never_rises_above_a_negative_top(self):
        """The ratio is applied through a min(), not by bare multiplication: for
        a negative top the product is *larger* than the top, and a crown base
        above its own top is a degenerate voxel column.

        Tested on the derivation directly. The classifier's below-minimum guard
        currently lifts every negative tree height before it gets here, so a
        scene-level test of this would pass with the min() removed -- it is a
        guard against the composition changing, not against today's data.
        """
        canopy = np.array([-3.0, 0.0, 8.0])
        bottom = pipeline._canopy_bottom(canopy)
        assert np.all(bottom <= canopy)
        assert bottom[2] == pytest.approx(8.0 * TRUNK_RATIO)

    def test_canopy_bottom_is_created_when_the_model_has_none(self, env):
        """Some models arrive with tree_canopy.bottom unset."""
        vc = _Model(building_lod=1)
        vc.tree_canopy.bottom = None
        assert _refine(vc) is True
        assert vc.tree_canopy.bottom is not None
        assert np.allclose(vc.tree_canopy.bottom,
                           vc.tree_canopy.top * TRUNK_RATIO)


class TestDegradedMode:
    def test_degraded_evidence_runs_footprint_only_end_to_end(self, env):
        """The COG on disk is single-band today, so this is the live path.

        Every cell is ambiguous: cells away from a footprint keep their nDSM
        height, and a cell whose height coincides with an adjacent building
        takes the local median of credible trees -- not a flat constant, which
        is the false cap this whole module exists to remove.
        """
        height = np.full(SHAPE, NDSM_HEIGHT)
        height[5, 0] = 24.0                  # the roof-leakage suspect
        env["evidence"] = _evidence(height=height, degraded=True)
        vc = _Model(building_lod=1)
        vc.buildings.heights[5, 1] = 25.0    # adjacent footprint, coincident

        assert _refine(vc) is True
        assert vc.tree_canopy.top[5, 0] == NDSM_HEIGHT, (
            "the coincident cell should have taken the median of its credible "
            "tree neighbours, which all sit at the nDSM height")
        assert vc.tree_canopy.top[3, 3] == NDSM_HEIGHT, (
            "a tree far from any footprint must be left alone in degraded mode")

    def test_degraded_mode_is_announced_once(self, env, capsys):
        env["evidence"] = _evidence(degraded=True)
        assert _refine(_Model(building_lod=1)) is True
        out = capsys.readouterr().out.lower()
        assert out.count("degraded") == 1, (
            "one line per generation, not one per cell")

    def test_the_verdict_counts_are_logged(self, env, capsys):
        assert _refine(_Model(building_lod=1)) is True
        out = capsys.readouterr().out
        assert "tree cells" in out
        n_tree = int((_asymmetric_land_cover() == _TREE_ID).sum())
        assert f"{n_tree} tree cells" in out

    def test_the_spread_distribution_is_logged_in_degraded_mode(
            self, env, capsys):
        """The single-band COG on disk is the only place the spread can be
        measured on real data before the evidence rebuild ships, and the
        measurement is what calibrates ``spread_max_m``. The veto is inert here;
        the *reporting* is not."""
        env["evidence"] = _evidence(degraded=True, spread=7.25)
        assert _refine(_Model(building_lod=1)) is True
        out = capsys.readouterr().out
        assert "pixel spread" in out
        assert "adjacent" in out
        assert "7.25" in out

    def test_no_spread_means_no_spread_log(self, env, capsys):
        """A reader that predates the signal must not make the pipeline print an
        empty stanza."""
        env["evidence"] = _evidence(degraded=True)
        del env["evidence"]["spread"]
        assert _refine(_Model(building_lod=1)) is True
        assert "pixel spread" not in capsys.readouterr().out


class TestDegenerateBuildingHeights:
    """Footprint adjacency is what makes the ambiguous bucket safe: without a
    building nearby a cell is never replaced. An all-zero building grid
    therefore makes *every* tree cell keep its nDSM height -- spikes included --
    while the counts still look healthy, since "kept" is a legitimate outcome.
    The failure is silent by construction, so it is announced.
    """

    @pytest.mark.parametrize("building_lod", [1, 2])
    def test_an_all_zero_building_grid_is_announced(
            self, env, capsys, building_lod):
        vc = _Model(building_lod=building_lod)          # heights are all zero
        assert _refine(vc) is True
        out = capsys.readouterr().out.lower()
        assert "building" in out and "no footprint" in out

    def test_a_grid_with_footprints_says_nothing(self, env, capsys):
        vc = _Model(building_lod=1)
        vc.buildings.heights[0, 0] = 12.0
        assert _refine(vc) is True
        assert "no footprint" not in capsys.readouterr().out.lower()

    def test_a_model_with_no_tree_cells_says_nothing(self, env, capsys):
        """Nothing is at risk when there is nothing to refine, and a warning
        every generation is a warning nobody reads."""
        vc = _Model(building_lod=1,
                    land_cover=np.zeros(SHAPE, dtype=np.int16))
        assert _refine(vc) is True
        assert "no footprint" not in capsys.readouterr().out.lower()


class TestInputs:
    def test_static_tree_height_below_the_minimum_is_lifted(self, env, capsys):
        """A static height under ``min_tree_height_m`` cannot satisfy the
        guard it is substituted by: the guard fires, installs the fallback, and
        the fallback is itself below the minimum. The pipeline takes a
        user-supplied value, so it lifts it rather than 500-ing an optional
        refinement."""
        env["evidence"] = _evidence(height=np.full(SHAPE, np.nan))
        params = RefineParams()
        vc = _Model(building_lod=1)

        assert _refine(vc, static_tree_height=0.5, params=params) is True
        tree = vc.land_cover.classes == _TREE_ID
        assert np.all(vc.tree_canopy.top[tree] == params.min_tree_height_m)
        assert "0.5" in capsys.readouterr().out, (
            "silently substituting a different height is how this went "
            "unnoticed in the first place")

    def test_esa_worldcover_really_puts_trees_at_zero_and_builtup_at_four(self):
        """Pin the premise against the real source, not just the fixture below.

        The next test hand-writes ESA WorldCover's ordering. If voxcity ever
        reorders that source the fixture stops modelling it, and the regression
        test would be asserting about nothing while staying green.
        """
        from voxcity.utils import get_land_cover_classes

        names = list(get_land_cover_classes("ESA WorldCover").values())
        assert names.index("Trees") == 0
        assert names.index("Built-up") == 4

    def test_a_tree_class_at_index_zero_is_not_lost(self, env, monkeypatch):
        """ESA WorldCover really does put 'Trees' at index 0. Resolving the id
        with a truthiness chain sends that source to the fallback id 4, which is
        'Built-up' there -- the canopy would be written onto the buildings."""
        classes = {"trees": "Trees", "shrubland": "Shrubland",
                   "grassland": "Grassland", "cropland": "Cropland",
                   "built_up": "Built-up"}
        monkeypatch.setattr(pipeline, "get_land_cover_classes",
                            lambda source: classes)
        lc = np.full(SHAPE, 4, dtype=np.int16)   # all "Built-up"
        lc[SHAPE[0] // 2:, :] = 0                # ... and these are the trees
        vc = _Model(building_lod=1, land_cover=lc)

        assert _refine(vc) is True
        assert np.array_equal(vc.tree_canopy.top,
                              np.where(lc == 0, NDSM_HEIGHT, 0.0))

    def test_the_cog_path_is_forwarded_to_the_reader(self, env, monkeypatch):
        """The caller owns the path -- the pipeline must not reach for a module
        constant of its own."""
        seen = {}

        def spy(verts, meshsize, cog_path, **kwargs):
            seen.update(verts=verts, meshsize=meshsize, cog_path=cog_path)
            return env["evidence"]

        monkeypatch.setattr(pipeline, "load_ndsm_evidence", spy)
        assert _refine(_Model(building_lod=1)) is True
        assert seen["cog_path"] == COG
        assert seen["verts"] == RECT
        assert seen["meshsize"] == 5.0

    def test_buildings_are_read_from_the_model_unreoriented(self, env):
        """Tree cells are compared against nearby building heights, so both
        grids must share a frame -- and in every voxcity model they do. A flip
        of either would test each tree against the buildings on the opposite
        side of the target."""
        height = np.full(SHAPE, 24.0)
        env["evidence"] = _evidence(height=height, degraded=True)
        vc = _Model(building_lod=1)
        # A footprint under the *southern* half only, where there are no trees.
        vc.buildings.heights[:SHAPE[0] // 2, :] = 25.0

        assert _refine(vc) is True
        tree = vc.land_cover.classes == _TREE_ID
        assert np.all(vc.tree_canopy.top[tree] == 24.0), (
            "the buildings were flipped into the tree rows: every tree was "
            "treated as roof leakage and replaced")


class TestResolveTreeId:
    """The resolver on its own, over a source's ordered class names.

    The id *is* the position in that ordered list, which is what makes index 0
    a real answer rather than a missing one -- and what made the chain this
    replaced (``get("Tree") or get("Trees") or ... or 4``) send ESA WorldCover,
    whose tree class is index 0, to a hard-coded 4 that is ``Built-up`` there.

    The integration consequences are pinned above and in
    ``TestSkipPathsLeaveTheModelAlone``; these cases pin the resolver's own
    contract, including the two halves that are invisible from a single
    land-cover source: the preference order between the three accepted names,
    and returning ``None`` rather than guessing.
    """

    @staticmethod
    def _classes(*names):
        """A ``get_land_cover_classes()`` return value listing *names* in order.

        The real function is keyed by RGB colour and the resolver reads only
        ``.values()``, so the keys are placeholders. The *order* is the whole
        content of these fixtures.
        """
        return {(i, i, i): name for i, name in enumerate(names)}

    @pytest.mark.parametrize("names, expected", [
        # Each accepted name, alone, at index 0.
        (["Tree"], 0),
        (["Trees"], 0),
        (["Tree Canopy"], 0),
        # ... and away from index 0, so the cases above are not passing by
        # agreeing with a hard-coded zero.
        (["Bareland", "Water", "Tree"], 2),
        # Preference order is over the *names*, not their positions: "Tree" is
        # checked first and wins from further down the list than "Trees".
        (["a", "b", "c", "Trees", "d", "Tree"], 5),
        # Index 0 is a real answer -- it must win over a later accepted name
        # rather than read as "missing, keep looking".
        (["Trees", "a", "b", "c", "d", "e", "f", "Tree Canopy"], 0),
        # No tree class -> no guess. The second list is six classes long, so a
        # fallback of 4 would return "Grassland": an in-range, plausible-looking
        # id, which is exactly how the guess did its damage unnoticed.
        (["Bareland"], None),
        (["Bareland", "Water", "Built-up", "Road", "Grassland", "Snow"], None),
    ])
    def test_resolve_tree_id(self, monkeypatch, names, expected):
        monkeypatch.setattr(pipeline, "get_land_cover_classes",
                            lambda source: self._classes(*names))
        assert pipeline._resolve_tree_id("any source") == expected


class TestTheCrossRepoContract:
    @pytest.mark.skipif(_real_reapply_canopy is None,
                        reason="voxcitygml with reapply_canopy is not installed")
    def test_the_overlay_call_matches_the_real_reapply_canopy_signature(
            self, env):
        """Bind the LOD2 overlay call against the *real* voxcitygml signature.

        ``env`` stubs reapply_canopy with a fake that swallows any argument
        list, so every other assertion in this file would stay green while
        production raised TypeError. This replays the captured call -- verbatim,
        keywords as keywords -- against the real signature, captured at import
        time before the fixture patched ``sys.modules``.

        This is the only test in the repo that pins that contract; a
        reapply_canopy rename or arity change in VoxCityGML is otherwise
        invisible until a generation fails.
        """
        vc = _Model(building_lod=2)
        assert _refine(vc) is True
        raw = env["reapply"]

        # bind() raises TypeError on an unknown/renamed keyword or a changed
        # arity.
        bound = inspect.signature(_real_reapply_canopy).bind(
            *raw["args"], **raw["kwargs"])
        # bind() alone is not enough: a *positional* argument binds to whatever
        # the parameter is now called, so a rename of canopy_top would pass
        # silently. Assert the names the call actually resolves to.
        assert bound.arguments.keys() == {"city", "canopy_top", "canopy_bottom"}
