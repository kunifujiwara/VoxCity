"""App entry point for evidence-based nDSM canopy refinement.

Reads the nDSM evidence once, classifies every land-cover tree cell with
:mod:`backend.ndsm_refine`, and writes the refined canopy back into a VoxCity
model in place. It is the only place those three steps meet; the classifier
knows nothing about voxcity models and the model knows nothing about rasters.

Separate from ``ndsm_refine`` on purpose: that module is pure array work with no
voxcity, voxcitygml or filesystem dependency, and is tested as such. Everything
that touches a model lives here.

Refinement is *optional*. Every failure mode below returns ``False`` rather than
raising, and -- the part that actually matters -- returns it with the model
bitwise unchanged. A half-applied refinement is worse than none: the 2.5-D
component grids would describe crowns the voxel grid does not contain, for the
rest of the session, reported only in a server log.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from voxcity.generator.update import regenerate_voxels
from voxcity.utils.lc import get_land_cover_classes

from .ndsm_refine import (
    DEFAULT_PARAMS,
    RefineParams,
    format_counts,
    format_spread_stats,
    load_ndsm_evidence,
    refine_from_evidence,
)

__all__ = ["refine_canopy_with_ndsm", "is_lod2_model", "TRUNK_HEIGHT_RATIO"]

#: Crown base as a fraction of crown top. Empirical mean trunk height over mean
#: total height; kept as the ratio it was measured as rather than rounded.
TRUNK_HEIGHT_RATIO = 11.76 / 19.98

#: Land-cover class names that mean "tree", in preference order.
_TREE_CLASS_NAMES = ("Tree", "Trees", "Tree Canopy")
#: Used when none of the above appears in the source's class list. A guess, and
#: the id of "Tree" in OpenEarthMapJapan and nothing else.
_TREE_ID_FALLBACK = 4


def is_lod2_model(voxcity_obj) -> bool:
    """True if this model's voxels came from LOD2 mesh voxelization.

    Such a grid holds true roof/wall geometry that the 2.5-D component grids
    (heights, min-heights, ids) cannot describe, so any ``regenerate_voxels()``
    call would silently replace it with extruded footprints -- i.e. downgrade it
    to LOD1. Callers that would regenerate must check this first.

    Public, and lives here rather than in ``main``, because it is the predicate
    that decides this module's write-back route -- and ``main`` imports this
    module, so the dependency can only run one way. ``main`` re-uses it for the
    apply-edits guard, which refuses the same rebuild for the same reason.
    """
    extras = getattr(voxcity_obj, "extras", None)
    if not isinstance(extras, dict):
        return False
    try:
        return int(extras.get("building_lod") or 1) >= 2
    except (TypeError, ValueError):
        return False


def _resolve_tree_id(land_cover_source: str) -> int:
    """Class id of the tree class in *land_cover_source*'s class list.

    The id is the class's *position* in the source's ordered class names, which
    is how the land-cover grid encodes it.

    Position 0 is a real answer, not a missing one: ESA WorldCover puts "Trees"
    first. Resolving this with a truthiness chain (``get("Tree") or get("Trees")
    or ... or 4``) therefore sends that source to the fallback, and index 4 there
    is "Built-up" -- the canopy would be written onto the buildings. Hence the
    explicit ``is not None``.
    """
    names = list(get_land_cover_classes(land_cover_source).values())
    index = {name: i for i, name in enumerate(names)}
    for name in _TREE_CLASS_NAMES:
        found = index.get(name)
        if found is not None:
            return int(found)
    return _TREE_ID_FALLBACK


def _canopy_bottom(canopy: np.ndarray) -> np.ndarray:
    """Crown base for each crown top.

    The ``minimum`` is not decoration: for a negative top the scaled value is
    *larger* than the top, and a crown base above its own top is a degenerate
    voxel column. Today the classifier's below-minimum guard lifts negatives
    before they reach here, so this only has to hold if that composition
    changes -- which is exactly when nobody would be looking.
    """
    canopy = np.asarray(canopy, dtype=np.float64)
    return np.minimum(canopy * TRUNK_HEIGHT_RATIO, canopy)


def refine_canopy_with_ndsm(
    voxcity_obj,
    rectangle_vertices: Sequence[Sequence[float]],
    meshsize: float,
    land_cover_source: str,
    cog_path: str,
    static_tree_height: float = 10.0,
    params: Optional[RefineParams] = None,
) -> bool:
    """Refine canopy heights from the nDSM, in place. False when skipped.

    Returns ``False`` -- model untouched -- when the COG is missing or does not
    overlap the target, when the evidence grid does not match the model grid, or
    when an LOD2 model meets a voxcitygml without ``reapply_canopy``. None of
    those is fatal; the caller logs and carries on with static tree heights.

    Args:
        voxcity_obj: the model to refine, mutated in place.
        rectangle_vertices: the target rectangle the model was built from.
        meshsize: the cell size the model was built at.
        land_cover_source: name of the land-cover source, used both to resolve
            the tree class id and, on the LOD1 path, to regenerate voxels.
        cog_path: path to the nDSM COG. Passed in rather than read from config
            so the caller owns which raster is used.
        static_tree_height: height for tree cells the nDSM cannot supply and
            whose neighbourhood holds no credible tree. Lifted to
            ``params.min_tree_height_m`` if it is below it -- see below.
        params: classifier thresholds; :data:`~backend.ndsm_refine.DEFAULT_PARAMS`
            when omitted.
    """
    params = DEFAULT_PARAMS if params is None else params
    land_cover_grid = voxcity_obj.land_cover.classes
    tree_id = _resolve_tree_id(land_cover_source)

    # static_tree_height comes from the request body. Below the classifier's
    # minimum it is unsatisfiable -- classify_and_refine raises rather than
    # produce a grid that violates its own documented floor. Refinement is
    # optional, so lift it here and say so rather than fail the generation.
    static_tree_height = float(static_tree_height)
    if static_tree_height < params.min_tree_height_m:
        print(
            f"[nDSM] static_tree_height {static_tree_height} m is below the "
            f"classifier minimum {params.min_tree_height_m} m — using the "
            "minimum"
        )
        static_tree_height = float(params.min_tree_height_m)

    evidence = load_ndsm_evidence(rectangle_vertices, meshsize, cog_path)
    if evidence is None:
        print(f"[nDSM] No usable nDSM at {cog_path} — skipping canopy refinement")
        return False

    if evidence["degraded"]:
        # The live path until the evidence COG is rebuilt: the raster on disk is
        # single band, so there is no multi-return fraction or roughness to
        # classify with and every cell falls to ambiguous. Still strictly better
        # than the heuristic this replaces -- cells away from a footprint are
        # never touched, and the ones that are get a neighbourhood median rather
        # than a flat constant. One line per generation, not per cell.
        print(
            "[nDSM] Evidence bands absent from the COG — running in degraded "
            "mode: footprint-adjacency only, no canopy/roof evidence"
        )

    # An LOD2 grid is mesh-voxelized: it holds true roof/wall geometry that the
    # 2.5-D component grids cannot describe, so regenerate_voxels() — which
    # rebuilds from those grids — would replace it with extruded footprints.
    # Measured on a Tsukiji target by test_ndsm_lod2_geometry.py, which owns the
    # metric: roof slope fraction 0.2868 -> 0.2039 and building voxels 117,270
    # -> 89,053 when the same model is refined through the rebuild route
    # instead. voxcitygml.reapply_canopy overlays the canopy onto the existing
    # grid instead, clearing and rewriting only canopy voxels — through it both
    # numbers come back bitwise identical.
    #
    # Resolved up front, before anything is mutated: an older voxcitygml has no
    # such entrypoint, and bailing out after the component grids were rewritten
    # would leave them describing crowns the voxel grid does not contain.
    reapply_canopy = None
    if is_lod2_model(voxcity_obj):
        try:
            from voxcitygml import reapply_canopy
        except ImportError as exc:
            # Deliberately no regenerate_voxels() fallback: that rebuild is
            # precisely what the missing entrypoint exists to avoid. Refinement
            # is optional, so degrade to a no-op and let the caller report it.
            print(
                "[nDSM] Skipping canopy refinement: this voxcitygml has no "
                "reapply_canopy(), and rebuilding the voxel grid would flatten "
                f"the LOD2 roof/wall geometry into extruded footprints ({exc}). "
                "Upgrade voxcitygml to use nDSM canopy in LOD2 mode."
            )
            return False

    # Both grids come from the same rectangle and meshsize, so a mismatch means
    # something upstream is wrong. Resampling here would paper over that and put
    # the canopy in subtly wrong places, which is unreviewable after the fact.
    grid_shape = tuple(land_cover_grid.shape)
    if tuple(evidence["shape"]) != grid_shape or \
            tuple(np.shape(evidence["height"])) != grid_shape:
        print(
            f"[nDSM] Evidence grid {evidence['shape']} does not match the model "
            f"grid {grid_shape} — skipping canopy refinement rather than "
            "resampling; both are built from the same rectangle and meshsize, "
            "so this is an upstream fault"
        )
        return False

    tree_mask = land_cover_grid == tree_id
    building_heights = np.asarray(voxcity_obj.buildings.heights, dtype=float)

    # Footprint adjacency is the only thing that can make a tree cell suspect:
    # a cell with no building near it is never replaced, in any mode. So an
    # all-zero building grid does not merely weaken the refinement, it disables
    # the half of it that removes spikes -- and it does so invisibly, because
    # every cell then reports the perfectly legitimate verdict "kept". Checked
    # here, above the LOD1/LOD2 write-back branch, so it covers both routes.
    if np.any(tree_mask) and not np.any(
            np.isfinite(building_heights) & (building_heights > 0)):
        print(
            "[nDSM] WARNING: the building height grid has no footprint at all "
            "(no cell > 0). Footprint adjacency is what marks a tree cell as "
            "suspect, so every tree cell will keep its nDSM height, roof "
            "leakage included — the counts below will still look healthy."
        )

    result = refine_from_evidence(
        evidence,
        tree_mask=tree_mask,
        building_heights=building_heights,
        static_tree_height=static_tree_height,
        params=params,
    )
    canopy = result["canopy"]
    print(f"[nDSM] {format_counts(result['counts'])}")
    # Printed in degraded mode too: the veto is inert without evidence bands,
    # but the distribution is measurable there and is what calibrates
    # spread_max_m. .get() because a stubbed classifier may not supply it.
    spread_report = format_spread_stats(result.get("spread_stats"), "[nDSM] ")
    if spread_report:
        print(spread_report)

    # Belt and braces. classify_and_refine fills every tree cell -- no_data
    # cells take the local median, or static_tree_height where the window holds
    # no credible tree -- so this is empty in practice. A NaN reaching the
    # component grids would propagate into the voxel grid silently, which is
    # worth one cheap line to make impossible.
    unfilled = (land_cover_grid == tree_id) & ~np.isfinite(canopy)
    if unfilled.any():
        print(f"[nDSM] {int(unfilled.sum())} tree cells left unfilled — "
              f"using the static height {static_tree_height} m")
        canopy = np.where(unfilled, static_tree_height, canopy)

    canopy_bottom = _canopy_bottom(canopy)

    # Branch on the binding, not on is_lod2_model: the two are equivalent only
    # because the probe above returns early when the import fails, and a future
    # non-returning path there would turn this into a call on None.
    if reapply_canopy is not None:
        # LOD2: overlay the canopy onto the existing grid. reapply_canopy owns
        # the whole update — it writes canopy_top/canopy_bottom into
        # tree_canopy itself, in place where the existing array can take them
        # (which keeps any extras alias current for free) and re-pointing the
        # alias explicitly when it has to rebind. So this path deliberately
        # mutates *nothing* beforehand. It raises ValueError on a missing
        # extras['voxel_min_z'], on a canopy_bottom that does not match
        # canopy_top, or on a mesh_vegetation_mask that does not match the
        # voxel grid; the caller's handler is non-fatal, so pre-writing the
        # component grids would leave them describing crowns the voxel grid
        # never received, with only a server log to say so.
        #
        # Pass the already-derived bottom rather than the trunk ratio.
        # voxcitygml's default ratio happens to equal ours today, so both would
        # produce the same array — passing the bottom keeps that a coincidence
        # this code does not depend on.
        #
        # Handed over at component-grid resolution, deliberately not resampled
        # to the voxel grid's shape: reapply_canopy resamples a mismatched
        # canopy itself (as it already does for the DEM) but *stores* what the
        # caller passed, so a voxel-resolution array would leave tree_canopy.top
        # out of step with land_cover, dem and buildings and break a later
        # update_voxcity. Resampling is the package's job, not a fallback.
        reapply_canopy(voxcity_obj, canopy, canopy_bottom=canopy_bottom)
    else:
        # LOD1: the voxels *are* extruded footprints, so revise the 2.5-D
        # component grids and rebuild the grid from them.
        voxcity_obj.tree_canopy.top[:] = canopy
        if voxcity_obj.tree_canopy.bottom is not None:
            voxcity_obj.tree_canopy.bottom[:] = canopy_bottom
        else:
            voxcity_obj.tree_canopy.bottom = canopy_bottom
        regenerate_voxels(voxcity_obj, land_cover_source=land_cover_source,
                          inplace=True)

    return True
