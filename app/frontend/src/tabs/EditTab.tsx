/**
 * Edit tab — buildings / trees / land cover.
 *
 * Edits buffer client-side as `PendingEdit` records and render immediately as
 * Leaflet overlays. The user clicks **Update 3D model** to ship the whole
 * buffer to `/api/model/apply_edits`; the server applies them in order,
 * voxelizes once, and returns a fresh Plotly figure that the 3D viewer renders.
 *
 * Adapted from `reference/optree_app/frontend/src/tabs/EditModelTab.tsx`.
 */

import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { Building, Trees, Layers, Plus, Pencil, Trash2, Undo2, Brush, Check } from 'lucide-react';
import {
  applyEdits,
  getModelGeo,
  listLandCoverClasses,
  ModelGeoResult,
  LandCoverClass,
  PendingEditDto,
} from '../api';
import { ChoiceGroup, GuidedSection } from '../components/guided';
import { useT } from '../i18n';
import ThreeViewer from '../components/ThreeViewer';
import PreviewDisabledNotice from '../components/PreviewDisabledNotice';
import PlanMapEditor, {
  Backdrop,
  BasemapKey,
  DrawColor,
  MapInteraction,
  PendingEdit,
} from '../components/PlanMapEditor';
import { polygonToCells, buildingsInPolygon, buildingsFullyContainedInPolygon } from '../lib/grid';
import {
  pendingBuildingHeightOverrides,
  validateBuildingHeightInput,
} from '../lib/buildingHeightEdits';
import {
  actionForWorkflow,
  defaultBackdropForTarget,
  defaultWorkflowForTarget,
  drawColorForTarget,
  interactionForWorkflow,
  methodOptionsForTask,
  normalizeWorkflow,
  taskOptionsForTarget,
  TARGET_OPTIONS,
  type EditMethod,
  type EditTarget,
  type EditTask,
  type ModeAction,
} from './editWorkflow';

interface EditTabProps {
  hasModel: boolean;
  figureJson: string;
  onFigureChange: (s: string) => void;
  /** Called after a successful commit so the parent can invalidate Solar /
   *  View / Landmark figures (they're now stale). */
  onModelEdited?: () => void;
  previewDisabled?: boolean;
  previewGridShape?: number[] | null;
}

interface MethodSelectorProps {
  target: import('./editWorkflow').EditTarget;
  task: import('./editWorkflow').EditTask;
  method: import('./editWorkflow').EditMethod;
  onMethodChange: (m: import('./editWorkflow').EditMethod) => void;
}

const MethodSelector: React.FC<MethodSelectorProps> = ({ target, task, method, onMethodChange }) => {
  const t = useT();
  return (
    <div className="form-group">
      <div className="guided-section-label">{t('editTab.methodLabel')}</div>
      <ChoiceGroup
        variant="checks"
        ariaLabel={t('editTab.editMethodAria')}
        value={method}
        onChange={onMethodChange}
        options={methodOptionsForTask(target, task).map(({ tone: _tone, ...option }) => option)}
      />
    </div>
  );
};

function parameterSectionMeta(
  t: ReturnType<typeof useT>,
  target: import('./editWorkflow').EditTarget,
  task: import('./editWorkflow').EditTask,
): { label: string; tone: 'default' | 'danger' } {
  if (target === 'building' && task === 'add') return { label: t('editTab.metaAddBuilding'), tone: 'default' };
  if (target === 'building' && task === 'height') return { label: t('editTab.metaEditHeight'), tone: 'default' };
  if (target === 'building' && task === 'remove') return { label: t('editTab.metaRemoveBuilding'), tone: 'default' };
  if (target === 'tree' && task === 'add') return { label: t('editTab.metaAddTree'), tone: 'default' };
  if (target === 'tree' && task === 'remove') return { label: t('editTab.metaRemoveTree'), tone: 'default' };
  if (target === 'land_cover' && task === 'paint') return { label: t('editTab.metaPaintLandCover'), tone: 'default' };
  return { label: t('editTab.metaParameters'), tone: 'default' };
}


const EditTab: React.FC<EditTabProps> = ({ hasModel, figureJson, onFigureChange, onModelEdited, previewDisabled = false, previewGridShape }) => {
  const t = useT();
  const [workflow, setWorkflow] = useState(() => defaultWorkflowForTarget('building'));
  const mode = workflow.target;
  const action = actionForWorkflow(workflow);
  const [geo, setGeo] = useState<ModelGeoResult | null>(null);
  const [classes, setClasses] = useState<LandCoverClass[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [info, setInfo] = useState<string | null>(null);

  const [backdrop, setBackdrop] = useState<Backdrop>('buildings');
  const [basemap, setBasemap] = useState<BasemapKey>('CartoDB Positron');

  const [buildingHeight, setBuildingHeight] = useState(15);
  const [buildingMinHeight, setBuildingMinHeight] = useState(0);

  // Building height edit state
  const [selectedBuildingIds, setSelectedBuildingIds] = useState<number[]>([]);
  const [heightEditValue, setHeightEditValue] = useState('10');
  const [heightEditMinValue, setHeightEditMinValue] = useState('0');
  const [useMinHeightEdit, setUseMinHeightEdit] = useState(false);
  const [showBuildingHeightLabels, setShowBuildingHeightLabels] = useState(true);
  // Tree shape defaults mirror voxcity.geoprocessor.draw.edit_tree:
  //   top=10 m, bottom (trunk)=4 m, crown diameter=6 m, fixed proportion ON.
  const [treeTop, setTreeTop] = useState(10);
  const [treeBottom, setTreeBottom] = useState(4);
  const [treeDiameter, setTreeDiameter] = useState(6);
  const [treeFixedProp, setTreeFixedProp] = useState(true);
  // Base ratios captured when 'Fixed proportion' is enabled. Updating any
  // one of top/bottom/diameter then rescales the others to preserve shape.
  const treeRatios = React.useRef({
    bottom: 4 / 10,
    crown: 6 / 10,
  });
  const treeUpdating = React.useRef(false);
  const [classIndex, setClassIndex] = useState<number>(2);

  const captureTreeRatios = useCallback(() => {
    const top = treeTop || 1;
    treeRatios.current = {
      bottom: Math.max(0, treeBottom / top),
      crown: Math.max(0, treeDiameter / top),
    };
  }, [treeTop, treeBottom, treeDiameter]);

  const onTreeTopChange = useCallback((v: number) => {
    if (treeUpdating.current) { setTreeTop(v); return; }
    setTreeTop(v);
    if (treeFixedProp && v > 0) {
      treeUpdating.current = true;
      setTreeBottom(Math.max(0, treeRatios.current.bottom * v));
      setTreeDiameter(Math.max(0, treeRatios.current.crown * v));
      treeUpdating.current = false;
    }
  }, [treeFixedProp]);

  const onTreeBottomChange = useCallback((v: number) => {
    if (treeUpdating.current) { setTreeBottom(v); return; }
    setTreeBottom(v);
    if (treeFixedProp && treeRatios.current.bottom > 0) {
      const newTop = Math.max(0, v / treeRatios.current.bottom);
      treeUpdating.current = true;
      setTreeTop(newTop);
      setTreeDiameter(Math.max(0, treeRatios.current.crown * newTop));
      treeUpdating.current = false;
    }
  }, [treeFixedProp]);

  const onTreeDiameterChange = useCallback((v: number) => {
    if (treeUpdating.current) { setTreeDiameter(v); return; }
    setTreeDiameter(v);
    if (treeFixedProp && treeRatios.current.crown > 0) {
      const newTop = Math.max(0, v / treeRatios.current.crown);
      treeUpdating.current = true;
      setTreeTop(newTop);
      setTreeBottom(Math.max(0, treeRatios.current.bottom * newTop));
      treeUpdating.current = false;
    }
  }, [treeFixedProp]);

  const onTreeFixedPropToggle = useCallback((checked: boolean) => {
    if (checked) captureTreeRatios();
    setTreeFixedProp(checked);
  }, [captureTreeRatios]);

  /** Buffered uncommitted edits. Drained automatically by the debounced
   *  commit effect below. */
  const [pendingEdits, setPendingEdits] = useState<PendingEdit[]>([]);
  const [committing, setCommitting] = useState(false);

  // Close the Display menu when clicking outside it
  const displayMenuRef = useRef<HTMLDetailsElement>(null);
  useEffect(() => {
    const handler = (e: MouseEvent) => {
      if (displayMenuRef.current && !displayMenuRef.current.contains(e.target as Node)) {
        displayMenuRef.current.removeAttribute('open');
      }
    };
    document.addEventListener('mousedown', handler);
    return () => document.removeEventListener('mousedown', handler);
  }, []);

  const setTarget = useCallback((target: EditTarget) => {
    setWorkflow(defaultWorkflowForTarget(target));
  }, []);

  const setTask = useCallback((task: EditTask) => {
    setWorkflow((current) => normalizeWorkflow({ ...current, task }));
  }, []);

  const setMethod = useCallback((method: EditMethod) => {
    setWorkflow((current) => normalizeWorkflow({ ...current, method }));
  }, []);

  /* ── Initial fetch ──────────────────────────────────────── */
  const reload = useCallback(async () => {
    if (!hasModel) return;
    setLoading(true);
    setError(null);
    try {
      const [g, c] = await Promise.all([getModelGeo(), listLandCoverClasses()]);
      setGeo(g);
      setClasses(c.classes);
      const editable = c.classes.filter((x) => x.editable);
      if (editable.length && !editable.some((x) => x.index === classIndex)) {
        setClassIndex(editable[0].index);
      }
    } catch (err: any) {
      setError(err.message || t('editTab.errLoadMap'));
    } finally {
      setLoading(false);
    }
  }, [hasModel, classIndex, t]);

  useEffect(() => { reload(); }, [reload]);

  useEffect(() => {
    setBackdrop(defaultBackdropForTarget(mode));
    setError(null);
    setInfo(null);
    setSelectedBuildingIds([]);
  }, [mode]);

  useEffect(() => {
    setError(null);
    setInfo(null);
    setSelectedBuildingIds([]);
  }, [action]);

  const interaction = useMemo(() => interactionForWorkflow(workflow), [workflow]);
  const drawColor = drawColorForTarget(mode);

  const editableClasses = useMemo(() => classes.filter((c) => c.editable), [classes]);

  const pendingBuildingHeightsMap = useMemo(
    () => pendingBuildingHeightOverrides(pendingEdits),
    [pendingEdits],
  );

  const visiblePendingEdits = useMemo(() => pendingEdits.filter((e) => {
    switch (mode) {
      case 'building': return e.kind === 'add_building' || e.kind === 'delete_building' || e.kind === 'set_building_height';
      case 'tree':     return e.kind === 'add_trees'    || e.kind === 'delete_trees';
      case 'land_cover': return e.kind === 'paint_lc';
      default: return false;
    }
  }), [pendingEdits, mode]);

  const meshsize = geo?.meshsize_m || 1;
  const treeBrushRadius = useMemo(() => {
    if (!geo) return 0;
    return Math.max(0, Math.floor(treeDiameter / 2 / meshsize));
  }, [geo, treeDiameter, meshsize]);

  const addPending = useCallback((e: PendingEdit, msg: string) => {
    setPendingEdits((p) => [...p, e]);
    setInfo(msg);
    setError(null);
  }, []);

  const discCells = (i: number, j: number, diameter_m: number): [number, number][] => {
    if (!geo) return [];
    const [nx, ny] = geo.grid_shape;
    const r_m = diameter_m / 2;
    const r_cells = Math.max(0, Math.floor(r_m / meshsize));
    const out: [number, number][] = [];
    for (let di = -r_cells; di <= r_cells; di++) {
      for (let dj = -r_cells; dj <= r_cells; dj++) {
        const d_m = Math.sqrt((di * meshsize) ** 2 + (dj * meshsize) ** 2);
        if (d_m > r_m) continue;
        const ii = i + di;
        const jj = j + dj;
        if (ii < 0 || ii >= nx || jj < 0 || jj >= ny) continue;
        out.push([ii, jj]);
      }
    }
    return out;
  };

  const ellipsoidStamp = (
    i: number,
    j: number,
    diameter_m: number,
    top_m: number,
    bottom_m: number,
  ): { cells: [number, number][]; tops: number[]; bottoms: number[] } => {
    if (!geo) return { cells: [], tops: [], bottoms: [] };
    const [nx, ny] = geo.grid_shape;
    const r_m = diameter_m / 2;
    const r_cells = Math.max(0, Math.floor(r_m / meshsize));
    const mid = (top_m + bottom_m) / 2;
    const halfV = (top_m - bottom_m) / 2;
    const cells: [number, number][] = [];
    const tops: number[] = [];
    const bottoms: number[] = [];
    for (let di = -r_cells; di <= r_cells; di++) {
      for (let dj = -r_cells; dj <= r_cells; dj++) {
        const d_m = Math.sqrt((di * meshsize) ** 2 + (dj * meshsize) ** 2);
        if (d_m > r_m) continue;
        const t = r_m > 0 ? d_m / r_m : 0;
        const s = Math.sqrt(Math.max(0, 1 - t * t));
        const cellTop = mid + halfV * s;
        const cellBottom = mid - halfV * s;
        if (cellTop - cellBottom < 0.01) continue;
        const ii = i + di;
        const jj = j + dj;
        if (ii < 0 || ii >= nx || jj < 0 || jj >= ny) continue;
        cells.push([ii, jj]);
        tops.push(cellTop);
        bottoms.push(cellBottom);
      }
    }
    return { cells, tops, bottoms };
  };

  const polygonCells = (ring: [number, number][]): [number, number][] => {
    if (!geo) return [];
    return polygonToCells(ring, {
      origin:    geo.grid_geom.origin,
      side_1:    geo.grid_geom.side_1,
      side_2:    geo.grid_geom.side_2,
      u_vec:     geo.grid_geom.u_vec,
      v_vec:     geo.grid_geom.v_vec,
      adj_mesh:  geo.grid_geom.adj_mesh,
      grid_size: geo.grid_geom.grid_size,
    });
  };

  /* ── Click handlers (buffer only) ──────────────────────── */
  const handlePointClick = useCallback(
    (_lat: number, _lon: number, cell: [number, number] | null) => {
      if (!cell) return;
      try {
        if (mode === 'tree' && action === 'add_click') {
          const { cells, tops, bottoms } =
            ellipsoidStamp(cell[0], cell[1], treeDiameter, treeTop, treeBottom);
          if (cells.length === 0) return;
          addPending(
            { kind: 'add_trees', cells, tops, bottoms, height_m: treeTop, bottom_m: treeBottom },
            t('editTab.msgAddTree', { n: cells.length }),
          );
        } else if (mode === 'tree' && action === 'remove_click') {
          const cells = discCells(cell[0], cell[1], treeDiameter);
          if (cells.length === 0) return;
          addPending(
            { kind: 'delete_trees', cells },
            t('editTab.msgClearCanopy', { n: cells.length }),
          );
        } else if (mode === 'land_cover' && action === 'paint_click') {
          const color =
            editableClasses.find((c) => c.index === classIndex)?.color || '#888888';
          addPending(
            { kind: 'paint_lc', cells: [cell], class_index: classIndex, color },
            t('editTab.msgRepaintOne'),
          );
        }
      } catch (err: any) {
        setError(err.message || t('editTab.errEditFailed'));
      }
    },
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [mode, action, treeDiameter, treeTop, treeBottom, classIndex, geo, editableClasses, addPending, meshsize, t],
  );

  const handlePickBuilding = useCallback((bid: number) => {
    if (mode !== 'building') return;
    if (action === 'remove_click') {
      addPending(
        { kind: 'delete_building', building_id: bid },
        t('editTab.msgDeleteBuilding', { id: bid }),
      );
    } else if (action === 'set_height_click') {
      setSelectedBuildingIds((prev) =>
        prev.includes(bid) ? prev.filter((id) => id !== bid) : [...prev, bid],
      );
    }
  }, [mode, action, addPending, t]);

  const handlePolygonComplete = useCallback(
    (ring: [number, number][]) => {
      if (!geo) return;
      try {
        if (mode === 'building' && (action === 'add_rect' || action === 'add_polygon')) {
          const cells = polygonCells(ring);
          if (cells.length === 0) throw new Error(t('editTab.errFootprintNoCells'));
          addPending(
            {
              kind: 'add_building',
              ring,
              cells,
              height_m: buildingHeight,
              min_height_m: buildingMinHeight,
            },
            t('editTab.msgAddBuilding', { n: cells.length }),
          );
        } else if (mode === 'building' && action === 'remove_area') {
          const ids = buildingsInPolygon(geo.building_geojson, ring);
          if (ids.length === 0) throw new Error(t('editTab.errNoBuildingsInPolygon'));
          setPendingEdits((p) => [
            ...p,
            ...ids.map<PendingEdit>((bid) => ({ kind: 'delete_building', building_id: bid })),
          ]);
          setInfo(t('editTab.msgDeleteBuildings', { n: ids.length }));
          setError(null);
        } else if (mode === 'building' && action === 'set_height_area') {
          const ids = buildingsFullyContainedInPolygon(geo.building_geojson, ring);
          if (ids.length === 0) throw new Error(t('editTab.errNoBuildingsFullyInPolygon'));
          setSelectedBuildingIds(ids);
          setInfo(t('editTab.msgSelectedForHeight', { n: ids.length }));
          setError(null);
        } else if (mode === 'tree' && action === 'add_area') {
          const cells = polygonCells(ring);
          if (cells.length === 0) throw new Error(t('editTab.errPolygonNoCells'));
          addPending(
            { kind: 'add_trees', cells, height_m: treeTop, bottom_m: treeBottom },
            t('editTab.msgAddTrees', { n: cells.length }),
          );
        } else if (mode === 'tree' && action === 'remove_area') {
          const cells = polygonCells(ring);
          if (cells.length === 0) throw new Error(t('editTab.errPolygonNoCells'));
          addPending(
            { kind: 'delete_trees', cells },
            t('editTab.msgClearCanopy', { n: cells.length }),
          );
        } else if (mode === 'land_cover' && action === 'paint_area') {
          const cells = polygonCells(ring);
          if (cells.length === 0) throw new Error(t('editTab.errPolygonNoCells'));
          const color =
            editableClasses.find((c) => c.index === classIndex)?.color || '#888888';
          addPending(
            { kind: 'paint_lc', cells, class_index: classIndex, color },
            t('editTab.msgRepaintCells', { n: cells.length }),
          );
        }
      } catch (err: any) {
        setError(err.message || t('editTab.errEditFailed'));
      }
    },
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [mode, action, geo, buildingHeight, buildingMinHeight, treeTop, treeBottom, classIndex, editableClasses, addPending, t],
  );

  const handleApplyHeightEdit = useCallback(() => {
    if (selectedBuildingIds.length === 0) return;
    const heightNum = parseFloat(heightEditValue);
    const minNum = useMinHeightEdit ? parseFloat(heightEditMinValue) : 0;
    const validated = validateBuildingHeightInput(heightNum, useMinHeightEdit, minNum);
    if (!validated.ok) {
      setError(validated.error);
      return;
    }
    addPending(
      {
        kind: 'set_building_height',
        building_ids: selectedBuildingIds,
        height_m: validated.height,
        ...(validated.minHeight != null ? { min_height_m: validated.minHeight } : {}),
      },
      t('editTab.msgSetHeight', { h: validated.height, n: selectedBuildingIds.length }),
    );
    setSelectedBuildingIds([]);
  }, [selectedBuildingIds, heightEditValue, useMinHeightEdit, heightEditMinValue, addPending, t]);

  /* ── Buffer management ─────────────────────────────────── */
  const handleUndoLast = useCallback(() => {
    setPendingEdits((p) => p.slice(0, -1));
  }, []);

  const handleClearEdits = useCallback(() => {
    setPendingEdits([]);
    setInfo(null);
    setError(null);
  }, []);

  /* ── Commit pending edits → 3D ─────────────────────────── */
  const handleUpdate3D = async () => {
    if (pendingEdits.length === 0) return;
    setError(null);
    setCommitting(true);
    try {
      const dtos: PendingEditDto[] = pendingEdits.map((e): PendingEditDto => {
        switch (e.kind) {
          case 'add_building':
            return {
              kind: 'add_building',
              cells: e.cells,
              height_m: e.height_m,
              min_height_m: e.min_height_m,
              ring: e.ring,
            };
          case 'delete_building':
            return { kind: 'delete_building', building_ids: [e.building_id] };
          case 'set_building_height':
            return {
              kind: 'set_building_height',
              building_ids: e.building_ids,
              height_m: e.height_m,
              ...(e.min_height_m != null ? { min_height_m: e.min_height_m } : {}),
            };
          case 'add_trees':
            return {
              kind: 'add_trees',
              cells: e.cells,
              height_m: e.height_m,
              bottom_m: e.bottom_m,
              ...(e.tops && e.bottoms ? { tops: e.tops, bottoms: e.bottoms } : {}),
            };
          case 'delete_trees':
            return { kind: 'delete_trees', cells: e.cells };
          case 'paint_lc':
            return { kind: 'paint_lc', cells: e.cells, class_index: e.class_index };
          default:
            throw new Error(`Unsupported edit kind: ${(e as PendingEdit).kind}`);
        }
      });
      const r = await applyEdits(dtos);
      onFigureChange(r.figure_json);
      onModelEdited?.();
      setPendingEdits([]);
      setInfo(t('editTab.msgCommitted', { edits: r.n_edits, cells: r.n_changed_total }));
      // Refresh baseline so newly created/deleted features appear in the
      // backdrop for the next round of edits.
      await reload();
    } catch (err: any) {
      setError(err.message || t('editTab.errRegenFailed'));
    } finally {
      setCommitting(false);
    }
  };

  if (!hasModel) {
    return (
      <div className="panel">
        <h2>{t('editTab.heading')}</h2>
        <div className="alert alert-info">{t('editTab.noModelBody')}</div>
      </div>
    );
  }

  const targetOptionsWithIcons = TARGET_OPTIONS.map((targetOption) => {
    const taskCount = taskOptionsForTarget(targetOption.id).length;
    return {
      ...targetOption,
      count: `${taskCount} task${taskCount === 1 ? '' : 's'}`,
      icon: targetOption.id === 'building' ? Building : targetOption.id === 'tree' ? Trees : Layers,
    };
  });

  const taskOptionsWithIcons = taskOptionsForTarget(workflow.target).map((taskOption) => {
    let icon;
    if (taskOption.id === 'add') icon = Plus;
    else if (taskOption.id === 'height') icon = Pencil;
    else if (taskOption.id === 'remove') icon = Trash2;
    else if (taskOption.id === 'paint') icon = Brush;
    const { tone: _tone, ...baseOption } = taskOption;
    return { ...baseOption, icon };
  });

  return (
    <div className="three-col">
      {/* Controls */}
      <div className="panel edit-control-panel">
        <div className="edit-control-scroll">
          <h2>{t('editTab.heading')}</h2>

          <GuidedSection index={1} label={t('editTab.targetHeading')}>
            <ChoiceGroup
              variant="checks"
              ariaLabel={t('editTab.editTargetAria')}
              value={mode}
              onChange={setTarget}
              options={targetOptionsWithIcons}
            />
          </GuidedSection>

          <GuidedSection index={2} label={t('editTab.operationsHeading')}>
            <ChoiceGroup
              variant="checks"
              ariaLabel={t('editTab.editWorkflowAria')}
              value={workflow.task}
              onChange={setTask}
              options={taskOptionsWithIcons}
            />
          </GuidedSection>

          {(() => {
            const meta = parameterSectionMeta(t, workflow.target, workflow.task);
            return (
              <GuidedSection index={3} label={meta.label} tone={meta.tone}>
                {mode === 'building' && workflow.task === 'add' && (
                  <div className="guided-tool-details">
                    <MethodSelector target={mode} task={workflow.task} method={workflow.method} onMethodChange={setMethod} />
                    <div className="form-group">
                      <label>{t('editTab.height')}</label>
                      <input type="number" min={1} step={0.5} value={buildingHeight}
                             onChange={(e) => setBuildingHeight(parseFloat(e.target.value))} />
                    </div>
                    <div className="form-group">
                      <label>{t('editTab.minHeightBase')}</label>
                      <input type="number" min={0} step={0.5} value={buildingMinHeight}
                             onChange={(e) => setBuildingMinHeight(parseFloat(e.target.value))} />
                    </div>
                  </div>
                )}

                {mode === 'building' && workflow.task === 'height' && (
                  <div className="guided-tool-details">
                    <MethodSelector target={mode} task={workflow.task} method={workflow.method} onMethodChange={setMethod} />
                    <div className="guided-tool-hint">
                      {action === 'set_height_click'
                        ? t('editTab.hintClickBuildings')
                        : t('editTab.hintDrawSelectBuildings')}
                    </div>
                    {selectedBuildingIds.length > 0 && (
                      <div style={{ fontSize: '0.8rem', marginBottom: '0.4rem', color: 'var(--vc-text)' }}>
                        {t('editTab.buildingsSelected', { n: selectedBuildingIds.length })}
                      </div>
                    )}
                    <div className="form-group">
                      <label>{t('editTab.topHeight')}</label>
                      <input
                        type="number"
                        min={0.1}
                        step={0.5}
                        value={heightEditValue}
                        onChange={(e) => setHeightEditValue(e.target.value)}
                      />
                    </div>
                    <div className="form-group">
                      <label style={{ display: 'flex', alignItems: 'center', gap: '6px', cursor: 'pointer' }}>
                        <input
                          type="checkbox"
                          checked={useMinHeightEdit}
                          onChange={(e) => setUseMinHeightEdit(e.target.checked)}
                        />
                        {t('editTab.setMinHeight')}
                      </label>
                    </div>
                    {useMinHeightEdit && (
                      <div className="form-group">
                        <input
                          type="number"
                          min={0}
                          step={0.5}
                          value={heightEditMinValue}
                          onChange={(e) => setHeightEditMinValue(e.target.value)}
                        />
                      </div>
                    )}
                    <button
                      className="btn btn-primary"
                      style={{ marginTop: '0.25rem', width: '100%' }}
                      onClick={handleApplyHeightEdit}
                      disabled={selectedBuildingIds.length === 0}
                      type="button"
                    >
                      <Check size={14} aria-hidden="true" style={{ marginRight: 6 }} />
                      {t('editTab.applyToBuildings', { n: selectedBuildingIds.length || 0 })}
                    </button>
                  </div>
                )}

                {mode === 'building' && workflow.task === 'remove' && (
                  <div className="guided-tool-details">
                    <MethodSelector target={mode} task={workflow.task} method={workflow.method} onMethodChange={setMethod} />
                    <div className="guided-tool-hint">
                      {action === 'remove_click'
                        ? t('editTab.hintClickDeleteFootprint')
                        : t('editTab.hintDrawDeleteBuildings')}
                    </div>
                  </div>
                )}

                {mode === 'tree' && workflow.task === 'add' && (
                  <div className="guided-tool-details">
                    <MethodSelector target={mode} task={workflow.task} method={workflow.method} onMethodChange={setMethod} />
                    <div className="form-group">
                      <label>{t('editTab.top')}</label>
                      <input type="number" min={1} step={0.5} value={treeTop}
                             onChange={(e) => onTreeTopChange(parseFloat(e.target.value))} />
                    </div>
                    <div className="form-group">
                      <label>{t('editTab.trunkBottom')}</label>
                      <input type="number" min={0} step={0.5} value={treeBottom}
                             onChange={(e) => onTreeBottomChange(parseFloat(e.target.value))} />
                    </div>
                    <div className="form-group">
                      <label>{t('editTab.diameterBrush', { n: treeBrushRadius })}</label>
                      <input type="number" min={1} step={0.5} value={treeDiameter}
                             onChange={(e) => onTreeDiameterChange(parseFloat(e.target.value))} />
                    </div>
                    <div className="form-group">
                      <label style={{ display: 'flex', alignItems: 'center', gap: '6px', cursor: 'pointer' }}>
                        <input type="checkbox" checked={treeFixedProp}
                               onChange={(e) => onTreeFixedPropToggle(e.target.checked)} />
                        {t('editTab.fixedProportion')}
                      </label>
                    </div>
                  </div>
                )}

                {mode === 'tree' && workflow.task === 'remove' && (
                  <div className="guided-tool-details">
                    <MethodSelector target={mode} task={workflow.task} method={workflow.method} onMethodChange={setMethod} />
                    <div className="guided-tool-hint">
                      {action === 'remove_click'
                        ? t('editTab.hintClickClearCanopy')
                        : t('editTab.hintDrawClearCanopy')}
                    </div>
                  </div>
                )}

                {mode === 'land_cover' && workflow.task === 'paint' && (
                  <div className="guided-tool-details">
                    <div className="land-cover-swatches">
                      {classes.map((c) => (
                        <button
                          key={c.index}
                          className={`lc-swatch ${classIndex === c.index ? 'active' : ''}`}
                          style={{ background: c.color }}
                          title={`${c.index} ${c.name}`}
                          onClick={() => setClassIndex(c.index)}
                          type="button"
                        />
                      ))}
                    </div>
                    <div className="lc-active-name">
                      {classes.find((c) => c.index === classIndex)?.name ?? '—'}
                    </div>
                    <MethodSelector target={mode} task={workflow.task} method={workflow.method} onMethodChange={setMethod} />
                  </div>
                )}
              </GuidedSection>
            );
          })()}

          <GuidedSection
            index={4}
            label={`${t('editTab.sessionHeading')}${pendingEdits.length > 0 ? t('editTab.sessionPending', { n: pendingEdits.length }) : ''}`}
            action={(
              <>
                <button
                  className="btn btn-secondary btn-sm"
                  onClick={handleUndoLast}
                  disabled={pendingEdits.length === 0 || committing}
                  title={t('editTab.discardRecent')}
                  type="button"
                >
                  <Undo2 size={12} aria-hidden="true" style={{ marginRight: 4 }} />
                  {t('editTab.undo')}
                </button>
                <button
                  className="btn btn-secondary btn-sm danger"
                  onClick={handleClearEdits}
                  disabled={pendingEdits.length === 0 || committing}
                  title={t('editTab.discardAll')}
                  type="button"
                  style={{ marginLeft: 6 }}
                >
                  <Trash2 size={12} aria-hidden="true" style={{ marginRight: 4 }} />
                  {t('editTab.clear')}
                </button>
              </>
            )}
          >
            <div style={{ color: 'var(--vc-muted)', fontSize: '0.78rem' }}>
              {t('editTab.bufferedHint')}
            </div>
          </GuidedSection>

          <div className="guided-feedback-slot">
            {error && <div className="alert alert-error">{error}</div>}
            {info && <div className="alert alert-success">{info}</div>}
          </div>
        </div>

        <div className="pending-edit-footer">
          <button
            className="btn btn-primary pending-update-btn"
            onClick={handleUpdate3D}
            disabled={pendingEdits.length === 0 || committing || loading}
            title={
              pendingEdits.length === 0
                ? t('editTab.noPendingEdits')
                : t('editTab.rerenderHint', { n: pendingEdits.length })
            }
            type="button"
          >
            {committing && <span className="spinner" />}
            <Layers size={14} aria-hidden="true" style={{ marginRight: 6 }} />
            {committing
              ? t('editTab.updating')
              : `${t('editTab.update3dModel')}${pendingEdits.length > 0 ? ` · ${pendingEdits.length}` : ''}`}
          </button>
        </div>
      </div>

      {/* 2D editor */}
      <div className="panel visual-panel">
        <div className="plan-panel-header">
          <details className="display-menu" ref={displayMenuRef}>
            <summary>{t('editTab.display')}</summary>
            <div className="display-menu-popover">
              <div className="form-group">
                <label>{t('editTab.basemap')}</label>
                <select value={basemap} onChange={(e) => setBasemap(e.target.value as BasemapKey)}>
                  <option value="CartoDB Positron">CartoDB Positron</option>
                  <option value="Google Satellite">Google Satellite</option>
                  <option value="OpenStreetMap">OpenStreetMap</option>
                </select>
              </div>
              <div className="form-group">
                <label>{t('editTab.overlay')}</label>
                <select value={backdrop} onChange={(e) => setBackdrop(e.target.value as Backdrop)}>
                  <option value="buildings">{t('editTab.ovBuildings')}</option>
                  <option value="canopy">{t('editTab.ovCanopy')}</option>
                  <option value="land_cover">{t('editTab.ovLandCover')}</option>
                  <option value="none">{t('editTab.ovNone')}</option>
                </select>
              </div>
              {backdrop === 'buildings' && (
                <label className="checkbox-row">
                  <input
                    type="checkbox"
                    checked={showBuildingHeightLabels}
                    onChange={(e) => setShowBuildingHeightLabels(e.target.checked)}
                  />
                  {t('editTab.showHeightLabels')}
                </label>
              )}
            </div>
          </details>
        </div>
        {loading && <div className="alert alert-info">{t('editTab.loadingMap')}</div>}
        <div className="visual-frame">
        {geo && (
          <PlanMapEditor
            geo={geo}
            interaction={interaction}
            backdrop={backdrop}
            basemap={basemap}
            drawColor={drawColor}
            pendingEdits={visiblePendingEdits}
            showBuildingHeightLabels={showBuildingHeightLabels && backdrop === 'buildings'}
            selectedBuildingIds={selectedBuildingIds}
            pendingBuildingHeights={pendingBuildingHeightsMap}
            onPointClick={handlePointClick}
            onPickBuilding={handlePickBuilding}
            onPolygonComplete={handlePolygonComplete}
          />
        )}
        </div>
      </div>

      {/* 3D viewer */}
      <div className="panel visual-panel">
        <div className="visual-frame">
          {previewDisabled ? (
            <PreviewDisabledNotice gridShape={previewGridShape} />
          ) : figureJson ? (
            <ThreeViewer figureJson={figureJson} />
          ) : (
            <div className="alert alert-info" style={{ marginTop: 0 }}>
              {t('editTab.applyHint', { action: t('editTab.update3dModel') })}
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default EditTab;
