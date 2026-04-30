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

import React, { useCallback, useEffect, useMemo, useState } from 'react';
import {
  applyEdits,
  getModelGeo,
  listLandCoverClasses,
  ModelGeoResult,
  LandCoverClass,
  PendingEditDto,
} from '../api';
import ThreeViewer from '../components/ThreeViewer';
import PlanMapEditor, {
  Backdrop,
  BasemapKey,
  DrawColor,
  MapInteraction,
  PendingEdit,
} from '../components/PlanMapEditor';
import { polygonToCells, buildingsInPolygon } from '../lib/grid';

interface EditTabProps {
  hasModel: boolean;
  figureJson: string;
  onFigureChange: (s: string) => void;
  /** Called after a successful commit so the parent can invalidate Solar /
   *  View / Landmark figures (they're now stale). */
  onModelEdited?: () => void;
}

type EditMode = 'building' | 'tree' | 'land_cover';

type BuildingAction = 'add_rect' | 'add_polygon' | 'remove_click' | 'remove_area';
type TreeAction = 'add_click' | 'add_area' | 'remove_click' | 'remove_area';
type LandCoverAction = 'paint_click' | 'paint_area';
type ModeAction = BuildingAction | TreeAction | LandCoverAction;

const MODE_OPTIONS: { id: EditMode; label: string }[] = [
  { id: 'building',   label: 'Building'   },
  { id: 'tree',       label: 'Tree'       },
  { id: 'land_cover', label: 'Land cover' },
];

function defaultActionFor(mode: EditMode): ModeAction {
  switch (mode) {
    case 'building':   return 'add_rect';
    case 'tree':       return 'add_click';
    case 'land_cover': return 'paint_click';
  }
}

function defaultBackdropFor(mode: EditMode): Backdrop {
  switch (mode) {
    case 'building':   return 'buildings';
    case 'tree':       return 'canopy';
    case 'land_cover': return 'land_cover';
  }
}

function drawColorFor(mode: EditMode): DrawColor {
  switch (mode) {
    case 'building':   return 'red';
    case 'tree':       return 'green';
    case 'land_cover': return 'blue';
  }
}

function interactionFor(mode: EditMode, action: ModeAction): MapInteraction {
  if (mode === 'building') {
    switch (action as BuildingAction) {
      case 'add_rect':     return 'draw_rect_3pt';
      case 'add_polygon':  return 'draw_polygon';
      case 'remove_click': return 'click_feature';
      case 'remove_area':  return 'draw_polygon';
    }
  }
  if (mode === 'tree') {
    switch (action as TreeAction) {
      case 'add_click':    return 'click_point';
      case 'add_area':     return 'draw_polygon';
      case 'remove_click': return 'click_point';
      case 'remove_area':  return 'draw_polygon';
    }
  }
  switch (action as LandCoverAction) {
    case 'paint_click': return 'click_point';
    case 'paint_area':  return 'draw_polygon';
  }
}

const EditTab: React.FC<EditTabProps> = ({ hasModel, figureJson, onFigureChange, onModelEdited }) => {
  const [mode, setMode] = useState<EditMode>('building');
  const [action, setAction] = useState<ModeAction>('add_rect');
  const [geo, setGeo] = useState<ModelGeoResult | null>(null);
  const [classes, setClasses] = useState<LandCoverClass[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [info, setInfo] = useState<string | null>(null);

  const [backdrop, setBackdrop] = useState<Backdrop>('buildings');
  const [basemap, setBasemap] = useState<BasemapKey>('CartoDB Positron');

  const [buildingHeight, setBuildingHeight] = useState(15);
  const [buildingMinHeight, setBuildingMinHeight] = useState(0);
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
      setError(err.message || 'Failed to load map');
    } finally {
      setLoading(false);
    }
  }, [hasModel, classIndex]);

  useEffect(() => { reload(); }, [reload]);

  useEffect(() => {
    setAction(defaultActionFor(mode));
    setBackdrop(defaultBackdropFor(mode));
    setError(null);
    setInfo(null);
  }, [mode]);

  useEffect(() => {
    setError(null);
    setInfo(null);
  }, [action]);

  const interaction = useMemo(() => interactionFor(mode, action), [mode, action]);
  const drawColor = drawColorFor(mode);

  const editableClasses = useMemo(() => classes.filter((c) => c.editable), [classes]);

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
            `Buffered: add ellipsoid tree over ${cells.length} cell(s).`,
          );
        } else if (mode === 'tree' && action === 'remove_click') {
          const cells = discCells(cell[0], cell[1], treeDiameter);
          if (cells.length === 0) return;
          addPending(
            { kind: 'delete_trees', cells },
            `Buffered: clear canopy on ${cells.length} cell(s).`,
          );
        } else if (mode === 'land_cover' && action === 'paint_click') {
          const color =
            editableClasses.find((c) => c.index === classIndex)?.color || '#888888';
          addPending(
            { kind: 'paint_lc', cells: [cell], class_index: classIndex, color },
            `Buffered: repaint 1 cell.`,
          );
        }
      } catch (err: any) {
        setError(err.message || 'Edit failed');
      }
    },
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [mode, action, treeDiameter, treeTop, treeBottom, classIndex, geo, editableClasses, addPending, meshsize],
  );

  const handlePickBuilding = useCallback((bid: number) => {
    if (mode !== 'building' || action !== 'remove_click') return;
    addPending(
      { kind: 'delete_building', building_id: bid },
      `Buffered: delete building #${bid}.`,
    );
  }, [mode, action, addPending]);

  const handlePolygonComplete = useCallback(
    (ring: [number, number][]) => {
      if (!geo) return;
      try {
        if (mode === 'building' && (action === 'add_rect' || action === 'add_polygon')) {
          const cells = polygonCells(ring);
          if (cells.length === 0) throw new Error('Footprint covers no cells.');
          addPending(
            {
              kind: 'add_building',
              ring,
              cells,
              height_m: buildingHeight,
              min_height_m: buildingMinHeight,
            },
            `Buffered: add building over ${cells.length} cell(s).`,
          );
        } else if (mode === 'building' && action === 'remove_area') {
          const ids = buildingsInPolygon(geo.building_geojson, ring);
          if (ids.length === 0) throw new Error('No buildings inside the polygon.');
          setPendingEdits((p) => [
            ...p,
            ...ids.map<PendingEdit>((bid) => ({ kind: 'delete_building', building_id: bid })),
          ]);
          setInfo(`Buffered: delete ${ids.length} building(s).`);
          setError(null);
        } else if (mode === 'tree' && action === 'add_area') {
          const cells = polygonCells(ring);
          if (cells.length === 0) throw new Error('Polygon covers no cells.');
          addPending(
            { kind: 'add_trees', cells, height_m: treeTop, bottom_m: treeBottom },
            `Buffered: add trees over ${cells.length} cell(s).`,
          );
        } else if (mode === 'tree' && action === 'remove_area') {
          const cells = polygonCells(ring);
          if (cells.length === 0) throw new Error('Polygon covers no cells.');
          addPending(
            { kind: 'delete_trees', cells },
            `Buffered: clear canopy on ${cells.length} cell(s).`,
          );
        } else if (mode === 'land_cover' && action === 'paint_area') {
          const cells = polygonCells(ring);
          if (cells.length === 0) throw new Error('Polygon covers no cells.');
          const color =
            editableClasses.find((c) => c.index === classIndex)?.color || '#888888';
          addPending(
            { kind: 'paint_lc', cells, class_index: classIndex, color },
            `Buffered: repaint ${cells.length} cell(s).`,
          );
        }
      } catch (err: any) {
        setError(err.message || 'Edit failed');
      }
    },
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [mode, action, geo, buildingHeight, buildingMinHeight, treeTop, treeBottom, classIndex, editableClasses, addPending],
  );

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
      setInfo(`Committed ${r.n_edits} edit(s); ${r.n_changed_total} cell(s) changed.`);
      // Refresh baseline so newly created/deleted features appear in the
      // backdrop for the next round of edits.
      await reload();
    } catch (err: any) {
      setError(err.message || '3D regeneration failed');
    } finally {
      setCommitting(false);
    }
  };

  if (!hasModel) {
    return (
      <div className="panel">
        <h2>Edit Model</h2>
        <div className="alert alert-info">Generate a model first to enable editing.</div>
      </div>
    );
  }

  const ActionBtn: React.FC<{ id: ModeAction; label: string; danger?: boolean }> = ({ id, label, danger }) => (
    <button
      className={
        `draw-toolbar-btn` +
        (action === id ? ' active' : '') +
        (danger ? ' danger' : '')
      }
      onClick={() => setAction(id)}
    >{label}</button>
  );

  return (
    <div className="three-col">
      {/* Controls */}
      <div className="panel">
        <h2>Edit Model</h2>

        <div className="mode-tabs">
          {MODE_OPTIONS.map((m) => (
            <button
              key={m.id}
              className={`mode-tab ${mode === m.id ? 'active' : ''}`}
              onClick={() => setMode(m.id)}
            >{m.label}</button>
          ))}
        </div>

        <div className="form-group">
          <label>Basemap</label>
          <select value={basemap} onChange={(e) => setBasemap(e.target.value as BasemapKey)}>
            <option value="CartoDB Positron">CartoDB Positron</option>
            <option value="Google Satellite">Google Satellite</option>
            <option value="OpenStreetMap">OpenStreetMap</option>
          </select>
        </div>

        <div className="form-group">
          <label>Overlay</label>
          <select value={backdrop} onChange={(e) => setBackdrop(e.target.value as Backdrop)}>
            <option value="buildings">Buildings</option>
            <option value="canopy">Canopy</option>
            <option value="land_cover">Land cover</option>
            <option value="none">None</option>
          </select>
        </div>

        {mode === 'building' && (
          <>
            <div className="mode-section">
              <h3>Add</h3>
              <div className="draw-toolbar">
                <ActionBtn id="add_rect"    label="Rectangle" />
                <ActionBtn id="add_polygon" label="Polygon" />
              </div>
              <div className="form-group">
                <label>Height (m)</label>
                <input type="number" min={1} step={0.5} value={buildingHeight}
                       onChange={(e) => setBuildingHeight(parseFloat(e.target.value))} />
              </div>
              <div className="form-group">
                <label>Min height / base (m)</label>
                <input type="number" min={0} step={0.5} value={buildingMinHeight}
                       onChange={(e) => setBuildingMinHeight(parseFloat(e.target.value))} />
              </div>
            </div>

            <div className="mode-section">
              <h3>Remove</h3>
              <div className="draw-toolbar">
                <ActionBtn id="remove_click" label="Click"  danger />
                <ActionBtn id="remove_area"  label="Area"   danger />
              </div>
              <div style={{ color: 'var(--vc-muted)', fontSize: '0.8rem' }}>
                {action === 'remove_click'
                  ? 'Click a footprint on the map to delete it.'
                  : action === 'remove_area'
                  ? 'Draw a polygon to delete every building inside.'
                  : 'Switch to a Remove tool above.'}
              </div>
            </div>
          </>
        )}

        {mode === 'tree' && (
          <>
            <div className="mode-section">
              <h3>Add</h3>
              <div className="draw-toolbar">
                <ActionBtn id="add_click" label="Click" />
                <ActionBtn id="add_area"  label="Area" />
              </div>
              <div className="form-group">
                <label>Top (m)</label>
                <input type="number" min={1} step={0.5} value={treeTop}
                       onChange={(e) => onTreeTopChange(parseFloat(e.target.value))} />
              </div>
              <div className="form-group">
                <label>Trunk / bottom (m)</label>
                <input type="number" min={0} step={0.5} value={treeBottom}
                       onChange={(e) => onTreeBottomChange(parseFloat(e.target.value))} />
              </div>
              <div className="form-group">
                <label>Diameter (m) — disc brush radius {treeBrushRadius} cell(s)</label>
                <input type="number" min={1} step={0.5} value={treeDiameter}
                       onChange={(e) => onTreeDiameterChange(parseFloat(e.target.value))} />
              </div>
              <div className="form-group">
                <label style={{ display: 'flex', alignItems: 'center', gap: '6px', cursor: 'pointer' }}>
                  <input type="checkbox" checked={treeFixedProp}
                         onChange={(e) => onTreeFixedPropToggle(e.target.checked)} />
                  Fixed proportion
                </label>
              </div>
            </div>

            <div className="mode-section">
              <h3>Remove</h3>
              <div className="draw-toolbar">
                <ActionBtn id="remove_click" label="Click" danger />
                <ActionBtn id="remove_area"  label="Area"  danger />
              </div>
            </div>
          </>
        )}

        {mode === 'land_cover' && (
          <>
            <div className="mode-section">
              <h3>Class</h3>
              <div className="land-cover-swatches">
                {classes.map((c) => (
                  <button
                    key={c.index}
                    className={`lc-swatch ${classIndex === c.index ? 'active' : ''}`}
                    style={{ background: c.color }}
                    title={`${c.index} ${c.name}`}
                    onClick={() => setClassIndex(c.index)}
                  />
                ))}
              </div>
              <div className="lc-active-name">
                {classes.find((c) => c.index === classIndex)?.name ?? '—'}
              </div>
            </div>

            <div className="mode-section">
              <h3>Paint</h3>
              <div className="draw-toolbar">
                <ActionBtn id="paint_click" label="Click" />
                <ActionBtn id="paint_area"  label="Area" />
              </div>
            </div>
          </>
        )}

        {error && <div className="alert alert-error" style={{ marginTop: '0.75rem' }}>{error}</div>}
        {info && <div className="alert alert-success" style={{ marginTop: '0.75rem' }}>{info}</div>}

        <div className="mode-section" style={{ marginTop: '0.75rem' }}>
          <h3>Edits</h3>
          <div className="draw-toolbar">
            <button
              className="draw-toolbar-btn"
              onClick={handleUndoLast}
              disabled={pendingEdits.length === 0 || committing}
              title="Discard the most recent buffered edit"
            >Undo last</button>
            <button
              className="draw-toolbar-btn"
              onClick={handleClearEdits}
              disabled={pendingEdits.length === 0 || committing}
              title="Discard all buffered edits"
            >Clear edits</button>
          </div>
          <button
            className="btn btn-primary"
            style={{ marginTop: '0.5rem', width: '100%' }}
            onClick={handleUpdate3D}
            disabled={pendingEdits.length === 0 || committing || loading}
            title={
              pendingEdits.length === 0
                ? 'No pending edits'
                : `Re-voxelize and render the ${pendingEdits.length} pending edit(s).`
            }
          >
            {committing && <span className="spinner" />}
            {committing
              ? 'Updating…'
              : `Update 3D model${pendingEdits.length > 0 ? ` (${pendingEdits.length})` : ''}`}
          </button>
        </div>
      </div>

      {/* 2D editor */}
      <div className="panel" style={{ overflow: 'hidden', display: 'flex', flexDirection: 'column' }}>
        <h2>2D plan editor</h2>
        {loading && <div className="alert alert-info">Loading map…</div>}
        {geo && (
          <PlanMapEditor
            geo={geo}
            interaction={interaction}
            backdrop={backdrop}
            basemap={basemap}
            drawColor={drawColor}
            pendingEdits={pendingEdits}
            onPointClick={handlePointClick}
            onPickBuilding={handlePickBuilding}
            onPolygonComplete={handlePolygonComplete}
          />
        )}
      </div>

      {/* 3D viewer */}
      <div className="panel" style={{ overflow: 'hidden', display: 'flex', flexDirection: 'column' }}>
        <h2>3D result</h2>
        <div style={{ flex: '1 1 auto', minHeight: 0, display: 'flex', flexDirection: 'column' }}>
          {figureJson ? (
            <ThreeViewer figureJson={figureJson} />
          ) : (
            <div className="alert alert-info" style={{ marginTop: 0 }}>
              Apply an edit and click <strong>Update 3D model</strong> to render the 3D result here.
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default EditTab;
