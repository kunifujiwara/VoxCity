/**
 * Import tab — upload an OBJ, position it, and stamp its buildings into the model.
 *
 * Placement lives in one `Placement` object (lib/objPlacement). The numeric form
 * here writes it; the 2D map and 3D gizmo read/write the same object. Commit
 * calls /api/model/import_obj/commit and renders the result.
 */
import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { Upload, Boxes } from 'lucide-react';
import {
  uploadImportObj,
  commitImportObj,
  uploadImportDxf,
  commitImportDxf,
  clearAuxiliaryLines,
  getModelGeo,
  getAnchorGround,
  AnchorGroundResult,
  ImportObjUploadResult,
  ImportDxfUploadResult,
  ModelGeoResult,
} from '../api';
import { GuidedSection } from '../components/guided';
import ThreeViewer from '../components/ThreeViewer';
import PreviewDisabledNotice from '../components/PreviewDisabledNotice';
import ObjPlacementMap from '../components/ObjPlacementMap';
import DxfPlacementMap from '../components/DxfPlacementMap';
import AuxiliaryLinesControl from '../components/AuxiliaryLinesControl';
import { SceneViewer } from '../three';
import { lonLatToUvM, domainRotationDeg } from '../lib/grid';
import {
  defaultPlacement,
  Placement,
  Units,
} from '../lib/objPlacement';
import { anchorSceneUp } from './importAnchorScene';

interface ImportTabProps {
  hasModel: boolean;
  figureJson: string;
  onFigureChange: (s: string) => void;
  onModelEdited?: () => void;
  previewDisabled?: boolean;
  previewGridShape?: number[] | null;
}

const UNIT_OPTIONS: Units[] = ['m', 'cm', 'mm', 'ft', 'in'];

const ImportTab: React.FC<ImportTabProps> = ({ hasModel, figureJson, onFigureChange, onModelEdited, previewDisabled = false, previewGridShape }) => {
  const [upload, setUpload] = useState<ImportObjUploadResult | null>(null);
  const [roles, setRoles] = useState<Record<string, string>>({});
  const [placement, setPlacement] = useState<Placement>(defaultPlacement);
  const [gizmoMode, setGizmoMode] = useState<'translate' | 'rotate'>('translate');
  const [advanced, setAdvanced] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [info, setInfo] = useState<string | null>(null);
  const [warning, setWarning] = useState<string | null>(null);
  const [geo, setGeo] = useState<ModelGeoResult | null>(null);
  // DEM datum at the current anchor cell, for the 3D preview's vertical seating.
  const [anchorGround, setAnchorGround] = useState<AnchorGroundResult | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const [importMode, setImportMode] = useState<'obj' | 'dxf'>('obj');
  // DXF auxiliary-line import state.
  const [dxfUpload, setDxfUpload] = useState<ImportDxfUploadResult | null>(null);
  const [dxfPlacement, setDxfPlacement] = useState<Placement>(defaultPlacement);
  const [dxfVisibility, setDxfVisibility] = useState<Record<string, boolean>>({});
  const [auxVisibility, setAuxVisibility] = useState<Record<string, Record<string, boolean>>>({});
  const dxfFileInputRef = useRef<HTMLInputElement>(null);

  const refreshGeo = useCallback(() => {
    getModelGeo().then(setGeo).catch(() => {});
  }, []);

  useEffect(() => {
    if (hasModel) getModelGeo().then(setGeo).catch(() => {});
  }, [hasModel]);

  // Default the placement anchor to the model centre once an OBJ is uploaded.
  // The 3D gizmo only writes `move`/`rotation`, never the anchor, so without
  // this a user who positions purely in 3D would never set `anchorLonLat` and
  // the "Import building(s)" button would stay disabled. Clicking the 2D map
  // still overrides this default. geo.center is Leaflet [lat, lon]; Placement
  // stores [lon, lat].
  useEffect(() => {
    if (!upload || !geo) return;
    setPlacement((p) =>
      p.anchorLonLat ? p : { ...p, anchorLonLat: [geo.center[1], geo.center[0]] },
    );
  }, [upload, geo]);

  // Same default-anchor behaviour as the OBJ effect above, for the DXF flow.
  useEffect(() => {
    if (!dxfUpload || !geo) return;
    setDxfPlacement((p) =>
      p.anchorLonLat ? p : { ...p, anchorLonLat: [geo.center[1], geo.center[0]] },
    );
  }, [dxfUpload, geo]);

  // Fetch the DEM datum at the anchor cell whenever the anchor moves. The 3D
  // preview uses this so `move_up = 0` seats the building on the ground at the
  // same height the commit transform does (see `anchorScene` below).
  useEffect(() => {
    const a = placement.anchorLonLat;
    if (!a) { setAnchorGround(null); return; }
    let cancelled = false;
    getAnchorGround(a[0], a[1])
      .then((r) => { if (!cancelled) setAnchorGround(r); })
      .catch(() => { if (!cancelled) setAnchorGround(null); });
    return () => { cancelled = true; };
  }, [placement.anchorLonLat]);

  const setMove = (idx: 0 | 1 | 2, v: number) =>
    setPlacement((p) => {
      const move = [...p.move] as [number, number, number];
      move[idx] = v;
      return { ...p, move };
    });

  // Stable identity so the gizmo's onObjectChange handler doesn't churn the
  // SceneViewer/PlacementGizmo props on every drag tick.
  const handlePlacementChange = useCallback(
    (next: Partial<Placement>) => setPlacement((p) => ({ ...p, ...next })),
    [],
  );

  // Scene-metre position [east, north, up] of the placement anchor. The 2D map
  // draws footprints at `anchorScene + transformModelPoint(...)`; the 3D gizmo
  // mesh must sit at `anchorScene + move` to stay in sync. East/north come from
  // the anchor lon/lat via the same grid projection the 2D map uses.
  //
  // The vertical component must match the commit transform so that `move_up = 0`
  // seats the building on the ground in BOTH the preview and the voxelized
  // result. The commit places model z=0 at scene-Z `(anchor_elevation - dem_min)
  // + meshsize` (per-cell terrain height + one ground voxel). We mirror that here
  // using the DEM datum fetched for the anchor cell; the effective elevation is
  // the user's manual override when set, else the auto DEM sample (matching the
  // commit endpoint's fallback). Falls back to 0 until the datum is available.
  const anchorScene = useMemo<[number, number, number]>(() => {
    if (!geo || !placement.anchorLonLat) return [0, 0, 0];
    const fwd = lonLatToUvM({ grid_geom: geo.grid_geom });
    if (!fwd) return [0, 0, 0];
    const [east, north] = fwd(placement.anchorLonLat[0], placement.anchorLonLat[1]);
    const up = anchorSceneUp(placement.anchorElevation, anchorGround);
    return [east, north, up];
  }, [geo, placement.anchorLonLat, placement.anchorElevation, anchorGround]);

  // Bearing (degrees) of the grid's own +u axis -- see lib/grid.ts's
  // domainRotationDeg(). Passed to the 3D gizmo so it applies the same
  // combined rotation (placement.rotation + phiDeg) as transformModelPoint
  // (used by the 2D footprint map), keeping the two previews in sync on
  // rotated grids.
  const phiDeg = useMemo(() => (geo ? domainRotationDeg(geo.grid_geom) : 0), [geo]);

  const handleFile = useCallback(async (files: FileList | null) => {
    if (!files || files.length === 0) return;
    const all = Array.from(files);
    // The .obj is the primary; anything else (the .mtl, textures) rides along
    // as a sidecar so the server can resolve material names for window detection.
    const obj = all.find((f) => f.name.toLowerCase().endsWith('.obj'));
    if (!obj) { setError('Please choose a .obj file (you can also select its .mtl).'); return; }
    const sidecars = all.filter((f) => f !== obj);
    setBusy(true); setError(null); setInfo(null); setWarning(null);
    try {
      const res = await uploadImportObj(obj, sidecars);
      setUpload(res);
      setRoles(Object.fromEntries(res.groups.map((g) => [g.name, g.role])));
      const mtlNote = sidecars.some((f) => f.name.toLowerCase().endsWith('.mtl'))
        ? '' : ' (tip: also select the .mtl so window materials are detected)';
      setInfo(`Loaded ${res.groups.length} group(s). Position it and import.${mtlNote}`);
      onFigureChange(''); // clear any previous committed result so the live preview shows
    } catch (err: any) {
      setError(err.message || 'Upload failed');
    } finally {
      setBusy(false);
    }
  }, [onFigureChange]);

  const handleImport = useCallback(async () => {
    if (!upload) return;
    const anchorLonLat = placement.anchorLonLat;
    if (!anchorLonLat) { setError('Click the map to set an anchor first.'); return; }
    setBusy(true); setError(null); setInfo(null); setWarning(null);
    try {
      const r = await commitImportObj({
        import_id: upload.import_id,
        placement: {
          anchor_lonlat: anchorLonLat,
          anchor_elevation: placement.anchorElevation,
          anchor_model_point: placement.anchorModelPoint,
          rotation: placement.rotation,
          move: placement.move,
          units: placement.units,
          z_up: placement.zUp,
          swap_yz: placement.swapYz,
        },
        roles,
        overwrite: true,
      });
      onFigureChange(r.figure_json);
      onModelEdited?.();
      if (r.warning) {
        setWarning(r.warning);
        setInfo(null);
      } else {
        setWarning(null);
        setInfo(
          `Imported ${r.imported_building_ids.length} building(s); ` +
          `${r.n_building_voxels_added} voxel(s) added` +
          (r.n_window_voxels_added > 0 ? `, ${r.n_window_voxels_added} window voxel(s)` : '') +
          `.`,
        );
      }
    } catch (err: any) {
      setError(err.message || 'Import failed');
    } finally {
      setBusy(false);
    }
  }, [upload, placement, roles, onFigureChange, onModelEdited]);

  const handleDxfFile = useCallback(async (files: FileList | null) => {
    if (!files || files.length === 0) return;
    const file = Array.from(files).find((f) => f.name.toLowerCase().endsWith('.dxf'));
    if (!file) { setError('Please choose a .dxf file.'); return; }
    setBusy(true); setError(null); setInfo(null); setWarning(null);
    try {
      const res = await uploadImportDxf(file);
      setDxfUpload(res);
      setDxfVisibility(Object.fromEntries(res.layers.map((l) => [l.name, true])));
      setDxfPlacement((p) => ({
        ...defaultPlacement(),
        anchorLonLat: p.anchorLonLat,
        units: (res.detected_units as Units) ?? 'm',
        anchorModelPoint: [res.model_center[0], res.model_center[1], 0],
      }));
      if (res.warning) setWarning(res.warning);
      setInfo(`Loaded ${res.layers.length} layer(s). Position it and add reference lines.`);
    } catch (err: any) {
      setError(err.message || 'DXF upload failed');
    } finally {
      setBusy(false);
    }
  }, []);

  const handleDxfImport = useCallback(async () => {
    if (!dxfUpload) return;
    const anchorLonLat = dxfPlacement.anchorLonLat;
    if (!anchorLonLat) { setError('Click the map to set an anchor first.'); return; }
    setBusy(true); setError(null); setInfo(null); setWarning(null);
    try {
      const r = await commitImportDxf({
        import_id: dxfUpload.import_id,
        placement: {
          anchor_lonlat: anchorLonLat,
          anchor_model_point: [dxfPlacement.anchorModelPoint[0], dxfPlacement.anchorModelPoint[1]],
          rotation: dxfPlacement.rotation,
          move: [dxfPlacement.move[0], dxfPlacement.move[1]],
          units: dxfPlacement.units,
        },
        layer_visibility: dxfVisibility,
      });
      const fileName = r.auxiliary_lines[0]?.file_name;
      if (fileName) {
        setAuxVisibility((v) => ({ ...v, [fileName]: { ...dxfVisibility } }));
      }
      setWarning(r.warning);
      setInfo(r.warning ? null : `Added ${r.auxiliary_lines.length} auxiliary line(s).`);
      setDxfUpload(null);
      refreshGeo();       // pull committed lines into geo.auxiliary_lines for the 3D overlay
      onModelEdited?.();
    } catch (err: any) {
      setError(err.message || 'DXF import failed');
    } finally {
      setBusy(false);
    }
  }, [dxfUpload, dxfPlacement, dxfVisibility, refreshGeo, onModelEdited]);

  const handleRemoveAuxFile = useCallback(async (fileName: string) => {
    try {
      await clearAuxiliaryLines({ fileName });
      setAuxVisibility((v) => { const n = { ...v }; delete n[fileName]; return n; });
      refreshGeo();
      onModelEdited?.();
    } catch (err: any) {
      setError(err.message || 'Failed to remove auxiliary lines');
    }
  }, [refreshGeo, onModelEdited]);

  if (!hasModel) {
    return (
      <div className="panel">
        <h2>Import OBJ</h2>
        <div className="alert alert-info">Generate a model first to enable import.</div>
      </div>
    );
  }

  return (
    <div className="three-col">
      <div className="panel edit-control-panel">
        <div className="edit-control-scroll">
          <h2>Import</h2>
          <div style={{ display: 'flex', gap: 6, marginBottom: 8 }}>
            <button type="button" disabled={busy}
                    className={`btn btn-xs${importMode === 'obj' ? ' btn-primary' : ' btn-ghost'}`}
                    onClick={() => { setError(null); setInfo(null); setWarning(null); setImportMode('obj'); }}>OBJ buildings</button>
            <button type="button" disabled={busy}
                    className={`btn btn-xs${importMode === 'dxf' ? ' btn-primary' : ' btn-ghost'}`}
                    onClick={() => { setError(null); setInfo(null); setWarning(null); setImportMode('dxf'); }}>DXF reference lines</button>
          </div>

          {importMode === 'obj' && (
          <>
          <GuidedSection index={1} label="UPLOAD">
            <button
              type="button"
              className="btn btn-secondary"
              style={{ width: '100%', cursor: busy ? 'not-allowed' : 'pointer', opacity: busy ? 0.6 : 1 }}
              disabled={busy}
              onClick={() => fileInputRef.current?.click()}
            >
              <Upload size={14} style={{ marginRight: 6 }} />
              {upload ? 'Replace OBJ…' : 'Choose OBJ (+ .mtl)…'}
            </button>
            <input
              ref={fileInputRef}
              type="file"
              accept=".obj,.mtl"
              multiple
              disabled={busy}
              style={{ position: 'absolute', width: 1, height: 1, padding: 0, margin: -1,
                       overflow: 'hidden', clip: 'rect(0,0,0,0)', whiteSpace: 'nowrap', border: 0 }}
              onChange={(e) => handleFile(e.target.files)}
            />
          </GuidedSection>

          {upload && (
            <GuidedSection index={2} label="GROUPS / ROLES">
              <table className="role-table" style={{ width: '100%', fontSize: '0.8rem' }}>
                <tbody>
                  {upload.groups.map((g) => (
                    <tr key={g.name}>
                      <td title={`${g.n_faces} faces`}>{g.name}</td>
                      <td style={{ textAlign: 'right' }}>
                        <select value={roles[g.name] ?? 'building'} disabled={busy}
                                onChange={(e) => setRoles((r) => ({ ...r, [g.name]: e.target.value }))}>
                          <option value="building">building</option>
                          <option value="window">window</option>
                          <option value="skip">skip</option>
                        </select>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </GuidedSection>
          )}

          {upload && (
            <GuidedSection index={3} label="PLACEMENT">
              <div className="guided-tool-hint">
                {placement.anchorLonLat
                  ? 'Edit the anchor below or click the map to set it.'
                  : 'Click the map or enter lat/lon below to set the anchor.'}
              </div>
              <div className="form-group">
                <label>Anchor latitude / longitude</label>
                <div style={{ display: 'flex', gap: 6 }}>
                  <input type="number" step="any" placeholder="lat" disabled={busy}
                         value={placement.anchorLonLat ? placement.anchorLonLat[1] : ''}
                         onChange={(e) => {
                           const lat = parseFloat(e.target.value);
                           if (Number.isNaN(lat)) return;
                           setPlacement((p) => ({
                             ...p,
                             anchorLonLat: [p.anchorLonLat ? p.anchorLonLat[0] : 0, lat],
                           }));
                         }} />
                  <input type="number" step="any" placeholder="lon" disabled={busy}
                         value={placement.anchorLonLat ? placement.anchorLonLat[0] : ''}
                         onChange={(e) => {
                           const lon = parseFloat(e.target.value);
                           if (Number.isNaN(lon)) return;
                           setPlacement((p) => ({
                             ...p,
                             anchorLonLat: [lon, p.anchorLonLat ? p.anchorLonLat[1] : 0],
                           }));
                         }} />
                </div>
              </div>
              <div className="form-group">
                <label>Anchor elevation (m, blank = auto from terrain)</label>
                <input type="number" step={0.5} disabled={busy}
                       value={placement.anchorElevation ?? ''}
                       onChange={(e) => setPlacement((p) => ({
                         ...p,
                         anchorElevation: e.target.value === '' ? null : parseFloat(e.target.value),
                       }))} />
              </div>
              <div className="form-group">
                <label>3D gizmo mode</label>
                <div style={{ display: 'flex', gap: 6 }}>
                  <button type="button" disabled={busy}
                          className={`btn btn-sm ${gizmoMode === 'translate' ? 'btn-primary' : 'btn-secondary'}`}
                          onClick={() => setGizmoMode('translate')}>
                    Move
                  </button>
                  <button type="button" disabled={busy}
                          className={`btn btn-sm ${gizmoMode === 'rotate' ? 'btn-primary' : 'btn-secondary'}`}
                          onClick={() => setGizmoMode('rotate')}>
                    Rotate
                  </button>
                </div>
              </div>
              <div className="form-group">
                <label>Rotation (deg)</label>
                <input type="number" step={1} value={placement.rotation} disabled={busy}
                       onChange={(e) => setPlacement((p) => ({ ...p, rotation: parseFloat(e.target.value) || 0 }))} />
              </div>
              <div className="form-group">
                <label>Move east / north / up (m)</label>
                <div style={{ display: 'flex', gap: 6 }}>
                  {[0, 1, 2].map((k) => (
                    <input key={k} type="number" step={0.5} value={placement.move[k]} disabled={busy}
                           onChange={(e) => setMove(k as 0 | 1 | 2, parseFloat(e.target.value) || 0)} />
                  ))}
                </div>
              </div>
              <div className="form-group">
                <label>Units</label>
                <select value={placement.units} disabled={busy}
                        onChange={(e) => setPlacement((p) => ({ ...p, units: e.target.value as Units }))}>
                  {UNIT_OPTIONS.map((u) => <option key={u} value={u}>{u}</option>)}
                </select>
              </div>

              <details open={advanced} onToggle={(e) => setAdvanced((e.target as HTMLDetailsElement).open)}>
                <summary>Advanced</summary>
                <label className="checkbox-row">
                  <input type="checkbox" checked={placement.zUp} disabled={busy}
                         onChange={(e) => setPlacement((p) => ({ ...p, zUp: e.target.checked }))} />
                  Z-up (uncheck for Y-up exports)
                </label>
                <label className="checkbox-row">
                  <input type="checkbox" checked={placement.swapYz} disabled={busy}
                         onChange={(e) => setPlacement((p) => ({ ...p, swapYz: e.target.checked }))} />
                  Swap Y/Z
                </label>
              </details>
            </GuidedSection>
          )}
          </>
          )}

          {importMode === 'dxf' && (
          <>
          <GuidedSection index={1} label="UPLOAD DXF">
            <button
              type="button"
              className="btn btn-secondary"
              style={{ width: '100%', cursor: busy ? 'not-allowed' : 'pointer', opacity: busy ? 0.6 : 1 }}
              disabled={busy}
              onClick={() => dxfFileInputRef.current?.click()}
            >
              <Upload size={14} style={{ marginRight: 6 }} />
              {dxfUpload ? 'Replace DXF…' : 'Choose DXF…'}
            </button>
            <input
              ref={dxfFileInputRef}
              type="file"
              accept=".dxf"
              disabled={busy}
              style={{ position: 'absolute', width: 1, height: 1, padding: 0, margin: -1,
                       overflow: 'hidden', clip: 'rect(0,0,0,0)', whiteSpace: 'nowrap', border: 0 }}
              onChange={(e) => handleDxfFile(e.target.files)}
            />
          </GuidedSection>

          {dxfUpload && (
            <GuidedSection index={2} label="LAYERS">
              <table className="role-table" style={{ width: '100%', fontSize: '0.8rem' }}>
                <tbody>
                  {dxfUpload.layers.map((l) => (
                    <tr key={l.name}>
                      <td>
                        <span style={{ display: 'inline-block', width: 10, height: 10,
                                       background: l.color, marginRight: 6, border: '1px solid #0003' }} />
                        {l.name}
                      </td>
                      <td style={{ textAlign: 'right' }} title={`${l.n_segments} segments`}>
                        <input type="checkbox" checked={dxfVisibility[l.name] !== false} disabled={busy}
                               onChange={(e) => setDxfVisibility((v) => ({ ...v, [l.name]: e.target.checked }))} />
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </GuidedSection>
          )}

          {dxfUpload && (
            <GuidedSection index={3} label="PLACEMENT">
              <div className="guided-tool-hint">
                {dxfPlacement.anchorLonLat
                  ? 'Edit the anchor below or click the map to set it.'
                  : 'Click the map or enter lat/lon below to set the anchor.'}
              </div>
              <div className="form-group">
                <label>Anchor latitude / longitude</label>
                <div style={{ display: 'flex', gap: 6 }}>
                  <input type="number" step="any" placeholder="lat" disabled={busy}
                         value={dxfPlacement.anchorLonLat ? dxfPlacement.anchorLonLat[1] : ''}
                         onChange={(e) => {
                           const lat = parseFloat(e.target.value);
                           if (Number.isNaN(lat)) return;
                           setDxfPlacement((p) => ({ ...p, anchorLonLat: [p.anchorLonLat ? p.anchorLonLat[0] : 0, lat] }));
                         }} />
                  <input type="number" step="any" placeholder="lon" disabled={busy}
                         value={dxfPlacement.anchorLonLat ? dxfPlacement.anchorLonLat[0] : ''}
                         onChange={(e) => {
                           const lon = parseFloat(e.target.value);
                           if (Number.isNaN(lon)) return;
                           setDxfPlacement((p) => ({ ...p, anchorLonLat: [lon, p.anchorLonLat ? p.anchorLonLat[1] : 0] }));
                         }} />
                </div>
              </div>
              <div className="form-group">
                <label>Rotation (deg)</label>
                <input type="number" step={1} value={dxfPlacement.rotation} disabled={busy}
                       onChange={(e) => setDxfPlacement((p) => ({ ...p, rotation: parseFloat(e.target.value) || 0 }))} />
              </div>
              <div className="form-group">
                <label>Move east / north (m)</label>
                <div style={{ display: 'flex', gap: 6 }}>
                  {[0, 1].map((k) => (
                    <input key={k} type="number" step={0.5} value={dxfPlacement.move[k]} disabled={busy}
                           onChange={(e) => setDxfPlacement((p) => {
                             const move = [...p.move] as [number, number, number];
                             move[k] = parseFloat(e.target.value) || 0;
                             return { ...p, move };
                           })} />
                  ))}
                </div>
              </div>
              <div className="form-group">
                <label>Units</label>
                <select value={dxfPlacement.units} disabled={busy}
                        onChange={(e) => setDxfPlacement((p) => ({ ...p, units: e.target.value as Units }))}>
                  {UNIT_OPTIONS.map((u) => <option key={u} value={u}>{u}</option>)}
                </select>
              </div>
            </GuidedSection>
          )}

          <AuxiliaryLinesControl
            geo={geo}
            visibility={auxVisibility}
            onToggle={(file, layer, visible) =>
              setAuxVisibility((v) => ({ ...v, [file]: { ...(v[file] ?? {}), [layer]: visible } }))}
            onRemoveFile={handleRemoveAuxFile}
            style={{ marginTop: 8 }}
          />
          </>
          )}

          <div className="guided-feedback-slot">
            {error && <div className="alert alert-error">{error}</div>}
            {warning && <div className="alert alert-warning">{warning}</div>}
            {info && <div className="alert alert-success">{info}</div>}
          </div>
        </div>

        <div className="pending-edit-footer">
          {importMode === 'obj' ? (
            <button className="btn btn-primary pending-update-btn"
                    onClick={handleImport}
                    disabled={!upload || busy || !placement.anchorLonLat}
                    type="button">
              {busy && <span className="spinner" />}
              <Boxes size={14} style={{ marginRight: 6 }} />
              {busy ? 'Importing…' : 'Import building(s)'}
            </button>
          ) : (
            <button className="btn btn-primary pending-update-btn"
                    onClick={handleDxfImport}
                    disabled={!dxfUpload || busy || !dxfPlacement.anchorLonLat}
                    type="button">
              {busy && <span className="spinner" />}
              <Boxes size={14} style={{ marginRight: 6 }} />
              {busy ? 'Adding…' : 'Add reference lines'}
            </button>
          )}
        </div>
      </div>

      {/* 2D map */}
      <div className="panel visual-panel">
        <div className="plan-panel-header"><h2>2D placement</h2></div>
        <div className="visual-frame">
          {importMode === 'obj' && (geo && upload ? (
            <ObjPlacementMap
              geo={geo}
              placement={placement}
              footprints={upload.preview.footprints}
              onAnchor={(lonLat) => setPlacement((p) => ({ ...p, anchorLonLat: lonLat }))}
            />
          ) : (
            <div className="alert alert-info">Upload an OBJ, then click the map to set the anchor.</div>
          ))}
          {importMode === 'dxf' && (geo && dxfUpload ? (
            <DxfPlacementMap
              geo={geo}
              placement={dxfPlacement}
              layers={dxfUpload.preview.layers}
              visibility={dxfVisibility}
              onAnchor={(lonLat) => setDxfPlacement((p) => ({ ...p, anchorLonLat: lonLat }))}
            />
          ) : (
            <div className="alert alert-info">Upload a DXF, then click the map to set the anchor.</div>
          ))}
        </div>
      </div>

      {/* 3D result */}
      <div className="panel visual-panel">
        <div className="plan-panel-header"><h2>3D result</h2></div>
        <div className="visual-frame">
          {importMode === 'dxf' ? (
            previewDisabled ? (
              <PreviewDisabledNotice gridShape={previewGridShape} />
            ) : geo?.auxiliary_lines && geo.auxiliary_lines.length > 0 ? (
              <SceneViewer
                geometryToken="import-dxf-preview"
                lonLatToXY={geo ? lonLatToUvM({ grid_geom: geo.grid_geom }) : undefined}
                auxiliaryLines={geo.auxiliary_lines}
                auxiliaryLineVisibility={auxVisibility}
              />
            ) : (
              <div className="alert alert-info">
                DXF reference lines are a flat overlay added to the 2D map and the
                3D scene without changing the voxel model. Add lines to see them here.
              </div>
            )
          ) : previewDisabled ? (
            <PreviewDisabledNotice gridShape={previewGridShape} />
          ) : upload && !figureJson ? (
            <SceneViewer
              geometryToken="import-preview"
              lonLatToXY={geo ? lonLatToUvM({ grid_geom: geo.grid_geom }) : undefined}
              auxiliaryLines={geo?.auxiliary_lines}
              auxiliaryLineVisibility={auxVisibility}
              placementPreview={{
                vertices: upload.preview.vertices,
                indices: upload.preview.indices,
                placement,
                anchorScene,
                domainRotationDeg: phiDeg,
                mode: gizmoMode,
                onChange: handlePlacementChange,
              }}
            />
          ) : figureJson ? (
            <ThreeViewer figureJson={figureJson} />
          ) : (
            <div className="alert alert-info">Upload an OBJ to place it in 3D.</div>
          )}
        </div>
      </div>
    </div>
  );
};

export default ImportTab;
