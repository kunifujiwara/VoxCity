/**
 * Import tab — upload an OBJ, position it, and stamp its buildings into the model.
 *
 * Placement lives in one `Placement` object (lib/objPlacement). The numeric form
 * here writes it; the 2D map (Task 8) and 3D gizmo (Task 9) read/write the same
 * object. Commit calls /api/model/import_obj/commit and renders the result.
 */
import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { Upload, Boxes } from 'lucide-react';
import {
  uploadImportObj,
  commitImportObj,
  getModelGeo,
  getAnchorGround,
  AnchorGroundResult,
  ImportObjUploadResult,
  ModelGeoResult,
} from '../api';
import { GuidedSection } from '../components/guided';
import ThreeViewer from '../components/ThreeViewer';
import ObjPlacementMap from '../components/ObjPlacementMap';
import { SceneViewer } from '../three';
import { lonLatToUvM } from '../lib/grid';
import {
  defaultPlacement,
  Placement,
  Units,
} from '../lib/objPlacement';

interface ImportTabProps {
  hasModel: boolean;
  figureJson: string;
  onFigureChange: (s: string) => void;
  onModelEdited?: () => void;
}

const UNIT_OPTIONS: Units[] = ['m', 'cm', 'mm', 'ft', 'in'];

const ImportTab: React.FC<ImportTabProps> = ({ hasModel, figureJson, onFigureChange, onModelEdited }) => {
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
    let up = 0;
    if (anchorGround) {
      const effElev = placement.anchorElevation ?? anchorGround.dem_elevation;
      up = effElev - anchorGround.dem_min + anchorGround.meshsize_m;
    }
    return [east, north, up];
  }, [geo, placement.anchorLonLat, placement.anchorElevation, anchorGround]);

  const handleFile = useCallback(async (file: File | null) => {
    if (!file) return;
    setBusy(true); setError(null); setInfo(null); setWarning(null);
    try {
      const res = await uploadImportObj(file);
      setUpload(res);
      setRoles(Object.fromEntries(res.groups.map((g) => [g.name, g.role])));
      setInfo(`Loaded ${res.groups.length} group(s). Position it and import.`);
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
        setInfo(`Imported ${r.imported_building_ids.length} building(s); ${r.n_building_voxels_added} voxel(s) added.`);
      }
    } catch (err: any) {
      setError(err.message || 'Import failed');
    } finally {
      setBusy(false);
    }
  }, [upload, placement, roles, onFigureChange, onModelEdited]);

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
          <h2>Import OBJ</h2>

          <GuidedSection index={1} label="UPLOAD">
            <button
              type="button"
              className="btn btn-secondary"
              style={{ width: '100%', cursor: busy ? 'not-allowed' : 'pointer', opacity: busy ? 0.6 : 1 }}
              disabled={busy}
              onClick={() => fileInputRef.current?.click()}
            >
              <Upload size={14} style={{ marginRight: 6 }} />
              {upload ? 'Replace OBJ…' : 'Choose OBJ file…'}
            </button>
            <input
              ref={fileInputRef}
              type="file"
              accept=".obj"
              disabled={busy}
              style={{ position: 'absolute', width: 1, height: 1, padding: 0, margin: -1,
                       overflow: 'hidden', clip: 'rect(0,0,0,0)', whiteSpace: 'nowrap', border: 0 }}
              onChange={(e) => handleFile(e.target.files?.[0] ?? null)}
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

          <div className="guided-feedback-slot">
            {error && <div className="alert alert-error">{error}</div>}
            {warning && <div className="alert alert-warning">{warning}</div>}
            {info && <div className="alert alert-success">{info}</div>}
          </div>
        </div>

        <div className="pending-edit-footer">
          <button className="btn btn-primary pending-update-btn"
                  onClick={handleImport}
                  disabled={!upload || busy || !placement.anchorLonLat}
                  type="button">
            {busy && <span className="spinner" />}
            <Boxes size={14} style={{ marginRight: 6 }} />
            {busy ? 'Importing…' : 'Import building(s)'}
          </button>
        </div>
      </div>

      {/* 2D map */}
      <div className="panel visual-panel">
        <div className="plan-panel-header"><h2>2D placement</h2></div>
        <div className="visual-frame">
          {geo && upload ? (
            <ObjPlacementMap
              geo={geo}
              placement={placement}
              footprints={upload.preview.footprints}
              onAnchor={(lonLat) => setPlacement((p) => ({ ...p, anchorLonLat: lonLat }))}
            />
          ) : (
            <div className="alert alert-info">Upload an OBJ, then click the map to set the anchor.</div>
          )}
        </div>
      </div>

      {/* 3D result */}
      <div className="panel visual-panel">
        <div className="plan-panel-header"><h2>3D result</h2></div>
        <div className="visual-frame">
          {upload && !figureJson ? (
            <SceneViewer
              geometryToken="import-preview"
              placementPreview={{
                vertices: upload.preview.vertices,
                indices: upload.preview.indices,
                placement,
                anchorScene,
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
