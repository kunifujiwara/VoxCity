/**
 * Zoning tab – draw 2D footprints over the current model and visualize them
 * as 3D curtains. Zones are pure frontend state owned by `App.tsx`; this tab
 * only mutates them via `onZonesChange`.
 */

import React, { useCallback, useEffect, useMemo, useState } from 'react';
import { getModelGeo, ModelGeoResult } from '../api';
import ThreeViewer from '../components/ThreeViewer';
import PlanMapEditor, {
  Backdrop,
  BasemapKey,
  MapInteraction,
  PendingEdit,
} from '../components/PlanMapEditor';
import { polygonToCells } from '../lib/grid';
import { buildZoneTraces } from '../lib/zoneTraces';
import {
  Zone,
  ZoneShape,
  ZONE_PALETTE,
  makeZoneId,
  nextZoneColor,
  nextZoneName,
} from '../types/zones';

interface ZoningTabProps {
  hasModel: boolean;
  figureJson: string;
  zones: Zone[];
  onZonesChange: (z: Zone[]) => void;
}

function maxBuildingHeight(geo: ModelGeoResult | null): number {
  if (!geo || !geo.building_geojson) return 30;
  let m = 0;
  for (const f of geo.building_geojson.features ?? []) {
    const h = Number(f.properties?.height ?? 0);
    if (Number.isFinite(h) && h > m) m = h;
  }
  return m > 0 ? m : 30;
}

const ZoningTab: React.FC<ZoningTabProps> = ({ hasModel, figureJson, zones, onZonesChange }) => {
  const [geo, setGeo] = useState<ModelGeoResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const [shape, setShape] = useState<ZoneShape>('rect');
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [renamingId, setRenamingId] = useState<string | null>(null);
  const [renameBuf, setRenameBuf] = useState('');
  const [colorPickerId, setColorPickerId] = useState<string | null>(null);
  const [backdrop, setBackdrop] = useState<Backdrop>('buildings');
  const [basemap, setBasemap] = useState<BasemapKey>('CartoDB Positron');

  // Initial fetch
  useEffect(() => {
    if (!hasModel) return;
    setLoading(true);
    setError(null);
    getModelGeo()
      .then(setGeo)
      .catch((e: Error) => setError(e.message))
      .finally(() => setLoading(false));
  }, [hasModel]);

  const maxH = useMemo(() => maxBuildingHeight(geo), [geo]);

  const interaction: MapInteraction = shape === 'rect' ? 'draw_rect_3pt' : 'draw_polygon';

  // Render existing zones as paint_zone overlays.
  const pendingEdits: PendingEdit[] = useMemo(() => {
    if (!geo) return [];
    return zones.map((z) => ({
      kind: 'paint_zone' as const,
      cells: polygonToCells(z.ring_lonlat, geo.grid_geom),
      color: z.color,
      target: 'evaluation' as const,
    }));
  }, [geo, zones]);

  const handlePolygonComplete = useCallback(
    (ring: [number, number][]) => {
      const newZone: Zone = {
        id: makeZoneId(),
        name: nextZoneName(zones),
        color: nextZoneColor(zones),
        shape,
        ring_lonlat: ring,
      };
      onZonesChange([...zones, newZone]);
      setSelectedId(newZone.id);
    },
    [zones, shape, onZonesChange],
  );

  const updateZone = (id: string, patch: Partial<Zone>) => {
    onZonesChange(zones.map((z) => (z.id === id ? { ...z, ...patch } : z)));
  };

  const deleteZone = (id: string) => {
    if (!window.confirm('Delete this zone?')) return;
    onZonesChange(zones.filter((z) => z.id !== id));
    if (selectedId === id) setSelectedId(null);
  };

  const clearAll = () => {
    if (zones.length === 0) return;
    if (!window.confirm(`Delete all ${zones.length} zones?`)) return;
    onZonesChange([]);
    setSelectedId(null);
  };

  // Build figure with zone curtain traces appended.
  const figureWithZones = useMemo(() => {
    if (!figureJson) return '';
    if (!geo || zones.length === 0) return figureJson;
    try {
      const fig = JSON.parse(figureJson);
      fig.data = [
        ...(fig.data ?? []),
        ...buildZoneTraces(zones, { grid_geom: geo.grid_geom }, {
          meshsize: geo.meshsize_m,
          ceilingM: Math.max(0.1 * maxH, 3),
          selectedZoneId: selectedId,
        }),
      ];
      return JSON.stringify(fig);
    } catch {
      return figureJson;
    }
  }, [figureJson, geo, zones, maxH, selectedId]);

  if (!hasModel) {
    return (
      <div className="alert alert-info">
        Generate a model on the <strong>Generation</strong> tab first, then come back here to define zones.
      </div>
    );
  }

  return (
    <div className="three-col">
      {/* Left: config + zone list */}
      <div className="panel" style={{ overflow: 'auto' }}>
        <h2>Zoning</h2>
        {error && <div className="alert alert-error">{error}</div>}
        <div className="form-group">
          <label>Shape</label>
          <div style={{ display: 'flex', gap: 8 }}>
            <button
              type="button"
              className={`btn ${shape === 'rect' ? 'btn-primary' : 'btn-secondary'}`}
              onClick={() => setShape('rect')}
            >
              Rectangle
            </button>
            <button
              type="button"
              className={`btn ${shape === 'polygon' ? 'btn-primary' : 'btn-secondary'}`}
              onClick={() => setShape('polygon')}
            >
              Polygon
            </button>
          </div>
        </div>

        <div className="form-group">
          <label>Backdrop</label>
          <select value={backdrop} onChange={(e) => setBackdrop(e.target.value as Backdrop)}>
            <option value="buildings">Buildings</option>
            <option value="canopy">Canopy</option>
            <option value="land_cover">Land cover</option>
            <option value="none">None</option>
          </select>
        </div>

        <div className="form-group">
          <label>Basemap</label>
          <select value={basemap} onChange={(e) => setBasemap(e.target.value as BasemapKey)}>
            <option>CartoDB Positron</option>
            <option>Google Satellite</option>
            <option>OpenStreetMap</option>
          </select>
        </div>

        <button className="btn btn-secondary" onClick={clearAll} disabled={zones.length === 0}>
          Clear all zones
        </button>

        <div className="zone-list">
          {zones.length === 0 && (
            <div className="alert alert-info" style={{ marginTop: 8 }}>
              Draw a {shape === 'rect' ? 'rectangle' : 'polygon'} on the map to add a zone.
            </div>
          )}
          {zones.map((z) => (
            <div
              key={z.id}
              className={`zone-row${selectedId === z.id ? ' selected' : ''}`}
              onClick={() => setSelectedId(z.id)}
            >
              <span
                className="swatch"
                style={{ background: z.color }}
                onClick={(e) => {
                  e.stopPropagation();
                  setColorPickerId(colorPickerId === z.id ? null : z.id);
                }}
              />
              {renamingId === z.id ? (
                <input
                  className="name"
                  autoFocus
                  value={renameBuf}
                  onChange={(e) => setRenameBuf(e.target.value)}
                  onBlur={() => {
                    if (renameBuf.trim()) updateZone(z.id, { name: renameBuf.trim() });
                    setRenamingId(null);
                  }}
                  onKeyDown={(e) => {
                    if (e.key === 'Enter') (e.target as HTMLInputElement).blur();
                    if (e.key === 'Escape') setRenamingId(null);
                  }}
                />
              ) : (
                <span className="name">{z.name}</span>
              )}
              <button
                title="Rename"
                onClick={(e) => {
                  e.stopPropagation();
                  setRenamingId(z.id);
                  setRenameBuf(z.name);
                }}
              >
                ✎
              </button>
              <button
                title="Delete"
                onClick={(e) => {
                  e.stopPropagation();
                  deleteZone(z.id);
                }}
              >
                🗑
              </button>
            </div>
          ))}
          {colorPickerId && (
            <div style={{ display: 'flex', gap: 4, marginTop: 4, flexWrap: 'wrap' }}>
              {ZONE_PALETTE.map((c) => (
                <span
                  key={c}
                  className="swatch"
                  style={{ background: c, width: 20, height: 20 }}
                  onClick={() => {
                    updateZone(colorPickerId, { color: c });
                    setColorPickerId(null);
                  }}
                />
              ))}
            </div>
          )}
        </div>
      </div>

      {/* Center: 2D editor */}
      <div className="panel" style={{ overflow: 'hidden', display: 'flex', flexDirection: 'column' }}>
        <h2>2D zone editor</h2>
        {loading && <div className="alert alert-info">Loading map…</div>}
        {geo && (
          <PlanMapEditor
            geo={geo}
            interaction={interaction}
            backdrop={backdrop}
            basemap={basemap}
            drawColor="blue"
            pendingEdits={pendingEdits}
            onPolygonComplete={handlePolygonComplete}
          />
        )}
      </div>

      {/* Right: 3D viewer */}
      <div className="panel" style={{ overflow: 'hidden', display: 'flex', flexDirection: 'column' }}>
        <h2>3D preview</h2>
        <div style={{ flex: '1 1 auto', minHeight: 0, display: 'flex', flexDirection: 'column' }}>
          {figureWithZones ? (
            <ThreeViewer figureJson={figureWithZones} />
          ) : (
            <div className="alert alert-info" style={{ marginTop: 0 }}>
              Generate a 3D figure (e.g. on the Generation tab) to preview zones in 3D.
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default ZoningTab;
