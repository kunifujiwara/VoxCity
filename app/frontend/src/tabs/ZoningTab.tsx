/**
 * Zoning tab – draw 2D footprints over the current model and visualize them
 * as 3D curtains. Zones are pure frontend state owned by `App.tsx`; this tab
 * only mutates them via `onZonesChange`.
 */

import React, { useCallback, useEffect, useMemo, useState } from 'react';
import { getModelGeo, ModelGeoResult } from '../api';
import { SceneViewer } from '../three';
import PlanMapEditor, {
  Backdrop,
  BasemapKey,
  MapInteraction,
  PendingEdit,
} from '../components/PlanMapEditor';
import { lonLatToWorldXY, polygonToCells } from '../lib/grid';
import {
  Zone,
  ZoneShape,
  ZONE_PALETTE,
  makeZoneGroupId,
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
  /**
   * Group id that subsequent drawn polygons attach to. Null means
   * "attach to the most recently used group" (or create the first one
   * implicitly).
   */
  const [activeGroupId, setActiveGroupId] = useState<string | null>(null);

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

  /**
   * Project (lon, lat) into voxel-world metres so zone outlines align with
   * the city geometry returned by `/api/scene/geometry`. Locked-in projection
   * is `(nx - u, v)` (see `lonLatToWorldXY` doc).
   */
  const lonLatToXY = useMemo(() => lonLatToWorldXY(geo), [geo]);

  const interaction: MapInteraction = shape === 'rect' ? 'draw_rect_3pt' : 'draw_polygon';

  // Render existing zones as paint_zone overlays.
  const pendingEdits: PendingEdit[] = useMemo(() => {
    if (!geo) return [];
    return zones.map((z) => ({
      kind: 'paint_zone' as const,
      cells: polygonToCells(z.ring_lonlat, geo.grid_geom),
      ring: z.ring_lonlat,
      selected:
        z.id === selectedId ||
        (activeGroupId != null && (z.groupId ?? z.id) === activeGroupId),
      color: z.color,
      target: 'evaluation' as const,
    }));
  }, [geo, zones, selectedId, activeGroupId]);

  /**
   * Group sibling zones (same `groupId`) together so the zone list shows
   * one row per logical zone. The first member's metadata acts as the
   * group's name/colour/shape.
   */
  const groups = useMemo(() => {
    const map = new Map<string, { id: string; name: string; color: string; members: Zone[] }>();
    for (const z of zones) {
      const key = z.groupId ?? z.id;
      const g = map.get(key);
      if (g) {
        g.members.push(z);
      } else {
        map.set(key, { id: key, name: z.name, color: z.color, members: [z] });
      }
    }
    return Array.from(map.values());
  }, [zones]);

  // Resolve which group new polygons attach to: explicit `activeGroupId`,
  // else the most recent group, else "start a new one".
  const effectiveActiveGroupId =
    activeGroupId ?? (groups.length > 0 ? groups[groups.length - 1].id : null);

  const handleAddZone = useCallback(() => {
    setActiveGroupId(makeZoneGroupId());
  }, []);

  const handlePolygonComplete = useCallback(
    (ring: [number, number][]) => {
      // Look up the active group, if any.
      const targetGroupId = effectiveActiveGroupId;
      const groupMembers = targetGroupId
        ? zones.filter((z) => (z.groupId ?? z.id) === targetGroupId)
        : [];
      const head = groupMembers[0];
      const newZone: Zone = head
        ? {
            id: makeZoneId(),
            name: head.name,
            color: head.color,
            shape,
            ring_lonlat: ring,
            groupId: head.groupId ?? head.id,
          }
        : {
            id: makeZoneId(),
            name: nextZoneName(zones),
            color: nextZoneColor(zones),
            shape,
            ring_lonlat: ring,
            groupId: targetGroupId ?? makeZoneGroupId(),
          };
      onZonesChange([...zones, newZone]);
      setSelectedId(newZone.id);
      setActiveGroupId(newZone.groupId ?? null);
    },
    [zones, shape, onZonesChange, effectiveActiveGroupId],
  );

  const updateGroup = (groupId: string, patch: Partial<Pick<Zone, 'name' | 'color'>>) => {
    onZonesChange(
      zones.map((z) => ((z.groupId ?? z.id) === groupId ? { ...z, ...patch } : z)),
    );
  };

  const deleteGroup = (groupId: string) => {
    if (!window.confirm('Delete this zone (all its rings)?')) return;
    onZonesChange(zones.filter((z) => (z.groupId ?? z.id) !== groupId));
    if (selectedId && zones.find((z) => z.id === selectedId)?.groupId === groupId) {
      setSelectedId(null);
    }
    if (activeGroupId === groupId) setActiveGroupId(null);
  };

  const clearAll = () => {
    if (zones.length === 0) return;
    if (!window.confirm(`Delete all ${groups.length} zones?`)) return;
    onZonesChange([]);
    setSelectedId(null);
    setActiveGroupId(null);
  };

  // Use each zone's own colour in 3D so the right-hand viewer matches the
  // left-hand 2D editor exactly. Selection emphasis happens on the 2D side.
  const colorOverride = undefined;

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

        <div style={{ display: 'flex', gap: 8 }}>
          <button
            type="button"
            className="btn btn-primary"
            onClick={handleAddZone}
            title="Start a new zone. Subsequent polygons join the active zone."
          >
            + Add a zone
          </button>
          <button className="btn btn-secondary" onClick={clearAll} disabled={zones.length === 0}>
            Clear all zones
          </button>
        </div>

        <div className="zone-list">
          {groups.length === 0 && (
            <div className="alert alert-info" style={{ marginTop: 8 }}>
              Draw a {shape === 'rect' ? 'rectangle' : 'polygon'} on the map to add a zone.
            </div>
          )}
          {groups.map((g) => {
            const isActive = (effectiveActiveGroupId ?? null) === g.id;
            return (
              <div
                key={g.id}
                className={`zone-row${isActive ? ' selected' : ''}`}
                onClick={() => {
                  setActiveGroupId(g.id);
                  setSelectedId(g.members[0]?.id ?? null);
                }}
              >
                <span
                  className="swatch"
                  style={{ background: g.color }}
                  onClick={(e) => {
                    e.stopPropagation();
                    setColorPickerId(colorPickerId === g.id ? null : g.id);
                  }}
                />
                {renamingId === g.id ? (
                  <input
                    className="name"
                    autoFocus
                    value={renameBuf}
                    onChange={(e) => setRenameBuf(e.target.value)}
                    onBlur={() => {
                      if (renameBuf.trim()) updateGroup(g.id, { name: renameBuf.trim() });
                      setRenamingId(null);
                    }}
                    onKeyDown={(e) => {
                      if (e.key === 'Enter') (e.target as HTMLInputElement).blur();
                      if (e.key === 'Escape') setRenamingId(null);
                    }}
                  />
                ) : (
                  <span className="name">
                    {g.name}
                    {g.members.length > 1 && (
                      <span style={{ opacity: 0.6, marginLeft: 6, fontSize: '0.85em' }}>
                        ×{g.members.length}
                      </span>
                    )}
                  </span>
                )}
                <button
                  title="Rename"
                  onClick={(e) => {
                    e.stopPropagation();
                    setRenamingId(g.id);
                    setRenameBuf(g.name);
                  }}
                >
                  ✎
                </button>
                <button
                  title="Delete"
                  onClick={(e) => {
                    e.stopPropagation();
                    deleteGroup(g.id);
                  }}
                >
                  🗑
                </button>
              </div>
            );
          })}
          {colorPickerId && (
            <div style={{ display: 'flex', gap: 4, marginTop: 4, flexWrap: 'wrap' }}>
              {ZONE_PALETTE.map((c) => (
                <span
                  key={c}
                  className="swatch"
                  style={{ background: c, width: 20, height: 20 }}
                  onClick={() => {
                    updateGroup(colorPickerId, { color: c });
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
          {hasModel ? (
            <SceneViewer
              geometryToken={hasModel ? 'loaded' : 'none'}
              downsample={1}
              zones={zones}
              lonLatToXY={lonLatToXY}
              showZones
              colorOverride={colorOverride}
            />
          ) : (
            <div className="alert alert-info" style={{ marginTop: 0 }}>
              Generate a model on the Generation tab to preview zones in 3D.
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default ZoningTab;
