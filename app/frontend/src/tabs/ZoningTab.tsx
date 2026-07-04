/**
 * Zoning tab – draw 2D footprints over the current model and visualize them
 * as 3D curtains. Zones are pure frontend state owned by `App.tsx`; this tab
 * only mutates them via `onZonesChange`.
 */

import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { Square, Box, Hexagon, Plus, Trash2 } from 'lucide-react';
import { getModelGeo, type ModelGeoResult, getBuildingAt } from '../api';
import { SceneViewer } from '../three';
import PreviewDisabledNotice from '../components/PreviewDisabledNotice';
import PlanMapEditor, {
  Backdrop,
  BasemapKey,
  MapInteraction,
  PendingEdit,
} from '../components/PlanMapEditor';
import { lonLatToUvM, polygonToCells } from '../lib/grid';
import {
  Zone,
  ZoneShape,
  ZONE_PALETTE,
  makeZoneGroupId,
  makeZoneId,
  nextZoneColor,
  nextZoneName,
  ZoneType,
  BuildingSurfaceZone,
  SurfaceSelector,
  toggleWholeBuilding,
  toggleSurfaceFace,
  toggleBulkSelector,
  buildingHasPositiveSelection,
  WallOrientation,
  resolveZoneGroupForMode,
  zoneGroupType,
  zoneTypeShortLabel,
} from '../types/zones';
import type { PickResult } from '../three/types';
import { useSurfaceZoneSelection } from '../hooks/useSurfaceZoneSelection';
import { getSurfaceZones, shouldEnableZoningSurfaceSelection } from '../three/surfaceSelection';
import { ChoiceGroup, GuidedFooter, GuidedPanel, GuidedSection, GuidedStatus } from '../components/guided';
import { prerequisiteMessageForTab } from './guidedTabState';

interface ZoningTabProps {
  hasModel: boolean;
  figureJson: string;
  zones: Zone[];
  onZonesChange: (z: Zone[]) => void;
  geometryToken?: string | number;
  previewDisabled?: boolean;
  previewGridShape?: number[] | null;
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

interface DraftZoneGroup {
  id: string;
  name: string;
  color: string;
  shape: ZoneShape;
}

const ZoningTab: React.FC<ZoningTabProps> = ({ hasModel, figureJson, zones, onZonesChange, geometryToken, previewDisabled = false, previewGridShape }) => {
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
  const [draftGroups, setDraftGroups] = useState<DraftZoneGroup[]>([]);
  const [groupOrder, setGroupOrder] = useState<string[]>([]);
  const [zoneType, setZoneType] = useState<ZoneType>('horizontal');
  const [refiningBuildingId, setRefiningBuildingId] = useState<number | null>(null);

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

  // Ensure there is always at least one visible row when the tab is empty.
  useEffect(() => {
    if (zones.length === 0 && draftGroups.length === 0) {
      const id = makeZoneGroupId();
      setDraftGroups([{ id, name: 'Zone 1', color: ZONE_PALETTE[0], shape }]);
      setGroupOrder([id]);
      setActiveGroupId(id);
    }
  }, [zones.length, draftGroups.length, shape]);

  /**
   * Project (lon, lat) into voxel-world metres so zone outlines align with
   * the city geometry returned by `/api/scene/geometry`. Locked-in projection
   * is `(u*du, v*dv)` (see `lonLatToUvM` doc).
   */
  const lonLatToXY = useMemo(() => lonLatToUvM(geo), [geo]);

  const interaction: MapInteraction = shape === 'rect' ? 'draw_rect_3pt' : 'draw_polygon';

  /**
   * Group sibling zones (same `groupId`) together so the zone list shows
   * one row per logical zone. The first member's metadata acts as the
   * group's name/colour/shape.
   */
  const committedGroups = useMemo(() => {
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

  const groups = useMemo(() => {
    const committed = committedGroups.map((g) => ({ ...g, draft: false as const }));
    const committedIds = new Set(committedGroups.map((g) => g.id));
    const drafts = draftGroups
      .filter((g) => !committedIds.has(g.id))
      .map((g) => ({ ...g, members: [] as Zone[], draft: true as const }));
    const all = new Map([...committed, ...drafts].map((g) => [g.id, g]));
    const ordered = groupOrder.filter((id) => all.has(id)).map((id) => all.get(id)!);
    const untracked = [...all.values()].filter((g) => !groupOrder.includes(g.id));
    return [...ordered, ...untracked];
  }, [committedGroups, draftGroups, groupOrder]);

  const effectiveActiveGroupId = useMemo(
    () => resolveZoneGroupForMode({ zones, candidates: groups, activeGroupId, zoneType }),
    [zones, groups, activeGroupId, zoneType],
  );

  // Render existing zones as paint_zone overlays.
  const pendingEdits: PendingEdit[] = useMemo(() => {
    if (!geo) return [];
    return zones.flatMap((z) => {
      if (z.type !== 'horizontal') return [];
      return [{
        kind: 'paint_zone' as const,
        cells: polygonToCells(z.ring_lonlat, geo.grid_geom),
        ring: z.ring_lonlat,
        selected: zoneType === 'horizontal' && (
          z.id === selectedId ||
          (effectiveActiveGroupId != null && (z.groupId ?? z.id) === effectiveActiveGroupId)
        ),
        color: z.color,
        target: 'evaluation' as const,
      }];
    });
  }, [geo, zones, selectedId, effectiveActiveGroupId, zoneType]);
  const surfaceZoneCount = useMemo(() => getSurfaceZones(zones).length, [zones]);
  const surfaceSelectionEnabled = shouldEnableZoningSurfaceSelection({ zoneType, surfaceZoneCount });

  const { surfaceSelection } = useSurfaceZoneSelection({
    hasModel,
    geometryToken,
    zones,
    enabled: surfaceSelectionEnabled,
    displayMode: 'fill',
    activeGroupId: effectiveActiveGroupId,
    requireSurfaceZones: false,
    requireSelectors: false,
    silentOnError: true,
  });

  const activeSurfaceZone = useMemo(
    () => zones.find(
      (zone): zone is BuildingSurfaceZone =>
        zone.type === 'building_surface' && (zone.groupId ?? zone.id) === effectiveActiveGroupId,
    ),
    [zones, effectiveActiveGroupId],
  );

  useEffect(() => {
    if (refiningBuildingId == null) return;
    if (
      zoneType !== 'building_surface' ||
      !activeSurfaceZone ||
      !buildingHasPositiveSelection(activeSurfaceZone.selectors, refiningBuildingId)
    ) {
      setRefiningBuildingId(null);
    }
  }, [zoneType, activeSurfaceZone, refiningBuildingId]);

  const nextDraftMeta = useCallback(() => {
    const usedNames = new Set([...zones.map((z) => z.name), ...draftGroups.map((g) => g.name)]);
    let n = committedGroups.length + draftGroups.length + 1;
    while (usedNames.has(`Zone ${n}`)) n += 1;
    return {
      name: `Zone ${n}`,
      color: ZONE_PALETTE[(committedGroups.length + draftGroups.length) % ZONE_PALETTE.length],
    };
  }, [zones, committedGroups.length, draftGroups]);

  const handleAddZone = useCallback(() => {
    const id = makeZoneGroupId();
    const { name, color } = nextDraftMeta();
    if (zoneType === 'building_surface') {
      // Immediately commit as a BuildingSurfaceZone (no drawing needed)
      const newZone: BuildingSurfaceZone = {
        type: 'building_surface',
        id: makeZoneId(),
        name,
        color,
        selectors: [],
        groupId: id,
      };
      onZonesChange([...zones, newZone]);
      setGroupOrder((prev) => [...prev, id]);
      setRefiningBuildingId(null);
      setActiveGroupId(id);
      setSelectedId(newZone.id);
    } else {
      setDraftGroups((prev) => [...prev, { id, name, color, shape }]);
      setGroupOrder((prev) => [...prev, id]);
      setActiveGroupId(id);
      setSelectedId(null);
    }
  }, [nextDraftMeta, shape, zoneType, zones, onZonesChange]);

  const handlePolygonComplete = useCallback(
    (ring: [number, number][]) => {
      // Look up the active group, if any.
      const targetGroupId = effectiveActiveGroupId;
      const draft = targetGroupId ? draftGroups.find((g) => g.id === targetGroupId) : undefined;
      const groupMembers = targetGroupId
        ? zones.filter((z) => (z.groupId ?? z.id) === targetGroupId)
        : [];
      const head = groupMembers[0];
      const fallbackGroupId = targetGroupId ?? makeZoneGroupId();
      const newZone: Zone = head
        ? {
            type: 'horizontal' as const,
            id: makeZoneId(),
            name: head.name,
            color: head.color,
            shape,
            ring_lonlat: ring,
            groupId: head.groupId ?? head.id,
          }
        : {
            type: 'horizontal' as const,
            id: makeZoneId(),
            name: draft?.name ?? nextZoneName(zones),
            color: draft?.color ?? nextZoneColor(zones),
            shape,
            ring_lonlat: ring,
            groupId: draft?.id ?? fallbackGroupId,
          };
      onZonesChange([...zones, newZone]);
      if (draft) {
        setDraftGroups((prev) => prev.filter((g) => g.id !== draft.id));
      }
      setSelectedId(newZone.id);
      setActiveGroupId(newZone.groupId ?? null);
    },
    [zones, shape, onZonesChange, effectiveActiveGroupId, draftGroups],
  );

  const handleSurfacePick = useCallback(
    async (hit: PickResult | null) => {
      if (!hit) return;
      if (!activeSurfaceZone) return;

      let buildingId = hit.buildingId;
      if (buildingId == null && hit.surface == null) {
        // Try to resolve building from world coordinates
        try {
          const res = await getBuildingAt(hit.point[0], hit.point[1]);
          buildingId = res.building_id;
        } catch {
          return;
        }
      }

      if (refiningBuildingId != null) {
        // Refining mode: face-level toggle
        if (hit.surface && hit.surface.buildingId === refiningBuildingId) {
          const updatedSelectors = toggleSurfaceFace(activeSurfaceZone.selectors, {
            buildingId: hit.surface.buildingId,
            faceKey: hit.surface.faceKey,
            surfaceKind: hit.surface.surfaceKind,
            orientation: hit.surface.orientation,
          });
          onZonesChange(
            zones.map((z) =>
              z.id === activeSurfaceZone.id
                ? { ...activeSurfaceZone, selectors: updatedSelectors }
                : z,
            ),
          );
        }
        return;
      }

      // Idle mode: whole-building toggle
      if (buildingId != null) {
        const updatedSelectors = toggleWholeBuilding(activeSurfaceZone.selectors, buildingId);
        onZonesChange(
          zones.map((z) =>
            z.id === activeSurfaceZone.id
              ? { ...activeSurfaceZone, selectors: updatedSelectors }
              : z,
          ),
        );
      }
    },
    [activeSurfaceZone, zones, refiningBuildingId, onZonesChange, activeGroupId],
  );

  const updateGroup = (groupId: string, patch: Partial<Pick<Zone, 'name' | 'color'>>) => {
    onZonesChange(
      zones.map((z) => ((z.groupId ?? z.id) === groupId ? { ...z, ...patch } : z)),
    );
    setDraftGroups((prev) => prev.map((g) => (g.id === groupId ? { ...g, ...patch } : g)));
  };

  const deleteGroup = (groupId: string) => {
    const hasCommittedMembers = zones.some((z) => (z.groupId ?? z.id) === groupId);
    if (hasCommittedMembers && !window.confirm('Delete this zone (all its rings)?')) return;
    onZonesChange(zones.filter((z) => (z.groupId ?? z.id) !== groupId));
    setDraftGroups((prev) => prev.filter((g) => g.id !== groupId));
    if (selectedId && zones.find((z) => z.id === selectedId)?.groupId === groupId) {
      setSelectedId(null);
    }
    setRefiningBuildingId(null);
    if (activeGroupId === groupId) setActiveGroupId(null);
  };

  const clearAll = () => {
    if (zones.length === 0 && draftGroups.length === 0) return;
    if (zones.length > 0 && !window.confirm(`Delete all ${committedGroups.length} zones?`)) return;
    onZonesChange([]);
    setDraftGroups([]);
    setGroupOrder([]);
    setSelectedId(null);
    setActiveGroupId(null);
  };

  const displayMenuRef = useRef<HTMLDetailsElement>(null);

  useEffect(() => {
    const handler = (event: MouseEvent) => {
      if (displayMenuRef.current && !displayMenuRef.current.contains(event.target as Node)) {
        displayMenuRef.current.removeAttribute('open');
      }
    };
    document.addEventListener('mousedown', handler);
    return () => document.removeEventListener('mousedown', handler);
  }, []);

  // Use each zone's own colour in 3D so the right-hand viewer matches the
  // left-hand 2D editor exactly. Selection emphasis happens on the 2D side.
  const colorOverride = undefined;

  if (!hasModel) {
    const message = prerequisiteMessageForTab('zoning');
    return (
      <div className="three-col">
        <GuidedStatus tone="warning">
          <strong>{message.title}</strong><br />
          {message.body}
        </GuidedStatus>
      </div>
    );
  }

  return (
    <div className="three-col">
      {/* Left: config + zone list */}
      <GuidedPanel
        title="Zoning"
        subtitle="Define evaluation zones for simulation summaries."
        status={error ? <GuidedStatus tone="error">{error}</GuidedStatus> : undefined}
        footer={(
          <GuidedFooter>
            <button
              type="button"
              className="btn btn-secondary"
              onClick={clearAll}
              disabled={zones.length === 0 && draftGroups.length === 0}
            >
              <Trash2 size={14} aria-hidden="true" style={{ marginRight: 6 }} />
              Clear all zones
            </button>
          </GuidedFooter>
        )}
      >
        <GuidedSection index={1} label="ZONE TYPE">
          <ChoiceGroup
            variant="checks"
            ariaLabel="Zone type"
            value={zoneType}
            onChange={(next) => setZoneType(next)}
            options={[
              { id: 'horizontal', label: '2D area', icon: Square },
              { id: 'building_surface', label: 'Building surfaces', icon: Box },
            ]}
          />
        </GuidedSection>
        {zoneType === 'horizontal' && (
          <GuidedSection index={2} label="SHAPE">
            <ChoiceGroup
              variant="checks"
              ariaLabel="Zone shape"
              value={shape}
              onChange={setShape}
              options={[
                { id: 'rect', label: 'Rectangle', icon: Square },
                { id: 'polygon', label: 'Polygon', icon: Hexagon },
              ]}
            />
          </GuidedSection>
        )}

        <GuidedSection index={3} label="ZONES" action={(
          <button
            type="button"
            className="btn btn-primary btn-sm"
            onClick={handleAddZone}
            title="Add a new zone row. Draw on the map to set its boundary."
          >
            <Plus size={12} aria-hidden="true" style={{ marginRight: 4 }} />
            Add zone
          </button>
        )}>
        <div className="zone-list">
          {groups.length === 0 && (
            <div className="alert alert-info" style={{ marginTop: 8 }}>
              Draw a {shape === 'rect' ? 'rectangle' : 'polygon'} on the map to add a zone.
            </div>
          )}
          {groups.map((g) => {
            const isActive = (effectiveActiveGroupId ?? null) === g.id;
            const groupType = g.draft ? 'horizontal' : zoneGroupType(zones, g.id);
            return (
              <div
                key={g.id}
                className={`zone-row${isActive ? ' selected' : ''}`}
                onClick={() => {
                  if (groupType) setZoneType(groupType);
                  setActiveGroupId(g.id);
                  setSelectedId(g.members[0]?.id ?? null);
                  setRefiningBuildingId(null);
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
                    {g.draft && (
                      <span style={{ opacity: 0.6, marginLeft: 6, fontSize: '0.85em' }}>
                        pending
                      </span>
                    )}
                    {!g.draft && g.members.length > 1 && (
                      <span style={{ opacity: 0.6, marginLeft: 6, fontSize: '0.85em' }}>
                        ×{g.members.length}
                      </span>
                    )}
                  </span>
                )}
                <span
                  className={`zone-type-badge ${groupType === 'building_surface' ? 'building' : 'two-d'}`}
                  title={groupType === 'building_surface' ? 'Building surface zone' : '2D area zone'}
                >
                  {groupType ? zoneTypeShortLabel(groupType) : 'Mixed'}
                </span>
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
        {zoneType === 'building_surface' && refiningBuildingId != null && (
          <div style={{ marginTop: 8 }}>
            <button
              className="btn btn-secondary"
              style={{ fontSize: '0.85em' }}
              onClick={() => setRefiningBuildingId(null)}
            >
              Cancel refine
            </button>
          </div>
        )}
        {zoneType === 'building_surface' && (() => {
          if (!activeSurfaceZone) return null;
          const selectedBuildingIds = [
            ...new Set(
              activeSurfaceZone.selectors
                .filter((s) => s.mode !== 'exclude_faces')
                .map((s) => s.buildingId),
            ),
          ];
          if (selectedBuildingIds.length === 0) return (
            <div className="alert alert-info" style={{ marginTop: 8 }}>
              Click buildings in the 3D viewer to select them.
            </div>
          );
          return (
            <div style={{ marginTop: 12 }}>
              <div style={{ fontSize: '0.85em', opacity: 0.7, marginBottom: 4 }}>Selected buildings:</div>
              {selectedBuildingIds.map((bid) => {
                const sel = activeSurfaceZone.selectors;
                const isMode = (mode: SurfaceSelector['mode'], dir?: WallOrientation) =>
                  sel.some((s) => s.buildingId === bid && s.mode === mode && (mode !== 'wall_orientation' || ('orientation' in s && s.orientation === dir)));
                const apply = (updated: SurfaceSelector[]) =>
                  onZonesChange(zones.map((z) => z.id === activeSurfaceZone.id ? { ...activeSurfaceZone, selectors: updated } : z));
                return (
                <div key={bid} className={`surface-building-row${refiningBuildingId === bid ? ' refining' : ''}`}>
                  <div className="surface-building-head">
                    <span className="surface-building-name" title={`Building ${bid}`}>Building {bid}</span>
                    {refiningBuildingId === bid ? (
                      <button className="btn btn-secondary btn-sm" onClick={() => setRefiningBuildingId(null)}>Done</button>
                    ) : (
                      <>
                        <button className="btn btn-secondary btn-sm" onClick={() => setRefiningBuildingId(bid)}>Refine</button>
                        <button
                          className="btn btn-secondary btn-sm"
                          title="Remove building"
                          onClick={() => {
                            setRefiningBuildingId(null);
                            apply(activeSurfaceZone.selectors.filter((s) => s.buildingId !== bid));
                          }}
                        >✕</button>
                      </>
                    )}
                  </div>
                  {refiningBuildingId === bid && (
                    <div className="surface-refine-group">
                      <button className={`btn btn-sm ${isMode('roof') ? 'btn-primary' : 'btn-secondary'}`}
                        onClick={() => apply(toggleBulkSelector(sel, bid, 'roof'))}>Roof</button>
                      <button className={`btn btn-sm ${isMode('all_walls') ? 'btn-primary' : 'btn-secondary'}`}
                        onClick={() => apply(toggleBulkSelector(sel, bid, 'all_walls'))}>All walls</button>
                      <button className={`btn btn-sm ${isMode('window') ? 'btn-primary' : 'btn-secondary'}`}
                        onClick={() => apply(toggleBulkSelector(sel, bid, 'window'))}>Window</button>
                      <span className="surface-refine-divider" />
                      {(['N', 'E', 'S', 'W'] as WallOrientation[]).map((dir) => (
                        <button key={dir} className={`btn btn-sm btn-icon ${isMode('wall_orientation', dir) ? 'btn-primary' : 'btn-secondary'}`}
                          title={`${dir} wall`}
                          onClick={() => apply(toggleBulkSelector(sel, bid, 'wall_orientation', dir))}>{dir}</button>
                      ))}
                    </div>
                  )}
                </div>
                );
              })}
            </div>
          );
        })()}
        </GuidedSection>
      </GuidedPanel>

      {/* Center: 2D editor */}
      <div className="panel visual-panel">
        <div className="plan-panel-header">
          <div>
            <h2>2D zone editor</h2>
          </div>
          <details className="display-menu" ref={displayMenuRef}>
            <summary>Display</summary>
            <div className="display-menu-popover">
              <div className="form-group">
                <label>Basemap</label>
                <select value={basemap} onChange={(e) => setBasemap(e.target.value as BasemapKey)}>
                  <option>CartoDB Positron</option>
                  <option>Google Satellite</option>
                  <option>OpenStreetMap</option>
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
            </div>
          </details>
        </div>
        {loading && <div className="alert alert-info">Loading map…</div>}
        <div className="visual-frame">
          {geo && (
            <PlanMapEditor
              geo={geo}
              interaction={zoneType === 'building_surface' ? 'none' : interaction}
              backdrop={backdrop}
              basemap={basemap}
              drawColor="blue"
              pendingEdits={pendingEdits}
              onPolygonComplete={handlePolygonComplete}
            />
          )}
        </div>
      </div>

      {/* Right: 3D viewer */}
      <div className="panel visual-panel">
        <div className="plan-panel-header">
          <h2>3D preview</h2>
        </div>
        <div className="visual-frame">
          {previewDisabled ? (
            <PreviewDisabledNotice gridShape={previewGridShape} />
          ) : hasModel ? (
            <SceneViewer
              geometryToken={hasModel ? (geometryToken ?? 'loaded') : 'none'}
              downsample={1}
              zones={zones.filter((z) => z.type === 'horizontal')}
              lonLatToXY={lonLatToXY}
              showZones
              colorOverride={colorOverride}
              onPick={zoneType === 'building_surface' ? handleSurfacePick : undefined}
              surfaceSelection={surfaceSelection}
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
