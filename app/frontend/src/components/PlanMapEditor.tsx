/**
 * Leaflet-based 2D plan editor for the Edit Model tab.
 *
 * Mirrors the interaction primitives of voxcity's draw editors
 * (`voxcity.geoprocessor.draw.{edit_building, edit_tree, edit_landcover}`):
 *
 *   - `none`            — pan/zoom only.
 *   - `click_point`     — single click → emit `(lat, lon)` + cell `(i, j)`.
 *   - `click_feature`   — single click on a building footprint → emit `id`.
 *   - `draw_rect_3pt`   — voxcity-style 3-corner rotated rectangle (auto 4th).
 *   - `draw_polygon`    — click vertices, double-click or click-first to close.
 *
 * Buildings whose IDs are listed in `highlightedBuildingIds` get a red outline
 * (mirrors voxcity's deletion-preview style).
 */

import React, { useEffect, useMemo, useRef } from 'react';
import L from 'leaflet';
import type { ModelGeoResult } from '../api';
import {
  canopyMaskFromGeoJson,
  cellKey,
  cellsToQuads,
  extrudeRectFromSide,
  keyToCell,
  type GridGeom,
} from '../lib/grid';

export type MapInteraction =
  | 'none'
  | 'click_point'
  | 'click_feature'
  | 'draw_rect_3pt'
  | 'draw_polygon';

export type Backdrop = 'buildings' | 'canopy' | 'land_cover' | 'none';
export type BasemapKey = 'CartoDB Positron' | 'Google Satellite' | 'OpenStreetMap';
export type DrawColor = 'red' | 'green' | 'blue';

/**
 * Vector record for a pending (uncommitted) edit. The 2D viewer renders these
 * as polygons on a dedicated overlay layer so the underlying baseline GeoJSON
 * never has to be re-fetched or re-rendered between clicks.
 */
export type PendingEdit =
  | { kind: 'add_building';   ring: [number, number][]; cells: [number, number][]; height_m: number; min_height_m: number }
  | { kind: 'delete_building'; building_id: number }
  | { kind: 'add_trees';      cells: [number, number][]; height_m: number; bottom_m: number; tops?: number[]; bottoms?: number[] }
  | { kind: 'delete_trees';   cells: [number, number][] }
  | { kind: 'paint_lc';       cells: [number, number][]; class_index: number; color: string }
  /** Visual-only zone overlay (no commit). `roof` = drawn with diagonal hatch
   *  to distinguish cells over buildings from ground cells. `target` controls
   *  the role styling: evaluation zones get a denser dashed outline and a
   *  higher fill opacity than placement zones, so the two are visually
   *  distinct even when they share neighbouring cells. */
  | { kind: 'paint_zone';     cells: [number, number][]; color: string; roof?: boolean; target?: 'placement' | 'evaluation'; ring?: [number, number][]; selected?: boolean };

export interface PlanMapEditorProps {
  geo: ModelGeoResult;
  interaction: MapInteraction;
  backdrop: Backdrop;
  basemap: BasemapKey;
  /** Color for the in-progress draw geometry (red/green/blue per voxcity). */
  drawColor?: DrawColor;
  /** Building feature IDs to highlight (red outline) — used for delete previews. */
  highlightedBuildingIds?: number[];
  /** Pending uncommitted edits — rendered as a separate overlay layer. */
  pendingEdits?: PendingEdit[];

  /** Single-click handlers. */
  onPointClick?: (lat: number, lon: number, cell: [number, number] | null) => void;
  onPickBuilding?: (buildingId: number) => void;

  /** Fired when a draw_rect_3pt or draw_polygon completes. Ring is `[lon, lat]` pairs (not closed). */
  onPolygonComplete?: (ring: [number, number][]) => void;

  /** Bumping this value clears any in-progress draw + completed-shape preview. */
  clearDrawNonce?: number;
}

/* ──────────────────────────────────────────────────────────────
   Basemap tile factory (URLs from voxcity.geoprocessor.draw._common)
   ────────────────────────────────────────────────────────────── */

function makeTile(key: BasemapKey): L.TileLayer {
  switch (key) {
    case 'Google Satellite':
      return L.tileLayer('https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}', {
        attribution: 'Google Satellite',
        maxZoom: 22,
      });
    case 'OpenStreetMap':
      return L.tileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png', {
        attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OSM</a>',
        maxZoom: 19,
      });
    case 'CartoDB Positron':
    default:
      return L.tileLayer(
        'https://cartodb-basemaps-a.global.ssl.fastly.net/light_all/{z}/{x}/{y}@2x.png',
        {
          attribution: '&copy; <a href="https://carto.com/">CARTO</a>',
          maxZoom: 20,
        },
      );
  }
}

/* ──────────────────────────────────────────────────────────────
   Geo ↔ cell helpers
   ────────────────────────────────────────────────────────────── */

function geoToCell(
  lon: number,
  lat: number,
  g: GridGeom,
): [number, number] | null {
  const dx = lon - g.origin[0];
  const dy = lat - g.origin[1];
  const det = g.side_1[0] * g.side_2[1] - g.side_1[1] * g.side_2[0];
  if (Math.abs(det) < 1e-15) return null;
  const alpha = (dx * g.side_2[1] - dy * g.side_2[0]) / det;
  const beta  = (g.side_1[0] * dy - g.side_1[1] * dx) / det;
  const i = Math.floor(alpha * g.grid_size[0]);
  const j = Math.floor(beta  * g.grid_size[1]);
  if (i < 0 || i >= g.grid_size[0] || j < 0 || j >= g.grid_size[1]) return null;
  return [i, j];
}

const RECT_STYLE: L.PathOptions = {
  color: '#3388ff',
  weight: 2,
  fillOpacity: 0,
  interactive: false,
};

function buildingStyle(): L.PathOptions {
  return {
    color: '#cd853f',
    weight: 1,
    fillColor: '#cd853f',
    fillOpacity: 0.55,
  };
}

const HIGHLIGHTED_BUILDING_STYLE: L.PathOptions = {
  color: '#ff0000',
  weight: 2,
  fillColor: '#ff5555',
  fillOpacity: 0.7,
};

/** Style for buildings that the user has queued for deletion. Drawn with
 *  `stroke: false` and `fillOpacity: 0` so the polygon vanishes from view
 *  in place — no DOM teardown, no GeoJSON re-fetch. */
const HIDDEN_BUILDING_STYLE: L.PathOptions = {
  stroke: false,
  fill: false,
  opacity: 0,
  fillOpacity: 0,
  interactive: false,
};

function canopyStyle(): L.PathOptions {
  return {
    color: '#228b22',
    weight: 0.5,
    fillColor: '#228b22',
    fillOpacity: 0.5,
  };
}

function lcStyle(feature: any): L.PathOptions {
  const s = feature?.properties?.style;
  return {
    color: s?.color ?? '#888',
    weight: s?.weight ?? 0.3,
    fillColor: s?.fillColor ?? '#888',
    fillOpacity: s?.fillOpacity ?? 0.55,
  };
}

const COLOR_HEX: Record<DrawColor, string> = {
  red:   '#ff0000',
  green: '#00b050',
  blue:  '#1a73e8',
};

/* ──────────────────────────────────────────────────────────────
   Component
   ────────────────────────────────────────────────────────────── */

const PlanMapEditor: React.FC<PlanMapEditorProps> = ({
  geo,
  interaction,
  backdrop,
  basemap,
  drawColor = 'red',
  highlightedBuildingIds,
  pendingEdits,
  onPointClick,
  onPickBuilding,
  onPolygonComplete,
  clearDrawNonce,
}) => {
  const mapRef = useRef<HTMLDivElement>(null);
  const mapInstanceRef = useRef<L.Map | null>(null);
  const tileLayerRef = useRef<L.TileLayer | null>(null);
  const rectLayerRef = useRef<L.Polygon | null>(null);
  const backdropLayerRef = useRef<L.Layer | null>(null);
  /** In-progress draw geometry (vertex circles + connecting line). */
  const drawLayerRef = useRef<L.LayerGroup | null>(null);
  /** Vertices of the polygon / rect being drawn (lon, lat). */
  const drawVerticesRef = useRef<[number, number][]>([]);
  /** Mousemove preview line (last vertex → cursor). */
  const previewLineRef = useRef<L.Polyline | null>(null);
  /** Live rectangle preview while the user moves the cursor toward p3. */
  const rectPreviewRef = useRef<L.Polygon | null>(null);
  /** Persistent overlay holding all uncommitted pending-edit polygons. */
  const pendingOverlayRef = useRef<L.LayerGroup | null>(null);

  // Stable refs so map handlers always see the latest props.
  const interactionRef = useRef(interaction);
  interactionRef.current = interaction;
  const drawColorRef = useRef(drawColor);
  drawColorRef.current = drawColor;
  const onPointClickRef = useRef(onPointClick);
  onPointClickRef.current = onPointClick;
  const onPickBuildingRef = useRef(onPickBuilding);
  onPickBuildingRef.current = onPickBuilding;
  const onPolygonCompleteRef = useRef(onPolygonComplete);
  onPolygonCompleteRef.current = onPolygonComplete;

  // Build the set of pending-delete ids once per render. These are rendered
  // invisible so the footprint disappears immediately on click.
  const pendingDeletedIds = useMemo(() => {
    const s = new Set<number>();
    for (const e of pendingEdits ?? []) {
      if (e.kind === 'delete_building') s.add(e.building_id);
    }
    return s;
  }, [pendingEdits]);
  const pendingDeletedRef = useRef<Set<number>>(new Set());
  pendingDeletedRef.current = pendingDeletedIds;
  const highlightedRef = useRef<Set<number>>(new Set());
  highlightedRef.current = new Set(highlightedBuildingIds ?? []);

  const grid: GridGeom = useMemo(() => ({
    origin:    geo.grid_geom.origin    as [number, number],
    side_1:    geo.grid_geom.side_1    as [number, number],
    side_2:    geo.grid_geom.side_2    as [number, number],
    u_vec:     geo.grid_geom.u_vec     as [number, number],
    v_vec:     geo.grid_geom.v_vec     as [number, number],
    adj_mesh:  geo.grid_geom.adj_mesh  as [number, number],
    grid_size: geo.grid_geom.grid_size as [number, number],
  }), [geo]);

  /** Stable identity of the grid geometry — used to gate map (re-)creation. */
  const gridKey = useMemo(
    () => JSON.stringify([geo.grid_geom, geo.rectangle_vertices, geo.center]),
    [geo.grid_geom, geo.rectangle_vertices, geo.center],
  );

  /* ──────── Per-cell canopy mask (derived once per geo) ──────── */
  const canopyMask = useMemo(
    () => canopyMaskFromGeoJson(geo.canopy_geojson, grid),
    [geo, grid],
  );
  const canopyMaskRef = useRef<Set<string>>(new Set());
  canopyMaskRef.current = canopyMask;

  /** Set of `"i,j"` cell keys queued for tree deletion — drives instant
   *  subtraction from the canopy backdrop. */
  const pendingDeletedTreeKeys = useMemo(() => {
    const s = new Set<string>();
    for (const e of pendingEdits ?? []) {
      if (e.kind === 'delete_trees') {
        for (const [i, j] of e.cells) s.add(cellKey(i, j));
      }
    }
    return s;
  }, [pendingEdits]);
  const pendingDeletedTreeKeysRef = useRef<Set<string>>(new Set());
  pendingDeletedTreeKeysRef.current = pendingDeletedTreeKeys;

  /* ──────── Draw helpers ──────── */

  const clearDraw = () => {
    drawVerticesRef.current = [];
    if (drawLayerRef.current) drawLayerRef.current.clearLayers();
    if (previewLineRef.current && mapInstanceRef.current) {
      mapInstanceRef.current.removeLayer(previewLineRef.current);
      previewLineRef.current = null;
    }
    if (rectPreviewRef.current && mapInstanceRef.current) {
      mapInstanceRef.current.removeLayer(rectPreviewRef.current);
      rectPreviewRef.current = null;
    }
  };

  const renderDrawProgress = () => {
    const layer = drawLayerRef.current;
    if (!layer) return;
    layer.clearLayers();
    const color = COLOR_HEX[drawColorRef.current];
    const verts = drawVerticesRef.current;
    // Vertex markers.
    verts.forEach(([lon, lat], idx) => {
      L.circleMarker([lat, lon], {
        radius: 4,
        color,
        fillColor: color,
        fillOpacity: 0.9,
        weight: 1,
        interactive: false,
      }).addTo(layer);
      // Once we have ≥3 vertices, telegraph the close target on vertex 0.
      if (idx === 0 && verts.length >= 3) {
        L.circleMarker([lat, lon], {
          radius: 9,
          color,
          fill: false,
          weight: 2,
          dashArray: '2,2',
          interactive: false,
        }).addTo(layer);
      }
    });
    // Edges between consecutive vertices.
    if (verts.length >= 2) {
      const latlngs = verts.map(([lon, lat]) => [lat, lon] as [number, number]);
      L.polyline(latlngs, {
        color, weight: 2, dashArray: '4,4', interactive: false,
      }).addTo(layer);
    }
  };

  const showCompletedShape = (ring: [number, number][]) => {
    const layer = drawLayerRef.current;
    if (!layer) return;
    layer.clearLayers();
    const color = COLOR_HEX[drawColorRef.current];
    const latlngs = ring.map(([lon, lat]) => [lat, lon] as [number, number]);
    L.polygon(latlngs, {
      color,
      weight: 2,
      fillColor: color,
      fillOpacity: 0.25,
      interactive: false,
    }).addTo(layer);
  };

  /* ──────── Initialise map ──────── */
  useEffect(() => {
    if (!mapRef.current || mapInstanceRef.current) return;

    const map = L.map(mapRef.current, {
      center: geo.center as [number, number],
      zoom: 17,
      zoomSnap: 0.25,
      doubleClickZoom: false, // we use dblclick to close polygons
    });

    tileLayerRef.current = makeTile(basemap).addTo(map);

    const rectLatLng: [number, number][] = geo.rectangle_vertices.map(
      ([lon, lat]) => [lat, lon],
    );
    rectLayerRef.current = L.polygon(rectLatLng, RECT_STYLE).addTo(map);
    map.fitBounds(rectLayerRef.current.getBounds(), { padding: [16, 16] });

    drawLayerRef.current = L.layerGroup().addTo(map);
    pendingOverlayRef.current = L.layerGroup().addTo(map);

    /* ── Click → routes by interaction mode ── */
    const onClick = (ev: L.LeafletMouseEvent) => {
      const mode = interactionRef.current;
      const lat = ev.latlng.lat;
      const lon = ev.latlng.lng;

      if (mode === 'click_point') {
        const cell = geoToCell(lon, lat, grid);
        onPointClickRef.current?.(lat, lon, cell);
        return;
      }

      if (mode === 'draw_rect_3pt') {
        // Don't accept additional clicks while a previous shape is being
        // committed. (clearDraw runs after the callback fires.)
        if (drawVerticesRef.current.length >= 3) return;
        drawVerticesRef.current.push([lon, lat]);
        if (drawVerticesRef.current.length === 3) {
          const [p1, p2, p3] = drawVerticesRef.current;
          const ring = extrudeRectFromSide(p1, p2, p3);
          if (!ring) {
            // Side or extrusion < 0.5 m — discard the third click and let the
            // user try again. Pop p3 so the preview line stays alive.
            drawVerticesRef.current.pop();
            renderDrawProgress();
            return;
          }
          onPolygonCompleteRef.current?.(ring);
          // Clear the in-editor preview — the committed shape now lives in
          // the overlay refresh fired by the parent.
          clearDraw();
        } else {
          renderDrawProgress();
        }
        return;
      }

      if (mode === 'draw_polygon') {
        // Click near the first vertex closes the polygon (≥3 vertices placed).
        // Threshold is metric (~voxcity's 0.0001° ≈ 11 m at the equator).
        if (drawVerticesRef.current.length >= 3) {
          const [lon0, lat0] = drawVerticesRef.current[0];
          const distM = ev.latlng.distanceTo(L.latLng(lat0, lon0));
          if (distM < 8) {
            const ring = drawVerticesRef.current.slice();
            onPolygonCompleteRef.current?.(ring);
            // Fully reset the editor — voxcity-style commit.
            clearDraw();
            return;
          }
        }
        drawVerticesRef.current.push([lon, lat]);
        renderDrawProgress();
      }
    };

    /* ── Mousemove → preview line / rectangle for in-progress draw. ── */
    const onMouseMove = (ev: L.LeafletMouseEvent) => {
      const mode = interactionRef.current;
      if (mode !== 'draw_polygon' && mode !== 'draw_rect_3pt') return;
      const verts = drawVerticesRef.current;
      if (verts.length === 0) return;

      // Live rectangle preview after the second click — track the cursor
      // through the same extrusion math the third click commits.
      if (mode === 'draw_rect_3pt' && verts.length === 2) {
        const cursor: [number, number] = [ev.latlng.lng, ev.latlng.lat];
        const ring = extrudeRectFromSide(verts[0], verts[1], cursor);
        if (previewLineRef.current) {
          map.removeLayer(previewLineRef.current);
          previewLineRef.current = null;
        }
        if (ring) {
          const latlngs = ring.map(([lo, la]) => [la, lo] as [number, number]);
          const color = COLOR_HEX[drawColorRef.current];
          if (rectPreviewRef.current) {
            rectPreviewRef.current.setLatLngs(latlngs);
          } else {
            rectPreviewRef.current = L.polygon(latlngs, {
              color, weight: 2,
              fillColor: color, fillOpacity: 0.2,
              dashArray: '4,4',
              interactive: false,
            }).addTo(map);
          }
        } else if (rectPreviewRef.current) {
          map.removeLayer(rectPreviewRef.current);
          rectPreviewRef.current = null;
        }
        return;
      }

      // Polygon (and rect with only 1 vertex) — show a thin preview line.
      const last = verts[verts.length - 1];
      const latlngs: [number, number][] = [
        [last[1], last[0]],
        [ev.latlng.lat, ev.latlng.lng],
      ];
      const color = COLOR_HEX[drawColorRef.current];
      if (previewLineRef.current) {
        previewLineRef.current.setLatLngs(latlngs);
      } else {
        previewLineRef.current = L.polyline(latlngs, {
          color, weight: 1.5, dashArray: '2,4', interactive: false,
        }).addTo(map);
      }
    };

    /* ── Double-click → close polygon. ── */
    const onDoubleClick = (ev: L.LeafletMouseEvent) => {
      const mode = interactionRef.current;
      if (mode !== 'draw_polygon') return;
      L.DomEvent.stopPropagation(ev);
      if (drawVerticesRef.current.length < 3) return;
      const ring = drawVerticesRef.current.slice();
      onPolygonCompleteRef.current?.(ring);
      clearDraw();
    };

    map.on('click', onClick);
    map.on('mousemove', onMouseMove);
    map.on('dblclick', onDoubleClick);

    mapInstanceRef.current = map;

    return () => {
      map.off('click', onClick);
      map.off('mousemove', onMouseMove);
      map.off('dblclick', onDoubleClick);
      map.remove();
      mapInstanceRef.current = null;
      tileLayerRef.current = null;
      rectLayerRef.current = null;
      backdropLayerRef.current = null;
      drawLayerRef.current = null;
      previewLineRef.current = null;
      pendingOverlayRef.current = null;
    };
    // Build the map once per stable grid geometry; geojson refreshes are
    // handled by the dedicated backdrop effect below so the user's pan/zoom
    // and any in-progress draw survive every mutation.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [gridKey]);

  /* ──────── Swap basemap ──────── */
  useEffect(() => {
    const map = mapInstanceRef.current;
    if (!map) return;
    if (tileLayerRef.current) map.removeLayer(tileLayerRef.current);
    tileLayerRef.current = makeTile(basemap).addTo(map);
    if (rectLayerRef.current) rectLayerRef.current.bringToFront();
  }, [basemap]);

  /* ──────── Swap backdrop overlay ──────── */
  useEffect(() => {
    const map = mapInstanceRef.current;
    if (!map) return;

    if (backdropLayerRef.current) {
      map.removeLayer(backdropLayerRef.current);
      backdropLayerRef.current = null;
    }

    let layer: L.Layer | null = null;
    if (backdrop === 'buildings') {
      layer = L.geoJSON(geo.building_geojson as any, {
        style: (feature) => {
          const id = feature?.properties?.idx;
          if (id != null) {
            const idn = Number(id);
            if (pendingDeletedRef.current.has(idn)) return HIDDEN_BUILDING_STYLE;
            if (highlightedRef.current.has(idn))   return HIGHLIGHTED_BUILDING_STYLE;
          }
          return buildingStyle();
        },
        onEachFeature: (feature, lyr) => {
          const id = feature?.properties?.idx;
          if (id == null) return;
          lyr.on('click', (e: L.LeafletMouseEvent) => {
            if (interactionRef.current === 'click_feature' && onPickBuildingRef.current) {
              // Skip clicks on already-deleted footprints so the same building
              // can't be queued twice.
              if (pendingDeletedRef.current.has(Number(id))) return;
              L.DomEvent.stopPropagation(e);
              onPickBuildingRef.current(Number(id));
            }
          });
        },
      });
    } else if (backdrop === 'canopy') {
      // Render canopy as a single multi-ring polygon built from the per-cell
      // mask, minus any pending `delete_trees` cells. This lets polygon-area
      // tree deletes vanish from the backdrop the instant the user closes the
      // polygon — no waiting for Update 3D, no GeoJSON re-fetch.
      const mask = canopyMaskRef.current;
      const deleted = pendingDeletedTreeKeysRef.current;
      const visible: [number, number][] = [];
      for (const k of mask) {
        if (!deleted.has(k)) visible.push(keyToCell(k));
      }
      if (visible.length > 0) {
        const quads = cellsToQuads(visible, grid);
        // Convert each [lon, lat] ring into [lat, lon] for Leaflet, then pass
        // them all as the multi-ring constructor of a single L.Polygon —
        // one DOM path with N subpaths.
        const llRings = quads.map(q => q.map(([lon, lat]) => [lat, lon] as [number, number]));
        layer = L.polygon(llRings, { ...canopyStyle(), interactive: false });
      }
    } else if (backdrop === 'land_cover') {
      layer = L.geoJSON(geo.land_cover_geojson as any, { style: lcStyle });
    }

    if (layer) {
      layer.addTo(map);
      backdropLayerRef.current = layer;
      // Keep pending overlay + in-progress draw on top.
      if (pendingOverlayRef.current) {
        map.removeLayer(pendingOverlayRef.current);
        map.addLayer(pendingOverlayRef.current);
      }
      if (drawLayerRef.current) {
        map.removeLayer(drawLayerRef.current);
        map.addLayer(drawLayerRef.current);
      }
    }
  }, [backdrop, geo, canopyMask, pendingDeletedTreeKeys]);

  /* ──────── Re-style buildings when highlights change ──────── */
  useEffect(() => {
    const layer = backdropLayerRef.current as L.GeoJSON | null;
    if (!layer || backdrop !== 'buildings') return;
    layer.setStyle((feature) => {
      const id = feature?.properties?.idx;
      if (id != null) {
        const idn = Number(id);
        if (pendingDeletedRef.current.has(idn)) return HIDDEN_BUILDING_STYLE;
        if (highlightedRef.current.has(idn))   return HIGHLIGHTED_BUILDING_STYLE;
      }
      return buildingStyle();
    });
  }, [highlightedBuildingIds, pendingDeletedIds, backdrop]);



  /* ──────── Pending-edit overlay (rebuild in place) ──────── */
  useEffect(() => {
    const layer = pendingOverlayRef.current;
    if (!layer) return;
    layer.clearLayers();
    if (!pendingEdits || pendingEdits.length === 0) return;

    const renderRing = (ring: [number, number][], style: L.PathOptions) => {
      const latlngs = ring.map(([lon, lat]) => [lat, lon] as [number, number]);
      L.polygon(latlngs, { ...style, interactive: false }).addTo(layer);
    };
    const renderCells = (cells: [number, number][], style: L.PathOptions) => {
      const quads = cellsToQuads(cells, grid);
      for (const q of quads) renderRing(q, style);
    };

    for (const edit of pendingEdits) {
      switch (edit.kind) {
        case 'add_building':
          renderRing(edit.ring, {
            color: '#ff0000', weight: 2,
            fillColor: '#ff0000', fillOpacity: 0.45,
          });
          break;
        case 'add_trees':
          renderCells(edit.cells, {
            color: '#00b050', weight: 0.5,
            fillColor: '#00b050', fillOpacity: 0.55,
          });
          break;
        case 'delete_trees':
          // No per-cell overlay — pending deletes are subtracted directly
          // from the canopy backdrop above (see `canopy` branch).
          break;
        case 'paint_lc':
          renderCells(edit.cells, {
            color: edit.color, weight: 0.5,
            fillColor: edit.color, fillOpacity: 0.6,
          });
          break;
        case 'paint_zone': {
          const isEval = edit.target === 'evaluation';
          // Prefer rendering the polygon outline directly (closer to how
          // the zone shows up on the 3D viewer) and only fall back to the
          // rasterised cell quads when no ring is provided.
          if (edit.ring && edit.ring.length >= 3) {
            renderRing(edit.ring, {
              color: edit.color,
              weight: edit.selected ? 3.5 : 2,
              fillColor: edit.color,
              fillOpacity: edit.selected ? 0.35 : 0.22,
              dashArray: isEval ? '4,3' : undefined,
            });
          } else {
            renderCells(edit.cells, {
              color: edit.color,
              weight: isEval ? 1.6 : (edit.roof ? 1.2 : 0.5),
              fillColor: edit.color,
              fillOpacity: edit.roof ? 0.35 : (isEval ? 0.5 : 0.55),
              dashArray: isEval ? '2,3' : (edit.roof ? '4,3' : undefined),
            });
          }
          break;
        }
        // delete_building is rendered via the buildings backdrop re-style.
      }
    }
  }, [pendingEdits, grid]);

  /* ──────── Clear in-progress draw on demand ──────── */
  useEffect(() => {
    clearDraw();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [clearDrawNonce, interaction]);

  /* ──────── Cursor + map dragging ──────── */
  useEffect(() => {
    const map = mapInstanceRef.current;
    if (!map) return;
    const c = map.getContainer();
    // Reset cursor classes (CSS overrides per-path .leaflet-interactive cursor).
    c.classList.remove('cursor-crosshair', 'cursor-pointer');
    switch (interaction) {
      case 'click_feature':
        c.style.cursor = 'pointer';
        c.classList.add('cursor-pointer');
        map.dragging.enable();
        break;
      case 'click_point':
        c.style.cursor = 'crosshair';
        c.classList.add('cursor-crosshair');
        map.dragging.enable();
        break;
      case 'draw_rect_3pt':
      case 'draw_polygon':
        c.style.cursor = 'crosshair';
        c.classList.add('cursor-crosshair');
        // Keep dragging enabled outside of explicit drag (clicks place vertices).
        map.dragging.enable();
        break;
      case 'none':
      default:
        c.style.cursor = '';
        map.dragging.enable();
        break;
    }
  }, [interaction]);

  return <div ref={mapRef} className="plan-map-wrap" />;
};

export default PlanMapEditor;
