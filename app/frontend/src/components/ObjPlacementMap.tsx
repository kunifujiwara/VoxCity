/**
 * Leaflet map for OBJ placement: basemap + footprint outline(s) at the current
 * placement; clicking sets the anchor lon/lat. Footprint vertices are computed
 * via transformModelPoint() (scene-metres offset from the anchor) composed with
 * the anchor's own scene position (via grid.ts's lonLatToUvM), then converted
 * back to lon/lat via the local sceneXYToLonLat() inverse.
 *
 * Coordinate convention note (see PlanMapEditor.tsx):
 *   - Voxcity / grid.ts use [lon, lat] order.
 *   - Leaflet uses [lat, lon] order.
 */
import React, { useEffect, useRef } from 'react';
import L from 'leaflet';
import type { ModelGeoResult } from '../api';
import { lonLatToUvM, type GridGeom } from '../lib/grid';
import { transformModelPoint, type Placement } from '../lib/objPlacement';

interface Props {
  geo: ModelGeoResult;
  placement: Placement;
  footprints: [number, number][][]; // model-XY rings from upload.preview
  onAnchor: (lonLat: [number, number]) => void;
}

/** Inverse of grid.ts's lonLatToUvM: scene metres [east, north] -> [lon, lat]. */
function sceneXYToLonLat(geo: GridGeom, eastM: number, northM: number): [number, number] {
  const [du, dv] = geo.adj_mesh;
  const [ox, oy] = geo.origin;
  const [ux, uy] = geo.u_vec;
  const [vx, vy] = geo.v_vec;
  const a = ux * du, b = vx * dv;
  const c = uy * du, d = vy * dv;
  const u_cell = northM / du;
  const v_cell = eastM / dv;
  const dlon = a * u_cell + b * v_cell;
  const dlat = c * u_cell + d * v_cell;
  return [ox + dlon, oy + dlat];
}

const ObjPlacementMap: React.FC<Props> = ({ geo, placement, footprints, onAnchor }) => {
  const mapRef = useRef<L.Map | null>(null);
  const layerRef = useRef<L.LayerGroup | null>(null);
  const containerRef = useRef<HTMLDivElement | null>(null);
  const onAnchorRef = useRef(onAnchor);
  onAnchorRef.current = onAnchor;

  useEffect(() => {
    if (!containerRef.current || mapRef.current) return;
    const map = L.map(containerRef.current).setView(geo.center as [number, number], 17);
    L.tileLayer(
      'https://cartodb-basemaps-a.global.ssl.fastly.net/light_all/{z}/{x}/{y}@2x.png',
      { attribution: '&copy; <a href="https://carto.com/">CARTO</a>', maxZoom: 20 },
    ).addTo(map);
    map.on('click', (e: L.LeafletMouseEvent) => onAnchorRef.current([e.latlng.lng, e.latlng.lat]));
    layerRef.current = L.layerGroup().addTo(map);
    mapRef.current = map;
    return () => { map.remove(); mapRef.current = null; layerRef.current = null; };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    const layer = layerRef.current;
    if (!layer) return;
    layer.clearLayers();
    if (!placement.anchorLonLat) return;
    const fwd = lonLatToUvM({ grid_geom: geo.grid_geom });
    if (!fwd) return;
    const [anchorEastM, anchorNorthM] = fwd(placement.anchorLonLat[0], placement.anchorLonLat[1]);
    for (const ring of footprints) {
      const latlngs = ring.map(([mx, my]) => {
        const [eastOffset, northOffset] = transformModelPoint([mx, my, 0], placement);
        const [lon, lat] = sceneXYToLonLat(geo.grid_geom, anchorEastM + eastOffset, anchorNorthM + northOffset);
        return L.latLng(lat, lon);
      });
      L.polygon(latlngs, { color: '#e8590c', weight: 2, fillOpacity: 0.25 }).addTo(layer);
    }
  }, [geo, placement, footprints]);

  return <div ref={containerRef} style={{ width: '100%', height: '100%' }} />;
};

export default ObjPlacementMap;
