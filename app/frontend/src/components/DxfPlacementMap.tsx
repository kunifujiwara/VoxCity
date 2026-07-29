/**
 * Leaflet map for DXF placement: basemap + per-layer polyline preview at the
 * current placement; clicking sets the anchor lon/lat. Mirrors ObjPlacementMap
 * but renders open polylines colored per DXF layer.
 */
import React, { useEffect, useRef } from 'react';
import L from 'leaflet';
import type { ModelGeoResult, DxfPreviewLayerDto } from '../api';
import { lonLatToUvM, sceneXYToLonLat, domainRotationDeg } from '../lib/grid';
import { transformModelPoint, type Placement } from '../lib/objPlacement';

interface Props {
  geo: ModelGeoResult;
  placement: Placement;
  layers: DxfPreviewLayerDto[];
  visibility: Record<string, boolean>;
  onAnchor: (lonLat: [number, number]) => void;
}

const DxfPlacementMap: React.FC<Props> = ({ geo, placement, layers, visibility, onAnchor }) => {
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
    const phiDeg = domainRotationDeg(geo.grid_geom);
    const [anchorEastM, anchorNorthM] = fwd(placement.anchorLonLat[0], placement.anchorLonLat[1]);
    for (const lyr of layers) {
      if (visibility[lyr.name] === false) continue;
      for (const ring of lyr.polylines) {
        const latlngs = ring.map(([mx, my]) => {
          const [eOff, nOff] = transformModelPoint([mx, my, 0], placement, phiDeg);
          const [lon, lat] = sceneXYToLonLat(geo.grid_geom, anchorEastM + eOff, anchorNorthM + nOff);
          return L.latLng(lat, lon);
        });
        L.polyline(latlngs, { color: lyr.color, weight: 2, dashArray: '4 3' }).addTo(layer);
      }
    }
  }, [geo, placement, layers, visibility]);

  return <div ref={containerRef} style={{ width: '100%', height: '100%' }} />;
};

export default DxfPlacementMap;
