import React, { useEffect, useRef, useCallback } from 'react';
import L from 'leaflet';
import { fetchDimensionRectangle } from '../lib/dimensionRectangle';
import { buildRotatedRectangleFromClicks } from '../lib/rectangleGeometry';

interface DimensionCenter {
  centerLon: number;
  centerLat: number;
}

interface MapPickerProps {
  center: [number, number]; // [lat, lon] — Leaflet convention (reversed from voxcity [lon, lat])
  zoom: number;
  rectangle: number[][] | null; // [[lon,lat], ...] or null
  selectionMode: 'draw' | 'dimensions' | 'rotated';
  widthM: number;
  heightM: number;
  rotationDeg?: number;
  onRectangleChange: (vertices: number[][]) => void;
}

const MapPicker: React.FC<MapPickerProps> = ({
  center,
  zoom,
  rectangle,
  selectionMode,
  widthM,
  heightM,
  rotationDeg = 0,
  onRectangleChange,
}) => {
  const mapRef = useRef<HTMLDivElement>(null);
  const mapInstanceRef = useRef<L.Map | null>(null);
  /** Stored (confirmed) rectangle overlay — always an L.Polygon to handle rotation. */
  const rectLayerRef = useRef<L.Polygon | null>(null);
  /** Live preview layer — L.Rectangle for 2-click mode, L.Polygon for rotated mode. */
  const previewLayerRef = useRef<L.Rectangle | L.Polygon | null>(null);
  const firstCornerRef = useRef<L.LatLng | null>(null);
  /** Second click in rotated mode. */
  const secondCornerRef = useRef<L.LatLng | null>(null);
  const markerRef = useRef<L.CircleMarker | null>(null);
  const markerRef2 = useRef<L.CircleMarker | null>(null);
  const dimensionCenterRef = useRef<DimensionCenter | null>(null);
  const dimensionRequestSeqRef = useRef(0);
  const dimensionParamsRef = useRef({ widthM, heightM, rotationDeg });

  // Stable callback ref so effects always see the latest callback
  const onRectangleChangeRef = useRef(onRectangleChange);
  onRectangleChangeRef.current = onRectangleChange;
  dimensionParamsRef.current = { widthM, heightM, rotationDeg };

  const updateDimensionRectangle = useCallback(async (centerPoint: DimensionCenter) => {
    const requestSeq = ++dimensionRequestSeqRef.current;
    const { widthM: width, heightM: height, rotationDeg: rotation } = dimensionParamsRef.current;
    try {
      const vertices = await fetchDimensionRectangle({
        centerLon: centerPoint.centerLon,
        centerLat: centerPoint.centerLat,
        widthM: width,
        heightM: height,
        rotationDeg: rotation,
      });
      if (requestSeq === dimensionRequestSeqRef.current && vertices.length >= 4) {
        onRectangleChangeRef.current(vertices);
      }
    } catch (err) {
      console.error('Failed to compute rectangle:', err);
    }
  }, []);

  // Initialize map
  useEffect(() => {
    if (!mapRef.current || mapInstanceRef.current) return;

    const map = L.map(mapRef.current, {
      center,
      zoom,
    });

    L.tileLayer('https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png', {
      attribution:
        '&copy; <a href="https://www.openstreetmap.org/copyright">OSM</a> &copy; <a href="https://carto.com/">CARTO</a>',
      maxZoom: 19,
    }).addTo(map);

    mapInstanceRef.current = map;

    return () => {
      map.remove();
      mapInstanceRef.current = null;
    };
  }, []);

  // Update center/zoom
  useEffect(() => {
    if (mapInstanceRef.current) {
      mapInstanceRef.current.setView(center, zoom);
    }
  }, [center, zoom]);

  // ── Clean up helpers ─────────────────────────────────────
  const clearPreview = useCallback(() => {
    const map = mapInstanceRef.current;
    if (!map) return;
    if (previewLayerRef.current) {
      map.removeLayer(previewLayerRef.current);
      previewLayerRef.current = null;
    }
    if (markerRef.current) {
      map.removeLayer(markerRef.current);
      markerRef.current = null;
    }
    if (markerRef2.current) {
      map.removeLayer(markerRef2.current);
      markerRef2.current = null;
    }
  }, []);

  // ── Handle interaction mode ──────────────────────────────
  useEffect(() => {
    const map = mapInstanceRef.current;
    if (!map) return;

    // Reset state when mode changes
    firstCornerRef.current = null;
    secondCornerRef.current = null;
    dimensionCenterRef.current = null;
    dimensionRequestSeqRef.current += 1;
    clearPreview();

    if (selectionMode === 'draw') {
      // ── Two-click aligned rectangle ──
      map.getContainer().style.cursor = 'crosshair';

      const onMouseMove = (e: L.LeafletMouseEvent) => {
        if (!firstCornerRef.current) return;
        const bounds = L.latLngBounds(firstCornerRef.current, e.latlng);
        if (previewLayerRef.current) {
          (previewLayerRef.current as L.Rectangle).setBounds(bounds);
        } else {
          previewLayerRef.current = L.rectangle(bounds, {
            color: '#3388ff',
            fillColor: '#3388ff',
            fillOpacity: 0.15,
            weight: 1,
            dashArray: '6 4',
          }).addTo(map);
        }
      };

      const onClick = (e: L.LeafletMouseEvent) => {
        if (!firstCornerRef.current) {
          firstCornerRef.current = e.latlng;
          markerRef.current = L.circleMarker(e.latlng, {
            radius: 5,
            color: '#3388ff',
            fillColor: '#3388ff',
            fillOpacity: 1,
          }).addTo(map);
        } else {
          const sw = L.latLng(
            Math.min(firstCornerRef.current.lat, e.latlng.lat),
            Math.min(firstCornerRef.current.lng, e.latlng.lng),
          );
          const ne = L.latLng(
            Math.max(firstCornerRef.current.lat, e.latlng.lat),
            Math.max(firstCornerRef.current.lng, e.latlng.lng),
          );
          onRectangleChangeRef.current([
            [sw.lng, sw.lat],
            [sw.lng, ne.lat],
            [ne.lng, ne.lat],
            [ne.lng, sw.lat],
          ]);
          firstCornerRef.current = null;
          clearPreview();
        }
      };

      map.on('click', onClick);
      map.on('mousemove', onMouseMove);
      return () => {
        map.off('click', onClick);
        map.off('mousemove', onMouseMove);
        map.getContainer().style.cursor = '';
        firstCornerRef.current = null;
        clearPreview();
      };

    } else if (selectionMode === 'rotated') {
      // ── Three-click rotated rectangle ──
      // Click 1: first edge point
      // Click 2: second edge point (defines the width side)
      // Mousemove after click 1 or 2: live polygon preview
      // Click 3: extrude perpendicular to form the rectangle
      map.getContainer().style.cursor = 'crosshair';

      const previewStyle = {
        color: '#ff7800',
        fillColor: '#ff7800',
        fillOpacity: 0.15,
        weight: 1,
        dashArray: '6 4',
      };

      const updatePreviewPolygon = (verts: [number, number][] | null) => {
        if (!verts) {
          if (previewLayerRef.current) {
            map.removeLayer(previewLayerRef.current);
            previewLayerRef.current = null;
          }
          return;
        }
        const latlngs = verts.map(([lon, lat]) => L.latLng(lat, lon));
        if (previewLayerRef.current) {
          (previewLayerRef.current as L.Polygon).setLatLngs(latlngs);
        } else {
          previewLayerRef.current = L.polygon(latlngs, previewStyle).addTo(map);
        }
      };

      const onMouseMove = (e: L.LeafletMouseEvent) => {
        const p3: [number, number] = [e.latlng.lng, e.latlng.lat];
        if (firstCornerRef.current && !secondCornerRef.current) {
          // Phase 1 preview: show the line from p1 to cursor as a tiny rect
          const p1: [number, number] = [firstCornerRef.current.lng, firstCornerRef.current.lat];
          updatePreviewPolygon([p1, p3, p3, p3]);
        } else if (firstCornerRef.current && secondCornerRef.current) {
          // Phase 2 preview: build the rotated rectangle
          const p1: [number, number] = [firstCornerRef.current.lng, firstCornerRef.current.lat];
          const p2: [number, number] = [secondCornerRef.current.lng, secondCornerRef.current.lat];
          updatePreviewPolygon(buildRotatedRectangleFromClicks(p1, p2, p3));
        }
      };

      const onClick = (e: L.LeafletMouseEvent) => {
        if (!firstCornerRef.current) {
          // Click 1
          firstCornerRef.current = e.latlng;
          markerRef.current = L.circleMarker(e.latlng, {
            radius: 5, color: '#ff7800', fillColor: '#ff7800', fillOpacity: 1,
          }).addTo(map);
        } else if (!secondCornerRef.current) {
          // Click 2 — fix the edge direction
          secondCornerRef.current = e.latlng;
          markerRef2.current = L.circleMarker(e.latlng, {
            radius: 5, color: '#ff7800', fillColor: '#ff7800', fillOpacity: 1,
          }).addTo(map);
        } else {
          // Click 3 — finalise
          const p1: [number, number] = [firstCornerRef.current.lng, firstCornerRef.current.lat];
          const p2: [number, number] = [secondCornerRef.current.lng, secondCornerRef.current.lat];
          const p3: [number, number] = [e.latlng.lng, e.latlng.lat];
          const verts = buildRotatedRectangleFromClicks(p1, p2, p3);
          if (verts) onRectangleChangeRef.current(verts);
          firstCornerRef.current = null;
          secondCornerRef.current = null;
          clearPreview();
        }
      };

      map.on('click', onClick);
      map.on('mousemove', onMouseMove);
      return () => {
        map.off('click', onClick);
        map.off('mousemove', onMouseMove);
        map.getContainer().style.cursor = '';
        firstCornerRef.current = null;
        secondCornerRef.current = null;
        clearPreview();
      };

    } else {
      // ── Dimensions mode: click to set centre ──
      map.getContainer().style.cursor = 'crosshair';

      const onClick = async (e: L.LeafletMouseEvent) => {
        const { lat, lng } = e.latlng;
        const centerPoint = { centerLon: lng, centerLat: lat };
        dimensionCenterRef.current = centerPoint;
        await updateDimensionRectangle(centerPoint);
      };

      map.on('click', onClick);
      return () => {
        map.off('click', onClick);
        map.getContainer().style.cursor = '';
      };
    }
  }, [selectionMode, clearPreview, updateDimensionRectangle]);

  // Recompute an existing fixed-dimension rectangle when its controls change.
  useEffect(() => {
    if (selectionMode !== 'dimensions' || !dimensionCenterRef.current) return;
    void updateDimensionRectangle(dimensionCenterRef.current);
  }, [selectionMode, widthM, heightM, rotationDeg, updateDimensionRectangle]);

  // ── Draw rectangle overlay from external state ───────────
  // Uses L.Polygon (not L.Rectangle) so rotated rectangles render correctly.
  useEffect(() => {
    const map = mapInstanceRef.current;
    if (!map) return;

    if (rectLayerRef.current) {
      map.removeLayer(rectLayerRef.current);
      rectLayerRef.current = null;
    }

    if (rectangle && rectangle.length >= 4) {
      const latlngs = rectangle.map(([lon, lat]) => L.latLng(lat, lon));
      const poly = L.polygon(latlngs, {
        color: '#3388ff',
        fillColor: '#3388ff',
        fillOpacity: 0.2,
        weight: 2,
      });
      poly.addTo(map);
      rectLayerRef.current = poly;
    }
  }, [rectangle]);

  return <div ref={mapRef} style={{ width: '100%', height: '100%' }} />;
};

export default MapPicker;
