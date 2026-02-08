import React, { useEffect, useRef, useCallback } from 'react';
import L from 'leaflet';

interface MapPickerProps {
  center: [number, number]; // [lat, lon]
  zoom: number;
  rectangle: number[][] | null; // [[lon,lat], ...] or null
  selectionMode: 'draw' | 'dimensions';
  widthM: number;
  heightM: number;
  onRectangleChange: (vertices: number[][]) => void;
}

const MapPicker: React.FC<MapPickerProps> = ({
  center,
  zoom,
  rectangle,
  selectionMode,
  widthM,
  heightM,
  onRectangleChange,
}) => {
  const mapRef = useRef<HTMLDivElement>(null);
  const mapInstanceRef = useRef<L.Map | null>(null);
  const rectLayerRef = useRef<L.Rectangle | null>(null);
  const previewLayerRef = useRef<L.Rectangle | null>(null);
  const firstCornerRef = useRef<L.LatLng | null>(null);
  const markerRef = useRef<L.CircleMarker | null>(null);

  // Stable callback ref so effects always see the latest callback
  const onRectangleChangeRef = useRef(onRectangleChange);
  onRectangleChangeRef.current = onRectangleChange;

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
  }, []);

  // ── Handle interaction mode ──────────────────────────────
  useEffect(() => {
    const map = mapInstanceRef.current;
    if (!map) return;

    // Reset state when mode changes
    firstCornerRef.current = null;
    clearPreview();

    if (selectionMode === 'draw') {
      // --- Two-click rectangle drawing ---
      map.getContainer().style.cursor = 'crosshair';

      const onMouseMove = (e: L.LeafletMouseEvent) => {
        if (!firstCornerRef.current) return;
        // Live preview rectangle while moving after first click
        const bounds = L.latLngBounds(firstCornerRef.current, e.latlng);
        if (previewLayerRef.current) {
          previewLayerRef.current.setBounds(bounds);
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
          // First click – record corner
          firstCornerRef.current = e.latlng;
          markerRef.current = L.circleMarker(e.latlng, {
            radius: 5,
            color: '#3388ff',
            fillColor: '#3388ff',
            fillOpacity: 1,
          }).addTo(map);
        } else {
          // Second click – finalise rectangle
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

          // Reset for next draw
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
    } else {
      // --- Dimensions mode: click to set centre ---
      map.getContainer().style.cursor = 'crosshair';

      const onClick = async (e: L.LeafletMouseEvent) => {
        const { lat, lng } = e.latlng;
        try {
          const res = await fetch('/api/rectangle-from-dimensions', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              center_lon: lng,
              center_lat: lat,
              width_m: widthM,
              height_m: heightM,
            }),
          });
          const data = await res.json();
          if (data.vertices) {
            onRectangleChangeRef.current(data.vertices);
          }
        } catch (err) {
          console.error('Failed to compute rectangle:', err);
        }
      };

      map.on('click', onClick);
      return () => {
        map.off('click', onClick);
        map.getContainer().style.cursor = '';
      };
    }
  }, [selectionMode, widthM, heightM, clearPreview]);

  // ── Draw rectangle overlay from external state ───────────
  useEffect(() => {
    const map = mapInstanceRef.current;
    if (!map) return;

    if (rectLayerRef.current) {
      map.removeLayer(rectLayerRef.current);
      rectLayerRef.current = null;
    }

    if (rectangle && rectangle.length >= 4) {
      const bounds: L.LatLngBoundsExpression = [
        [rectangle[0][1], rectangle[0][0]], // SW
        [rectangle[2][1], rectangle[2][0]], // NE
      ];
      const rect = L.rectangle(bounds, {
        color: '#3388ff',
        fillColor: '#3388ff',
        fillOpacity: 0.2,
        weight: 2,
      });
      rect.addTo(map);
      rectLayerRef.current = rect;
    }
  }, [rectangle]);

  return <div ref={mapRef} style={{ width: '100%', height: '100%' }} />;
};

export default MapPicker;
