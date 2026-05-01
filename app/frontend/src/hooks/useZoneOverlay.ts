import { useEffect, useMemo, useState } from 'react';
import { getModelGeo, ModelGeoResult } from '../api';
import { Zone } from '../types/zones';
import { buildZoneTraces } from '../lib/zoneTraces';

/**
 * Helper used by Solar/View/Landmark tabs to overlay zone curtains on top of
 * the simulation figure. Fetches the model geometry once, derives a sensible
 * curtain height from the tallest building, and (when the toggle is on)
 * appends `Mesh3d`/`Scatter3d` traces from `buildZoneTraces` to the figure.
 */
export function useZoneOverlay(
  hasModel: boolean,
  figureJson: string,
  zones: Zone[],
  show: boolean,
): { figure: string; geo: ModelGeoResult | null } {
  const [geo, setGeo] = useState<ModelGeoResult | null>(null);

  useEffect(() => {
    if (!hasModel) {
      setGeo(null);
      return;
    }
    let cancelled = false;
    getModelGeo()
      .then((g) => {
        if (!cancelled) setGeo(g);
      })
      .catch(() => {
        /* swallow — overlay is best-effort */
      });
    return () => {
      cancelled = true;
    };
  }, [hasModel]);

  const maxH = useMemo(() => {
    if (!geo?.building_geojson) return 30;
    let m = 0;
    for (const f of geo.building_geojson.features ?? []) {
      const h = Number(f.properties?.height ?? 0);
      if (Number.isFinite(h) && h > m) m = h;
    }
    return m > 0 ? m : 30;
  }, [geo]);

  const figure = useMemo(() => {
    if (!figureJson) return '';
    if (!show || !geo || zones.length === 0) return figureJson;
    try {
      const fig = JSON.parse(figureJson);
      fig.data = [
        ...(fig.data ?? []),
        ...buildZoneTraces(zones, { grid_geom: geo.grid_geom }, {
          meshsize: geo.meshsize_m,
          ceilingM: Math.max(0.1 * maxH, 3),
        }),
      ];
      return JSON.stringify(fig);
    } catch {
      return figureJson;
    }
  }, [figureJson, show, geo, zones, maxH]);

  return { figure, geo };
}
