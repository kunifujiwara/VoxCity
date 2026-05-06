import { useEffect, useState } from 'react';
import type { Zone } from '../types/zones';
import { getSurfaceZoneEdges } from '../api';
import type { SurfaceZoneEdgeRenderSpec } from '../three/surfaceZoneEdges';
import {
  shouldFetchSurfaceZoneEdges,
  toSurfaceZoneEdgeRenderSpecs,
} from '../three/surfaceZoneEdges';

interface UseSurfaceZoneEdgesOptions {
  hasModel: boolean;
  enabled: boolean;
  zones: Zone[];
}

export function useSurfaceZoneEdges({ hasModel, enabled, zones }: UseSurfaceZoneEdgesOptions): {
  surfaceZoneEdges: SurfaceZoneEdgeRenderSpec[] | null;
  loading: boolean;
} {
  const [surfaceZoneEdges, setSurfaceZoneEdges] = useState<SurfaceZoneEdgeRenderSpec[] | null>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (!shouldFetchSurfaceZoneEdges({ hasModel, enabled, zones })) {
      setSurfaceZoneEdges(null);
      setLoading(false);
      return;
    }

    let cancelled = false;
    setLoading(true);

    getSurfaceZoneEdges(zones)
      .then((response) => {
        if (cancelled) return;
        setSurfaceZoneEdges(toSurfaceZoneEdgeRenderSpecs(zones, response));
      })
      .catch((err: unknown) => {
        if (cancelled) return;
        const name = err instanceof Error ? err.name : null;
        if (name !== 'AbortError') {
          setSurfaceZoneEdges(null);
        }
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });

    return () => {
      cancelled = true;
    };
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [hasModel, enabled, JSON.stringify(zones)]);

  return { surfaceZoneEdges, loading };
}
