import { useEffect, useMemo, useState } from 'react';
import type { Zone } from '../types/zones';
import { getSurfaceZoneEdges, type SurfaceZoneEdgesResponse } from '../api';
import type { SurfaceZoneEdgeRenderSpec } from '../three/surfaceZoneEdges';
import {
  getSurfaceZonesWithSelectors,
  shouldFetchSurfaceZoneEdges,
  surfaceZoneEdgeRequestKey,
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
  const surfaceZones = useMemo(() => getSurfaceZonesWithSelectors(zones), [zones]);
  const requestKey = useMemo(() => surfaceZoneEdgeRequestKey(surfaceZones), [surfaceZones]);
  const requestZones = useMemo(() => surfaceZones, [requestKey]);
  const [response, setResponse] = useState<SurfaceZoneEdgesResponse | null>(null);
  const [loading, setLoading] = useState(false);

  const surfaceZoneEdges = useMemo<SurfaceZoneEdgeRenderSpec[] | null>(
    () => (response ? toSurfaceZoneEdgeRenderSpecs(zones, response) : null),
    [zones, response],
  );

  useEffect(() => {
    if (!shouldFetchSurfaceZoneEdges({ hasModel, enabled, zones: requestZones })) {
      setResponse(null);
      setLoading(false);
      return;
    }

    const controller = new AbortController();
    setLoading(true);

    getSurfaceZoneEdges(requestZones, controller.signal)
      .then((response) => {
        if (controller.signal.aborted) return;
        setResponse(response);
      })
      .catch((err: unknown) => {
        if (controller.signal.aborted) return;
        const name = err instanceof Error ? err.name : null;
        if (name !== 'AbortError') {
          setResponse(null);
        }
      })
      .finally(() => {
        if (!controller.signal.aborted) setLoading(false);
      });

    return () => {
      controller.abort();
    };
  }, [hasModel, enabled, requestKey, requestZones]);

  return { surfaceZoneEdges, loading };
}
