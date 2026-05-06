import { useEffect, useMemo, useRef, useState } from 'react';
import { getBuildingSurfaces, type MeshChunkDto } from '../api';
import type { Zone } from '../types/zones';
import type { SurfaceFaceMeta } from '../three/types';
import {
  buildSceneSurfaceSelectionSpec,
  emptySurfaceGeometryState,
  getSurfaceZones,
  shouldFetchSurfaceSelection,
  surfaceLoadErrorResult,
  toSurfaceFaceMetaArray,
  type SceneSurfaceSelectionSpec,
  type SurfaceSelectionDisplayMode,
} from '../three/surfaceSelection';

interface UseSurfaceZoneSelectionOptions {
  hasModel: boolean;
  zones: Zone[];
  enabled: boolean;
  displayMode: SurfaceSelectionDisplayMode;
  geometryToken?: string | number;
  activeGroupId?: string | null;
  requireSurfaceZones?: boolean;
  requireSelectors?: boolean;
  silentOnError?: boolean;
  onError?: (message: string) => void;
}

export function useSurfaceZoneSelection({
  hasModel,
  zones,
  enabled,
  displayMode,
  geometryToken,
  activeGroupId = null,
  requireSurfaceZones = true,
  requireSelectors = true,
  silentOnError = true,
  onError,
}: UseSurfaceZoneSelectionOptions): {
  surfaceChunk: MeshChunkDto | null;
  faceToSurface: SurfaceFaceMeta[];
  surfaceSelection: SceneSurfaceSelectionSpec | null;
  loading: boolean;
} {
  const silentOnErrorRef = useRef(silentOnError);
  silentOnErrorRef.current = silentOnError;
  const onErrorRef = useRef(onError);
  onErrorRef.current = onError;

  const [surfaceChunk, setSurfaceChunk] = useState<MeshChunkDto | null>(null);
  const [faceToSurface, setFaceToSurface] = useState<SurfaceFaceMeta[]>([]);
  const [loading, setLoading] = useState(false);

  const surfaceZones = useMemo(() => getSurfaceZones(zones), [zones]);
  const shouldFetch = shouldFetchSurfaceSelection({
    hasModel,
    enabled,
    surfaceZoneCount: surfaceZones.length,
    requireSurfaceZones,
  });

  useEffect(() => {
    if (!shouldFetch) {
      const next = emptySurfaceGeometryState();
      setSurfaceChunk(next.surfaceChunk);
      setFaceToSurface(next.faceToSurface);
      setLoading(false);
      return;
    }

    let cancelled = false;
    setLoading(true);
    getBuildingSurfaces()
      .then((response) => {
        if (cancelled) return;
        const nextFaceToSurface = toSurfaceFaceMetaArray(response.face_to_surface);
        if (response.chunk.positions.length === 0 || nextFaceToSurface.length === 0) {
          setSurfaceChunk(null);
          setFaceToSurface([]);
          return;
        }
        setSurfaceChunk(response.chunk);
        setFaceToSurface(nextFaceToSurface);
      })
      .catch((error: unknown) => {
        if (cancelled) return;
        const next = surfaceLoadErrorResult(error, silentOnErrorRef.current, onErrorRef.current);
        setSurfaceChunk(next.surfaceChunk);
        setFaceToSurface(next.faceToSurface);
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });

    return () => {
      cancelled = true;
    };
  }, [shouldFetch, geometryToken]);

  const surfaceSelection = buildSceneSurfaceSelectionSpec({
    surfaceChunk,
    faceToSurface,
    zones: surfaceZones,
    enabled,
    displayMode,
    activeGroupId,
    requireSelectors,
  });

  return { surfaceChunk, faceToSurface, surfaceSelection, loading };
}
