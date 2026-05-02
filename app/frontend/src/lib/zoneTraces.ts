/**
 * Stub kept only to satisfy `useZoneOverlay.ts` until the R3F migration
 * (Chunk 5) deletes both files. The legacy implementation produced
 * Plotly Mesh3d / Scatter3d "curtain" traces around each zone polygon;
 * the new architecture renders zone outlines directly in `<ZoneOutlines/>`
 * inside the R3F SceneViewer, so this no-op just returns nothing.
 */
import type { Zone } from '../types/zones';

export interface ZoneTraceCtx {
  meshsize: number;
  ceilingM: number;
  /** Optional id of the currently-selected zone (legacy field; ignored). */
  selectedZoneId?: string | null;
  [extra: string]: unknown;
}

export function buildZoneTraces(
  _zones: Zone[],
  _geo: { grid_geom: unknown },
  _ctx: ZoneTraceCtx,
): unknown[] {
  return [];
}
