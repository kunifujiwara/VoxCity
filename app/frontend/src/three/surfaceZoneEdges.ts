import type { SurfaceZoneEdgesResponse, SurfaceZoneEdgeSegmentDto } from '../api';
import type { BuildingSurfaceZone, Zone } from '../types/zones';

export type SurfaceZoneEdgeSegment = SurfaceZoneEdgeSegmentDto;
export type SurfaceZoneEdgePoint = [number, number, number];

export interface SurfaceZoneEdgeRenderSpec {
  id: string;
  color: string;
  segments: SurfaceZoneEdgeSegment[];
}

export interface SurfaceZoneEdgeLineSpec {
  id: string;
  color: string;
  lineWidth: number;
  opacity: number;
  points: SurfaceZoneEdgePoint[];
}

function isSurfaceZoneWithSelectors(zone: Zone): zone is BuildingSurfaceZone {
  return zone.type === 'building_surface' && zone.selectors.length > 0;
}

export function getSurfaceZonesWithSelectors(zones: Zone[]): BuildingSurfaceZone[] {
  return zones.filter(isSurfaceZoneWithSelectors);
}

export function surfaceZoneEdgeRequestKey(zones: Zone[]): string {
  return JSON.stringify(
    getSurfaceZonesWithSelectors(zones).map((zone) => ({
      id: zone.id,
      selectors: zone.selectors,
    })),
  );
}

export function shouldFetchSurfaceZoneEdges(options: { hasModel: boolean; enabled: boolean; zones: Zone[] }): boolean {
  return options.hasModel && options.enabled && getSurfaceZonesWithSelectors(options.zones).length > 0;
}

export function toSurfaceZoneEdgeRenderSpecs(zones: Zone[], response: SurfaceZoneEdgesResponse | null): SurfaceZoneEdgeRenderSpec[] {
  if (!response) return [];
  const colors = new Map(zones.map((zone) => [zone.id, zone.color]));
  return response.zones
    .filter((payload) => payload.segments.length > 0 && colors.has(payload.id))
    .map((payload) => ({ id: payload.id, color: colors.get(payload.id)!, segments: payload.segments }));
}

export function segmentsToLinePoints(segments: SurfaceZoneEdgeSegment[]): SurfaceZoneEdgePoint[] {
  return segments.flatMap(([x1, y1, z1, x2, y2, z2]) => [[x1, y1, z1], [x2, y2, z2]] as SurfaceZoneEdgePoint[]);
}

export function buildSurfaceZoneEdgeLineSpecs(zones: SurfaceZoneEdgeRenderSpec[]): SurfaceZoneEdgeLineSpec[] {
  return zones.flatMap((zone) => {
    const points = segmentsToLinePoints(zone.segments);
    if (points.length === 0) return [];
    return [
      { id: zone.id, color: zone.color, lineWidth: 2, opacity: 1, points },
    ];
  });
}
