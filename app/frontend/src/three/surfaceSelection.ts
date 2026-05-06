import * as THREE from 'three';
import type { BuildingSurfacesResponse, MeshChunkDto, SurfaceFaceMetaDto } from '../api';
import type { BuildingSurfaceZone, SurfaceSelector, WallOrientation, Zone } from '../types/zones';
import type { SurfaceFaceMeta } from './types';

export type SurfaceSelectionDisplayMode = 'fill' | 'boundary';
export type BoundaryPoint = [number, number, number];

export interface SurfaceSelectionLayerSpec {
  id: string;
  color: string;
  selectors: SurfaceSelector[];
  active: boolean;
}

export interface SceneSurfaceSelectionSpec {
  surfaceChunk: MeshChunkDto | null;
  faceToSurface: SurfaceFaceMeta[];
  zones: SurfaceSelectionLayerSpec[];
  enabled: boolean;
  displayMode: SurfaceSelectionDisplayMode;
}

type SurfaceFaceMetaInput = SurfaceFaceMetaDto[] | BuildingSurfacesResponse['face_to_surface'] | Record<string, SurfaceFaceMetaDto>;

const EDGE_KEY_PRECISION = 6;

export function toSurfaceFaceMetaArray(faceToSurface: SurfaceFaceMetaInput): SurfaceFaceMeta[] {
  const values = Array.isArray(faceToSurface) ? faceToSurface : Object.values(faceToSurface);
  return values.map((face) => ({
    faceKey: face.face_key,
    buildingId: face.building_id,
    surfaceKind: face.surface_kind as SurfaceFaceMeta['surfaceKind'],
    orientation: face.orientation as WallOrientation | null | undefined,
  }));
}

export function isSurfaceFaceSelected(meta: SurfaceFaceMeta, selectors: SurfaceSelector[]): boolean {
  const buildingSelectors = selectors.filter((selector) => selector.buildingId === meta.buildingId);
  if (buildingSelectors.length === 0) return false;

  const excluded = buildingSelectors.some(
    (selector) =>
      selector.mode === 'exclude_faces' &&
      'faceKeys' in selector &&
      selector.faceKeys.includes(meta.faceKey),
  );
  if (excluded) return false;

  return buildingSelectors.some((selector) => {
    switch (selector.mode) {
      case 'whole':
        return true;
      case 'roof':
        return meta.surfaceKind === 'roof';
      case 'all_walls':
        return meta.surfaceKind === 'wall';
      case 'wall_orientation':
        return meta.surfaceKind === 'wall' && meta.orientation === selector.orientation;
      case 'faces':
        return selector.faceKeys.includes(meta.faceKey);
      case 'exclude_faces':
        return false;
    }
  });
}

export function getSurfaceZones(zones: Zone[]): BuildingSurfaceZone[] {
  return zones.filter((zone): zone is BuildingSurfaceZone => zone.type === 'building_surface');
}

export function toSurfaceSelectionLayerSpecs(
  zones: BuildingSurfaceZone[],
  activeGroupId: string | null = null,
): SurfaceSelectionLayerSpec[] {
  return zones.map((zone) => ({
    id: zone.id,
    color: zone.color,
    selectors: zone.selectors,
    active: activeGroupId != null && (zone.groupId ?? zone.id) === activeGroupId,
  }));
}

export function hasUsableSurfaceGeometry(
  surfaceChunk: MeshChunkDto | null,
  faceToSurface: SurfaceFaceMeta[],
): surfaceChunk is MeshChunkDto {
  return !!surfaceChunk && surfaceChunk.positions.length > 0 && faceToSurface.length > 0 && faceToSurface.length === surfaceTriangleCount(surfaceChunk);
}

export function surfaceTriangleCount(surfaceChunk: MeshChunkDto): number {
  const indices = surfaceChunk.indices ?? [];
  if (indices.length > 0) return Math.floor(indices.length / 3);
  return Math.floor(surfaceChunk.positions.length / 9);
}

export function shouldFetchSurfaceSelection(options: {
  hasModel: boolean;
  enabled: boolean;
  surfaceZoneCount: number;
  requireSurfaceZones: boolean;
}): boolean {
  return options.hasModel && options.enabled && (!options.requireSurfaceZones || options.surfaceZoneCount > 0);
}

export function shouldMountPickableSurface(
  onPick: unknown,
  surfaceSelection: SceneSurfaceSelectionSpec | null | undefined,
): boolean {
  return typeof onPick === 'function' && !!surfaceSelection?.enabled && !!surfaceSelection.surfaceChunk;
}

export function emptySurfaceGeometryState(): { surfaceChunk: null; faceToSurface: SurfaceFaceMeta[] } {
  return { surfaceChunk: null, faceToSurface: [] };
}

export function surfaceLoadErrorResult(
  error: unknown,
  silentOnError: boolean,
  onError?: (message: string) => void,
): { surfaceChunk: null; faceToSurface: SurfaceFaceMeta[] } {
  if (!silentOnError) {
    const message = error instanceof Error ? error.message : String(error);
    onError?.(message);
  }
  return emptySurfaceGeometryState();
}

export function buildSceneSurfaceSelectionSpec(options: {
  surfaceChunk: MeshChunkDto | null;
  faceToSurface: SurfaceFaceMeta[];
  zones: BuildingSurfaceZone[];
  enabled: boolean;
  displayMode: SurfaceSelectionDisplayMode;
  activeGroupId?: string | null;
  requireSelectors: boolean;
}): SceneSurfaceSelectionSpec | null {
  const layerZones = toSurfaceSelectionLayerSpecs(options.zones, options.activeGroupId ?? null);
  const hasRenderableSelectors = layerZones.some((zone) => zone.selectors.length > 0);

  if (!options.enabled) return null;
  if (!hasUsableSurfaceGeometry(options.surfaceChunk, options.faceToSurface)) return null;
  if (options.requireSelectors && !hasRenderableSelectors) return null;

  return {
    surfaceChunk: options.surfaceChunk,
    faceToSurface: options.faceToSurface,
    zones: layerZones,
    enabled: true,
    displayMode: options.displayMode,
  };
}

export function buildSelectedTriangleIndices(
  faceToSurface: SurfaceFaceMeta[],
  selectors: SurfaceSelector[],
): number[] {
  const selectedTriangleIndices: number[] = [];
  for (let index = 0; index < faceToSurface.length; index++) {
    const meta = faceToSurface[index];
    if (meta && isSurfaceFaceSelected(meta, selectors)) {
      selectedTriangleIndices.push(index);
    }
  }
  return selectedTriangleIndices;
}

function vertexOffset(surfaceChunk: MeshChunkDto, triangleIndex: number, cornerIndex: 0 | 1 | 2): number {
  const indexOffset = triangleIndex * 3 + cornerIndex;
  const indices = surfaceChunk.indices ?? [];
  const vertexIndex = indices.length > indexOffset
    ? indices[indexOffset]
    : indexOffset;
  return vertexIndex * 3;
}

function readPoint(surfaceChunk: MeshChunkDto, triangleIndex: number, cornerIndex: 0 | 1 | 2): BoundaryPoint {
  const offset = vertexOffset(surfaceChunk, triangleIndex, cornerIndex);
  const positions = surfaceChunk.positions;
  return [positions[offset], positions[offset + 1], positions[offset + 2]];
}

export function buildZoneFillGeometry(
  surfaceChunk: MeshChunkDto,
  faceToSurface: SurfaceFaceMeta[],
  selectors: SurfaceSelector[],
): THREE.BufferGeometry | null {
  const selectedTriangleIndices = buildSelectedTriangleIndices(faceToSurface, selectors);
  if (selectedTriangleIndices.length === 0) return null;

  const out = new Float32Array(selectedTriangleIndices.length * 9);
  let outOffset = 0;
  for (const triangleIndex of selectedTriangleIndices) {
    for (const cornerIndex of [0, 1, 2] as const) {
      const point = readPoint(surfaceChunk, triangleIndex, cornerIndex);
      out[outOffset++] = point[0];
      out[outOffset++] = point[1];
      out[outOffset++] = point[2];
    }
  }

  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.BufferAttribute(out, 3));
  geometry.computeVertexNormals();
  return geometry;
}

function pointKey(point: BoundaryPoint): string {
  return point.map((value) => value.toFixed(EDGE_KEY_PRECISION)).join(',');
}

function edgeKey(a: BoundaryPoint, b: BoundaryPoint): string {
  const aKey = pointKey(a);
  const bKey = pointKey(b);
  return aKey < bKey ? `${aKey}|${bKey}` : `${bKey}|${aKey}`;
}

export function buildBoundaryLinePoints(
  surfaceChunk: MeshChunkDto,
  faceToSurface: SurfaceFaceMeta[],
  selectors: SurfaceSelector[],
): BoundaryPoint[] {
  const selectedTriangleIndices = buildSelectedTriangleIndices(faceToSurface, selectors);
  const edgeCounts = new Map<string, { count: number; start: BoundaryPoint; end: BoundaryPoint }>();

  for (const triangleIndex of selectedTriangleIndices) {
    const a = readPoint(surfaceChunk, triangleIndex, 0);
    const b = readPoint(surfaceChunk, triangleIndex, 1);
    const c = readPoint(surfaceChunk, triangleIndex, 2);
    for (const [start, end] of [[a, b], [b, c], [c, a]] as const) {
      const key = edgeKey(start, end);
      const previous = edgeCounts.get(key);
      if (previous) {
        previous.count += 1;
      } else {
        edgeCounts.set(key, { count: 1, start, end });
      }
    }
  }

  const points: BoundaryPoint[] = [];
  for (const edge of edgeCounts.values()) {
    if (edge.count === 1) {
      points.push(edge.start, edge.end);
    }
  }
  return points;
}
