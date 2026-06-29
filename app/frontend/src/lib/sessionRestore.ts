import type { SurfaceSelector, WallOrientation, Zone } from '../types/zones';

export interface PersistedFrontendState {
  zones?: unknown;
}

export interface RestoredFrontendState {
  zones?: Zone[];
}

export interface BuiltRestoredFrontendState {
  restored: RestoredFrontendState;
  skippedFrontendState: boolean;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

function isNumberPair(value: unknown): value is [number, number] {
  return (
    Array.isArray(value) &&
    value.length === 2 &&
    typeof value[0] === 'number' &&
    typeof value[1] === 'number'
  );
}

function isNumberPairArray(value: unknown): value is [number, number][] {
  return Array.isArray(value) && value.every(isNumberPair);
}

const WALL_ORIENTATIONS: WallOrientation[] = ['N', 'E', 'S', 'W'];

function isSurfaceSelector(value: unknown): value is SurfaceSelector {
  if (!isRecord(value) || typeof value.buildingId !== 'number' || typeof value.mode !== 'string')
    return false;
  if (value.mode === 'whole' || value.mode === 'roof' || value.mode === 'all_walls' || value.mode === 'window') return true;
  if (value.mode === 'wall_orientation') {
    return WALL_ORIENTATIONS.includes(value.orientation as WallOrientation);
  }
  if (value.mode === 'faces' || value.mode === 'exclude_faces') {
    return Array.isArray(value.faceKeys) && value.faceKeys.every((k) => typeof k === 'string');
  }
  return false;
}

function isZone(value: unknown): value is Zone {
  if (!isRecord(value)) return false;
  if (
    typeof value.id !== 'string' ||
    typeof value.name !== 'string' ||
    typeof value.color !== 'string'
  )
    return false;
  if (value.type === 'horizontal') {
    return (
      (value.shape === 'rect' || value.shape === 'polygon') &&
      isNumberPairArray(value.ring_lonlat)
    );
  }
  if (value.type === 'building_surface') {
    return Array.isArray(value.selectors) && value.selectors.every(isSurfaceSelector);
  }
  return false;
}

function isZoneArray(value: unknown): value is Zone[] {
  return Array.isArray(value) && value.every(isZone);
}

export function parsePersistedFrontendState(
  frontendState: string | null,
): PersistedFrontendState | null {
  if (!frontendState) return null;
  try {
    const parsed: unknown = JSON.parse(frontendState);
    return isRecord(parsed) ? parsed : null;
  } catch {
    return null;
  }
}

export function buildRestoredFrontendState(
  persisted: PersistedFrontendState | null,
): BuiltRestoredFrontendState {
  const restored: RestoredFrontendState = {};
  let skippedFrontendState = false;

  if (!persisted) {
    return { restored, skippedFrontendState };
  }

  if ('zones' in persisted) {
    if (isZoneArray(persisted.zones)) {
      restored.zones = persisted.zones;
    } else {
      skippedFrontendState = true;
    }
  }

  return { restored, skippedFrontendState };
}
