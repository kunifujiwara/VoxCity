// ─────────────────────────────────────────────────────────────────────────────
// Zone type discriminated union for VoxCity
// ─────────────────────────────────────────────────────────────────────────────

export type ZoneType = 'horizontal' | 'building_surface';
export type WallOrientation = 'N' | 'E' | 'S' | 'W';
export type ZoneShape = 'rect' | 'polygon';

// ── Surface selector union ────────────────────────────────────────────────────

export type SurfaceSelector =
  | { buildingId: number; mode: 'whole' }
  | { buildingId: number; mode: 'roof' }
  | { buildingId: number; mode: 'all_walls' }
  | { buildingId: number; mode: 'wall_orientation'; orientation: WallOrientation }
  | { buildingId: number; mode: 'faces'; faceKeys: string[] }
  | { buildingId: number; mode: 'exclude_faces'; faceKeys: string[] };

export interface SurfacePickMeta {
  buildingId: number;
  faceKey: string;
  surfaceKind: 'roof' | 'wall' | 'bottom' | 'other';
  orientation?: WallOrientation | null;
}

// ── Zone interfaces ───────────────────────────────────────────────────────────

export interface HorizontalZone {
  id: string;
  name: string;
  color: string;
  type: 'horizontal';
  shape: ZoneShape;
  ring_lonlat: [number, number][];
  groupId?: string;
}

export interface BuildingSurfaceZone {
  id: string;
  name: string;
  color: string;
  type: 'building_surface';
  selectors: SurfaceSelector[];
  groupId?: string;
}

export type Zone = HorizontalZone | BuildingSurfaceZone;

// ── Color palette and ID generation ──────────────────────────────────────────

export const ZONE_PALETTE: string[] = [
  '#e6194B', '#3cb44b', '#ffe119', '#4363d8',
  '#f58231', '#911eb4', '#42d4f4', '#f032e6',
];

export function makeZoneId(): string {
  return `z_${Date.now().toString(36)}_${Math.random().toString(36).slice(2, 8)}`;
}

export function makeZoneGroupId(): string {
  return `g_${Date.now().toString(36)}_${Math.random().toString(36).slice(2, 8)}`;
}

// ── Zone group helpers ────────────────────────────────────────────────────────

/** Distinct group keys, in insertion order. */
export function zoneGroupKeys(existing: Zone[]): string[] {
  const seen = new Set<string>();
  const out: string[] = [];
  for (const z of existing) {
    const key = z.groupId ?? z.id;
    if (!seen.has(key)) {
      seen.add(key);
      out.push(key);
    }
  }
  return out;
}

export function nextZoneName(existing: Zone[]): string {
  const used = new Set(existing.map((z) => z.name));
  let n = zoneGroupKeys(existing).length + 1;
  while (used.has(`Zone ${n}`)) n += 1;
  return `Zone ${n}`;
}

export function nextZoneColor(existing: Zone[]): string {
  return ZONE_PALETTE[zoneGroupKeys(existing).length % ZONE_PALETTE.length];
}

export function hashZones(zones: Zone[]): string {
  return zones
    .map((z) => {
      if (z.type === 'building_surface') {
        const selStr = z.selectors
          .map((s) => `${s.buildingId}:${s.mode}${'orientation' in s ? ':' + s.orientation : ''}${'faceKeys' in s ? ':' + s.faceKeys.join(',') : ''}`)
          .join('|');
        return `${z.id}:surface:${selStr}`;
      }
      return `${z.id}:${z.ring_lonlat.map((p) => p.join(',')).join('|')}`;
    })
    .join(';');
}

/** Returns the type of zones in a group, or null if the group is empty/mixed. */
export function zoneGroupType(zones: Zone[], groupId: string): ZoneType | null {
  const members = zones.filter((z) => (z.groupId ?? z.id) === groupId);
  if (members.length === 0) return null;
  const types = new Set(members.map((z) => z.type));
  return types.size === 1 ? (members[0].type as ZoneType) : null;
}

/** Human-readable summary for a building surface zone. */
export function surfaceZoneSummary(zone: BuildingSurfaceZone): string {
  const buildingIds = [...new Set(zone.selectors.map((s) => s.buildingId))];
  if (buildingIds.length === 0) return 'No buildings selected';
  if (buildingIds.length === 1) return `Building ${buildingIds[0]}`;
  return `${buildingIds.length} buildings`;
}

// ── Surface selector normalization ────────────────────────────────────────────

const POSITIVE_MODES = new Set(['whole', 'roof', 'all_walls', 'wall_orientation', 'faces']);
const BULK_MODES = new Set(['whole', 'roof', 'all_walls', 'wall_orientation']);

/**
 * Normalize selectors: for each building, if 'whole' is present, remove narrower
 * positive selectors (roof, all_walls, wall_orientation) — keep exclude_faces and faces.
 */
export function normalizeSurfaceSelectors(selectors: SurfaceSelector[]): SurfaceSelector[] {
  const buildingIds = [...new Set(selectors.map((s) => s.buildingId))];
  const result: SurfaceSelector[] = [];

  for (const bid of buildingIds) {
    const forBuilding = selectors.filter((s) => s.buildingId === bid);
    const hasWhole = forBuilding.some((s) => s.mode === 'whole');

    if (hasWhole) {
      // Keep only 'whole' and 'exclude_faces'
      result.push({ buildingId: bid, mode: 'whole' });
      for (const s of forBuilding) {
        if (s.mode === 'exclude_faces') result.push(s);
      }
    } else {
      result.push(...forBuilding);
    }
  }

  return result;
}

/** Whether a building has any positive (non-exclude) selectors. */
export function buildingHasPositiveSelection(selectors: SurfaceSelector[], buildingId: number): boolean {
  return selectors.some((s) => s.buildingId === buildingId && POSITIVE_MODES.has(s.mode));
}

/** Toggle whole-building selection: add if not selected, remove all if selected. */
export function toggleWholeBuilding(selectors: SurfaceSelector[], buildingId: number): SurfaceSelector[] {
  const hasPositive = buildingHasPositiveSelection(selectors, buildingId);
  if (hasPositive) {
    // Remove all selectors for this building
    return selectors.filter((s) => s.buildingId !== buildingId);
  }
  // Add whole selector
  const next = [...selectors, { buildingId, mode: 'whole' } as SurfaceSelector];
  return normalizeSurfaceSelectors(next);
}

/**
 * Toggle a bulk selector (roof, all_walls, wall_orientation) for a building.
 * If this is the only selector for the building, removes it. Otherwise replaces.
 */
export function toggleBulkSelector(
  selectors: SurfaceSelector[],
  buildingId: number,
  mode: 'roof' | 'all_walls' | 'wall_orientation',
  orientation?: WallOrientation,
): SurfaceSelector[] {
  // Check if this exact selector already exists
  const existing = selectors.find(
    (s) =>
      s.buildingId === buildingId &&
      s.mode === mode &&
      (!('orientation' in s) || s.orientation === orientation),
  );
  if (existing) {
    // Remove it (and keep other selectors for this building)
    return selectors.filter((s) => s !== existing);
  }
  // Remove other positive selectors for this building (not exclude_faces)
  const filtered = selectors.filter(
    (s) => s.buildingId !== buildingId || !POSITIVE_MODES.has(s.mode),
  );
  const newSelector =
    mode === 'wall_orientation' && orientation
      ? ({ buildingId, mode, orientation } as SurfaceSelector)
      : ({ buildingId, mode } as SurfaceSelector);
  return normalizeSurfaceSelectors([...filtered, newSelector]);
}

/**
 * Toggle a specific face:
 * - If the face is currently excluded → remove exclusion (re-include it)
 * - If the face is bulk-included (via whole/roof/all_walls/wall_orientation) → add exclude
 * - If the face is not selected → add it via 'faces' selector
 */
export function toggleSurfaceFace(
  selectors: SurfaceSelector[],
  surface: SurfacePickMeta,
): SurfaceSelector[] {
  const { buildingId, faceKey } = surface;

  // Check if face is currently excluded
  const excludeIdx = selectors.findIndex(
    (s) => s.buildingId === buildingId && s.mode === 'exclude_faces' && 'faceKeys' in s && s.faceKeys.includes(faceKey),
  );
  if (excludeIdx !== -1) {
    // Remove from exclusion list
    const excl = selectors[excludeIdx] as { buildingId: number; mode: 'exclude_faces'; faceKeys: string[] };
    const newKeys = excl.faceKeys.filter((k) => k !== faceKey);
    if (newKeys.length === 0) {
      return selectors.filter((_, i) => i !== excludeIdx);
    }
    return selectors.map((s, i) =>
      i === excludeIdx ? { ...excl, faceKeys: newKeys } : s,
    );
  }

  // Check if face is bulk-included
  const hasBulk = selectors.some(
    (s) => s.buildingId === buildingId && BULK_MODES.has(s.mode),
  );
  if (hasBulk) {
    // Add to exclude
    const existingExclude = selectors.find(
      (s) => s.buildingId === buildingId && s.mode === 'exclude_faces',
    ) as { buildingId: number; mode: 'exclude_faces'; faceKeys: string[] } | undefined;
    if (existingExclude) {
      return selectors.map((s) =>
        s === existingExclude
          ? { ...existingExclude, faceKeys: [...existingExclude.faceKeys, faceKey] }
          : s,
      );
    }
    return [...selectors, { buildingId, mode: 'exclude_faces', faceKeys: [faceKey] }];
  }

  // Check if face is in a 'faces' selector
  const facesIdx = selectors.findIndex(
    (s) => s.buildingId === buildingId && s.mode === 'faces' && 'faceKeys' in s && s.faceKeys.includes(faceKey),
  );
  if (facesIdx !== -1) {
    // Remove from faces selector
    const facesSelector = selectors[facesIdx] as { buildingId: number; mode: 'faces'; faceKeys: string[] };
    const newKeys = facesSelector.faceKeys.filter((k) => k !== faceKey);
    if (newKeys.length === 0) {
      return selectors.filter((_, i) => i !== facesIdx);
    }
    return selectors.map((s, i) =>
      i === facesIdx ? { ...facesSelector, faceKeys: newKeys } : s,
    );
  }

  // Add to faces selector
  const existingFaces = selectors.find(
    (s) => s.buildingId === buildingId && s.mode === 'faces',
  ) as { buildingId: number; mode: 'faces'; faceKeys: string[] } | undefined;
  if (existingFaces) {
    return selectors.map((s) =>
      s === existingFaces
        ? { ...existingFaces, faceKeys: [...existingFaces.faceKeys, faceKey] }
        : s,
    );
  }
  return [...selectors, { buildingId, mode: 'faces', faceKeys: [faceKey] }];
}
