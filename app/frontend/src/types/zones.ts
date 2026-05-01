export type ZoneShape = 'rect' | 'polygon';

export interface Zone {
  id: string;
  name: string;
  color: string;
  shape: ZoneShape;
  ring_lonlat: [number, number][];
  /**
   * Optional UI-only group identifier. Multiple `Zone` records sharing the
   * same `groupId` are presented as one logical zone (e.g. "Zone 1 has two
   * polygons"). When unset, the zone is treated as its own one-ring group.
   */
  groupId?: string;
}

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
    .map((z) => `${z.id}:${z.ring_lonlat.map((p) => p.join(',')).join('|')}`)
    .join(';');
}
