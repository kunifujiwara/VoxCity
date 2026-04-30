export type ZoneShape = 'rect' | 'polygon';

export interface Zone {
  id: string;
  name: string;
  color: string;
  shape: ZoneShape;
  ring_lonlat: [number, number][];
}

export const ZONE_PALETTE: string[] = [
  '#e6194B', '#3cb44b', '#ffe119', '#4363d8',
  '#f58231', '#911eb4', '#42d4f4', '#f032e6',
];

export function makeZoneId(): string {
  return `z_${Date.now().toString(36)}_${Math.random().toString(36).slice(2, 8)}`;
}

export function nextZoneName(existing: Zone[]): string {
  const used = new Set(existing.map((z) => z.name));
  let n = existing.length + 1;
  while (used.has(`Zone ${n}`)) n += 1;
  return `Zone ${n}`;
}

export function nextZoneColor(existing: Zone[]): string {
  return ZONE_PALETTE[existing.length % ZONE_PALETTE.length];
}

export function hashZones(zones: Zone[]): string {
  return zones
    .map((z) => `${z.id}:${z.ring_lonlat.map((p) => p.join(',')).join('|')}`)
    .join(';');
}
