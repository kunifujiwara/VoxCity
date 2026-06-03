import { describe, expect, it } from 'vitest';
import { aggregateByGroup } from './useZoneStats';
import type { HorizontalZone, Zone } from '../types/zones';
import type { ZoneStatsResponse } from '../api';

function makeZone(id: string, groupId?: string): HorizontalZone {
  return {
    id,
    name: id,
    color: '#ff0000',
    type: 'horizontal',
    shape: 'polygon',
    ring_lonlat: [[0, 0], [1, 0], [1, 1]],
    ...(groupId ? { groupId } : {}),
  };
}

function makeResponse(stats: ZoneStatsResponse['stats']): ZoneStatsResponse {
  return { target: 'ground', sim_type: 'solar', unit_label: 'kWh/m²', stats };
}

describe('aggregateByGroup – std behaviour', () => {
  it('passes std through for a single-ring zone', () => {
    const zones = [makeZone('z1')];
    const response = makeResponse([
      { zone_id: 'z1', cell_count: 10, valid_count: 10, mean: 0.5, min: 0.1, max: 0.9, std: 0.25 },
    ]);
    const result = aggregateByGroup(zones, response);
    expect(result.stats[0].std).toBe(0.25);
  });

  it('sets std to null when two rings merge into one group', () => {
    const zones = [makeZone('r1', 'g1'), makeZone('r2', 'g1')];
    const response = makeResponse([
      { zone_id: 'r1', cell_count: 5, valid_count: 5, mean: 0.4, min: 0.1, max: 0.7, std: 0.1 },
      { zone_id: 'r2', cell_count: 5, valid_count: 5, mean: 0.6, min: 0.3, max: 0.9, std: 0.2 },
    ]);
    const result = aggregateByGroup(zones, response);
    expect(result.stats[0].std).toBeNull();
  });

  it('passes null std through when backend returns null', () => {
    const zones = [makeZone('z1')];
    const response = makeResponse([
      { zone_id: 'z1', cell_count: 10, valid_count: 10, mean: 0.5, min: 0.1, max: 0.9, std: null },
    ]);
    const result = aggregateByGroup(zones, response);
    expect(result.stats[0].std).toBeNull();
  });
});
