import { describe, expect, it } from 'vitest';
import { toZoneSpecDto } from './api';
import type { HorizontalZone, BuildingSurfaceZone } from './types/zones';

describe('toZoneSpecDto', () => {
  it('converts horizontal zone to DTO', () => {
    const zone: HorizontalZone = {
      id: 'z1',
      name: 'Zone 1',
      color: '#ff0000',
      type: 'horizontal',
      shape: 'polygon',
      ring_lonlat: [[1, 2], [3, 4], [5, 6]],
    };
    const dto = toZoneSpecDto(zone);
    expect(dto).toEqual({
      id: 'z1',
      name: 'Zone 1',
      type: 'horizontal',
      ring_lonlat: [[1, 2], [3, 4], [5, 6]],
      selectors: [],
      group_id: undefined,
    });
  });

  it('converts surface zone to DTO with snake_case selectors', () => {
    const zone: BuildingSurfaceZone = {
      id: 'z2',
      name: 'Roof',
      color: '#00ff00',
      type: 'building_surface',
      groupId: 'g1',
      selectors: [
        { buildingId: 42, mode: 'whole' },
        { buildingId: 42, mode: 'exclude_faces', faceKeys: ['f1', 'f2'] },
        { buildingId: 5, mode: 'wall_orientation', orientation: 'N' },
      ],
    };
    const dto = toZoneSpecDto(zone);
    expect(dto).toEqual({
      id: 'z2',
      name: 'Roof',
      type: 'building_surface',
      ring_lonlat: null,
      selectors: [
        { building_id: 42, mode: 'whole', orientation: null, face_keys: null },
        { building_id: 42, mode: 'exclude_faces', orientation: null, face_keys: ['f1', 'f2'] },
        { building_id: 5, mode: 'wall_orientation', orientation: 'N', face_keys: null },
      ],
      group_id: 'g1',
    });
  });
});
