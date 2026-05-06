import { describe, expect, it } from 'vitest';
import {
  normalizeSurfaceSelectors,
  toggleSurfaceFace,
  toggleWholeBuilding,
  toggleBulkSelector,
  buildingHasPositiveSelection,
  resolveZoneGroupForMode,
  zoneGroupType,
  zoneTypeShortLabel,
  type SurfaceSelector,
  type SurfacePickMeta,
  type Zone,
} from './zones';

describe('normalizeSurfaceSelectors', () => {
  it('keeps exclude_faces when whole is present', () => {
    const selectors: SurfaceSelector[] = [
      { buildingId: 1, mode: 'whole' },
      { buildingId: 1, mode: 'roof' },
      { buildingId: 1, mode: 'exclude_faces', faceKeys: ['f1'] },
    ];
    expect(normalizeSurfaceSelectors(selectors)).toEqual([
      { buildingId: 1, mode: 'whole' },
      { buildingId: 1, mode: 'exclude_faces', faceKeys: ['f1'] },
    ]);
  });

  it('removes narrow selectors that are subsumed by whole', () => {
    const selectors: SurfaceSelector[] = [
      { buildingId: 1, mode: 'whole' },
      { buildingId: 1, mode: 'roof' },
    ];
    expect(normalizeSurfaceSelectors(selectors)).toEqual([
      { buildingId: 1, mode: 'whole' },
    ]);
  });

  it('keeps multi-building selectors independent', () => {
    const selectors: SurfaceSelector[] = [
      { buildingId: 1, mode: 'roof' },
      { buildingId: 2, mode: 'all_walls' },
    ];
    expect(normalizeSurfaceSelectors(selectors)).toEqual([
      { buildingId: 1, mode: 'roof' },
      { buildingId: 2, mode: 'all_walls' },
    ]);
  });
});

describe('toggleSurfaceFace', () => {
  it('clicking a bulk-selected face creates an exclusion', () => {
    const next = toggleSurfaceFace(
      [{ buildingId: 1, mode: 'roof' }],
      { buildingId: 1, faceKey: 'roof-a', surfaceKind: 'roof', orientation: null },
    );
    expect(next).toContainEqual({ buildingId: 1, mode: 'exclude_faces', faceKeys: ['roof-a'] });
  });

  it('clicking an excluded face re-includes it (removes exclusion)', () => {
    const next = toggleSurfaceFace(
      [
        { buildingId: 1, mode: 'whole' },
        { buildingId: 1, mode: 'exclude_faces', faceKeys: ['f1'] },
      ],
      { buildingId: 1, faceKey: 'f1', surfaceKind: 'roof', orientation: null },
    );
    const excl = next.find(
      (s) => s.mode === 'exclude_faces' && 'faceKeys' in s && s.faceKeys?.includes('f1'),
    );
    expect(excl).toBeUndefined();
  });
});

describe('toggleWholeBuilding', () => {
  it('adds whole selector for unselected building', () => {
    const next = toggleWholeBuilding([], 1);
    expect(next).toContainEqual({ buildingId: 1, mode: 'whole' });
  });

  it('removes all selectors for selected building', () => {
    const next = toggleWholeBuilding(
      [{ buildingId: 1, mode: 'whole' }],
      1,
    );
    expect(next.filter((s) => s.buildingId === 1)).toHaveLength(0);
  });
});

describe('buildingHasPositiveSelection', () => {
  it('returns true when building has whole selector', () => {
    expect(
      buildingHasPositiveSelection([{ buildingId: 1, mode: 'whole' }], 1),
    ).toBe(true);
  });

  it('returns false when building only has exclude selectors', () => {
    expect(
      buildingHasPositiveSelection(
        [{ buildingId: 1, mode: 'exclude_faces', faceKeys: ['f1'] }],
        1,
      ),
    ).toBe(false);
  });
});

describe('zone group edit mode compatibility', () => {
  const zones: Zone[] = [
    {
      id: 'h1',
      name: 'Area',
      color: '#ff0000',
      type: 'horizontal',
      shape: 'rect',
      ring_lonlat: [[0, 0], [1, 0], [1, 1]],
      groupId: 'area-group',
    },
    {
      id: 's1',
      name: 'Surface',
      color: '#00ff00',
      type: 'building_surface',
      selectors: [],
      groupId: 'surface-group',
    },
  ];

  it('reports the committed type for a zone group', () => {
    expect(zoneGroupType(zones, 'area-group')).toBe('horizontal');
    expect(zoneGroupType(zones, 'surface-group')).toBe('building_surface');
  });

  it('does not resolve a building surface group while editing 2D areas', () => {
    expect(resolveZoneGroupForMode({
      zones,
      candidates: [
        { id: 'area-group', draft: false },
        { id: 'surface-group', draft: false },
      ],
      activeGroupId: 'surface-group',
      zoneType: 'horizontal',
    })).toBeNull();
  });

  it('treats draft groups as 2D area groups only', () => {
    expect(resolveZoneGroupForMode({
      zones,
      candidates: [{ id: 'draft-group', draft: true }],
      activeGroupId: 'draft-group',
      zoneType: 'horizontal',
    })).toBe('draft-group');
    expect(resolveZoneGroupForMode({
      zones,
      candidates: [{ id: 'draft-group', draft: true }],
      activeGroupId: 'draft-group',
      zoneType: 'building_surface',
    })).toBeNull();
  });
});

describe('zone type display labels', () => {
  it('uses compact labels for zone-list badges', () => {
    expect(zoneTypeShortLabel('horizontal')).toBe('2D');
    expect(zoneTypeShortLabel('building_surface')).toBe('Building');
  });
});
