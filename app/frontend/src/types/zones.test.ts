import { describe, expect, it } from 'vitest';
import {
  normalizeSurfaceSelectors,
  toggleSurfaceFace,
  toggleWholeBuilding,
  toggleBulkSelector,
  buildingHasPositiveSelection,
  type SurfaceSelector,
  type SurfacePickMeta,
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
