import { describe, expect, it } from 'vitest';
import {
  buildingHeightLabelValue,
  pendingBuildingHeightOverrides,
  readBuildingHeightValue,
  toSetBuildingHeightDto,
  validateBuildingHeightInput,
} from './buildingHeightEdits';

function feature(id: unknown, height: unknown, minHeight?: unknown) {
  return {
    type: 'Feature',
    properties: {
      idx: id,
      height,
      ...(minHeight !== undefined ? { min_height: minHeight } : {}),
    },
    geometry: null,
  };
}

describe('readBuildingHeightValue', () => {
  it('reads committed height and min height from a feature', () => {
    expect(readBuildingHeightValue(feature(42, 18.5, 3))).toEqual({
      buildingId: 42,
      height: 18.5,
      minHeight: 3,
    });
  });

  it('returns null when id or height is missing or non-finite', () => {
    expect(readBuildingHeightValue(feature(null, 18))).toBeNull();
    expect(readBuildingHeightValue(feature(42, undefined))).toBeNull();
    expect(readBuildingHeightValue(feature(42, Number.NaN))).toBeNull();
  });

  it('defaults missing or non-finite min height to zero', () => {
    expect(readBuildingHeightValue(feature(42, 18))?.minHeight).toBe(0);
    expect(readBuildingHeightValue(feature(42, 18, Number.NaN))?.minHeight).toBe(0);
  });
});

describe('pendingBuildingHeightOverrides', () => {
  it('replays pending height edits in order and ignores other edits', () => {
    const overrides = pendingBuildingHeightOverrides([
      { kind: 'set_building_height', building_ids: [1, 2], height_m: 10 },
      { kind: 'paint_lc', cells: [[0, 0]], class_index: 2 },
      { kind: 'set_building_height', building_ids: [2], height_m: 12, min_height_m: 1.5 },
    ]);

    expect(overrides.get(1)).toEqual({ height_m: 10 });
    expect(overrides.get(2)).toEqual({ height_m: 12, min_height_m: 1.5 });
    expect(overrides.has(3)).toBe(false);
  });
});

describe('buildingHeightLabelValue', () => {
  it('uses committed values when no pending override exists', () => {
    expect(buildingHeightLabelValue(feature(1, 8, 2), new Map())).toEqual({
      buildingId: 1,
      height: 8,
      minHeight: 2,
    });
  });

  it('uses pending height and preserves committed min height when omitted', () => {
    const overrides = new Map([[1, { height_m: 20 }]]);

    expect(buildingHeightLabelValue(feature(1, 8, 2), overrides)).toEqual({
      buildingId: 1,
      height: 20,
      minHeight: 2,
    });
  });

  it('uses pending min height when present', () => {
    const overrides = new Map([[1, { height_m: 20, min_height_m: 4 }]]);

    expect(buildingHeightLabelValue(feature(1, 8, 2), overrides)).toEqual({
      buildingId: 1,
      height: 20,
      minHeight: 4,
    });
  });
});

describe('validateBuildingHeightInput', () => {
  it('accepts finite positive height without min height', () => {
    expect(validateBuildingHeightInput(15, false, Number.NaN)).toEqual({ ok: true, height: 15 });
  });

  it('accepts finite min height below height when enabled', () => {
    expect(validateBuildingHeightInput(15, true, 3)).toEqual({ ok: true, height: 15, minHeight: 3 });
  });

  it('rejects invalid height values', () => {
    expect(validateBuildingHeightInput(0, false, 0)).toEqual({ ok: false, error: 'Height must be greater than 0.' });
    expect(validateBuildingHeightInput(Number.NaN, false, 0)).toEqual({ ok: false, error: 'Height must be greater than 0.' });
  });

  it('rejects invalid enabled min height values', () => {
    expect(validateBuildingHeightInput(10, true, -1)).toEqual({ ok: false, error: 'Min height / base must be at least 0 and less than height.' });
    expect(validateBuildingHeightInput(10, true, 10)).toEqual({ ok: false, error: 'Min height / base must be at least 0 and less than height.' });
    expect(validateBuildingHeightInput(10, true, Number.NaN)).toEqual({ ok: false, error: 'Min height / base must be at least 0 and less than height.' });
  });
});

describe('toSetBuildingHeightDto', () => {
  it('converts building ids and height to a pending edit DTO', () => {
    expect(toSetBuildingHeightDto([1, 2], 18)).toEqual({
      kind: 'set_building_height',
      building_ids: [1, 2],
      height_m: 18,
    });
  });

  it('includes min_height_m only when provided', () => {
    expect(toSetBuildingHeightDto([1], 18, 2)).toEqual({
      kind: 'set_building_height',
      building_ids: [1],
      height_m: 18,
      min_height_m: 2,
    });
  });
});