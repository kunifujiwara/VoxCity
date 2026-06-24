import { describe, it, expect } from 'vitest';
import { defaultPlacement, unitScale, transformModelPoint, type Placement } from './objPlacement';

describe('unitScale', () => {
  it('maps known units to metres', () => {
    expect(unitScale('m')).toBe(1);
    expect(unitScale('cm')).toBeCloseTo(0.01);
    expect(unitScale('ft')).toBeCloseTo(0.3048);
  });
});

describe('transformModelPoint', () => {
  const base: Placement = { ...defaultPlacement(), units: 'm' };

  it('places the anchor_model_point at move offset (rotation 0)', () => {
    const p = { ...base, move: [5, 7, 2] as [number, number, number] };
    const out = transformModelPoint([0, 0, 0], p); // anchor model point -> move
    expect(out[0]).toBeCloseTo(5); // east
    expect(out[1]).toBeCloseTo(7); // north
    expect(out[2]).toBeCloseTo(2); // up
  });

  it('rotates model +X toward north at rotation=90', () => {
    const p = { ...base, rotation: 90, move: [0, 0, 0] as [number, number, number] };
    const out = transformModelPoint([1, 0, 0], p); // +X, 1 m
    expect(out[0]).toBeCloseTo(0, 5);  // east ~ 0
    expect(out[1]).toBeCloseTo(1, 5);  // north ~ 1
  });

  it('applies unit scale', () => {
    const p = { ...base, units: 'ft' };
    const out = transformModelPoint([1, 0, 0], p);
    expect(out[0]).toBeCloseTo(0.3048, 4);
  });
});
