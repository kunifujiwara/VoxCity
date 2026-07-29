import { describe, it, expect } from 'vitest';
import { isAuxLayerVisible, groupAuxLineLayers } from './auxiliaryLines';
import type { AuxiliaryLineDto } from '../api';

describe('isAuxLayerVisible', () => {
  it('defaults to visible when unset', () => {
    expect(isAuxLayerVisible({}, 'a.dxf', 'x')).toBe(true);
    expect(isAuxLayerVisible({ 'a.dxf': {} }, 'a.dxf', 'x')).toBe(true);
  });

  it('is hidden only when explicitly false', () => {
    expect(isAuxLayerVisible({ 'a.dxf': { x: false } }, 'a.dxf', 'x')).toBe(false);
    expect(isAuxLayerVisible({ 'a.dxf': { x: true } }, 'a.dxf', 'x')).toBe(true);
  });
});

describe('groupAuxLineLayers', () => {
  it('groups by file then unique layer with first-seen color, in order', () => {
    const lines: AuxiliaryLineDto[] = [
      { id: '1', file_name: 'a.dxf', layer: 'w', color: '#111', points: [] },
      { id: '2', file_name: 'a.dxf', layer: 'w', color: '#999', points: [] },
      { id: '3', file_name: 'a.dxf', layer: 'r', color: '#222', points: [] },
      { id: '4', file_name: 'b.dxf', layer: 'w', color: '#333', points: [] },
    ];
    expect(groupAuxLineLayers(lines)).toEqual([
      { fileName: 'a.dxf', layers: [{ layer: 'w', color: '#111' }, { layer: 'r', color: '#222' }] },
      { fileName: 'b.dxf', layers: [{ layer: 'w', color: '#333' }] },
    ]);
  });

  it('returns an empty array for no lines', () => {
    expect(groupAuxLineLayers([])).toEqual([]);
  });
});
