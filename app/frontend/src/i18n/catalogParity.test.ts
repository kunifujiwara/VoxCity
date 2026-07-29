import { describe, expect, it } from 'vitest';
import { en } from './locales/en';
import { ja } from './locales/ja';

function flatKeys(obj: Record<string, unknown>, prefix = ''): string[] {
  return Object.entries(obj).flatMap(([k, v]) =>
    v && typeof v === 'object'
      ? flatKeys(v as Record<string, unknown>, `${prefix}${k}.`)
      : [`${prefix}${k}`],
  );
}

function values(obj: Record<string, unknown>): string[] {
  return Object.values(obj).flatMap((v) =>
    v && typeof v === 'object' ? values(v as Record<string, unknown>) : [String(v)],
  );
}

describe('catalog parity', () => {
  it('ja has exactly the same keys as en', () => {
    expect(flatKeys(ja).sort()).toEqual(flatKeys(en).sort());
  });

  it('has no empty values in either catalog', () => {
    for (const v of [...values(en), ...values(ja)]) {
      expect(v.trim().length).toBeGreaterThan(0);
    }
  });
});
