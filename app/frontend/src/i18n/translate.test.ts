import { describe, expect, it } from 'vitest';
import { translate } from './translate';

describe('translate', () => {
  it('resolves a nested key in the active language', () => {
    expect(translate('en', 'nav.export')).toBe('File');
    expect(translate('ja', 'nav.export')).toBe('ファイル');
  });

  it('interpolates named placeholders', () => {
    expect(
      translate('en', 'previewNotice.bodyWithDims', { dims: '1500×900', cells: '1,000,000' }),
    ).toBe(
      'The grid (1500×900) exceeds the preview limit of 1,000,000 cells. Generation, editing, simulation results, and export still work.',
    );
  });

  it('returns the Japanese and English values for a shared key', () => {
    expect(translate('en', 'common.newSession')).toBe('New session');
    expect(translate('ja', 'common.newSession')).toBe('新規セッション');
  });
});
