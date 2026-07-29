import React from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { afterEach, describe, expect, it } from 'vitest';
import { LanguageProvider } from './LanguageContext';
import { useT } from './useT';

function Probe() {
  const t = useT();
  return <span>{t('nav.area')}</span>;
}

afterEach(() => {
  delete (globalThis as unknown as { localStorage?: unknown }).localStorage;
});

describe('LanguageProvider + useT (SSR)', () => {
  it('renders English by default without a provider', () => {
    expect(renderToStaticMarkup(<Probe />)).toContain('Target');
  });

  it('renders English inside the provider by default', () => {
    expect(renderToStaticMarkup(<LanguageProvider><Probe /></LanguageProvider>)).toContain('Target');
  });

  it('initializes from localStorage when set to ja', () => {
    (globalThis as unknown as { localStorage: Storage }).localStorage = {
      getItem: (k: string) => (k === 'voxcity.lang' ? 'ja' : null),
      setItem: () => {},
    } as unknown as Storage;
    expect(renderToStaticMarkup(<LanguageProvider><Probe /></LanguageProvider>)).toContain('対象エリア');
  });
});
