import React from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { describe, expect, it } from 'vitest';
import PreviewDisabledNotice from './PreviewDisabledNotice';

describe('PreviewDisabledNotice', () => {
  it('shows the grid dimensions when provided', () => {
    const html = renderToStaticMarkup(<PreviewDisabledNotice gridShape={[1500, 900, 30]} />);
    expect(html).toContain('1500');
    expect(html).toContain('900');
    expect(html).toContain('preview disabled');
  });

  it('renders a generic message without a grid shape', () => {
    const html = renderToStaticMarkup(<PreviewDisabledNotice />);
    expect(html.toLowerCase()).toContain('preview disabled');
  });
});
