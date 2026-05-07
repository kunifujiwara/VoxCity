import React from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { describe, expect, it, vi } from 'vitest';
import {
  ChoiceGroup,
  GuidedFooter,
  GuidedPanel,
  GuidedSection,
  GuidedStatus,
} from './index';

describe('guided components', () => {
  it('renders a panel with title, subtitle, body, status, and footer', () => {
    const html = renderToStaticMarkup(
      <GuidedPanel
        title="Generation"
        subtitle="Build the model"
        status={<GuidedStatus tone="success">Ready</GuidedStatus>}
        footer={<GuidedFooter><button type="button">Generate</button></GuidedFooter>}
      >
        <GuidedSection label="Mode">Normal</GuidedSection>
      </GuidedPanel>,
    );

    expect(html).toContain('guided-panel');
    expect(html).toContain('Generation');
    expect(html).toContain('Build the model');
    expect(html).toContain('Mode');
    expect(html).toContain('Ready');
    expect(html).toContain('Generate');
  });

  it('marks the active choice and preserves disabled choices', () => {
    const onChange = vi.fn();
    const html = renderToStaticMarkup(
      <ChoiceGroup
        ariaLabel="Export format"
        value="cityles"
        onChange={onChange}
        options={[
          { id: 'cityles', label: 'CityLES', description: 'Simulation archive' },
          { id: 'obj', label: 'OBJ', description: 'Mesh export', disabled: true },
        ]}
      />,
    );

    expect(html).toContain('choice-group');
    expect(html).toContain('choice-btn active');
    expect(html).toContain('CityLES');
    expect(html).toContain('Simulation archive');
    expect(html).toContain('disabled=""');
  });
});
