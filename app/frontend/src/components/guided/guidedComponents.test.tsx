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

  it('renders check-style choices as radio inputs', () => {
    const onChange = vi.fn();
    const html = renderToStaticMarkup(
      <ChoiceGroup
        variant="checks"
        ariaLabel="Generation mode"
        value="normal"
        onChange={onChange}
        options={[
          { id: 'normal', label: 'Normal', description: 'Global open data sources' },
          { id: 'plateau', label: 'PLATEAU', description: 'Japanese CityGML data', disabled: true },
        ]}
      />,
    );

    expect(html).toContain('choice-group-checks');
    expect(html).toContain('choice-check-row active');
    expect(html).toContain('type="radio"');
    expect(html).toContain('checked=""');
    expect(html).not.toContain('Global open data sources');
    expect(html).toContain('disabled=""');
  });

  it('applies column classes to check-style choices', () => {
    const html = renderToStaticMarkup(
      <ChoiceGroup
        variant="checks"
        columns={2}
        ariaLabel="Analysis target"
        value="ground"
        onChange={() => {}}
        options={[
          { id: 'ground', label: 'Ground level' },
          { id: 'building', label: 'Building surfaces' },
        ]}
      />,
    );

    expect(html).toContain('choice-group-checks-2');
  });

  it('renders a bare panel without heading or bottom section', () => {
    const html = renderToStaticMarkup(<GuidedPanel>content</GuidedPanel>);
    expect(html).not.toContain('guided-panel-bottom');
    expect(html).not.toContain('guided-panel-heading');
  });

  it('renders a numbered index when GuidedSection.index is set', () => {
    const html = renderToStaticMarkup(
      <GuidedSection index={2} label="Solar">body</GuidedSection>,
    );
    expect(html).toContain('guided-section-index');
    expect(html).toContain('>2<');
  });

  it('applies tone-danger class when GuidedSection.tone="danger"', () => {
    const html = renderToStaticMarkup(
      <GuidedSection label="Reset" tone="danger">body</GuidedSection>,
    );
    expect(html).toContain('tone-danger');
  });

  it('renders a collapsible affordance with chevron when GuidedSection.collapsible', () => {
    const html = renderToStaticMarkup(
      <GuidedSection label="Solar" collapsible>body</GuidedSection>,
    );
    expect(html).toContain('guided-section-header-collapsible');
    expect(html).toContain('guided-section-collapse-chevron');
  });

  it('keeps descriptions visible for card-style choices', () => {
    const html = renderToStaticMarkup(
      <ChoiceGroup
        ariaLabel="Export format"
        value="cityles"
        onChange={() => {}}
        options={[{ id: 'cityles', label: 'CityLES', description: 'Simulation archive' }]}
      />,
    );
    expect(html).toContain('Simulation archive');
  });

  it('renders an icon glyph before the label when ChoiceOption.icon is set', () => {
    const Stub: any = (props: any) => <svg {...props} />;
    const html = renderToStaticMarkup(
      <ChoiceGroup
        variant="checks"
        ariaLabel="Action"
        value="add"
        onChange={() => {}}
        options={[{ id: 'add', label: 'Add', icon: Stub }]}
      />,
    );
    expect(html).toContain('<svg');
    expect(html).toContain('choice-check-icon');
    expect(html).toContain('aria-hidden="true"');
    expect(html).toMatch(/<svg[^>]*class="choice-check-icon"[^>]*aria-hidden="true"/);
  });
});
