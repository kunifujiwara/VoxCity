import { describe, expect, it } from 'vitest';
import { translate } from '../i18n/translate';
import {
  exportActionLabel,
  generationActionLabel,
  prerequisiteMessageForTab,
  simulationActionLabel,
  targetAreaActionLabel,
} from './guidedTabState';

const t = (key: string, vars?: Record<string, string | number>) =>
  translate('en', key as never, vars);

describe('guided tab prerequisite messages', () => {
  it('returns the generation prerequisite message', () => {
    expect(prerequisiteMessageForTab(t, 'generation')).toEqual({
      title: 'Set a target area first',
      body: 'Use the Target Area tab to choose the city area before generating a model.',
    });
  });

  it('returns the model-required message for other tabs', () => {
    expect(prerequisiteMessageForTab(t, 'export')).toEqual({
      title: 'Generate a model first',
      body: 'Use the Generation tab to create a VoxCity model before using this workflow.',
    });
  });

  it('returns action labels', () => {
    expect(targetAreaActionLabel(t, 'coordinates', false)).toBe('Set Rectangle');
    expect(generationActionLabel(t, true)).toBe('Generating...');
    expect(simulationActionLabel(t, false)).toBe('Run Simulation');
    expect(exportActionLabel(t, 'geotiff', false)).toBe('Export GeoTIFF');
  });
});
