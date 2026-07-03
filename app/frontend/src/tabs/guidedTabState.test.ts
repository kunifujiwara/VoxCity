import { describe, expect, it } from 'vitest';
import {
  exportActionLabel,
  generationActionLabel,
  prerequisiteMessageForTab,
  simulationActionLabel,
  targetAreaActionLabel,
} from './guidedTabState';

describe('guided tab prerequisite messages', () => {
  it('returns the generation prerequisite message', () => {
    expect(prerequisiteMessageForTab('generation')).toEqual({
      title: 'Set a target area first',
      body: 'Use the Target Area tab to choose the city area before generating a model.',
    });
  });

  it.each(['zoning', 'solar', 'view', 'landmark', 'export'] as const)(
    'returns the model prerequisite message for %s',
    (tab) => {
      expect(prerequisiteMessageForTab(tab).title).toBe('Generate a model first');
      expect(prerequisiteMessageForTab(tab).body).toContain('Generation tab');
    },
  );
});

describe('guided tab action labels', () => {
  it('labels target-area actions by input mode and loading state', () => {
    expect(targetAreaActionLabel('draw', false)).toBe('Load Map');
    expect(targetAreaActionLabel('draw', true)).toBe('Loading map...');
    expect(targetAreaActionLabel('coordinates', false)).toBe('Set Rectangle');
    expect(targetAreaActionLabel('coordinates', true)).toBe('Set Rectangle');
  });

  it('labels generation and simulation actions by loading state', () => {
    expect(generationActionLabel(false)).toBe('Generate VoxCity Model');
    expect(generationActionLabel(true)).toBe('Generating...');
    expect(simulationActionLabel(false)).toBe('Run Simulation');
    expect(simulationActionLabel(true)).toBe('Running...');
  });

  it('labels export actions by selected format and loading state', () => {
    expect(exportActionLabel('cityles', false)).toBe('Export CityLES');
    expect(exportActionLabel('cityles', true)).toBe('Exporting...');
    expect(exportActionLabel('obj', false)).toBe('Export OBJ');
    expect(exportActionLabel('obj', true)).toBe('Exporting...');
    expect(exportActionLabel('geotiff', false)).toBe('Export GeoTIFF');
    expect(exportActionLabel('geotiff', true)).toBe('Exporting...');
  });
});
