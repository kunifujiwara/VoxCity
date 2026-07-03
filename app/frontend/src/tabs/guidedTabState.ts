export type PrerequisiteTab = 'generation' | 'zoning' | 'solar' | 'view' | 'landmark' | 'export';
export type TargetAreaMethod = 'draw' | 'coordinates';
export type ExportFormat = 'cityles' | 'obj' | 'geotiff';

const MODEL_REQUIRED_BODY = 'Use the Generation tab to create a VoxCity model before using this workflow.';

export function prerequisiteMessageForTab(tab: PrerequisiteTab): { title: string; body: string } {
  if (tab === 'generation') {
    return {
      title: 'Set a target area first',
      body: 'Use the Target Area tab to choose the city area before generating a model.',
    };
  }
  return {
    title: 'Generate a model first',
    body: MODEL_REQUIRED_BODY,
  };
}

export function targetAreaActionLabel(method: TargetAreaMethod, loading: boolean) {
  if (method === 'coordinates') return 'Set Rectangle';
  return loading ? 'Loading map...' : 'Load Map';
}

export function generationActionLabel(loading: boolean) {
  return loading ? 'Generating...' : 'Generate VoxCity Model';
}

export function simulationActionLabel(loading: boolean) {
  return loading ? 'Running...' : 'Run Simulation';
}

export function exportActionLabel(format: ExportFormat, loading: boolean) {
  if (loading) return 'Exporting...';
  if (format === 'cityles') return 'Export CityLES';
  if (format === 'geotiff') return 'Export GeoTIFF';
  return 'Export OBJ';
}
