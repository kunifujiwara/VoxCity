export type PrerequisiteTab = 'generation' | 'zoning' | 'solar' | 'view' | 'landmark' | 'export';
export type TargetAreaMethod = 'draw' | 'coordinates';
export type ExportFormat = 'cityles' | 'obj' | 'geotiff';

// Typed loosely (`key: string`) so this module stays decoupled from the app's
// `TranslationKey` union. `useT()`'s return type is a stricter function (its
// `key` param is the `TranslationKey` union, not plain `string`), so callers
// pass it in as `useT() as Translate` — safe because every literal passed to
// `t(...)` below is a valid `TranslationKey`, TypeScript just can't see that
// through this deliberately-widened signature.
export type Translate = (key: string, vars?: Record<string, string | number>) => string;

export function prerequisiteMessageForTab(
  t: Translate,
  tab: PrerequisiteTab,
): { title: string; body: string } {
  if (tab === 'generation') {
    return { title: t('guided.setAreaTitle'), body: t('guided.setAreaBody') };
  }
  return { title: t('guided.modelRequiredTitle'), body: t('guided.modelRequiredBody') };
}

export function targetAreaActionLabel(t: Translate, method: TargetAreaMethod, loading: boolean) {
  if (method === 'coordinates') return t('guided.actionSetRectangle');
  return loading ? t('guided.actionLoadingMap') : t('guided.actionLoadMap');
}

export function generationActionLabel(t: Translate, loading: boolean) {
  return loading ? t('guided.actionGenerating') : t('guided.actionGenerate');
}

export function simulationActionLabel(t: Translate, loading: boolean) {
  return loading ? t('guided.actionRunning') : t('guided.actionRunSimulation');
}

export function exportActionLabel(t: Translate, format: ExportFormat, loading: boolean) {
  if (loading) return t('guided.actionExporting');
  if (format === 'cityles') return t('guided.actionExportCityles');
  if (format === 'geotiff') return t('guided.actionExportGeotiff');
  return t('guided.actionExportObj');
}
