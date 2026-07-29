// English catalog — the single source of truth. `ja.ts` is typed against this.
export const en = {
  common: {
    close: 'Close',
    newSession: 'New session',
    openSession: 'Open session...',
    getStarted: 'GET STARTED',
    dontShowAgain: "Don't show this again",
  },
  language: {
    label: 'Language',
    english: 'EN',
    japanese: '日本語',
  },
  nav: {
    area: 'Target',
    generation: 'Generate',
    edit: 'Edit',
    import: 'Import',
    zoning: 'Zone',
    solar: 'Solar',
    view: 'View',
    landmark: 'Landmark',
    export: 'File',
  },
  splash: {
    title: 'Welcome to VoxCity',
    subtitle: 'Start a new urban model, or open a saved session.',
    loadFailed: 'Failed to load session.',
  },
  guided: {
    setAreaTitle: 'Set a target area first',
    setAreaBody: 'Use the Target Area tab to choose the city area before generating a model.',
    modelRequiredTitle: 'Generate a model first',
    modelRequiredBody: 'Use the Generation tab to create a VoxCity model before using this workflow.',
    actionSetRectangle: 'Set Rectangle',
    actionLoadingMap: 'Loading map...',
    actionLoadMap: 'Load Map',
    actionGenerating: 'Generating...',
    actionGenerate: 'Generate VoxCity Model',
    actionRunning: 'Running...',
    actionRunSimulation: 'Run Simulation',
    actionExporting: 'Exporting...',
    actionExportCityles: 'Export CityLES',
    actionExportGeotiff: 'Export GeoTIFF',
    actionExportObj: 'Export OBJ',
  },
  previewNotice: {
    heading: '3D preview disabled',
    bodyWithDims: 'The grid ({dims}) exceeds the preview limit of {cells} cells. Generation, editing, simulation results, and export still work.',
    bodyNoDims: 'This grid exceeds the preview limit of {cells} cells. Generation, editing, simulation results, and export still work.',
  },
  voxelVisibility: {
    heading: 'Visualization Settings',
    hideClasses: 'Hide element classes',
  },
};

export type Messages = typeof en;
