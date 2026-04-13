import React, { useEffect, useRef, useState, useCallback } from 'react';
import { runLandmark, getLandmarkPreview, getBuildingsList, BuildingInfo } from '../api';
import ThreeViewer, { SelectionMode, BuildingCentroid } from '../components/ThreeViewer';
import ColorSettings from '../components/ColorSettings';
import SamplingSettings from '../components/SamplingSettings';
import VoxelClassVisibility from '../components/VoxelClassVisibility';
import { useManualRerender } from '../hooks/useDebouncedRerender';

interface LandmarkTabProps {
  hasModel: boolean;
  figureJson: string;
  onFigureChange: (json: string) => void;
}

const LandmarkTab: React.FC<LandmarkTabProps> = ({ hasModel, figureJson, onFigureChange }) => {
  const [analysisTarget, setAnalysisTarget] = useState<'ground' | 'building'>('ground');
  const [landmarkIdsText, setLandmarkIdsText] = useState('');
  const [nAzimuth, setNAzimuth] = useState(60);
  const [nElevation, setNElevation] = useState(10);
  const [elevMin, setElevMin] = useState(-30);
  const [elevMax, setElevMax] = useState(30);
  const [colormap, setColormap] = useState('viridis');
  const [vmin, setVmin] = useState(0);
  const [vmax, setVmax] = useState(1);
  const [hiddenClasses, setHiddenClasses] = useState<Set<number>>(new Set());
  const [loading, setLoading] = useState(false);
  const [rerendering, setRerendering] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const hasSimResult = useRef(false);
  const simDone = figureJson !== '';

  // Selection state
  const [selectionMode, setSelectionMode] = useState<SelectionMode>('click');
  const [selectedBuildingIds, setSelectedBuildingIds] = useState<number[]>([]);
  const [previewJson, setPreviewJson] = useState('');
  const [previewLoading, setPreviewLoading] = useState(false);
  const [buildings, setBuildings] = useState<BuildingInfo[]>([]);
  const [showingSimResult, setShowingSimResult] = useState(!!figureJson);

  useEffect(() => {
    if (figureJson) {
      hasSimResult.current = true;
    }
  }, []);

  // Load preview and buildings list when tab opens with a model
  useEffect(() => {
    if (!hasModel || previewJson || simDone) return;
    let cancelled = false;
    setPreviewLoading(true);
    Promise.all([getLandmarkPreview(), getBuildingsList()])
      .then(([preview, bList]) => {
        if (cancelled) return;
        setPreviewJson(preview.figure_json);
        setBuildings(bList.buildings);
      })
      .catch((err) => {
        if (!cancelled) setError(`Failed to load preview: ${err.message}`);
      })
      .finally(() => { if (!cancelled) setPreviewLoading(false); });
    return () => { cancelled = true; };
  }, [hasModel]);

  const handleUpdate = useManualRerender(hasSimResult, { colormap, vmin, vmax, hiddenClasses }, onFigureChange, setRerendering);

  // Bidirectional sync: text → selection
  const handleIdsTextChange = useCallback((text: string) => {
    setLandmarkIdsText(text);
    const ids = text
      .split(',')
      .map((s) => parseInt(s.trim(), 10))
      .filter((n) => !isNaN(n) && n > 0);
    setSelectedBuildingIds(ids);
  }, []);

  // Bidirectional sync: selection → text
  const handleBuildingSelect = useCallback((ids: number[]) => {
    setSelectedBuildingIds(ids);
    setLandmarkIdsText(ids.join(', '));
  }, []);

  const handleClearSelection = useCallback(() => {
    setSelectedBuildingIds([]);
    setLandmarkIdsText('');
  }, []);

  const handleBackToSelection = useCallback(() => {
    setShowingSimResult(false);
  }, []);

  // Build centroids for box selection
  const buildingCentroids: BuildingCentroid[] = buildings.map((b) => ({
    id: b.id,
    cx: b.cx,
    cy: b.cy,
    cz: b.cz,
  }));

  if (!hasModel) {
    return <div className="alert alert-warning">Please generate a VoxCity model first in the "Generation" tab.</div>;
  }

  const handleRun = async () => {
    setLoading(true);
    setError(null);
    try {
      const ids = landmarkIdsText
        .split(',')
        .map((s) => parseInt(s.trim(), 10))
        .filter((n) => !isNaN(n));

      const result = await runLandmark({
        analysis_target: analysisTarget,
        landmark_ids: ids,
        view_point_height: 1.5,
        n_azimuth: nAzimuth,
        n_elevation: nElevation,
        elevation_min_degrees: elevMin,
        elevation_max_degrees: elevMax,
        colormap,
        vmin,
        vmax,
        hidden_classes: Array.from(hiddenClasses),
      });
      if (!result.figure_json || result.figure_json === '{}') {
        setError('Visualization failed – the generated figure was empty. Check the backend logs for details.');
      } else {
        onFigureChange(result.figure_json);
        hasSimResult.current = true;
        setShowingSimResult(true);
      }
    } catch (err: any) {
      setError(err.message);
    }
    setLoading(false);
  };

  // Determine which figure to show in the viewer
  const viewerFigure = showingSimResult && figureJson ? figureJson : previewJson;
  const isSelecting = !showingSimResult || !figureJson;

  return (
    <div className="two-col">
      <div className="panel">
        <h2>Landmark Visibility Analysis</h2>

        <div className="form-group">
          <label>Analysis Target</label>
          <div className="radio-group">
            <label>
              <input type="radio" checked={analysisTarget === 'ground'} onChange={() => setAnalysisTarget('ground')} />
              Ground Level
            </label>
            <label>
              <input type="radio" checked={analysisTarget === 'building'} onChange={() => setAnalysisTarget('building')} />
              Building Surfaces
            </label>
          </div>
        </div>

        {/* Selection toolbar */}
        <div className="form-group">
          <label>Select Landmark Buildings</label>
          {isSelecting && (
            <div className="selection-toolbar">
              <button
                className={`selection-toolbar-btn ${selectionMode === 'click' ? 'active' : ''}`}
                onClick={() => setSelectionMode('click')}
                title="Click buildings to select/deselect"
              >
                Click
              </button>
              <button
                className={`selection-toolbar-btn ${selectionMode === 'box' ? 'active' : ''}`}
                onClick={() => setSelectionMode('box')}
                title="Drag a box to select multiple buildings"
              >
                Box
              </button>
              <button
                className={`selection-toolbar-btn ${selectionMode === 'none' ? 'active' : ''}`}
                onClick={() => setSelectionMode('none')}
                title="Orbit mode (rotate/pan/zoom only)"
              >
                Orbit
              </button>
              <button
                className="selection-toolbar-btn clear-btn"
                onClick={handleClearSelection}
                title="Clear all selections"
                disabled={selectedBuildingIds.length === 0}
              >
                Clear
              </button>
            </div>
          )}
        </div>

        {/* Selected buildings chips */}
        {selectedBuildingIds.length > 0 && (
          <div className="form-group">
            <label>Selected Buildings ({selectedBuildingIds.length})</label>
            <div className="building-chips">
              {selectedBuildingIds.map((id) => (
                <span key={id} className="building-chip">
                  {id}
                  <button
                    className="chip-remove"
                    onClick={() => handleBuildingSelect(selectedBuildingIds.filter((i) => i !== id))}
                  >
                    ×
                  </button>
                </span>
              ))}
            </div>
          </div>
        )}

        <div className="form-group">
          <label>Landmark Building IDs (comma-separated, empty = center building)</label>
          <input
            type="text"
            value={landmarkIdsText}
            onChange={(e) => handleIdsTextChange(e.target.value)}
            placeholder="e.g. 12, 34, 56"
          />
        </div>

        <SamplingSettings
          nAzimuth={nAzimuth}
          onNAzimuthChange={setNAzimuth}
          nElevation={nElevation}
          onNElevationChange={setNElevation}
          elevMin={elevMin}
          onElevMinChange={setElevMin}
          elevMax={elevMax}
          onElevMaxChange={setElevMax}
          showElevationRange={analysisTarget === 'ground'}
        />

        <ColorSettings
          colormap={colormap}
          onColormapChange={setColormap}
          vmin={vmin}
          onVminChange={setVmin}
          vmax={vmax}
          onVmaxChange={(v) => setVmax(Number(v))}
        />

        <VoxelClassVisibility
          hiddenClasses={hiddenClasses}
          onHiddenClassesChange={setHiddenClasses}
        />

        {showingSimResult && figureJson && (
          <button className="btn btn-secondary" onClick={handleBackToSelection} disabled={loading} style={{ marginBottom: '0.5rem' }}>
            Back to Selection
          </button>
        )}

        {simDone && showingSimResult && (
          <button className="btn btn-secondary" onClick={handleUpdate} disabled={loading || rerendering} style={{ marginBottom: '0.5rem' }}>
            {rerendering && <span className="spinner" />}
            {rerendering ? 'Updating...' : 'Update View'}
          </button>
        )}

        <button className="btn btn-primary" onClick={handleRun} disabled={loading}>
          {loading && <span className="spinner" />}
          {loading ? 'Running...' : 'Run Simulation'}
        </button>

        {error && <div className="alert alert-error" style={{ marginTop: '0.75rem' }}>{error}</div>}
      </div>

      <div className="panel">
        {previewLoading && <div className="alert alert-info">Loading 3D preview...</div>}
        <ThreeViewer
          figureJson={viewerFigure}
          selectionMode={isSelecting ? selectionMode : 'none'}
          selectedBuildingIds={selectedBuildingIds}
          onBuildingSelect={handleBuildingSelect}
          buildingCentroids={buildingCentroids}
        />
      </div>
    </div>
  );
};

export default LandmarkTab;
