import React, { useEffect, useRef, useState } from 'react';
import { runView } from '../api';
import ThreeViewer from '../components/ThreeViewer';
import ColorSettings from '../components/ColorSettings';
import SamplingSettings from '../components/SamplingSettings';
import VoxelClassVisibility from '../components/VoxelClassVisibility';
import { CUSTOM_CLASSES } from '../constants';
import { useManualRerender } from '../hooks/useDebouncedRerender';

interface ViewTabProps {
  hasModel: boolean;
  figureJson: string;
  onFigureChange: (json: string) => void;
}

const ViewTab: React.FC<ViewTabProps> = ({ hasModel, figureJson, onFigureChange }) => {
  const [viewType, setViewType] = useState('green');
  const [analysisTarget, setAnalysisTarget] = useState<'ground' | 'building'>('ground');
  const [viewPointHeight, setViewPointHeight] = useState(1.5);
  const [customClasses, setCustomClasses] = useState<Set<number>>(new Set());
  const [inclusionMode, setInclusionMode] = useState(true);
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

  useEffect(() => {
    if (figureJson) hasSimResult.current = true;
  }, []);

  const handleUpdate = useManualRerender(hasSimResult, { colormap, vmin, vmax, hiddenClasses }, onFigureChange, setRerendering);

  if (!hasModel) {
    return <div className="alert alert-warning">Please generate a VoxCity model first in the "Generation" tab.</div>;
  }

  const toggleClass = (id: number) => {
    const next = new Set(customClasses);
    next.has(id) ? next.delete(id) : next.add(id);
    setCustomClasses(next);
  };

  const handleRun = async () => {
    setLoading(true);
    setError(null);
    try {
      const result = await runView({
        view_type: viewType,
        analysis_target: analysisTarget,
        view_point_height: viewPointHeight,
        custom_classes: Array.from(customClasses),
        inclusion_mode: inclusionMode,
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
      }
    } catch (err: any) {
      setError(err.message);
    }
    setLoading(false);
  };

  return (
    <div className="two-col">
      <div className="panel">
        <h2>View Index Analysis</h2>

        <div className="form-group">
          <label>View Type</label>
          <select value={viewType} onChange={(e) => setViewType(e.target.value)}>
            <option value="green">Green View Index</option>
            <option value="sky">Sky View Index</option>
            <option value="custom">Custom (Select Classes)</option>
          </select>
        </div>

        {viewType === 'custom' && (
          <div className="form-group" style={{ maxHeight: 200, overflowY: 'auto', padding: '0.5rem', border: '1px solid var(--vc-ring)', borderRadius: 6 }}>
            <div className="radio-group" style={{ marginBottom: '0.5rem' }}>
              <label>
                <input type="radio" checked={inclusionMode} onChange={() => setInclusionMode(true)} />
                Inclusion
              </label>
              <label>
                <input type="radio" checked={!inclusionMode} onChange={() => setInclusionMode(false)} />
                Exclusion
              </label>
            </div>
            {CUSTOM_CLASSES.map((cls) => (
              <div className="checkbox-row" key={cls.id}>
                <input
                  type="checkbox"
                  checked={customClasses.has(cls.id)}
                  onChange={() => toggleClass(cls.id)}
                />
                <span>{cls.label}</span>
              </div>
            ))}
          </div>
        )}

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

        <div className="form-group">
          <label>View Point Height (m)</label>
          <input type="number" value={viewPointHeight} min={0} max={10} step={0.5} onChange={(e) => setViewPointHeight(Number(e.target.value))} />
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

        {simDone && (
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
        <ThreeViewer figureJson={figureJson} />
      </div>
    </div>
  );
};

export default ViewTab;
