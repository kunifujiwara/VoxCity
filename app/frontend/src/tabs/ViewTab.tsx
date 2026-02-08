import React, { useState } from 'react';
import { runView } from '../api';
import PlotlyViewer from '../components/PlotlyViewer';

interface ViewTabProps {
  hasModel: boolean;
}

const COLORMAPS = [
  'viridis', 'plasma', 'magma', 'inferno', 'cividis', 'turbo',
  'Greens', 'Blues', 'BuPu_r', 'coolwarm', 'RdYlBu_r', 'gray',
];

const CUSTOM_CLASSES = [
  { id: -3, label: 'Building' },
  { id: -2, label: 'Tree' },
  { id: 1, label: 'Bareland' },
  { id: 2, label: 'Rangeland' },
  { id: 3, label: 'Shrub' },
  { id: 4, label: 'Agriculture land' },
  { id: 6, label: 'Moss and lichen' },
  { id: 7, label: 'Wet land' },
  { id: 8, label: 'Mangrove' },
  { id: 9, label: 'Water' },
  { id: 10, label: 'Snow and ice' },
  { id: 11, label: 'Developed space' },
  { id: 12, label: 'Road' },
];

const ViewTab: React.FC<ViewTabProps> = ({ hasModel }) => {
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
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [figureJson, setFigureJson] = useState('');
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [showColor, setShowColor] = useState(false);

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
        hidden_classes: [],
      });
      setFigureJson(result.figure_json);
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

        <div className="expander">
          <div className="expander-header" onClick={() => setShowAdvanced(!showAdvanced)}>
            Sampling Settings <span>{showAdvanced ? '▲' : '▼'}</span>
          </div>
          {showAdvanced && (
            <div className="expander-body">
              <div className="form-row">
                <div>
                  <label>N_azimuth</label>
                  <input type="number" value={nAzimuth} min={1} max={360} onChange={(e) => setNAzimuth(Number(e.target.value))} />
                </div>
                <div>
                  <label>N_elevation</label>
                  <input type="number" value={nElevation} min={1} max={180} onChange={(e) => setNElevation(Number(e.target.value))} />
                </div>
              </div>
              {analysisTarget === 'ground' && (
                <div className="form-row">
                  <div>
                    <label>Elev min (°)</label>
                    <input type="number" value={elevMin} onChange={(e) => setElevMin(Number(e.target.value))} />
                  </div>
                  <div>
                    <label>Elev max (°)</label>
                    <input type="number" value={elevMax} onChange={(e) => setElevMax(Number(e.target.value))} />
                  </div>
                </div>
              )}
            </div>
          )}
        </div>

        <div className="expander">
          <div className="expander-header" onClick={() => setShowColor(!showColor)}>
            Color Settings <span>{showColor ? '▲' : '▼'}</span>
          </div>
          {showColor && (
            <div className="expander-body">
              <div className="form-group">
                <label>Colormap</label>
                <select value={colormap} onChange={(e) => setColormap(e.target.value)}>
                  {COLORMAPS.map((cm) => <option key={cm} value={cm}>{cm}</option>)}
                </select>
              </div>
              <div className="form-row">
                <div>
                  <label>vmin</label>
                  <input type="number" value={vmin} step={0.1} onChange={(e) => setVmin(Number(e.target.value))} />
                </div>
                <div>
                  <label>vmax</label>
                  <input type="number" value={vmax} step={0.1} onChange={(e) => setVmax(Number(e.target.value))} />
                </div>
              </div>
            </div>
          )}
        </div>

        <button className="btn btn-primary" onClick={handleRun} disabled={loading}>
          {loading && <span className="spinner" />}
          {loading ? 'Running...' : 'Run Simulation'}
        </button>

        {error && <div className="alert alert-error" style={{ marginTop: '0.75rem' }}>{error}</div>}
      </div>

      <div className="panel">
        <PlotlyViewer figureJson={figureJson} />
      </div>
    </div>
  );
};

export default ViewTab;
