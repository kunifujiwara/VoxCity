import React, { useState } from 'react';
import { runLandmark } from '../api';
import PlotlyViewer from '../components/PlotlyViewer';

interface LandmarkTabProps {
  hasModel: boolean;
}

const COLORMAPS = [
  'viridis', 'plasma', 'magma', 'inferno', 'cividis', 'turbo',
  'Greens', 'Blues', 'coolwarm', 'RdYlBu_r', 'gray',
];

const LandmarkTab: React.FC<LandmarkTabProps> = ({ hasModel }) => {
  const [analysisTarget, setAnalysisTarget] = useState<'ground' | 'building'>('ground');
  const [landmarkIdsText, setLandmarkIdsText] = useState('');
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

        <div className="form-group">
          <label>Landmark Building IDs (comma-separated, empty = center building)</label>
          <input
            type="text"
            value={landmarkIdsText}
            onChange={(e) => setLandmarkIdsText(e.target.value)}
            placeholder="e.g. 12, 34, 56"
          />
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

export default LandmarkTab;
