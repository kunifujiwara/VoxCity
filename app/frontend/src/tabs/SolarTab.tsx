import React, { useState } from 'react';
import { runSolar } from '../api';
import ThreeViewer from '../components/ThreeViewer';

interface SolarTabProps {
  hasModel: boolean;
}

const COLORMAPS = [
  'magma', 'viridis', 'plasma', 'inferno', 'cividis', 'turbo',
  'Greens', 'Blues', 'coolwarm', 'RdYlBu_r', 'Spectral', 'gray',
];

const SolarTab: React.FC<SolarTabProps> = ({ hasModel }) => {
  const [calcType, setCalcType] = useState<'instantaneous' | 'cumulative'>('instantaneous');
  const [analysisTarget, setAnalysisTarget] = useState<'ground' | 'building'>('ground');
  const [calcDate, setCalcDate] = useState('01-01');
  const [calcTime, setCalcTime] = useState('12:00:00');
  const [startTime, setStartTime] = useState('01-01 01:00:00');
  const [endTime, setEndTime] = useState('01-31 23:00:00');
  const [colormap, setColormap] = useState('magma');
  const [vmin, setVmin] = useState<number>(0);
  const [vmax, setVmax] = useState<string>('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [figureJson, setFigureJson] = useState('');
  const [showColorSettings, setShowColorSettings] = useState(false);

  if (!hasModel) {
    return <div className="alert alert-warning">Please generate a VoxCity model first in the "Generation" tab.</div>;
  }

  const handleRun = async () => {
    setLoading(true);
    setError(null);
    try {
      const params: any = {
        calc_type: calcType,
        analysis_target: analysisTarget,
        epw_source: 'default',
        colormap,
        vmin,
        vmax: vmax ? parseFloat(vmax) : null,
        hidden_classes: [],
      };
      if (calcType === 'instantaneous') {
        params.calc_time = `${calcDate} ${calcTime}`;
      } else {
        params.start_time = startTime;
        params.end_time = endTime;
      }
      const result = await runSolar(params);
      setFigureJson(result.figure_json);
    } catch (err: any) {
      setError(err.message);
    }
    setLoading(false);
  };

  return (
    <div className="two-col">
      <div className="panel">
        <h2>Solar Radiation Analysis</h2>

        <div className="form-group">
          <label>Calculation Type</label>
          <div className="radio-group">
            <label>
              <input type="radio" checked={calcType === 'instantaneous'} onChange={() => setCalcType('instantaneous')} />
              Instantaneous
            </label>
            <label>
              <input type="radio" checked={calcType === 'cumulative'} onChange={() => setCalcType('cumulative')} />
              Cumulative
            </label>
          </div>
        </div>

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

        {calcType === 'instantaneous' ? (
          <div className="form-row">
            <div>
              <label>Date (MM-DD)</label>
              <input type="text" value={calcDate} onChange={(e) => setCalcDate(e.target.value)} />
            </div>
            <div>
              <label>Time (HH:MM:SS)</label>
              <input type="text" value={calcTime} onChange={(e) => setCalcTime(e.target.value)} />
            </div>
          </div>
        ) : (
          <>
            <div className="form-group">
              <label>Start (MM-DD HH:MM:SS)</label>
              <input type="text" value={startTime} onChange={(e) => setStartTime(e.target.value)} />
            </div>
            <div className="form-group">
              <label>End (MM-DD HH:MM:SS)</label>
              <input type="text" value={endTime} onChange={(e) => setEndTime(e.target.value)} />
            </div>
          </>
        )}

        <div className="expander">
          <div className="expander-header" onClick={() => setShowColorSettings(!showColorSettings)}>
            Color Settings <span>{showColorSettings ? '▲' : '▼'}</span>
          </div>
          {showColorSettings && (
            <div className="expander-body">
              <div className="form-group">
                <label>Colormap</label>
                <select value={colormap} onChange={(e) => setColormap(e.target.value)}>
                  {COLORMAPS.map((cm) => (
                    <option key={cm} value={cm}>{cm}</option>
                  ))}
                </select>
              </div>
              <div className="form-row">
                <div>
                  <label>vmin</label>
                  <input type="number" value={vmin} onChange={(e) => setVmin(Number(e.target.value))} />
                </div>
                <div>
                  <label>vmax (empty = auto)</label>
                  <input type="text" value={vmax} onChange={(e) => setVmax(e.target.value)} />
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
        <ThreeViewer figureJson={figureJson} />
      </div>
    </div>
  );
};

export default SolarTab;
