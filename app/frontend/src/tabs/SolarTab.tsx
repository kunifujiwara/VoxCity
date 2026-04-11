import React, { useRef, useState } from 'react';
import { runSolar } from '../api';
import ThreeViewer from '../components/ThreeViewer';
import ColorSettings from '../components/ColorSettings';
import VoxelClassVisibility from '../components/VoxelClassVisibility';
import { useManualRerender } from '../hooks/useDebouncedRerender';

interface SolarTabProps {
  hasModel: boolean;
}

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
  const [hiddenClasses, setHiddenClasses] = useState<Set<number>>(new Set());
  const [loading, setLoading] = useState(false);
  const [rerendering, setRerendering] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [figureJson, setFigureJson] = useState('');
  const [simDone, setSimDone] = useState(false);
  const hasSimResult = useRef(false);

  const handleUpdate = useManualRerender(hasSimResult, { colormap, vmin, vmax, hiddenClasses }, setFigureJson, setRerendering);

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
        hidden_classes: Array.from(hiddenClasses),
      };
      if (calcType === 'instantaneous') {
        params.calc_time = `${calcDate} ${calcTime}`;
      } else {
        params.start_time = startTime;
        params.end_time = endTime;
      }
      const result = await runSolar(params);
      if (!result.figure_json || result.figure_json === '{}') {
        setError('Visualization failed – the generated figure was empty. Check the backend logs for details.');
      } else {
        setFigureJson(result.figure_json);
        hasSimResult.current = true;
        setSimDone(true);
      }
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

        <ColorSettings
          colormap={colormap}
          onColormapChange={setColormap}
          vmin={vmin}
          onVminChange={setVmin}
          vmax={vmax}
          onVmaxChange={(v) => setVmax(String(v))}
          vmaxAsText
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

export default SolarTab;
