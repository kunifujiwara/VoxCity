import React, { useState } from 'react';
import { runLandmark } from '../api';
import ThreeViewer from '../components/ThreeViewer';
import ColorSettings from '../components/ColorSettings';
import SamplingSettings from '../components/SamplingSettings';
import VoxelClassVisibility from '../components/VoxelClassVisibility';

interface LandmarkTabProps {
  hasModel: boolean;
}

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
  const [hiddenClasses, setHiddenClasses] = useState<Set<number>>(new Set());
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [figureJson, setFigureJson] = useState('');

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

export default LandmarkTab;
