import React, { useState } from 'react';
import { generateModel } from '../api';
import ThreeViewer from '../components/ThreeViewer';

interface GenerationTabProps {
  rectangle: number[][] | null;
  figureJson: string;
  onFigureChange: (json: string) => void;
  onModelReady: () => void;
}

const GenerationTab: React.FC<GenerationTabProps> = ({
  rectangle,
  figureJson,
  onFigureChange,
  onModelReady,
}) => {
  const [meshsize, setMeshsize] = useState(5);
  const [buildingComplementHeight, setBuildingComplementHeight] = useState(10);
  const [staticTreeHeight, setStaticTreeHeight] = useState(10);
  const [demInterpolation, setDemInterpolation] = useState(true);
  const [useCitygmlCache, setUseCitygmlCache] = useState(true);
  const [useNdsmCanopy, setUseNdsmCanopy] = useState(true);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [gridShape, setGridShape] = useState<number[] | null>(null);
  const [showAdvanced, setShowAdvanced] = useState(false);

  const handleGenerate = async () => {
    if (!rectangle) return;
    setLoading(true);
    setError(null);
    try {
      const result = await generateModel({
        rectangle_vertices: rectangle,
        meshsize,
        building_complement_height: buildingComplementHeight,
        static_tree_height: staticTreeHeight,
        dem_interpolation: demInterpolation,
        use_citygml_cache: useCitygmlCache,
        use_ndsm_canopy: useNdsmCanopy,
      });
      setGridShape(result.grid_shape);
      onFigureChange(result.figure_json);
      onModelReady();
    } catch (err: any) {
      setError(err.message);
    }
    setLoading(false);
  };

  if (!rectangle) {
    return <div className="alert alert-warning">Set the target area first in the "Target Area" tab.</div>;
  }

  return (
    <div className="two-col">
      {/* Left – controls */}
      <div className="panel">
        <h2>Parameters</h2>

        <div className="form-group">
          <label>Mesh Size (meters)</label>
          <input
            type="number"
            value={meshsize}
            min={1}
            max={50}
            onChange={(e) => setMeshsize(Number(e.target.value))}
          />
        </div>

        <div className="expander">
          <div className="expander-header" onClick={() => setShowAdvanced(!showAdvanced)}>
            Advanced Parameters
            <span>{showAdvanced ? '▲' : '▼'}</span>
          </div>
          {showAdvanced && (
            <div className="expander-body">
              <div className="form-group">
                <label>Building Complement Height (m)</label>
                <input
                  type="number"
                  value={buildingComplementHeight}
                  onChange={(e) => setBuildingComplementHeight(Number(e.target.value))}
                />
              </div>
              <div className="form-group">
                <label>Static Tree Height (m)</label>
                <input
                  type="number"
                  value={staticTreeHeight}
                  min={0}
                  max={100}
                  onChange={(e) => setStaticTreeHeight(Number(e.target.value))}
                />
              </div>
              <div className="checkbox-row">
                <input
                  type="checkbox"
                  checked={demInterpolation}
                  onChange={(e) => setDemInterpolation(e.target.checked)}
                />
                <span>DEM Interpolation</span>
              </div>
              <div className="checkbox-row">
                <input
                  type="checkbox"
                  checked={useCitygmlCache}
                  onChange={(e) => setUseCitygmlCache(e.target.checked)}
                />
                <span>Use CityGML Cache</span>
              </div>
              <div className="checkbox-row">
                <input
                  type="checkbox"
                  checked={useNdsmCanopy}
                  onChange={(e) => setUseNdsmCanopy(e.target.checked)}
                />
                <span>Use nDSM for Canopy</span>
              </div>
            </div>
          )}
        </div>

        <button className="btn btn-primary" onClick={handleGenerate} disabled={loading}>
          {loading && <span className="spinner" />}
          {loading ? 'Generating...' : 'Generate VoxCity Model'}
        </button>

        {error && <div className="alert alert-error" style={{ marginTop: '0.75rem' }}>{error}</div>}

        {gridShape && (
          <div className="alert alert-success" style={{ marginTop: '0.75rem' }}>
            Model generated! Grid: {gridShape.join(' × ')} • Mesh: {meshsize}m
          </div>
        )}
      </div>

      {/* Right – 3D preview */}
      <div className="panel">
        <ThreeViewer figureJson={figureJson} />
      </div>
    </div>
  );
};

export default GenerationTab;
