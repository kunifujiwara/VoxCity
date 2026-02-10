import React, { useState, useEffect } from 'react';
import { generateModel, autoDetectSources, AutoDetectResult } from '../api';
import ThreeViewer from '../components/ThreeViewer';

// Available data source options for each category
const BUILDING_SOURCES = [
  'OpenStreetMap',
  'Microsoft Building Footprints',
  'Open Building 2.5D Temporal',
  'EUBUCCO v0.1',
  'Overture',
  'GBA',
];

const BUILDING_COMPLEMENTARY_SOURCES = [
  'None',
  'OpenStreetMap',
  'Microsoft Building Footprints',
  'Open Building 2.5D Temporal',
  'EUBUCCO v0.1',
  'Overture',
  'GBA',
  'England 1m DSM - DTM',
  'Netherlands 0.5m DSM - DTM',
];

const LAND_COVER_SOURCES = [
  'OpenStreetMap',
  'OpenEarthMapJapan',
  'Urbanwatch',
  'ESA WorldCover',
  'ESRI 10m Annual Land Cover',
  'Dynamic World V1',
];

const CANOPY_HEIGHT_SOURCES = [
  'Static',
  'OpenStreetMap',
  'High Resolution 1m Global Canopy Height Maps',
  'ETH Global Sentinel-2 10m Canopy Height (2020)',
];

const DEM_SOURCES = [
  'Flat',
  'FABDEM',
  'DeltaDTM',
  'USGS 3DEP 1m',
  'England 1m DTM',
  'DEM France 1m',
  'DEM France 5m',
  'AUSTRALIA 5M DEM',
  'Netherlands 0.5m DTM',
];

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
  // Mode: "plateau" or "normal"
  const [mode, setMode] = useState<'plateau' | 'normal'>('normal');

  // Common parameters
  const [meshsize, setMeshsize] = useState(5);
  const [buildingComplementHeight, setBuildingComplementHeight] = useState(10);
  const [staticTreeHeight, setStaticTreeHeight] = useState(10);
  const [demInterpolation, setDemInterpolation] = useState(true);
  const [useCitygmlCache, setUseCitygmlCache] = useState(true);
  const [useNdsmCanopy, setUseNdsmCanopy] = useState(true);

  // Normal-mode data sources (null = auto)
  const [useAutoSources, setUseAutoSources] = useState(true);
  const [buildingSource, setBuildingSource] = useState<string | null>(null);
  const [buildingCompSource, setBuildingCompSource] = useState<string | null>(null);
  const [landCoverSource, setLandCoverSource] = useState<string | null>(null);
  const [canopyHeightSource, setCanopyHeightSource] = useState<string | null>(null);
  const [demSource, setDemSource] = useState<string | null>(null);
  const [autoDetected, setAutoDetected] = useState<AutoDetectResult | null>(null);
  const [detectingAuto, setDetectingAuto] = useState(false);

  // UI state
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [gridShape, setGridShape] = useState<number[] | null>(null);
  const [showAdvanced, setShowAdvanced] = useState(false);

  // Auto-detect sources when rectangle changes and mode is normal + auto
  useEffect(() => {
    if (rectangle && mode === 'normal' && useAutoSources) {
      handleAutoDetect();
    }
  }, [rectangle, mode]);

  const handleAutoDetect = async () => {
    if (!rectangle) return;
    setDetectingAuto(true);
    try {
      const result = await autoDetectSources(rectangle);
      setAutoDetected(result);
      setBuildingSource(result.building_source);
      setBuildingCompSource(result.building_complementary_source);
      setLandCoverSource(result.land_cover_source);
      setCanopyHeightSource(result.canopy_height_source);
      setDemSource(result.dem_source);
    } catch (err: any) {
      console.warn('Auto-detect failed:', err.message);
    }
    setDetectingAuto(false);
  };

  const handleGenerate = async () => {
    if (!rectangle) return;
    setLoading(true);
    setError(null);
    try {
      const params: Record<string, any> = {
        rectangle_vertices: rectangle,
        meshsize,
        mode,
        building_complement_height: buildingComplementHeight,
        static_tree_height: staticTreeHeight,
        dem_interpolation: demInterpolation,
      };

      if (mode === 'plateau') {
        params.use_citygml_cache = useCitygmlCache;
        params.use_ndsm_canopy = useNdsmCanopy;
      } else {
        // Normal mode: pass sources (null = auto)
        params.building_source = useAutoSources ? null : buildingSource;
        params.land_cover_source = useAutoSources ? null : landCoverSource;
        params.canopy_height_source = useAutoSources ? null : canopyHeightSource;
        params.dem_source = useAutoSources ? null : demSource;
        params.building_complementary_source = useAutoSources ? null : buildingCompSource;
      }

      const result = await generateModel(params as any);
      setGridShape(result.grid_shape);
      onFigureChange(result.figure_json);
      onModelReady();
    } catch (err: any) {
      setError(err.message);
    }
    setLoading(false);
  };

  if (!rectangle) {
    return <div className="alert alert-warning">Set the target area first in the &quot;Target Area&quot; tab.</div>;
  }

  return (
    <div className="two-col">
      {/* Left – controls */}
      <div className="panel">
        <h2>Generation Mode</h2>

        {/* Mode toggle */}
        <div className="mode-toggle">
          <button
            className={`mode-btn ${mode === 'normal' ? 'active' : ''}`}
            onClick={() => setMode('normal')}
          >
            🌍 Normal
          </button>
          <button
            className={`mode-btn ${mode === 'plateau' ? 'active' : ''}`}
            onClick={() => setMode('plateau')}
          >
            🏯 PLATEAU
          </button>
        </div>

        <div className="mode-description">
          {mode === 'plateau'
            ? 'Uses CityGML data from PLATEAU (Japan). Best for Japanese cities with high-precision 3D building data.'
            : 'Uses global open data sources. Works anywhere in the world with automatic or custom data source selection.'}
        </div>

        <hr style={{ margin: '0.75rem 0', border: 'none', borderTop: '1px solid var(--vc-ring)' }} />

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

        {/* Normal mode: data source configuration */}
        {mode === 'normal' && (
          <div className="expander" style={{ marginBottom: '0.75rem' }}>
            <div className="expander-header" onClick={() => {}}>
              Data Sources
            </div>
            <div className="expander-body">
              <div className="checkbox-row" style={{ marginBottom: '0.5rem' }}>
                <input
                  type="checkbox"
                  checked={useAutoSources}
                  onChange={(e) => setUseAutoSources(e.target.checked)}
                />
                <span>Auto-select sources based on location</span>
              </div>

              {useAutoSources && autoDetected && (
                <div className="alert alert-info" style={{ fontSize: '0.78rem', marginBottom: '0.75rem' }}>
                  <strong>Auto-detected:</strong><br />
                  Buildings: {autoDetected.building_source}<br />
                  Complementary: {autoDetected.building_complementary_source}<br />
                  Land Cover: {autoDetected.land_cover_source}<br />
                  Canopy: {autoDetected.canopy_height_source}<br />
                  DEM: {autoDetected.dem_source}
                </div>
              )}

              {useAutoSources && !autoDetected && (
                <button
                  className="btn btn-sm"
                  onClick={handleAutoDetect}
                  disabled={detectingAuto}
                  style={{ marginBottom: '0.5rem' }}
                >
                  {detectingAuto ? 'Detecting...' : 'Detect Sources'}
                </button>
              )}

              {!useAutoSources && (
                <>
                  <div className="form-group">
                    <label>Building Source</label>
                    <select
                      value={buildingSource || 'OpenStreetMap'}
                      onChange={(e) => setBuildingSource(e.target.value)}
                    >
                      {BUILDING_SOURCES.map((s) => (
                        <option key={s} value={s}>{s}</option>
                      ))}
                    </select>
                  </div>

                  <div className="form-group">
                    <label>Building Complementary Source</label>
                    <select
                      value={buildingCompSource || 'None'}
                      onChange={(e) => setBuildingCompSource(e.target.value)}
                    >
                      {BUILDING_COMPLEMENTARY_SOURCES.map((s) => (
                        <option key={s} value={s}>{s}</option>
                      ))}
                    </select>
                  </div>

                  <div className="form-group">
                    <label>Land Cover Source</label>
                    <select
                      value={landCoverSource || 'OpenStreetMap'}
                      onChange={(e) => setLandCoverSource(e.target.value)}
                    >
                      {LAND_COVER_SOURCES.map((s) => (
                        <option key={s} value={s}>{s}</option>
                      ))}
                    </select>
                  </div>

                  <div className="form-group">
                    <label>Canopy Height Source</label>
                    <select
                      value={canopyHeightSource || 'Static'}
                      onChange={(e) => setCanopyHeightSource(e.target.value)}
                    >
                      {CANOPY_HEIGHT_SOURCES.map((s) => (
                        <option key={s} value={s}>{s}</option>
                      ))}
                    </select>
                  </div>

                  <div className="form-group">
                    <label>DEM Source</label>
                    <select
                      value={demSource || 'Flat'}
                      onChange={(e) => setDemSource(e.target.value)}
                    >
                      {DEM_SOURCES.map((s) => (
                        <option key={s} value={s}>{s}</option>
                      ))}
                    </select>
                  </div>
                </>
              )}
            </div>
          </div>
        )}

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
              {mode === 'plateau' && (
                <>
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
                </>
              )}
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
