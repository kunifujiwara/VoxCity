import React, { useState, useEffect } from 'react';
import { Globe, Building2, Layers } from 'lucide-react';
import { generateModel, autoDetectSources, AutoDetectResult } from '../api';
import ThreeViewer from '../components/ThreeViewer';
import PreviewDisabledNotice from '../components/PreviewDisabledNotice';
import { estimateGridShape } from '../lib/grid';
import {
  BUILDING_SOURCES,
  BUILDING_COMPLEMENTARY_SOURCES,
  LAND_COVER_SOURCES,
  CANOPY_HEIGHT_SOURCES,
  DEM_SOURCES,
  PREVIEW_MAX_CELLS,
} from '../constants';
import { ChoiceGroup, GuidedFooter, GuidedPanel, GuidedSection, GuidedStatus } from '../components/guided';
import { generationActionLabel, prerequisiteMessageForTab } from './guidedTabState';

interface GenerationTabProps {
  rectangle: number[][] | null;
  figureJson: string;
  onFigureChange: (json: string) => void;
  onModelReady: (info: { grid_shape: number[]; preview_disabled?: boolean }) => void;
  previewDisabled?: boolean;
  previewGridShape?: number[] | null;
}

const GenerationTab: React.FC<GenerationTabProps> = ({
  rectangle,
  figureJson,
  onFigureChange,
  onModelReady,
  previewDisabled = false,
  previewGridShape,
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
      onModelReady({ grid_shape: result.grid_shape, preview_disabled: result.preview_disabled });
    } catch (err: any) {
      setError(err.message);
    }
    setLoading(false);
  };

  const estimate = rectangle ? estimateGridShape(rectangle, meshsize) : null;
  const willDisablePreview = !!estimate && estimate[0] * estimate[1] > PREVIEW_MAX_CELLS;

  if (!rectangle) {
    const message = prerequisiteMessageForTab('generation');
    return (
      <div className="two-col">
        <GuidedStatus tone="warning">
          <strong>{message.title}</strong><br />
          {message.body}
        </GuidedStatus>
      </div>
    );
  }

  return (
    <div className="two-col">
      {/* Left – controls */}
      <GuidedPanel
        title="Generate Model"
        subtitle="Build the VoxCity 3D model from city data."
        status={
          error ? (
            <GuidedStatus tone="error">{error}</GuidedStatus>
          ) : gridShape ? (
            <GuidedStatus tone="success">
              Model generated. Grid: {gridShape.join(' × ')}. Mesh: {meshsize}m.
            </GuidedStatus>
          ) : undefined
        }
        footer={(
          <GuidedFooter>
            <button className="btn btn-primary" onClick={handleGenerate} disabled={loading} type="button">
              {loading && <span className="spinner" />}
              <Layers size={14} aria-hidden="true" style={{ marginRight: 6 }} />
              {generationActionLabel(loading)}
            </button>
          </GuidedFooter>
        )}
      >
        <GuidedSection index={1} label="GENERATION MODE">
          <ChoiceGroup
            variant="checks"
            ariaLabel="Generation mode"
            value={mode}
            onChange={setMode}
            options={[
              { id: 'normal', label: 'Normal', description: 'Global open data sources', icon: Globe },
              { id: 'plateau', label: 'PLATEAU', description: 'Japanese CityGML data', icon: Building2 },
            ]}
          />
        </GuidedSection>

        <GuidedSection index={2} label="GRID RESOLUTION">
          <div className="form-group">
            <label>Mesh size (meters)</label>
            <input type="number" value={meshsize} min={1} max={50} onChange={(e) => setMeshsize(Number(e.target.value))} />
          </div>
        </GuidedSection>

        {willDisablePreview && estimate && (
          <div className="alert alert-info" style={{ fontSize: '0.78rem', margin: '0 0 0.75rem' }}>
            Estimated grid ~{estimate[0]}×{estimate[1]} — the 3D preview will be
            disabled at this size. Generation and export still work.
          </div>
        )}

        {mode === 'normal' && (
          <GuidedSection
            index={3}
            label="DATA SOURCES"
            collapsible
            defaultOpen={!useAutoSources}
          >
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
          </GuidedSection>
        )}

        <GuidedSection
          index={4}
          label="ADVANCED"
          collapsible
          defaultOpen={false}
        >
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
        </GuidedSection>
      </GuidedPanel>

      {/* Right – 3D preview */}
      <div className="panel visual-panel">
        <div className="visual-frame">
          {previewDisabled
            ? <PreviewDisabledNotice gridShape={previewGridShape} />
            : <ThreeViewer figureJson={figureJson} />}
        </div>
      </div>
    </div>
  );
};

export default GenerationTab;
