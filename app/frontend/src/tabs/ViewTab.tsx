/**
 * View tab — runs a view-index simulation (sky/green/custom) and previews
 * the result on the new R3F `<SceneViewer>`.
 *
 * Migrated from the legacy `figureJson` + `<ThreeViewer>` flow during
 * Chunk 4 of the Three.js migration.
 */
import React, { useEffect, useMemo, useState } from 'react';
import { getModelGeo, ModelGeoResult, runView } from '../api';
import { SceneViewer } from '../three';
import ColorSettings from '../components/ColorSettings';
import SamplingSettings from '../components/SamplingSettings';
import VoxelClassVisibility from '../components/VoxelClassVisibility';
import ZoneStatsTable from '../components/ZoneStatsTable';
import { CUSTOM_CLASSES } from '../constants';
import { lonLatToUvM } from '../lib/grid';
import { useZoneStats } from '../hooks/useZoneStats';
import { useSurfaceZoneEdges } from '../hooks/useSurfaceZoneEdges';
import { Zone } from '../types/zones';
import { ChoiceGroup, GuidedFooter, GuidedPanel, GuidedSection, GuidedStatus } from '../components/guided';
import { prerequisiteMessageForTab, simulationActionLabel } from './guidedTabState';

interface ViewTabProps {
  hasModel: boolean;
  /** @deprecated kept for App-level state compatibility, unused here. */
  figureJson: string;
  /** @deprecated kept for App-level state compatibility, unused here. */
  onFigureChange: (json: string) => void;
  zones: Zone[];
  simRunNonce: number;
  onSimRun: () => void;
  geometryToken?: string | number;
}

const ViewTab: React.FC<ViewTabProps> = ({ hasModel, zones, simRunNonce, onSimRun, geometryToken }) => {
  const [showZones3D, setShowZones3D] = useState(true);
  const { stats: zoneStats, loading: zoneStatsLoading } = useZoneStats(zones, 'view', simRunNonce);
  const [viewType, setViewType] = useState<'green' | 'sky' | 'custom'>('green');
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
  const [error, setError] = useState<string | null>(null);
  const [hasSimResult, setHasSimResult] = useState(false);

  const { surfaceZoneEdges } = useSurfaceZoneEdges({
    hasModel,
    enabled: showZones3D,
    zones,
  });

  const [geo, setGeo] = useState<ModelGeoResult | null>(null);
  useEffect(() => {
    if (!hasModel) {
      setGeo(null);
      return;
    }
    let cancelled = false;
    getModelGeo()
      .then((g) => { if (!cancelled) setGeo(g); })
      .catch(() => {});
    return () => { cancelled = true; };
  }, [hasModel]);
  const lonLatToXY = useMemo(() => lonLatToUvM(geo), [geo]);

  if (!hasModel) {
    const message = prerequisiteMessageForTab('view');
    return (
      <div className="two-col">
        <GuidedStatus tone="warning">
          <strong>{message.title}</strong><br />
          {message.body}
        </GuidedStatus>
      </div>
    );
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
      await runView({
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
      setHasSimResult(true);
      onSimRun();
    } catch (err: any) {
      setError(err.message);
    }
    setLoading(false);
  };

  return (
    <div className="two-col">
      <GuidedPanel
        title="View Index"
        subtitle="Analyse sky, green, or custom view from any point."
        status={
          error ? (
            <GuidedStatus tone="error">{error}</GuidedStatus>
          ) : hasSimResult ? (
            <GuidedStatus tone="success">Simulation complete.</GuidedStatus>
          ) : undefined
        }
        footer={(
          <GuidedFooter>
            <button className="btn btn-primary" onClick={handleRun} disabled={loading} type="button">
              {loading && <span className="spinner" />}
              {simulationActionLabel(loading)}
            </button>
          </GuidedFooter>
        )}
      >
        <GuidedSection label="View type">
          <ChoiceGroup
            ariaLabel="View type"
            value={viewType}
            onChange={setViewType}
            columns={1}
            options={[
              { id: 'green', label: 'Green View Index' },
              { id: 'sky', label: 'Sky View Index' },
              { id: 'custom', label: 'Custom (select classes)' },
            ]}
          />
        </GuidedSection>

        {viewType === 'custom' && (
          <GuidedSection label="Custom classes">
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
            <div style={{ maxHeight: 160, overflowY: 'auto' }}>
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
          </GuidedSection>
        )}

        <GuidedSection label="Sampling">
          <ChoiceGroup
            ariaLabel="Analysis target"
            value={analysisTarget}
            onChange={setAnalysisTarget}
            options={[
              { id: 'ground', label: 'Ground level' },
              { id: 'building', label: 'Building surfaces' },
            ]}
          />
          <div className="form-group">
            <label>View point height (m)</label>
            <input
              type="number"
              value={viewPointHeight}
              min={0}
              max={10}
              step={0.5}
              onChange={(e) => setViewPointHeight(Number(e.target.value))}
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
        </GuidedSection>

        <GuidedSection label="Display">
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
        </GuidedSection>

        {zones.length > 0 && (
          <GuidedSection label="Zones and results">
            <div className="form-group">
              <label>
                <input
                  type="checkbox"
                  checked={showZones3D}
                  onChange={(e) => setShowZones3D(e.target.checked)}
                />{' '}
                Show zones in 3D
              </label>
            </div>
            <ZoneStatsTable zones={zones} stats={zoneStats} loading={zoneStatsLoading} />
          </GuidedSection>
        )}
      </GuidedPanel>

      <div className="panel visual-panel">
        <div className="visual-frame">
          <SceneViewer
            geometryToken={hasModel ? (geometryToken ?? 'loaded') : 'none'}
            downsample={1}
            colorScheme="grayscale"
            simKind={hasSimResult ? 'view' : null}
            simToken={simRunNonce}
            colormap={colormap}
            vmin={vmin}
            vmax={vmax}
            zones={zones}
            lonLatToXY={lonLatToXY}
            showZones={showZones3D}
            hiddenClasses={hiddenClasses}
            surfaceZoneEdges={surfaceZoneEdges}
          />
        </div>
      </div>
    </div>
  );
};

export default ViewTab;
