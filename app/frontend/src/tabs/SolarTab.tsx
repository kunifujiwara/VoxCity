/**
 * Solar tab — runs a solar-radiation simulation and previews the result on
 * the new R3F `<SceneViewer>` (overlay + zone outlines + colorbar).
 *
 * The legacy `figureJson` / `useZoneOverlay` / `<ThreeViewer>` path was
 * removed during the Three.js migration (Chunk 4). The tab still accepts
 * `figureJson` / `onFigureChange` props for App-level state compatibility,
 * but they are intentionally unused.
 */
import React, { useEffect, useMemo, useRef, useState } from 'react';
import { Clock, CalendarRange, Layers, Box, Sun } from 'lucide-react';
import { getModelGeo, ModelGeoResult, runSolar } from '../api';
import { SceneViewer } from '../three';
import ColorSettings from '../components/ColorSettings';
import VoxelClassVisibility from '../components/VoxelClassVisibility';
import ZoneStatsTable from '../components/ZoneStatsTable';
import { lonLatToUvM } from '../lib/grid';
import { useZoneStats } from '../hooks/useZoneStats';
import { useSurfaceZoneEdges } from '../hooks/useSurfaceZoneEdges';
import { Zone } from '../types/zones';
import { ChoiceGroup, GuidedFooter, GuidedPanel, GuidedSection, GuidedStatus } from '../components/guided';
import { prerequisiteMessageForTab, simulationActionLabel } from './guidedTabState';

interface SolarTabProps {
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

const SolarTab: React.FC<SolarTabProps> = ({
  hasModel,
  zones,
  simRunNonce,
  onSimRun,
  geometryToken,
}) => {
  const [showZones3D, setShowZones3D] = useState(true);
  const { stats: zoneStats, loading: zoneStatsLoading } = useZoneStats(zones, 'solar', simRunNonce);
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
  const [error, setError] = useState<string | null>(null);
  const [hasSimResult, setHasSimResult] = useState(false);

  const { surfaceZoneEdges } = useSurfaceZoneEdges({
    hasModel,
    enabled: showZones3D,
    zones,
  });

  // Geometry for the lon/lat -> world projection used by zone outlines.
  const [geo, setGeo] = useState<ModelGeoResult | null>(null);
  useEffect(() => {
    if (!hasModel) {
      setGeo(null);
      return;
    }
    let cancelled = false;
    getModelGeo()
      .then((g) => { if (!cancelled) setGeo(g); })
      .catch(() => { /* ignore — zones just won't project */ });
    return () => { cancelled = true; };
  }, [hasModel]);
  const lonLatToXY = useMemo(() => lonLatToUvM(geo), [geo]);

  if (!hasModel) {
    const message = prerequisiteMessageForTab('solar');
    return (
      <div className="two-col">
        <GuidedStatus tone="warning">
          <strong>{message.title}</strong><br />
          {message.body}
        </GuidedStatus>
      </div>
    );
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
      await runSolar(params);
      setHasSimResult(true);
      onSimRun();
    } catch (err: any) {
      setError(err.message);
    }
    setLoading(false);
  };

  const vmaxNum = vmax ? parseFloat(vmax) : null;

  return (
    <div className="two-col">
      <GuidedPanel
        title="Solar Radiation"
        subtitle="Compute irradiance on ground or building surfaces."
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
              <Sun size={14} aria-hidden="true" style={{ marginRight: 6 }} />
              {simulationActionLabel(loading)}
            </button>
          </GuidedFooter>
        )}
      >
        <GuidedSection index={1} label="TEMPORAL TYPE">
          <ChoiceGroup
            variant="checks"
            ariaLabel="Calculation type"
            value={calcType}
            onChange={setCalcType}
            options={[
              { id: 'instantaneous', label: 'Instantaneous', icon: Clock },
              { id: 'cumulative', label: 'Cumulative', icon: CalendarRange },
            ]}
          />
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
        </GuidedSection>

        <GuidedSection index={2} label="SPATIAL TYPE">
          <ChoiceGroup
            variant="checks"
            ariaLabel="Analysis target"
            value={analysisTarget}
            onChange={setAnalysisTarget}
            options={[
              { id: 'ground', label: 'Ground level', icon: Layers },
              { id: 'building', label: 'Building surfaces', icon: Box },
            ]}
          />
        </GuidedSection>

        <GuidedSection index={3} label="DISPLAY">
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
        </GuidedSection>

        {zones.length > 0 && (
          <GuidedSection index={4} label="ZONES AND RESULTS">
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
            simKind={hasSimResult ? 'solar' : null}
            simToken={simRunNonce}
            colormap={colormap}
            vmin={vmin}
            vmax={vmaxNum}
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

export default SolarTab;
