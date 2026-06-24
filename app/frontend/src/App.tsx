import React, { useCallback, useEffect, useRef, useState } from 'react';
import TargetAreaTab from './tabs/TargetAreaTab';
import GenerationTab from './tabs/GenerationTab';
import EditTab from './tabs/EditTab';
import ImportTab from './tabs/ImportTab';
import SolarTab from './tabs/SolarTab';
import ViewTab from './tabs/ViewTab';
import LandmarkTab from './tabs/LandmarkTab';
import ExportTab from './tabs/ExportTab';
import ZoningTab from './tabs/ZoningTab';
import StartSplash, { SPLASH_DISMISSED_KEY } from './components/StartSplash';
import type { RestoredFrontendState } from './lib/sessionRestore';
export type { RestoredFrontendState };
import type { SessionLoadSummary } from './api';
import {
  MapPin, Layers, Pencil, Grid3x3, Sun, Camera,
  Landmark as LandmarkIcon, FolderOpen, Boxes,
} from 'lucide-react';
import type { Zone } from './types/zones';
import { healthCheck, resetSession } from './api';

const TABS = [
  { id: 'area',       label: 'Target',   Icon: MapPin },
  { id: 'generation', label: 'Generate', Icon: Layers },
  { id: 'edit',       label: 'Edit',     Icon: Pencil },
  { id: 'import',     label: 'Import',   Icon: Boxes },
  { id: 'zoning',     label: 'Zone',     Icon: Grid3x3 },
  { id: 'solar',      label: 'Solar',    Icon: Sun },
  { id: 'view',       label: 'View',     Icon: Camera },
  { id: 'landmark',   label: 'Landmark', Icon: LandmarkIcon },
  { id: 'export',     label: 'File',     Icon: FolderOpen },
] as const;

type TabId = (typeof TABS)[number]['id'];

const App: React.FC = () => {
  const [activeTab, setActiveTab] = useState<TabId>('area');
  const [rectangle, setRectangle] = useState<number[][] | null>(null);
  const [figureJson, setFigureJson] = useState('');
  const [editFigureJson, setEditFigureJson] = useState('');
  const [importFigureJson, setImportFigureJson] = useState('');
  const [solarFigureJson, setSolarFigureJson] = useState('');
  const [viewFigureJson, setViewFigureJson] = useState('');
  const [landmarkFigureJson, setLandmarkFigureJson] = useState('');
  const [hasModel, setHasModel] = useState(false);
  const [geometryToken, setGeometryToken] = useState(0);
  const [zones, setZones] = useState<Zone[]>([]);
  const [solarRunNonce, setSolarRunNonce] = useState(0);
  const [viewRunNonce, setViewRunNonce] = useState(0);
  const [landmarkRunNonce, setLandmarkRunNonce] = useState(0);

  const [splashOpen, setSplashOpen] = useState(() => {
    try { return localStorage.getItem(SPLASH_DISMISSED_KEY) !== '1'; } catch { return true; }
  });
  const [initialResetPending, setInitialResetPending] = useState(true);
  const restoringFromSessionRef = useRef<RestoredFrontendState>({});
  const sessionLoadedRef = useRef(false);
  // Sim types restored by a session load, consumed by the rectangle effect so it
  // seeds the matching sim nonces (instead of 0) — a non-zero nonce is what makes
  // useZoneStats fetch stats for the restored result.
  const restoreSimNoncesRef = useRef<string[]>([]);
  // When a session is loaded that carried cached simulation results, this holds
  // every restored sim type ('solar' | 'view' | 'landmark') so each matching sim
  // tab can show its overlay without requiring a re-run.
  const [restoredSimTypes, setRestoredSimTypes] = useState<string[]>([]);
  // Landmark building IDs recovered from a restored landmark sim, so the
  // Landmark tab can re-apply its selection highlights.
  const [restoredLandmarkIds, setRestoredLandmarkIds] = useState<number[]>([]);

  // When the user changes the target rectangle, the previous zones and any
  // cached simulation figures no longer correspond to the area on screen.
  useEffect(() => {
    const restoredZones = restoringFromSessionRef.current.zones;
    if (restoredZones !== undefined) {
      setZones(restoredZones);
      restoringFromSessionRef.current.zones = undefined;
    } else {
      setZones([]);
    }
    setFigureJson('');
    setEditFigureJson('');
    setImportFigureJson('');
    setSolarFigureJson('');
    setViewFigureJson('');
    setLandmarkFigureJson('');
    // On a session restore, the cached sim results are valid for this rectangle,
    // so seed the matching nonces (non-zero) to make zone stats fetch. A normal
    // rectangle change leaves the list empty and resets all nonces to 0.
    const restoredSims = restoreSimNoncesRef.current;
    setSolarRunNonce(restoredSims.includes('solar') ? 1 : 0);
    setViewRunNonce(restoredSims.includes('view') ? 1 : 0);
    setLandmarkRunNonce(restoredSims.includes('landmark') ? 1 : 0);
    restoreSimNoncesRef.current = [];
  }, [rectangle]);

  // After an edit commit, the cached Solar / View / Landmark figures and the
  // Generation tab's preview are stale (they referenced the old voxel grid).
  // Also bump geometryToken so SceneViewer re-fetches city geometry.
  const handleModelEdited = useCallback(() => {
    setFigureJson('');
    setSolarFigureJson('');
    setViewFigureJson('');
    setLandmarkFigureJson('');
    setGeometryToken((t) => t + 1);
    setZones((prev) => prev.filter((z) => z.type === 'horizontal'));
    setRestoredSimTypes([]);
    setRestoredLandmarkIds([]);
  }, []);

  const handleSessionLoaded = useCallback(
    (summary: SessionLoadSummary, restored?: RestoredFrontendState) => {
      sessionLoadedRef.current = true;
      restoringFromSessionRef.current = { zones: restored?.zones };
      restoreSimNoncesRef.current = summary.sim_result_types ?? [];
      setHasModel(summary.has_voxcity);
      setRectangle(summary.rectangle_vertices);
      setFigureJson('');
      setEditFigureJson('');
      setImportFigureJson('');
      setSolarFigureJson('');
      setViewFigureJson('');
      setLandmarkFigureJson('');
      setGeometryToken((token) => token + 1);
      setRestoredSimTypes(summary.sim_result_types ?? []);
      setRestoredLandmarkIds(summary.landmark_building_ids ?? []);
    },
    [],
  );

  // On page load, reset the backend so Taichi caches are cleared and a
  // new target area / model / simulation cycle can run cleanly.
  // Also check whether a model already exists (backend may still hold one).
  const didReset = useRef(false);
  useEffect(() => {
    if (didReset.current) return;
    didReset.current = true;
    resetSession()
      .then(() => healthCheck())
      .then((h) => { if (!sessionLoadedRef.current) setHasModel(h.has_model); })
      .catch(() => {})
      .finally(() => setInitialResetPending(false));
  }, []);

  return (
    <div className="app-container">
      <StartSplash
        open={splashOpen}
        onClose={() => setSplashOpen(false)}
        onSessionLoaded={handleSessionLoaded}
        disableOpen={initialResetPending}
      />
      {/* Header */}
      <header className="app-header">
        <img src="/logo.png" alt="VoxCity" className="logo" />
        <nav className="tab-bar">
          {TABS.map((tab) => {
            const Icon = tab.Icon;
            return (
              <button
                key={tab.id}
                className={`tab-btn ${activeTab === tab.id ? 'active' : ''}`}
                onClick={() => setActiveTab(tab.id)}
              >
                <Icon size={14} aria-hidden="true" />
                <span>{tab.label}</span>
              </button>
            );
          })}
        </nav>
      </header>

      {/* Tab content */}
      <main className="tab-content">
        {activeTab === 'area' && (
          <TargetAreaTab rectangle={rectangle} onRectangleChange={setRectangle} />
        )}
        {/*
          Generation tab is kept mounted so the 3D viewer's WebGL context
          (and any rendered model) survives tab switches.
        */}
        <div style={{ display: activeTab === 'generation' ? 'contents' : 'none' }}>
          <GenerationTab
            rectangle={rectangle}
            figureJson={figureJson}
            onFigureChange={setFigureJson}
            onModelReady={() => {
              setHasModel(true);
              setZones([]);
              setGeometryToken((t) => t + 1);
              setEditFigureJson('');
              setImportFigureJson('');
              setSolarFigureJson('');
              setViewFigureJson('');
              setLandmarkFigureJson('');
              setRestoredSimTypes([]);
              setRestoredLandmarkIds([]);
            }}
          />
        </div>
        {activeTab === 'edit' && (
          <EditTab
            hasModel={hasModel}
            figureJson={editFigureJson}
            onFigureChange={setEditFigureJson}
            onModelEdited={handleModelEdited}
          />
        )}
        {activeTab === 'import' && (
          <ImportTab
            hasModel={hasModel}
            figureJson={importFigureJson}
            onFigureChange={setImportFigureJson}
            onModelEdited={handleModelEdited}
          />
        )}
        {activeTab === 'zoning' && (
          <ZoningTab
            hasModel={hasModel}
            figureJson={figureJson}
            zones={zones}
            onZonesChange={setZones}
            geometryToken={geometryToken}
          />
        )}
        {/*
          Sim tabs are kept mounted (once a model exists) so their internal
          overlay/result state survives tab switches. Inactive tabs are hidden
          via CSS rather than unmounted.
        */}
        {hasModel && (
          <>
            <div style={{ display: activeTab === 'solar' ? 'contents' : 'none' }}>
              <SolarTab
                hasModel={hasModel}
                figureJson={solarFigureJson}
                onFigureChange={setSolarFigureJson}
                zones={zones}
                simRunNonce={solarRunNonce}
                onSimRun={() => setSolarRunNonce((n) => n + 1)}
                geometryToken={geometryToken}
                restoredSimTypes={restoredSimTypes}
              />
            </div>
            <div style={{ display: activeTab === 'view' ? 'contents' : 'none' }}>
              <ViewTab
                hasModel={hasModel}
                figureJson={viewFigureJson}
                onFigureChange={setViewFigureJson}
                zones={zones}
                simRunNonce={viewRunNonce}
                onSimRun={() => setViewRunNonce((n) => n + 1)}
                geometryToken={geometryToken}
                restoredSimTypes={restoredSimTypes}
              />
            </div>
            <div style={{ display: activeTab === 'landmark' ? 'contents' : 'none' }}>
              <LandmarkTab
                hasModel={hasModel}
                figureJson={landmarkFigureJson}
                onFigureChange={setLandmarkFigureJson}
                zones={zones}
                simRunNonce={landmarkRunNonce}
                onSimRun={() => setLandmarkRunNonce((n) => n + 1)}
                geometryToken={geometryToken}
                restoredSimTypes={restoredSimTypes}
                restoredLandmarkIds={restoredLandmarkIds}
              />
            </div>
          </>
        )}
        {!hasModel && activeTab === 'solar' && (
          <SolarTab
            hasModel={hasModel}
            figureJson={solarFigureJson}
            onFigureChange={setSolarFigureJson}
            zones={zones}
            simRunNonce={solarRunNonce}
            onSimRun={() => setSolarRunNonce((n) => n + 1)}
            geometryToken={geometryToken}
          />
        )}
        {!hasModel && activeTab === 'view' && (
          <ViewTab
            hasModel={hasModel}
            figureJson={viewFigureJson}
            onFigureChange={setViewFigureJson}
            zones={zones}
            simRunNonce={viewRunNonce}
            onSimRun={() => setViewRunNonce((n) => n + 1)}
            geometryToken={geometryToken}
          />
        )}
        {!hasModel && activeTab === 'landmark' && (
          <LandmarkTab
            hasModel={hasModel}
            figureJson={landmarkFigureJson}
            onFigureChange={setLandmarkFigureJson}
            zones={zones}
            simRunNonce={landmarkRunNonce}
            onSimRun={() => setLandmarkRunNonce((n) => n + 1)}
            geometryToken={geometryToken}
          />
        )}
        {activeTab === 'export' && (
          <ExportTab
            hasModel={hasModel}
            zones={zones}
            onSessionLoaded={handleSessionLoaded}
          />
        )}
      </main>

    </div>
  );
};

export default App;
