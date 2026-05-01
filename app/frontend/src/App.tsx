import React, { useCallback, useEffect, useRef, useState } from 'react';
import TargetAreaTab from './tabs/TargetAreaTab';
import GenerationTab from './tabs/GenerationTab';
import EditTab from './tabs/EditTab';
import SolarTab from './tabs/SolarTab';
import ViewTab from './tabs/ViewTab';
import LandmarkTab from './tabs/LandmarkTab';
import ExportTab from './tabs/ExportTab';
import ZoningTab from './tabs/ZoningTab';
import type { Zone } from './types/zones';
import { healthCheck, resetSession } from './api';

const TABS = [
  { id: 'area', label: 'Target Area' },
  { id: 'generation', label: 'Generation' },
  { id: 'edit', label: 'Edit' },
  { id: 'zoning', label: 'Zoning' },
  { id: 'solar', label: 'Solar' },
  { id: 'view', label: 'View' },
  { id: 'landmark', label: 'Landmark' },
  { id: 'export', label: 'Export' },
] as const;

type TabId = (typeof TABS)[number]['id'];

const App: React.FC = () => {
  const [activeTab, setActiveTab] = useState<TabId>('area');
  const [rectangle, setRectangle] = useState<number[][] | null>(null);
  const [figureJson, setFigureJson] = useState('');
  const [editFigureJson, setEditFigureJson] = useState('');
  const [solarFigureJson, setSolarFigureJson] = useState('');
  const [viewFigureJson, setViewFigureJson] = useState('');
  const [landmarkFigureJson, setLandmarkFigureJson] = useState('');
  const [hasModel, setHasModel] = useState(false);
  const [zones, setZones] = useState<Zone[]>([]);
  const [solarRunNonce, setSolarRunNonce] = useState(0);
  const [viewRunNonce, setViewRunNonce] = useState(0);
  const [landmarkRunNonce, setLandmarkRunNonce] = useState(0);

  // When the user changes the target rectangle, the previous zones and any
  // cached simulation figures no longer correspond to the area on screen.
  useEffect(() => {
    setZones([]);
    setFigureJson('');
    setEditFigureJson('');
    setSolarFigureJson('');
    setViewFigureJson('');
    setLandmarkFigureJson('');
    setSolarRunNonce(0);
    setViewRunNonce(0);
    setLandmarkRunNonce(0);
  }, [rectangle]);

  // After an edit commit, the cached Solar / View / Landmark figures and the
  // Generation tab's preview are stale (they referenced the old voxel grid).
  // Reset them so the user re-runs simulations on the edited model.
  const handleModelEdited = useCallback(() => {
    setFigureJson('');
    setSolarFigureJson('');
    setViewFigureJson('');
    setLandmarkFigureJson('');
  }, []);

  // On page load, reset the backend so Taichi caches are cleared and a
  // new target area / model / simulation cycle can run cleanly.
  // Also check whether a model already exists (backend may still hold one).
  const didReset = useRef(false);
  useEffect(() => {
    if (didReset.current) return;
    didReset.current = true;
    resetSession()
      .then(() => healthCheck())
      .then((h) => setHasModel(h.has_model))
      .catch(() => {});
  }, []);

  return (
    <div className="app-container">
      {/* Header */}
      <header className="app-header">
        <img src="/logo.png" alt="VoxCity" className="logo" />
      </header>

      {/* Tab bar */}
      <nav className="tab-bar">
        {TABS.map((tab) => (
          <button
            key={tab.id}
            className={`tab-btn ${activeTab === tab.id ? 'active' : ''}`}
            onClick={() => setActiveTab(tab.id)}
          >
            {tab.label}
          </button>
        ))}
      </nav>

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
              setEditFigureJson('');
              setSolarFigureJson('');
              setViewFigureJson('');
              setLandmarkFigureJson('');
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
        {activeTab === 'zoning' && (
          <ZoningTab
            hasModel={hasModel}
            figureJson={figureJson}
            zones={zones}
            onZonesChange={setZones}
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
          />
        )}
        {activeTab === 'export' && <ExportTab hasModel={hasModel} />}
      </main>

    </div>
  );
};

export default App;
