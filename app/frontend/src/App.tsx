import React, { useCallback, useEffect, useRef, useState } from 'react';
import TargetAreaTab from './tabs/TargetAreaTab';
import GenerationTab from './tabs/GenerationTab';
import EditTab from './tabs/EditTab';
import SolarTab from './tabs/SolarTab';
import ViewTab from './tabs/ViewTab';
import LandmarkTab from './tabs/LandmarkTab';
import ExportTab from './tabs/ExportTab';
import { healthCheck, resetSession } from './api';

const TABS = [
  { id: 'area', label: 'Target Area' },
  { id: 'generation', label: 'Generation' },
  { id: 'edit', label: 'Edit' },
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
        {activeTab === 'generation' && (
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
        )}
        {activeTab === 'edit' && (
          <EditTab
            hasModel={hasModel}
            figureJson={editFigureJson}
            onFigureChange={setEditFigureJson}
            onModelEdited={handleModelEdited}
          />
        )}
        {activeTab === 'solar' && <SolarTab hasModel={hasModel} figureJson={solarFigureJson} onFigureChange={setSolarFigureJson} />}
        {activeTab === 'view' && <ViewTab hasModel={hasModel} figureJson={viewFigureJson} onFigureChange={setViewFigureJson} />}
        {activeTab === 'landmark' && <LandmarkTab hasModel={hasModel} figureJson={landmarkFigureJson} onFigureChange={setLandmarkFigureJson} />}
        {activeTab === 'export' && <ExportTab hasModel={hasModel} />}
      </main>

    </div>
  );
};

export default App;
