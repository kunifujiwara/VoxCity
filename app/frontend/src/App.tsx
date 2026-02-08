import React, { useState } from 'react';
import TargetAreaTab from './tabs/TargetAreaTab';
import GenerationTab from './tabs/GenerationTab';
import SolarTab from './tabs/SolarTab';
import ViewTab from './tabs/ViewTab';
import LandmarkTab from './tabs/LandmarkTab';
import ExportTab from './tabs/ExportTab';

const TABS = [
  { id: 'area', label: 'Target Area' },
  { id: 'generation', label: 'Generation' },
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
  const [hasModel, setHasModel] = useState(false);

  return (
    <div className="app-container">
      {/* Header */}
      <header className="app-header">
        <h1>VoxCity Web App</h1>
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
            onModelReady={() => setHasModel(true)}
          />
        )}
        {activeTab === 'solar' && <SolarTab hasModel={hasModel} />}
        {activeTab === 'view' && <ViewTab hasModel={hasModel} />}
        {activeTab === 'landmark' && <LandmarkTab hasModel={hasModel} />}
        {activeTab === 'export' && <ExportTab hasModel={hasModel} />}
      </main>

      {/* Footer */}
      <footer className="app-footer">
        VoxCity Web App | Based on{' '}
        <a href="https://github.com/kunifujiwara/VoxCity" target="_blank" rel="noreferrer">
          VoxCity
        </a>{' '}
        by Kunihiko Fujiwara
      </footer>
    </div>
  );
};

export default App;
