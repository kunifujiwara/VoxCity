import React, { useCallback, useEffect, useRef, useState } from 'react';
import { Plus, Upload, X } from 'lucide-react';
import { loadSession } from '../api';
import type { SessionLoadSummary } from '../api';
import {
  buildRestoredFrontendState,
  parsePersistedFrontendState,
  type RestoredFrontendState,
} from '../lib/sessionRestore';
import { GuidedFooter, GuidedPanel, GuidedSection, GuidedStatus } from './guided';

export const SPLASH_DISMISSED_KEY = 'voxcity:splash:dismissed';

export interface StartSplashProps {
  open: boolean;
  onClose: () => void;
  onSessionLoaded: (summary: SessionLoadSummary, restored?: RestoredFrontendState) => void;
  disableOpen?: boolean;
}

const StartSplash: React.FC<StartSplashProps> = ({
  open,
  onClose,
  onSessionLoaded,
  disableOpen = false,
}) => {
  const [dontShowAgain, setDontShowAgain] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  const handleClose = useCallback(() => {
    if (dontShowAgain) {
      try {
        localStorage.setItem(SPLASH_DISMISSED_KEY, '1');
      } catch {
        // Ignore storage failures.
      }
    }
    onClose();
  }, [dontShowAgain, onClose]);

  useEffect(() => {
    if (!open) return undefined;
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape') handleClose();
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [handleClose, open]);

  if (!open) return null;

  const handleFileChange = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;
    setLoading(true);
    setError(null);
    try {
      const summary = await loadSession(file);
      const persisted = parsePersistedFrontendState(summary.frontend_state);
      const { restored } = buildRestoredFrontendState(persisted);
      onSessionLoaded(summary, restored);
      handleClose();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load session.');
    } finally {
      if (fileInputRef.current) fileInputRef.current.value = '';
      setLoading(false);
    }
  };

  return (
    <div
      data-testid="start-splash-backdrop"
      onClick={handleClose}
      style={{
        position: 'fixed',
        inset: 0,
        zIndex: 1000,
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        padding: '1rem',
        background: 'rgba(15, 23, 42, 0.55)',
      }}
    >
      <div
        role="dialog"
        aria-modal="true"
        aria-labelledby="start-splash-title"
        onClick={(event) => event.stopPropagation()}
        style={{ width: 'min(100%, 440px)', position: 'relative' }}
      >
        <button
          type="button"
          className="btn btn-icon"
          aria-label="Close"
          onClick={handleClose}
          style={{ position: 'absolute', top: 8, right: 8, zIndex: 1 }}
        >
          <X size={16} aria-hidden="true" />
        </button>
        <GuidedPanel
          title={<span id="start-splash-title">Welcome to VoxCity</span>}
          subtitle="Start a new urban model, or open a saved session."
          status={error ? <GuidedStatus tone="error">{error}</GuidedStatus> : undefined}
          footer={(
            <GuidedFooter>
              <button type="button" className="btn btn-primary" onClick={handleClose}>
                <Plus size={14} aria-hidden="true" />
                New session
              </button>
              <button
                type="button"
                className="btn"
                onClick={() => fileInputRef.current?.click()}
                disabled={loading || disableOpen}
              >
                <Upload size={14} aria-hidden="true" />
                Open session...
              </button>
              <input
                ref={fileInputRef}
                type="file"
                accept=".zip,application/zip"
                style={{ display: 'none' }}
                onChange={handleFileChange}
              />
            </GuidedFooter>
          )}
        >
          <GuidedSection index={1} label="GET STARTED">
            <label className="checkbox-row">
              <input
                type="checkbox"
                checked={dontShowAgain}
                onChange={(event) => setDontShowAgain(event.target.checked)}
              />
              <span>Don't show this again</span>
            </label>
          </GuidedSection>
        </GuidedPanel>
      </div>
    </div>
  );
};

export default StartSplash;
