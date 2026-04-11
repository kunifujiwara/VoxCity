import { useCallback, useRef } from 'react';
import { rerenderSimulation } from '../api';

/**
 * Manual rerender hook.
 *
 * Returns an `handleUpdate` callback the caller wires to an "Update View"
 * button.  Settings changes are NOT applied until the button is clicked.
 * Cancels any in-flight request when a new one is triggered so that stale
 * responses never overwrite fresh state.
 */
export function useManualRerender(
  hasSimResult: React.MutableRefObject<boolean>,
  deps: {
    colormap: string;
    vmin: number;
    vmax: string | number | null;
    hiddenClasses: Set<number>;
  },
  setFigureJson: (json: string) => void,
  setRerendering: (v: boolean) => void,
) {
  const abortRef = useRef<AbortController | null>(null);

  const handleUpdate = useCallback(async () => {
    if (!hasSimResult.current) return;

    // Abort any in-flight request
    if (abortRef.current) abortRef.current.abort();

    setRerendering(true);
    const controller = new AbortController();
    abortRef.current = controller;
    try {
      const result = await rerenderSimulation({
        colormap: deps.colormap,
        vmin: deps.vmin,
        vmax: typeof deps.vmax === 'string'
          ? (deps.vmax ? parseFloat(deps.vmax) : null)
          : deps.vmax,
        hidden_classes: Array.from(deps.hiddenClasses),
      });
      if (!controller.signal.aborted) {
        setFigureJson(result.figure_json);
      }
    } catch {
      // Ignore aborted / network errors during rerender
    }
    if (!controller.signal.aborted) {
      setRerendering(false);
    }
  }, [deps.colormap, deps.vmin, deps.vmax, deps.hiddenClasses, hasSimResult, setFigureJson, setRerendering]);

  return handleUpdate;
}
