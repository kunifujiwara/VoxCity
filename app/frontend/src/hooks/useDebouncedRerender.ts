import { useCallback, useEffect, useRef } from 'react';
import { rerenderSimulation } from '../api';

const DEBOUNCE_MS = 400;

/**
 * Debounced rerender hook.
 *
 * Waits DEBOUNCE_MS after the last settings change before firing an API call.
 * Cancels any in-flight request when a newer one is scheduled so that stale
 * responses never overwrite fresh state.
 */
export function useDebouncedRerender(
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
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const abortRef = useRef<AbortController | null>(null);
  const initialMount = useRef(true);

  const doRerender = useCallback(() => {
    if (!hasSimResult.current) return;

    // Cancel any pending debounce timer
    if (timerRef.current) clearTimeout(timerRef.current);
    // Abort any in-flight request
    if (abortRef.current) abortRef.current.abort();

    setRerendering(true);

    timerRef.current = setTimeout(async () => {
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
        // Only update if this request wasn't superseded
        if (!controller.signal.aborted) {
          setFigureJson(result.figure_json);
        }
      } catch {
        // Ignore aborted / network errors during rerender
      }
      if (!controller.signal.aborted) {
        setRerendering(false);
      }
    }, DEBOUNCE_MS);
  }, [deps.colormap, deps.vmin, deps.vmax, deps.hiddenClasses, hasSimResult, setFigureJson, setRerendering]);

  // Trigger rerender when deps change (skip initial mount)
  useEffect(() => {
    if (initialMount.current) {
      initialMount.current = false;
      return;
    }
    doRerender();
  }, [doRerender]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (timerRef.current) clearTimeout(timerRef.current);
      if (abortRef.current) abortRef.current.abort();
    };
  }, []);
}
