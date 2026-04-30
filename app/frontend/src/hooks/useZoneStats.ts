import { useEffect, useState } from 'react';
import { getZoneStats, ZoneStatsResponse } from '../api';
import { Zone, hashZones } from '../types/zones';

/**
 * Fetch per-zone statistics for the most recent simulation cached in the
 * backend. Re-fetches when:
 *   - `simRunNonce` changes (a new simulation has produced fresh values)
 *   - the set / shape / order of zones changes
 *
 * Returns `stats = null` when there are no zones or no simulation has been run.
 */
export function useZoneStats(
  zones: Zone[],
  simRunNonce: number,
): { stats: ZoneStatsResponse | null; loading: boolean; error: string | null } {
  const [stats, setStats] = useState<ZoneStatsResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const key = `${simRunNonce}|${hashZones(zones)}`;

  useEffect(() => {
    if (zones.length === 0 || simRunNonce === 0) {
      setStats(null);
      setError(null);
      return;
    }
    let cancelled = false;
    setLoading(true);
    setError(null);
    const handle = window.setTimeout(() => {
      getZoneStats(
        zones.map((z) => ({ id: z.id, name: z.name, ring_lonlat: z.ring_lonlat })),
      )
        .then((r) => {
          if (!cancelled) setStats(r);
        })
        .catch((e: Error) => {
          if (cancelled) return;
          // Treat "no simulation run" as no stats rather than an error.
          if (/simulation/i.test(e.message)) {
            setStats(null);
            setError(null);
          } else {
            setError(e.message);
          }
        })
        .finally(() => {
          if (!cancelled) setLoading(false);
        });
    }, 150);
    return () => {
      cancelled = true;
      window.clearTimeout(handle);
    };
  }, [key]);

  return { stats, loading, error };
}
