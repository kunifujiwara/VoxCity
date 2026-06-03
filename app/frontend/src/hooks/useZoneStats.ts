import { useEffect, useState } from 'react';
import { getZoneStats, ZoneStatsResponse, ZoneStat } from '../api';
import { Zone, hashZones } from '../types/zones';

/**
 * Aggregate per-ring stats returned by the backend into one stat per logical
 * zone group (zones sharing the same `groupId`).
 */
export function aggregateByGroup(zones: Zone[], response: ZoneStatsResponse): ZoneStatsResponse {
  // Map each ring id → its group key
  const ringToGroup = new Map<string, string>();
  for (const z of zones) {
    ringToGroup.set(z.id, z.groupId ?? z.id);
  }

  type Acc = ZoneStat & { _wMeanSum: number; _wSum: number };
  const groups = new Map<string, Acc>();

  for (const stat of response.stats) {
    const key = ringToGroup.get(stat.zone_id) ?? stat.zone_id;
    const prev = groups.get(key);
    if (!prev) {
      groups.set(key, {
        zone_id: key,
        cell_count: stat.cell_count,
        valid_count: stat.valid_count,
        mean: stat.mean,
        min: stat.min,
        max: stat.max,
        std: null,
        _wMeanSum: stat.mean != null ? stat.mean * stat.valid_count : 0,
        _wSum: stat.mean != null ? stat.valid_count : 0,
      });
    } else {
      prev.cell_count += stat.cell_count;
      prev.valid_count += stat.valid_count;
      if (stat.mean != null) {
        prev._wMeanSum += stat.mean * stat.valid_count;
        prev._wSum += stat.valid_count;
      }
      if (stat.min != null && (prev.min == null || stat.min < prev.min)) prev.min = stat.min;
      if (stat.max != null && (prev.max == null || stat.max > prev.max)) prev.max = stat.max;
    }
  }

  const stats: ZoneStat[] = [];
  for (const acc of groups.values()) {
    stats.push({
      zone_id: acc.zone_id,
      cell_count: acc.cell_count,
      valid_count: acc.valid_count,
      mean: acc._wSum > 0 ? acc._wMeanSum / acc._wSum : null,
      min: acc.min,
      max: acc.max,
      std: null,
    });
  }
  return { ...response, stats };
}

/**
 * Fetch per-zone statistics for the specified simulation type cached in the
 * backend. Re-fetches when:
 *   - `simRunNonce` changes (a new simulation has produced fresh values)
 *   - the set / shape / order of zones changes
 *   - `simType` changes
 *
 * Stats are aggregated by `groupId` so multi-ring logical zones produce a
 * single row.  Returns `stats = null` when there are no zones or the
 * requested simulation has not been run yet.
 */
export function useZoneStats(
  zones: Zone[],
  simType: 'solar' | 'view' | 'landmark',
  simRunNonce: number,
): { stats: ZoneStatsResponse | null; loading: boolean; error: string | null } {
  const [stats, setStats] = useState<ZoneStatsResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const key = `${simType}|${simRunNonce}|${hashZones(zones)}`;

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
      getZoneStats(zones, simType)
        .then((r) => {
          if (!cancelled) setStats(aggregateByGroup(zones, r));
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
