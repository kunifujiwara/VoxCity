import React from 'react';
import { Zone } from '../types/zones';
import { ZoneStatsResponse } from '../api';

interface Props {
  zones: Zone[];
  stats: ZoneStatsResponse | null;
  loading?: boolean;
}

function fmt(n: number | null | undefined, d = 2): string {
  return n == null || !Number.isFinite(n) ? '—' : n.toFixed(d);
}

function downloadCsv(zones: Zone[], stats: ZoneStatsResponse): void {
  const rows: string[][] = [
    ['zone_id', 'name', 'cell_count', 'valid_count', 'mean', 'min', 'max', 'std'],
  ];
  const byId = new Map(stats.stats.map((s) => [s.zone_id, s]));
  for (const z of zones) {
    const s = byId.get(z.id);
    rows.push([
      z.id,
      JSON.stringify(z.name),
      String(s?.cell_count ?? 0),
      String(s?.valid_count ?? 0),
      String(s?.mean ?? ''),
      String(s?.min ?? ''),
      String(s?.max ?? ''),
      String(s?.std ?? ''),
    ]);
  }
  const csv = rows.map((r) => r.join(',')).join('\n');
  const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = 'zone_stats.csv';
  a.click();
  URL.revokeObjectURL(url);
}

export const ZoneStatsTable: React.FC<Props> = ({ zones, stats, loading }) => {
  if (zones.length === 0) return null;
  const unit = stats?.unit_label ? ` (${stats.unit_label})` : '';
  const byId = new Map((stats?.stats ?? []).map((s) => [s.zone_id, s]));

  return (
    <div className="zone-stats-table">
      <div className="header">
        Zone statistics{unit}
        {loading ? ' …' : ''}
      </div>
      <table>
        <thead>
          <tr>
            <th>Zone</th>
            <th>cells</th>
            <th>mean</th>
            <th>min</th>
            <th>max</th>
            <th>std</th>
          </tr>
        </thead>
        <tbody>
          {zones.map((z) => {
            const s = byId.get(z.id);
            const noData = !s || s.valid_count === 0;
            return (
              <tr key={z.id} className={noData ? 'muted' : ''}>
                <td>
                  <span className="swatch" style={{ background: z.color }} />
                  {z.name}
                </td>
                <td>{s?.cell_count ?? 0}</td>
                <td>{fmt(s?.mean)}</td>
                <td>{fmt(s?.min)}</td>
                <td>{fmt(s?.max)}</td>
                <td>{fmt(s?.std)}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
      <button
        className="btn btn-secondary"
        disabled={!stats}
        onClick={() => stats && downloadCsv(zones, stats)}
        style={{ marginTop: 8 }}
      >
        Export CSV
      </button>
    </div>
  );
};

export default ZoneStatsTable;
