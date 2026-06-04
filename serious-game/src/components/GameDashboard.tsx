import type { Stats } from '../types';
import { STAT_META } from '../data/stats';
import ScoreBar from './ScoreBar';

interface GameDashboardProps {
  stats: Stats;
  /** Deltas der letzten Entscheidung je Spielwert. */
  deltas?: Partial<Record<keyof Stats, number>>;
}

/** Dashboard mit allen neun Spielwerten als Fortschrittsbalken. */
export default function GameDashboard({ stats, deltas }: GameDashboardProps) {
  return (
    <section
      aria-label="Spielwerte"
      className="rounded-2xl border border-slate-200 bg-white/80 p-4 shadow-sm backdrop-blur sm:p-5"
    >
      <h2 className="mb-3 flex items-center gap-2 text-sm font-bold uppercase tracking-wide text-slate-500">
        <span aria-hidden="true">📊</span> Stadt-Dashboard
      </h2>
      <div className="grid grid-cols-1 gap-x-6 gap-y-3 sm:grid-cols-2 lg:grid-cols-3">
        {STAT_META.map((meta) => (
          <ScoreBar
            key={meta.key}
            meta={meta}
            value={stats[meta.key]}
            delta={deltas?.[meta.key]}
            compact
          />
        ))}
      </div>
    </section>
  );
}
