import type { StatMeta } from '../types';

interface ScoreBarProps {
  meta: StatMeta;
  value: number;
  /** Veränderung seit der letzten Entscheidung (für die Anzeige eines Deltas). */
  delta?: number;
  /** Kompakte Darstellung (z. B. im Dashboard-Grid). */
  compact?: boolean;
}

/** Fortschrittsbalken für einen einzelnen Spielwert. */
export default function ScoreBar({ meta, value, delta, compact = false }: ScoreBarProps) {
  const showDelta = typeof delta === 'number' && delta !== 0;

  return (
    <div className="w-full" title={meta.description}>
      <div className="mb-1 flex items-baseline justify-between gap-2">
        <span className={`font-medium text-slate-700 ${compact ? 'text-xs' : 'text-sm'}`}>
          {compact ? meta.short : meta.label}
        </span>
        <span className="flex items-baseline gap-1">
          {showDelta && (
            <span
              className={`text-xs font-semibold ${delta! > 0 ? 'text-eco-600' : 'text-red-600'}`}
            >
              {delta! > 0 ? `+${delta}` : delta}
            </span>
          )}
          <span className={`font-bold tabular-nums text-slate-800 ${compact ? 'text-xs' : 'text-sm'}`}>
            {value}
          </span>
        </span>
      </div>
      <div
        className="h-2.5 w-full overflow-hidden rounded-full bg-slate-200"
        role="progressbar"
        aria-valuenow={value}
        aria-valuemin={0}
        aria-valuemax={100}
        aria-label={meta.label}
      >
        <div
          className={`h-full rounded-full transition-all duration-500 ease-out ${meta.barColor}`}
          style={{ width: `${value}%` }}
        />
      </div>
    </div>
  );
}
