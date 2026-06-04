import type { Stats } from '../types';
import { STAT_META } from '../data/stats';
import { calculateScore, determineResultTier, rateValue } from '../logic/gameLogic';
import ScoreBar from './ScoreBar';

interface ResultScreenProps {
  stats: Stats;
  completedCount: number;
  onRestart: () => void;
  onBackToMissions: () => void;
}

const RATING_LABEL: Record<'low' | 'mid' | 'high', string> = {
  low: 'kritisch',
  mid: 'solide',
  high: 'stark',
};

const RATING_COLOR: Record<'low' | 'mid' | 'high', string> = {
  low: 'text-red-600',
  mid: 'text-amber-600',
  high: 'text-eco-600',
};

/** Endbildschirm mit Resilienz-Score, Ergebnis-Stufe und Werteübersicht. */
export default function ResultScreen({
  stats,
  completedCount,
  onRestart,
  onBackToMissions,
}: ResultScreenProps) {
  const score = calculateScore(stats);
  const tier = determineResultTier(stats, score);

  // Stärkste und schwächste Werte für ein kurzes Feedback.
  const sorted = [...STAT_META].sort((a, b) => stats[b.key] - stats[a.key]);
  const strongest = sorted[0];
  const weakest = sorted[sorted.length - 1];

  return (
    <div className="mx-auto max-w-3xl px-4 py-8">
      <div className={`animate-scale-in rounded-3xl bg-gradient-to-br ${tier.accent} p-1 shadow-xl`}>
        <div className="rounded-[22px] bg-white p-6 sm:p-8">
          <p className="text-center text-sm font-semibold uppercase tracking-wide text-slate-400">
            Dein Resilienz-Score
          </p>

          {/* Score-Ring */}
          <div className="mx-auto my-4 flex h-40 w-40 items-center justify-center">
            <div
              className="flex h-40 w-40 items-center justify-center rounded-full"
              style={{
                background: `conic-gradient(currentColor ${score * 3.6}deg, #e2e8f0 0deg)`,
                color:
                  score >= 68 ? '#16a34a' : score >= 52 ? '#2563eb' : score >= 40 ? '#d97706' : '#dc2626',
              }}
            >
              <div className="flex h-32 w-32 flex-col items-center justify-center rounded-full bg-white">
                <span className="text-4xl font-extrabold text-slate-800 tabular-nums">{score}</span>
                <span className="text-xs font-medium text-slate-400">von 100</span>
              </div>
            </div>
          </div>

          <h1 className="text-center text-2xl font-extrabold text-slate-900">{tier.title}</h1>
          <p className="mt-3 text-center text-sm leading-relaxed text-slate-600">
            {tier.explanation}
          </p>

          <div className="mt-5 grid grid-cols-1 gap-3 sm:grid-cols-2">
            <div className="rounded-xl border border-eco-200 bg-eco-50/60 p-3 text-sm">
              <span className="font-semibold text-eco-700">Stärkster Wert: </span>
              <span className="text-slate-700">
                {strongest.label} ({stats[strongest.key]})
              </span>
            </div>
            <div className="rounded-xl border border-red-200 bg-red-50/60 p-3 text-sm">
              <span className="font-semibold text-red-700">Schwächster Wert: </span>
              <span className="text-slate-700">
                {weakest.label} ({stats[weakest.key]})
              </span>
            </div>
          </div>
        </div>
      </div>

      {/* Detail-Übersicht aller Werte */}
      <section className="mt-6 rounded-2xl border border-slate-200 bg-white p-5 shadow-sm sm:p-6">
        <h2 className="mb-4 text-lg font-bold text-slate-800">Werteübersicht</h2>
        <div className="grid grid-cols-1 gap-x-8 gap-y-4 sm:grid-cols-2">
          {STAT_META.map((meta) => {
            const rating = rateValue(stats[meta.key]);
            return (
              <div key={meta.key}>
                <div className="mb-1 flex items-center justify-between">
                  <span className="text-sm font-medium text-slate-700">{meta.label}</span>
                  <span className={`text-xs font-semibold ${RATING_COLOR[rating]}`}>
                    {RATING_LABEL[rating]}
                  </span>
                </div>
                <ScoreBar meta={meta} value={stats[meta.key]} />
              </div>
            );
          })}
        </div>
        <p className="mt-5 text-xs text-slate-400">
          Basierend auf {completedCount} gespielten{' '}
          {completedCount === 1 ? 'Mission' : 'Missionen'}.
        </p>
      </section>

      <div className="mt-6 flex flex-col justify-center gap-3 sm:flex-row">
        <button
          type="button"
          onClick={onBackToMissions}
          className="rounded-xl bg-brand-600 px-6 py-3 font-bold text-white transition hover:bg-brand-700 focus:outline-none focus:ring-4 focus:ring-brand-300"
        >
          Weitere Mission spielen
        </button>
        <button
          type="button"
          onClick={onRestart}
          className="rounded-xl border border-slate-300 bg-white px-6 py-3 font-bold text-slate-700 transition hover:bg-slate-50 focus:outline-none focus:ring-4 focus:ring-slate-200"
        >
          Neues Spiel starten
        </button>
      </div>
    </div>
  );
}
