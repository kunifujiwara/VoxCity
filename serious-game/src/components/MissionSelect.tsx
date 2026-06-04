import type { Mission } from '../types';

interface MissionSelectProps {
  missions: Mission[];
  /** IDs bereits abgeschlossener Missionen. */
  completedMissionIds: string[];
  onSelect: (mission: Mission) => void;
  /** Öffnet die Endauswertung (sobald mindestens eine Mission gespielt wurde). */
  onFinish: () => void;
}

/** Auswahlbildschirm für die fünf Missionen. */
export default function MissionSelect({
  missions,
  completedMissionIds,
  onSelect,
  onFinish,
}: MissionSelectProps) {
  const completedCount = completedMissionIds.length;

  return (
    <div className="mx-auto max-w-5xl px-4 py-8">
      <div className="mb-6 text-center">
        <h1 className="text-2xl font-extrabold text-brand-900 sm:text-3xl">Wähle eine Mission</h1>
        <p className="mt-2 text-slate-600">
          Jede Mission besteht aus mehreren Entscheidungskarten. {completedCount} von{' '}
          {missions.length} abgeschlossen.
        </p>
      </div>

      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3">
        {missions.map((mission) => {
          const done = completedMissionIds.includes(mission.id);
          return (
            <button
              key={mission.id}
              type="button"
              onClick={() => onSelect(mission)}
              className={`group relative flex flex-col rounded-2xl border bg-white p-5 text-left shadow-sm transition hover:-translate-y-0.5 hover:shadow-md focus:outline-none focus:ring-2 focus:ring-brand-300 ${
                done ? 'border-eco-300' : 'border-slate-200'
              }`}
            >
              {done && (
                <span className="absolute right-3 top-3 rounded-full bg-eco-100 px-2 py-0.5 text-xs font-semibold text-eco-700">
                  ✓ gespielt
                </span>
              )}
              <span className="text-4xl" aria-hidden="true">
                {mission.icon}
              </span>
              <h2 className="mt-3 text-lg font-bold text-slate-900">{mission.title}</h2>
              <p className="mt-1 text-sm font-medium text-brand-600">{mission.subtitle}</p>
              <p className="mt-2 flex-1 text-sm leading-relaxed text-slate-600">
                {mission.description}
              </p>
              <p className="mt-3 border-t border-slate-100 pt-3 text-xs italic text-slate-400">
                {mission.reference}
              </p>
              <span className="mt-3 inline-flex items-center gap-1 text-sm font-semibold text-brand-700">
                {done ? 'Erneut spielen' : 'Mission starten'} →
              </span>
            </button>
          );
        })}
      </div>

      <div className="mt-8 text-center">
        <button
          type="button"
          onClick={onFinish}
          disabled={completedCount === 0}
          className="rounded-xl bg-eco-600 px-6 py-3 font-bold text-white shadow-lg shadow-eco-600/20 transition hover:bg-eco-700 focus:outline-none focus:ring-4 focus:ring-eco-300 disabled:cursor-not-allowed disabled:opacity-40"
        >
          Endauswertung anzeigen
        </button>
        {completedCount === 0 && (
          <p className="mt-2 text-xs text-slate-400">
            Spiele zuerst mindestens eine Mission, um deinen Resilienz-Score zu sehen.
          </p>
        )}
      </div>
    </div>
  );
}
