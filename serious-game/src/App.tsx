import { useState } from 'react';
import type { DecisionOption, InfoContent, Mission, Stats } from './types';
import { MISSIONS, createInitialStats } from './data';
import { STAT_META } from './data/stats';
import { applyOption } from './logic/gameLogic';
import StartScreen from './components/StartScreen';
import MissionSelect from './components/MissionSelect';
import GameDashboard from './components/GameDashboard';
import DecisionCard from './components/DecisionCard';
import ResultScreen from './components/ResultScreen';
import InfoModal from './components/InfoModal';

type Phase = 'start' | 'missionSelect' | 'playing' | 'missionSummary' | 'result';

/** Berechnet die Differenz zweier Wertesätze (nur veränderte Werte). */
function diffStats(before: Stats, after: Stats): Partial<Record<keyof Stats, number>> {
  const deltas: Partial<Record<keyof Stats, number>> = {};
  for (const meta of STAT_META) {
    const d = after[meta.key] - before[meta.key];
    if (d !== 0) deltas[meta.key] = d;
  }
  return deltas;
}

export default function App() {
  const [phase, setPhase] = useState<Phase>('start');
  const [stats, setStats] = useState<Stats>(createInitialStats);
  const [learnMode, setLearnMode] = useState(true);

  const [activeMission, setActiveMission] = useState<Mission | null>(null);
  const [cardIndex, setCardIndex] = useState(0);
  const [completedMissionIds, setCompletedMissionIds] = useState<string[]>([]);

  // Werte zu Beginn der aktuellen Mission – für die Runden-Auswertung.
  const [missionStartStats, setMissionStartStats] = useState<Stats>(createInitialStats);
  // Delta der letzten Entscheidung – für die kurze Anzeige im Dashboard.
  const [lastDelta, setLastDelta] = useState<Partial<Record<keyof Stats, number>>>({});

  const [infoContent, setInfoContent] = useState<InfoContent | null>(null);

  const startNewGame = () => {
    setStats(createInitialStats());
    setCompletedMissionIds([]);
    setActiveMission(null);
    setCardIndex(0);
    setLastDelta({});
    setPhase('missionSelect');
  };

  const beginMission = (mission: Mission) => {
    setActiveMission(mission);
    setCardIndex(0);
    setMissionStartStats(stats);
    setLastDelta({});
    setPhase('playing');
  };

  const handleConfirm = (option: DecisionOption) => {
    if (!activeMission) return;
    const next = applyOption(stats, option);
    setLastDelta(diffStats(stats, next));
    setStats(next);

    const isLastCard = cardIndex >= activeMission.cards.length - 1;
    if (isLastCard) {
      setCompletedMissionIds((prev) =>
        prev.includes(activeMission.id) ? prev : [...prev, activeMission.id],
      );
      setPhase('missionSummary');
    } else {
      setCardIndex((i) => i + 1);
    }
  };

  const currentCard = activeMission?.cards[cardIndex];
  const missionDeltas = diffStats(missionStartStats, stats);

  return (
    <div className="min-h-screen">
      {/* Kopfzeile mit Spieltitel und Lernmodus-Schalter */}
      {phase !== 'start' && (
        <header className="sticky top-0 z-40 border-b border-slate-200 bg-white/85 backdrop-blur">
          <div className="mx-auto flex max-w-5xl items-center justify-between gap-3 px-4 py-3">
            <button
              type="button"
              onClick={() => setPhase('missionSelect')}
              className="flex items-center gap-2 text-left focus:outline-none"
            >
              <span className="text-xl" aria-hidden="true">
                🏙️
              </span>
              <span className="font-bold text-brand-900">
                Resilienz Stadt
                <span className="ml-1 hidden font-normal text-slate-400 sm:inline">
                  · Daten, KI und Klima
                </span>
              </span>
            </button>

            <label className="flex cursor-pointer select-none items-center gap-2 text-sm text-slate-600">
              <span className="hidden sm:inline">Lernmodus</span>
              <button
                type="button"
                role="switch"
                aria-checked={learnMode}
                aria-label="Lernmodus umschalten"
                onClick={() => setLearnMode((v) => !v)}
                className={`relative h-6 w-11 rounded-full transition ${
                  learnMode ? 'bg-eco-500' : 'bg-slate-300'
                }`}
              >
                <span
                  className={`absolute top-0.5 h-5 w-5 rounded-full bg-white shadow transition-all ${
                    learnMode ? 'left-[22px]' : 'left-0.5'
                  }`}
                />
              </button>
            </label>
          </div>
        </header>
      )}

      <main>
        {phase === 'start' && <StartScreen onStart={startNewGame} />}

        {phase === 'missionSelect' && (
          <MissionSelect
            missions={MISSIONS}
            completedMissionIds={completedMissionIds}
            onSelect={beginMission}
            onFinish={() => setPhase('result')}
          />
        )}

        {phase === 'playing' && activeMission && currentCard && (
          <div className="mx-auto max-w-5xl space-y-5 px-4 py-6">
            <div>
              <div className="mb-2 flex items-center justify-between gap-3">
                <h1 className="flex items-center gap-2 text-lg font-bold text-slate-800">
                  <span aria-hidden="true">{activeMission.icon}</span>
                  {activeMission.title}
                </h1>
                <span className="text-sm font-medium text-slate-500">
                  {cardIndex + 1} / {activeMission.cards.length}
                </span>
              </div>
              {/* Fortschrittsanzeige pro Mission */}
              <div className="h-2 w-full overflow-hidden rounded-full bg-slate-200">
                <div
                  className="h-full rounded-full bg-brand-500 transition-all duration-500"
                  style={{
                    width: `${(cardIndex / activeMission.cards.length) * 100}%`,
                  }}
                />
              </div>
            </div>

            <GameDashboard stats={stats} deltas={lastDelta} />

            <DecisionCard
              key={currentCard.id}
              card={currentCard}
              cardIndex={cardIndex}
              cardCount={activeMission.cards.length}
              learnMode={learnMode}
              onConfirm={handleConfirm}
              onShowInfo={() => currentCard.info && setInfoContent(currentCard.info)}
            />
          </div>
        )}

        {phase === 'missionSummary' && activeMission && (
          <div className="mx-auto max-w-2xl px-4 py-10">
            <div className="animate-scale-in rounded-3xl border border-slate-200 bg-white p-6 text-center shadow-lg sm:p-8">
              <div className="text-5xl" aria-hidden="true">
                ✅
              </div>
              <h1 className="mt-3 text-2xl font-extrabold text-slate-900">Runde abgeschlossen</h1>
              <p className="mt-1 font-semibold text-brand-600">{activeMission.title}</p>
              <p className="mt-3 text-sm text-slate-600">
                So haben sich deine Spielwerte in dieser Mission verändert:
              </p>

              <div className="mt-5 grid grid-cols-1 gap-2 text-left sm:grid-cols-2">
                {STAT_META.filter((m) => missionDeltas[m.key] !== undefined).length === 0 && (
                  <p className="col-span-full text-sm text-slate-400">
                    Keine Werte verändert.
                  </p>
                )}
                {STAT_META.map((meta) => {
                  const d = missionDeltas[meta.key];
                  if (d === undefined) return null;
                  return (
                    <div
                      key={meta.key}
                      className="flex items-center justify-between rounded-lg border border-slate-100 bg-slate-50 px-3 py-2 text-sm"
                    >
                      <span className="text-slate-700">{meta.label}</span>
                      <span
                        className={`font-bold tabular-nums ${
                          d > 0 ? 'text-eco-600' : 'text-red-600'
                        }`}
                      >
                        {d > 0 ? `+${d}` : d} → {stats[meta.key]}
                      </span>
                    </div>
                  );
                })}
              </div>

              <div className="mt-7 flex flex-col justify-center gap-3 sm:flex-row">
                <button
                  type="button"
                  onClick={() => setPhase('missionSelect')}
                  className="rounded-xl bg-brand-600 px-6 py-3 font-bold text-white transition hover:bg-brand-700 focus:outline-none focus:ring-4 focus:ring-brand-300"
                >
                  Zur Missionsauswahl
                </button>
                <button
                  type="button"
                  onClick={() => setPhase('result')}
                  className="rounded-xl border border-slate-300 bg-white px-6 py-3 font-bold text-slate-700 transition hover:bg-slate-50 focus:outline-none focus:ring-4 focus:ring-slate-200"
                >
                  Endauswertung
                </button>
              </div>
            </div>
          </div>
        )}

        {phase === 'result' && (
          <ResultScreen
            stats={stats}
            completedCount={completedMissionIds.length}
            onRestart={startNewGame}
            onBackToMissions={() => setPhase('missionSelect')}
          />
        )}
      </main>

      <footer className="mx-auto max-w-5xl px-4 py-8 text-center text-xs text-slate-400">
        Resilienz Stadt: Daten, KI und Klima · Ein Serious Game zur kommunalen Datenkompetenz ·
        Läuft lokal im Browser
      </footer>

      {infoContent && <InfoModal content={infoContent} onClose={() => setInfoContent(null)} />}
    </div>
  );
}
