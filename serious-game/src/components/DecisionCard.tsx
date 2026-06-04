import { useState } from 'react';
import type { DecisionCard as DecisionCardType, DecisionOption } from '../types';
import { STAT_META } from '../data/stats';

interface DecisionCardProps {
  card: DecisionCardType;
  /** Aktuelle Kartennummer und Gesamtzahl für die Fortschrittsanzeige. */
  cardIndex: number;
  cardCount: number;
  /** Ob der Lernmodus aktiviert ist. */
  learnMode: boolean;
  /** Callback, wenn eine Option endgültig bestätigt wurde. */
  onConfirm: (option: DecisionOption) => void;
  /** Öffnet das Info-Popup zur Karte. */
  onShowInfo: () => void;
}

/** Liefert ein Kürzel-Label zu einem Spielwert für die Effekt-Vorschau. */
function effectLabel(key: string): string {
  const meta = STAT_META.find((m) => m.key === key);
  return meta ? meta.short : key;
}

/** Einzelne Entscheidungskarte mit Szenario, Optionen und optionalem Lernmodus. */
export default function DecisionCard({
  card,
  cardIndex,
  cardCount,
  learnMode,
  onConfirm,
  onShowInfo,
}: DecisionCardProps) {
  const [selected, setSelected] = useState<DecisionOption | null>(null);

  const handleSelect = (option: DecisionOption) => {
    // Im Lernmodus wird erst die Erklärung gezeigt, danach bestätigt.
    if (learnMode) {
      setSelected(option);
    } else {
      onConfirm(option);
    }
  };

  return (
    <article className="animate-fade-in rounded-2xl border border-slate-200 bg-white p-5 shadow-md sm:p-6">
      <div className="mb-3 flex items-center justify-between gap-3">
        <span className="rounded-full bg-brand-50 px-3 py-1 text-xs font-semibold text-brand-700">
          Entscheidung {cardIndex + 1} / {cardCount}
        </span>
        {card.info && (
          <button
            type="button"
            onClick={onShowInfo}
            className="flex items-center gap-1 text-xs font-semibold text-brand-600 hover:text-brand-800 focus:outline-none focus:ring-2 focus:ring-brand-300 rounded"
          >
            <span aria-hidden="true">💡</span> Wissen
          </button>
        )}
      </div>

      <h2 className="text-xl font-bold text-slate-900">{card.title}</h2>
      <p className="mt-2 text-sm leading-relaxed text-slate-600">{card.scenario}</p>

      <div className="mt-5 space-y-3">
        {card.options.map((option, i) => {
          const isSelected = selected?.id === option.id;
          const letter = String.fromCharCode(65 + i); // A, B, C
          return (
            <div key={option.id}>
              <button
                type="button"
                onClick={() => handleSelect(option)}
                aria-pressed={isSelected}
                className={`group w-full rounded-xl border p-4 text-left transition focus:outline-none focus:ring-2 focus:ring-brand-300 ${
                  isSelected
                    ? 'border-brand-500 bg-brand-50'
                    : 'border-slate-200 bg-white hover:border-brand-300 hover:bg-slate-50'
                }`}
              >
                <div className="flex items-start gap-3">
                  <span
                    className={`mt-0.5 flex h-6 w-6 flex-none items-center justify-center rounded-full text-xs font-bold ${
                      isSelected ? 'bg-brand-600 text-white' : 'bg-slate-100 text-slate-600'
                    }`}
                  >
                    {letter}
                  </span>
                  <div className="min-w-0">
                    <p className="font-semibold text-slate-800">{option.label}</p>
                    <p className="mt-0.5 text-sm text-slate-500">{option.consequence}</p>
                    <div className="mt-2 flex flex-wrap gap-1.5">
                      {Object.entries(option.effects).map(([key, delta]) => (
                        <span
                          key={key}
                          className={`rounded px-1.5 py-0.5 text-[11px] font-semibold ${
                            (delta as number) > 0
                              ? 'bg-eco-100 text-eco-700'
                              : 'bg-red-100 text-red-700'
                          }`}
                        >
                          {effectLabel(key)} {(delta as number) > 0 ? `+${delta}` : delta}
                        </span>
                      ))}
                    </div>
                  </div>
                </div>
              </button>

              {/* Lernmodus: Erklärung nach Auswahl, vor Bestätigung */}
              {learnMode && isSelected && (
                <div className="animate-fade-in mt-2 rounded-xl border border-brand-200 bg-brand-50/70 p-4">
                  <p className="text-sm leading-relaxed text-slate-700">
                    <span className="font-semibold text-brand-800">Lernmodus: </span>
                    {option.learn}
                  </p>
                  <div className="mt-3 flex flex-wrap gap-2">
                    <button
                      type="button"
                      onClick={() => onConfirm(option)}
                      className="rounded-lg bg-brand-600 px-4 py-2 text-sm font-semibold text-white transition hover:bg-brand-700 focus:outline-none focus:ring-2 focus:ring-brand-400"
                    >
                      Entscheidung bestätigen
                    </button>
                    <button
                      type="button"
                      onClick={() => setSelected(null)}
                      className="rounded-lg border border-slate-300 bg-white px-4 py-2 text-sm font-semibold text-slate-600 transition hover:bg-slate-50"
                    >
                      Andere Option wählen
                    </button>
                  </div>
                </div>
              )}
            </div>
          );
        })}
      </div>

      {!learnMode && (
        <p className="mt-4 text-center text-xs text-slate-400">
          Tipp: Aktiviere oben den Lernmodus, um nach jeder Wahl eine kurze Erklärung zu erhalten.
        </p>
      )}
    </article>
  );
}
