interface StartScreenProps {
  onStart: () => void;
}

/** Startbildschirm mit Einführung in das Spiel und seine Kernbotschaft. */
export default function StartScreen({ onStart }: StartScreenProps) {
  return (
    <div className="mx-auto flex min-h-[80vh] max-w-3xl flex-col items-center justify-center px-4 py-10 text-center">
      <div className="animate-fade-in">
        <div className="mb-4 text-6xl" aria-hidden="true">
          🏙️
        </div>
        <h1 className="text-3xl font-extrabold tracking-tight text-brand-900 sm:text-4xl">
          Resilienz Stadt
        </h1>
        <p className="mt-1 text-lg font-semibold text-eco-700">Daten, KI und Klima</p>

        <p className="mt-6 text-base leading-relaxed text-slate-600">
          Du übernimmst eine kommunale Entscheidungsstelle. Deine Aufgabe: deine Stadt
          klimaresilienter, datenbasierter und nachhaltiger machen – mit begrenztem Budget und
          wachsenden Erwartungen.
        </p>
        <p className="mt-4 text-base leading-relaxed text-slate-600">
          In mehreren Runden triffst du Entscheidungen zu Geodaten, IoT-Sensorik, KI, digitalen
          Zwillingen, Datenschutz und offenen Standards. Jede Wahl verändert deine Spielwerte.
        </p>

        <div className="mt-6 rounded-2xl border border-brand-100 bg-brand-50/60 p-5 text-left">
          <p className="text-sm font-semibold text-brand-800">Die zentrale Botschaft:</p>
          <p className="mt-1 text-sm leading-relaxed text-slate-700">
            KI allein ist keine Lösung. Erst das fachliche Problem verstehen, dann Datenqualität,
            Governance, Datenschutz und offene Standards sichern – und danach KI sinnvoll einsetzen.
          </p>
        </div>

        <div className="mt-6 grid grid-cols-2 gap-3 text-left text-sm text-slate-600 sm:grid-cols-4">
          {[
            { icon: '🧭', text: 'Aufgabe vor Technologie' },
            { icon: '🗺️', text: 'Geodaten sind das Gold' },
            { icon: '🔓', text: 'Offenheit als Pflicht' },
            { icon: '🤝', text: 'Allianzen schmieden' },
          ].map((item) => (
            <div
              key={item.text}
              className="flex items-center gap-2 rounded-xl border border-slate-200 bg-white px-3 py-2"
            >
              <span aria-hidden="true">{item.icon}</span>
              <span className="font-medium">{item.text}</span>
            </div>
          ))}
        </div>

        <button
          type="button"
          onClick={onStart}
          className="mt-8 rounded-xl bg-brand-600 px-8 py-3.5 text-lg font-bold text-white shadow-lg shadow-brand-600/20 transition hover:bg-brand-700 focus:outline-none focus:ring-4 focus:ring-brand-300"
        >
          Spiel starten
        </button>
        <p className="mt-3 text-xs text-slate-400">
          Läuft vollständig lokal im Browser – kein Login, keine Datenbank.
        </p>
      </div>
    </div>
  );
}
