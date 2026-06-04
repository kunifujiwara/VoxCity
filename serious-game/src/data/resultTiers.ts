import type { ResultTier } from '../types';

/**
 * Die fünf Ergebnis-Stufen am Spielende.
 * Welche Stufe erreicht wird, entscheidet die Logik in gameLogic.ts
 * anhand von Gesamtscore und ausgewählten Spielwerten.
 */
export const RESULT_TIERS: Record<string, ResultTier> = {
  vorreiter: {
    id: 'vorreiter',
    title: 'Datengetriebene Vorreiterstadt',
    explanation:
      'Hervorragend! Du hast erst das fachliche Problem verstanden, dann Datenqualität, Governance und Datenschutz gesichert – und KI gezielt und offen eingesetzt. Deine Stadt ist klimaresilient, souverän und ein Vorbild für andere Kommunen.',
    accent: 'from-eco-600 to-brand-600',
  },
  solide: {
    id: 'solide',
    title: 'Solide Smart-City-Verwaltung',
    explanation:
      'Gut gemacht. Deine Stadt arbeitet datenbasiert und verantwortungsvoll. An einzelnen Stellen – etwa bei Offenheit, Kooperation oder dauerhaftem Betrieb – ist noch Luft nach oben, aber das Fundament stimmt.',
    accent: 'from-brand-600 to-brand-800',
  },
  technisch: {
    id: 'technisch',
    title: 'Technisch aktiv, aber organisatorisch schwach',
    explanation:
      'Du hast viel Technik eingeführt, aber die Organisation hinkt hinterher: Zuständigkeiten, Datenkompetenz oder Datenqualität sind nicht mitgewachsen. Technik allein trägt nicht – die Verwaltung muss sie tragen können.',
    accent: 'from-amber-500 to-orange-600',
  },
  kiGekauft: {
    id: 'kiGekauft',
    title: 'KI gekauft, Problem nicht gelöst',
    explanation:
      'Es wurde in Technik und KI investiert, aber ohne saubere Datengrundlage. Die Ergebnisse bleiben unsicher und das Budget ist aufgebraucht. Die Lehre: Erst Aufgabe und Datenqualität klären, dann KI – nicht umgekehrt.',
    accent: 'from-orange-600 to-red-600',
  },
  datenchaos: {
    id: 'datenchaos',
    title: 'Datenchaos und Vendor Lock-in',
    explanation:
      'Geschlossene Systeme, Datensilos und schlechte Datenqualität haben die Stadt in Abhängigkeit und Unübersichtlichkeit geführt. Ohne offene Standards und Governance fehlt die Souveränität. Zeit für einen Neustart mit Offenheit als Pflicht.',
    accent: 'from-red-600 to-rose-700',
  },
};
