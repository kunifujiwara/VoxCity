import type { StatKey, StatMeta, Stats } from '../types';

/**
 * Metadaten zu allen neun Spielwerten.
 * Reihenfolge bestimmt die Anzeige im Dashboard.
 */
export const STAT_META: StatMeta[] = [
  {
    key: 'klimaresilienz',
    label: 'Klimaresilienz',
    short: 'Klima',
    description: 'Wie gut ist die Stadt gegen Hitze, Starkregen und Klimafolgen gewappnet?',
    barColor: 'bg-eco-500',
  },
  {
    key: 'budget',
    label: 'Budget',
    short: 'Budget',
    description: 'Verfügbare finanzielle Mittel der Kommune. Geht es zur Neige, wird jede Maßnahme schwieriger.',
    barColor: 'bg-amber-500',
  },
  {
    key: 'datenqualitaet',
    label: 'Datenqualität',
    short: 'Daten',
    description: 'Vollständigkeit und Verlässlichkeit der Geo- und Sensordaten – die Basis jeder KI.',
    barColor: 'bg-brand-500',
  },
  {
    key: 'buergerzufriedenheit',
    label: 'Bürgerzufriedenheit',
    short: 'Bürger',
    description: 'Wie zufrieden sind die Menschen in der Stadt mit deinen Entscheidungen?',
    barColor: 'bg-teal-500',
  },
  {
    key: 'datenschutzVertrauen',
    label: 'Datenschutz-Vertrauen',
    short: 'Datenschutz',
    description: 'Vertrauen der Bürgerinnen in den verantwortungsvollen Umgang mit ihren Daten.',
    barColor: 'bg-indigo-500',
  },
  {
    key: 'verwaltungsfaehigkeit',
    label: 'Verwaltungsfähigkeit',
    short: 'Verwaltung',
    description: 'Organisatorische Fähigkeit der Verwaltung, Projekte zu steuern und Zuständigkeiten zu klären.',
    barColor: 'bg-slate-500',
  },
  {
    key: 'openSourceReife',
    label: 'Open-Source-Reife',
    short: 'Open Source',
    description: 'Grad an Offenheit, offenen Standards und Unabhängigkeit von einzelnen Anbietern.',
    barColor: 'bg-green-600',
  },
  {
    key: 'interkommunaleKooperation',
    label: 'Interkommunale Kooperation',
    short: 'Kooperation',
    description: 'Wie stark arbeitet die Stadt mit anderen Kommunen zusammen und teilt Lösungen?',
    barColor: 'bg-cyan-600',
  },
  {
    key: 'technischeNachhaltigkeit',
    label: 'Technische Nachhaltigkeit',
    short: 'Nachhaltigkeit',
    description: 'Dauerhafter Betrieb, Wartbarkeit und Zukunftsfähigkeit der eingesetzten Technik.',
    barColor: 'bg-emerald-600',
  },
];

/** Startwert für jeden Spielwert. */
export const START_VALUE = 50;

/** Erzeugt den Anfangszustand: alle Werte starten bei 50. */
export function createInitialStats(): Stats {
  const stats = {} as Stats;
  for (const meta of STAT_META) {
    stats[meta.key] = START_VALUE;
  }
  return stats;
}

/** Liefert die Metadaten zu einem bestimmten Spielwert. */
export function getStatMeta(key: StatKey): StatMeta {
  const meta = STAT_META.find((m) => m.key === key);
  if (!meta) {
    throw new Error(`Unbekannter Spielwert: ${key}`);
  }
  return meta;
}
