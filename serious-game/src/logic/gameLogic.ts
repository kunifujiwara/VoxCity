import type { DecisionOption, ResultTier, Stats } from '../types';
import { RESULT_TIERS } from '../data/resultTiers';
import { STAT_META } from '../data/stats';

/** Begrenzt einen Wert auf den gültigen Bereich 0–100. */
export function clamp(value: number): number {
  return Math.max(0, Math.min(100, value));
}

/**
 * Wendet die Effekte einer gewählten Option auf die Spielwerte an.
 * Werte bleiben dabei immer im Bereich 0–100.
 */
export function applyOption(stats: Stats, option: DecisionOption): Stats {
  const next: Stats = { ...stats };
  for (const meta of STAT_META) {
    const delta = option.effects[meta.key];
    if (typeof delta === 'number') {
      next[meta.key] = clamp(next[meta.key] + delta);
    }
  }
  return next;
}

/**
 * Berechnet den Gesamtscore (0–100) als Durchschnitt aller Spielwerte.
 * Budget zählt etwas geringer, da es ein Mittel und kein Ziel ist.
 */
export function calculateScore(stats: Stats): number {
  const weights: Partial<Record<keyof Stats, number>> = {
    budget: 0.5,
    klimaresilienz: 1.3,
    datenqualitaet: 1.2,
  };

  let weightedSum = 0;
  let weightTotal = 0;
  for (const meta of STAT_META) {
    const weight = weights[meta.key] ?? 1;
    weightedSum += stats[meta.key] * weight;
    weightTotal += weight;
  }
  return Math.round(weightedSum / weightTotal);
}

/**
 * Bestimmt die Ergebnis-Stufe abhängig von Gesamtscore und Profil der Spielwerte.
 * Die Reihenfolge der Prüfungen kodiert die zentrale Botschaft des Spiels:
 * KI ohne Datenqualität und Organisation führt nicht zum Ziel.
 */
export function determineResultTier(stats: Stats, score: number): ResultTier {
  const {
    datenqualitaet,
    verwaltungsfaehigkeit,
    openSourceReife,
    technischeNachhaltigkeit,
    budget,
  } = stats;

  const organisation = (verwaltungsfaehigkeit + datenqualitaet) / 2;
  const offenheit = (openSourceReife + technischeNachhaltigkeit) / 2;

  // KI gekauft, Problem nicht gelöst: Budget verbraucht, aber Datenqualität schlecht.
  if (datenqualitaet < 40 && budget < 35) {
    return RESULT_TIERS.kiGekauft;
  }

  // Datenchaos und Vendor Lock-in: weder Offenheit noch saubere Daten.
  if (offenheit < 40 && datenqualitaet < 45) {
    return RESULT_TIERS.datenchaos;
  }

  // Technisch aktiv, aber organisatorisch schwach: Technik da, Organisation fehlt.
  if (offenheit >= 55 && organisation < 45) {
    return RESULT_TIERS.technisch;
  }

  // Datengetriebene Vorreiterstadt: rundum stark.
  if (score >= 68 && datenqualitaet >= 60 && offenheit >= 55) {
    return RESULT_TIERS.vorreiter;
  }

  // Solide Smart-City-Verwaltung: ordentliches Mittelfeld.
  if (score >= 52) {
    return RESULT_TIERS.solide;
  }

  // Fällt der Score insgesamt zu niedrig aus, ist es Datenchaos.
  return RESULT_TIERS.datenchaos;
}

/** Bewertet einen einzelnen Wert qualitativ für die UI-Darstellung. */
export function rateValue(value: number): 'low' | 'mid' | 'high' {
  if (value < 35) return 'low';
  if (value < 65) return 'mid';
  return 'high';
}
