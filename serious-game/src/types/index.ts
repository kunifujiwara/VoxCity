// Zentrale Typdefinitionen für das Serious Game "Resilienz Stadt: Daten, KI und Klima".

/** Schlüssel aller neun Spielwerte, die im Dashboard angezeigt werden. */
export type StatKey =
  | 'klimaresilienz'
  | 'budget'
  | 'datenqualitaet'
  | 'buergerzufriedenheit'
  | 'datenschutzVertrauen'
  | 'verwaltungsfaehigkeit'
  | 'openSourceReife'
  | 'interkommunaleKooperation'
  | 'technischeNachhaltigkeit';

/** Aktueller Zustand aller Spielwerte (jeweils 0–100). */
export type Stats = Record<StatKey, number>;

/** Teilweise Veränderung der Spielwerte durch eine Option (Delta-Werte). */
export type StatEffects = Partial<Record<StatKey, number>>;

/** Metadaten zu einem einzelnen Spielwert (Anzeigename, Farbe, Beschreibung). */
export interface StatMeta {
  key: StatKey;
  label: string;
  short: string;
  description: string;
  /** Tailwind-Klassen für den Fortschrittsbalken. */
  barColor: string;
}

/** Eine einzelne Entscheidungsoption innerhalb einer Karte. */
export interface DecisionOption {
  id: string;
  label: string;
  /** Kurzbeschreibung der direkten Folge der Entscheidung. */
  consequence: string;
  /** Veränderung der Spielwerte. */
  effects: StatEffects;
  /** Lernmodus-Erklärung: Warum ist die Entscheidung gut oder riskant? */
  learn: string;
}

/** Eine Entscheidungskarte mit Szenario und mehreren Optionen. */
export interface DecisionCard {
  id: string;
  title: string;
  scenario: string;
  /** Optionaler Lerninhalt, der als Info-Popup zur Karte angezeigt werden kann. */
  info?: InfoContent;
  options: DecisionOption[];
}

/** Lerninhalt für Info-Popups (Wissensvermittlung). */
export interface InfoContent {
  title: string;
  body: string;
}

/** Eine Mission besteht aus mehreren Entscheidungskarten. */
export interface Mission {
  id: string;
  title: string;
  subtitle: string;
  /** Reale Referenzstadt / Use Case aus dem Klimaresilienz-Kontext. */
  reference: string;
  description: string;
  /** Emoji-Icon zur schnellen visuellen Unterscheidung. */
  icon: string;
  cards: DecisionCard[];
}

/** Ergebnis-Stufe am Spielende abhängig vom erreichten Score. */
export interface ResultTier {
  id: string;
  title: string;
  explanation: string;
  /** Tailwind-Klassen für die farbliche Darstellung der Ergebnisstufe. */
  accent: string;
}

/** Phasen des Spiels für die Navigation. */
export type GamePhase = 'start' | 'missionSelect' | 'playing' | 'result';
