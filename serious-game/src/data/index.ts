import type { Mission } from '../types';
import missionsData from './missions.json';

/**
 * Lokales JSON-Datenmodell für alle Missionen, Karten, Optionen und Effekte.
 * Die JSON-Datei wird beim Build typisiert eingebunden.
 */
export const MISSIONS = missionsData as Mission[];

export { STAT_META, START_VALUE, createInitialStats, getStatMeta } from './stats';
export { RESULT_TIERS } from './resultTiers';
