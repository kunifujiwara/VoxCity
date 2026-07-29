import { en } from './en';
import { ja } from './ja';

export const catalogs = { en, ja } as const;
export type Lang = keyof typeof catalogs; // 'en' | 'ja'
