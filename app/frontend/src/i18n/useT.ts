import { useCallback } from 'react';
import { useLanguage } from './LanguageContext';
import { translate, type TranslationKey } from './translate';

export function useT() {
  const { lang } = useLanguage();
  return useCallback(
    (key: TranslationKey, vars?: Record<string, string | number>) => translate(lang, key, vars),
    [lang],
  );
}
