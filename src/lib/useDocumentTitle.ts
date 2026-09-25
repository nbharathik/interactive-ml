import { useEffect } from 'react';

const DEFAULT_TITLE = 'Interactive ML · Interactive Machine Learning';

export function useDocumentTitle(title: string) {
  useEffect(() => {
    document.title = title + ' · Interactive ML';
    return () => {
      document.title = DEFAULT_TITLE;
    };
  }, [title]);
}
