import { useCallback, useSyncExternalStore } from 'react';

// Mirrors the media blocks in studio.css.
export const PHONE_QUERY = '(max-width: 760px)';
export const COMPACT_QUERY = '(max-width: 1080px)';

/** True while the media query matches; re-renders when it flips. */
export function useMediaQuery(query: string): boolean {
  const subscribe = useCallback(
    (notify: () => void) => {
      const list = window.matchMedia(query);
      list.addEventListener('change', notify);
      return () => list.removeEventListener('change', notify);
    },
    [query],
  );
  return useSyncExternalStore(
    subscribe,
    () => window.matchMedia(query).matches,
    () => false,
  );
}
