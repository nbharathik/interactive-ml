/** A copy of generated data that the reader can edit; a new source replaces the edits in the same render. */

import { useCallback, useRef, useState } from 'react';
import type { Dispatch, SetStateAction } from 'react';

export function useEditable<T>(source: T): [T, Dispatch<SetStateAction<T>>] {
  const [edit, setEdit] = useState<{ source: T; value: T } | null>(null);
  const sourceRef = useRef(source);
  sourceRef.current = source;

  const set = useCallback((action: SetStateAction<T>) => {
    setEdit((prev) => {
      const from = sourceRef.current;
      const current = prev && prev.source === from ? prev.value : from;
      const value = typeof action === 'function' ? (action as (prev: T) => T)(current) : action;
      return { source: from, value };
    });
  }, []);

  return [edit && edit.source === source ? edit.value : source, set];
}
