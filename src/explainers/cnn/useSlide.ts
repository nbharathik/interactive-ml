/** Steps the window across its map a few cells a second, unless held or the pointer is choosing the cell. */

import { useCallback, useEffect, useMemo, useState } from 'react';

import type { Window } from './scene';

const SLIDE_MS = 160;

export interface Slide {
  paused: boolean;
  toggle(): void;
  /** The pointer is picking the cell: the slide waits. */
  setHovering(hovering: boolean): void;
  /** Back to sliding, for the chapter reset. */
  reset(): void;
}

export function useSlide(
  active: boolean,
  window: Window | null,
  dims: { cols: number; rows: number },
  onWindow: (window: Window) => void,
): Slide {
  const [paused, setPaused] = useState(false);
  const [hovering, setHovering] = useState(false);
  useEffect(() => {
    if (!active || paused || hovering || !window) return undefined;
    const id = globalThis.setInterval(() => {
      // A hidden tab has nobody watching; the training loop pauses there too.
      if (typeof document !== 'undefined' && document.hidden) return;
      const next = window.x + 1 < dims.cols ? { x: window.x + 1, y: window.y } : { x: 0, y: (window.y + 1) % dims.rows };
      onWindow({ ...window, ...next });
    }, SLIDE_MS);
    return () => globalThis.clearInterval(id);
  }, [active, paused, hovering, window, dims.cols, dims.rows, onWindow]);
  const toggle = useCallback(() => setPaused((v) => !v), []);
  const reset = useCallback(() => {
    setPaused(false);
    setHovering(false);
  }, []);
  return useMemo(() => ({ paused, toggle, setHovering, reset }), [paused, toggle, reset]);
}
