/** Drag and arrow keys turn a surface; a click that did not move is left to the caller. */

import { useCallback, useMemo, useRef, useState } from 'react';
import type { KeyboardEvent } from 'react';

import { DEFAULT_SURFACE_VIEW, dragView, turnView, viewKey } from '../lib/viz/surface';
import type { SurfaceView } from '../lib/viz/surface';

const CLICK_SLOP = 3;
const KEY_TURN = 0.12;
const KEY_TILT = 0.08;

export interface Turntable {
  view: SurfaceView;
  /** Changes whenever the view does, for redraw keys. */
  key: string;
  dragging: boolean;
  /** Start a drag at a canvas point. */
  down(pos: { x: number; y: number }): void;
  /** Feed a pointer move; true while the drag is turning the view. */
  move(pos: { x: number; y: number } | null): boolean;
  /** End the drag; true when it was a click rather than a turn. */
  up(): boolean;
  /** Arrow keys turn and tilt; true when the key was used. */
  onKey(event: KeyboardEvent): boolean;
  turn(dAzimuth: number, dElevation: number): void;
  reset(): void;
}

export function useTurntable(initial: SurfaceView = DEFAULT_SURFACE_VIEW): Turntable {
  const [view, setView] = useState<SurfaceView>(initial);
  const [dragging, setDragging] = useState(false);
  const drag = useRef<{ x: number; y: number; from: SurfaceView; moved: boolean } | null>(null);

  const down = useCallback(
    (pos: { x: number; y: number }) => {
      drag.current = { x: pos.x, y: pos.y, from: view, moved: false };
    },
    [view],
  );
  const move = useCallback((pos: { x: number; y: number } | null) => {
    const d = drag.current;
    if (!d || !pos) return false;
    const dx = pos.x - d.x;
    const dy = pos.y - d.y;
    if (!d.moved && Math.abs(dx) + Math.abs(dy) > CLICK_SLOP) {
      d.moved = true;
      setDragging(true);
    }
    if (!d.moved) return false;
    setView(dragView(d.from, dx, dy));
    return true;
  }, []);
  const up = useCallback(() => {
    const d = drag.current;
    drag.current = null;
    setDragging(false);
    return d !== null && !d.moved;
  }, []);
  const turn = useCallback((dAzimuth: number, dElevation: number) => setView((v) => turnView(v, dAzimuth, dElevation)), []);
  const reset = useCallback(() => setView(initial), [initial]);
  const onKey = useCallback(
    (event: KeyboardEvent) => {
      switch (event.key) {
        case 'ArrowLeft':
          turn(-KEY_TURN, 0);
          return true;
        case 'ArrowRight':
          turn(KEY_TURN, 0);
          return true;
        case 'ArrowUp':
          turn(0, KEY_TILT);
          return true;
        case 'ArrowDown':
          turn(0, -KEY_TILT);
          return true;
        default:
          return false;
      }
    },
    [turn],
  );

  return useMemo(
    () => ({ view, key: viewKey(view), dragging, down, move, up, onKey, turn, reset }),
    [view, dragging, down, move, up, onKey, turn, reset],
  );
}
