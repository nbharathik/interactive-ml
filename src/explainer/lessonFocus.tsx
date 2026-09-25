/** The lesson's spotlight: targets register with useLessonTarget, SpotTag pins the label. */

import { createContext, useContext, useEffect, useState } from 'react';
import { createPortal } from 'react-dom';

import type { FocusTarget } from './lessons';
import { focusKey, focusLabel } from './lessons';
import { IconTarget } from './components/Icons';

export const LessonFocusContext = createContext<FocusTarget | null>(null);

/** True while a lesson is open; pages with a hero layout provide it. */
export const LessonOpenContext = createContext(false);

export interface LessonTargetAttrs {
  'data-lesson-target': string;
  'data-mlx-spot'?: '';
}

/** Attributes for an element that a lesson step can spotlight. */
export function useLessonTarget(kind: FocusTarget['kind'], id: string): { spot: boolean; attrs: LessonTargetAttrs } {
  const target = useContext(LessonFocusContext);
  const key = kind + ':' + id;
  const spot =
    target !== null &&
    target.kind === kind &&
    (focusKey(target) === key || focusKey(target).startsWith(key + ':'));
  return { spot, attrs: spot ? { 'data-lesson-target': key, 'data-mlx-spot': '' } : { 'data-lesson-target': key } };
}

/** Whether the target is a knob with this key, for groups that open themselves. */
export function focusesControl(target: FocusTarget | null, keys: readonly string[]): boolean {
  return target !== null && target.kind === 'control' && keys.includes(target.key);
}

interface TagPlace {
  left: number;
  top: number;
  side: 'above' | 'below' | 'inside';
}

/** The label pinned to the highlighted element. Diagram components draw their own. */
export function SpotTag({ within }: { within?: string } = {}) {
  const target = useContext(LessonFocusContext);
  const [place, setPlace] = useState<TagPlace | null>(null);
  const label = target ? focusLabel(target) : '';

  useEffect(() => {
    if (!target || target.kind === 'node' || target.kind === 'edge' || typeof window === 'undefined') {
      setPlace(null);
      return undefined;
    }
    let raf = 0;
    const tick = () => {
      // The first spotlit element that is actually on screen; a knob in a hidden tab is skipped.
      const el = [...document.querySelectorAll<HTMLElement>((within ?? '') + ' [data-mlx-spot]')].find((e) => {
        const box = e.getBoundingClientRect();
        return box.width > 0 && box.height > 0;
      });
      const r = el?.getBoundingClientRect();
      let next: TagPlace | null = null;
      if (r && r.width > 0 && r.height > 0 && r.bottom > 0 && r.top < window.innerHeight) {
        // A tall target (a panel) carries the tag inside, under its own header.
        const side = r.height > 160 ? 'inside' : r.top > 44 ? 'above' : 'below';
        const head = side === 'inside' ? el?.querySelector('.mlx-panel__head, .mlx-reg-pane__head')?.getBoundingClientRect() : undefined;
        const top = side === 'above' ? r.top - 10 : side === 'below' ? r.bottom + 10 : (head?.bottom ?? r.top) + 8;
        next = { left: Math.max(8, r.left + (side === 'inside' ? 8 : 0)), top, side };
      }
      setPlace((prev) =>
        prev === next || (prev && next && prev.left === next.left && prev.top === next.top && prev.side === next.side) ? prev : next,
      );
      raf = window.requestAnimationFrame(tick);
    };
    raf = window.requestAnimationFrame(tick);
    return () => window.cancelAnimationFrame(raf);
  }, [target, within]);

  if (!place || !label) return null;
  return createPortal(
    <span
      className="mlx-spot-tag"
      data-side={place.side}
      style={{ left: place.left, top: place.top, transform: place.side === 'above' ? 'translateY(-100%)' : undefined }}
      aria-hidden="true"
    >
      <IconTarget size={12} />
      {label}
    </span>,
    document.body,
  );
}
