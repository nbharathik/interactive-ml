/** Keyboard access to a canvas diagram: arrows move, Enter opens, Esc closes. */

import { useCallback } from 'react';
import type { KeyboardEvent, KeyboardEventHandler } from 'react';

export interface DiagramKeysOptions<T> {
  /** Every component, in the order the arrows walk them. */
  targets: readonly T[];
  cursor: T | null;
  same: (a: T | null, b: T | null) => boolean;
  onCursor: (target: T | null) => void;
  onOpen: (target: T) => void;
  onClose: () => void;
}

export function useDiagramKeys<T>({
  targets,
  cursor,
  same,
  onCursor,
  onOpen,
  onClose,
}: DiagramKeysOptions<T>): KeyboardEventHandler<HTMLDivElement> {
  return useCallback(
    (event: KeyboardEvent<HTMLDivElement>) => {
      // Keys inside an open card belong to the card, except on the card itself.
      const target = event.target as HTMLElement;
      if (target !== event.currentTarget && !target.classList.contains('mlx-detail')) return;
      const count = targets.length;
      const index = cursor !== null ? targets.findIndex((t) => same(t, cursor)) : -1;
      let next: number | null = null;
      switch (event.key) {
        case 'ArrowRight':
        case 'ArrowDown':
          next = count ? (index + 1) % count : null;
          break;
        case 'ArrowLeft':
        case 'ArrowUp':
          next = count ? (index <= 0 ? count - 1 : index - 1) : null;
          break;
        case 'Home':
          next = count ? 0 : null;
          break;
        case 'End':
          next = count ? count - 1 : null;
          break;
        case 'Enter':
        case ' ':
          if (cursor === null) return;
          event.preventDefault();
          event.stopPropagation();
          onOpen(cursor);
          return;
        case 'Escape':
          // With nothing picked, Esc belongs to the page: it leaves the expanded view.
          if (cursor === null) return;
          event.stopPropagation();
          onClose();
          return;
        default:
          return;
      }
      if (next === null) return;
      // Stops the transport from stepping on the same arrow key.
      event.preventDefault();
      event.stopPropagation();
      onCursor(targets[next]);
    },
    [targets, cursor, same, onCursor, onOpen, onClose],
  );
}
