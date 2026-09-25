import { useEffect } from 'react';
import type { RefObject } from 'react';

const FOCUSABLE = 'a[href], button:not(:disabled), select, input, [tabindex]:not([tabindex="-1"])';

/** Dialog behaviour for a full-screen layer: focus, inert chrome, Tab cycling, focus restore. */
export function useModalLayer(open: boolean, ref: RefObject<HTMLElement | null>) {
  useEffect(() => {
    const layer = ref.current;
    if (!open || !layer) return undefined;
    const previous = document.activeElement as HTMLElement | null;
    const chrome = document.querySelectorAll<HTMLElement>('.mlx-skip-link, .mlx-header, .mlx-footer');
    chrome.forEach((el) => {
      el.inert = true;
    });
    layer.focus({ preventScroll: true });

    const onKey = (event: KeyboardEvent) => {
      if (event.key !== 'Tab') return;
      const items = [...layer.querySelectorAll<HTMLElement>(FOCUSABLE)].filter(
        (el) => el.offsetParent !== null || el === document.activeElement,
      );
      if (items.length === 0) {
        event.preventDefault();
        return;
      }
      const first = items[0];
      const last = items[items.length - 1];
      const active = document.activeElement;
      if (event.shiftKey && (active === first || active === layer)) {
        event.preventDefault();
        last.focus();
      } else if (!event.shiftKey && (active === last || !layer.contains(active))) {
        event.preventDefault();
        first.focus();
      }
    };
    layer.addEventListener('keydown', onKey);

    return () => {
      layer.removeEventListener('keydown', onKey);
      chrome.forEach((el) => {
        el.inert = false;
      });
      if (previous && previous.isConnected) previous.focus({ preventScroll: true });
    };
  }, [open, ref]);
}
