/** Track an element's content-box size with a ResizeObserver. */

import { useCallback, useLayoutEffect, useRef, useState } from 'react';

export interface Size {
  width: number;
  height: number;
}

export function useElementSize<T extends HTMLElement>(): [
  (node: T | null) => void,
  Size,
  React.MutableRefObject<T | null>,
] {
  const ref = useRef<T | null>(null);
  const [node, setNodeState] = useState<T | null>(null);
  const [size, setSize] = useState<Size>({ width: 0, height: 0 });

  // Observe in an effect keyed on the node so StrictMode's double invoke re-observes.
  const setNode = useCallback((next: T | null) => {
    ref.current = next;
    setNodeState(next);
  }, []);

  // A layout effect, so the first measurement lands before the first paint.
  useLayoutEffect(() => {
    if (!node) return undefined;

    const measure = () => {
      const rect = node.getBoundingClientRect();
      const width = Math.round(rect.width);
      const height = Math.round(rect.height);
      setSize((prev) => (prev.width === width && prev.height === height ? prev : { width, height }));
    };

    measure();

    if (typeof ResizeObserver === 'undefined') {
      window.addEventListener('resize', measure);
      return () => window.removeEventListener('resize', measure);
    }

    const observer = new ResizeObserver(measure);
    observer.observe(node);
    return () => observer.disconnect();
  }, [node]);

  return [setNode, size, ref];
}
