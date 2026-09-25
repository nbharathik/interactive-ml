/** The explainer pages, one chunk each. The router loads them; links warm them ahead of the click. */

import type { ComponentType } from 'react';

type PageModule = { default: ComponentType };

/** Pages still in the works: routed by link on the dev server or in a VITE_PREVIEW=1 build, listed nowhere. */
export const PREVIEW = import.meta.env.DEV || (typeof __MLX_PREVIEW__ !== 'undefined' && __MLX_PREVIEW__);

const PREVIEW_MODULES: Record<string, () => Promise<PageModule>> = PREVIEW ? { transformers: () => import('./transformers/Page') } : {};

export const PAGE_MODULES: Record<string, () => Promise<PageModule>> = {
  'linear-regression': () => import('./linear-regression/Page'),
  'polynomial-regression': () => import('./polynomial-regression/Page'),
  'logistic-regression': () => import('./logistic-regression/Page'),
  'k-means': () => import('./k-means/Page'),
  'decision-tree': () => import('./decision-tree/Page'),
  'neural-network': () => import('./neural-network/Page'),
  'knn': () => import('./knn/Page'),
  'regularization': () => import('./regularization/Page'),
  'activation-functions': () => import('./activation-functions/Page'),
  'cnn': () => import('./cnn/Page'),
  ...PREVIEW_MODULES,
};

const warmed = new Set<string>();

/** Fetches and evaluates a page's chunk so the click only has to render. */
export function preloadExplainer(slug: string): Promise<void> {
  const load = PAGE_MODULES[slug];
  if (!load || warmed.has(slug)) return Promise.resolve();
  warmed.add(slug);
  return load().then(
    () => undefined,
    () => {
      warmed.delete(slug);
    },
  );
}

/** Link handlers that start the page's chunk on hover, focus or touch, ahead of the click. */
export function warmOn(slug: string) {
  const warm = () => {
    void preloadExplainer(slug);
  };
  return { onPointerEnter: warm, onFocus: warm, onTouchStart: warm };
}

function metered(): boolean {
  const connection = (navigator as { connection?: { saveData?: boolean } }).connection;
  return Boolean(connection?.saveData);
}

function whenIdle(fn: () => void, timeout: number): () => void {
  if (typeof requestIdleCallback === 'function') {
    const id = requestIdleCallback(fn, { timeout });
    return () => cancelIdleCallback(id);
  }
  const id = window.setTimeout(fn, 200);
  return () => window.clearTimeout(id);
}

/** Warms pages one at a time while the browser is idle. Skipped when the user asked to save data. */
export function preloadOnIdle(slugs: readonly string[], timeout = 2000): () => void {
  if (metered()) return () => undefined;
  const queue = slugs.filter((slug) => !warmed.has(slug));
  let cancelled = false;
  let cancel = () => undefined as void;
  const next = () => {
    const slug = queue.shift();
    if (!slug || cancelled) return;
    cancel = whenIdle(() => {
      preloadExplainer(slug).then(next);
    }, timeout);
  };
  next();
  return () => {
    cancelled = true;
    cancel();
  };
}

/** Starts the KaTeX font downloads so the first formula does not reflow when they land. */
export function warmMathFonts(): void {
  if (metered() || typeof document === 'undefined' || !document.fonts?.load) return;
  const faces = ['1em KaTeX_Main', 'italic 1em KaTeX_Math', 'bold 1em KaTeX_Main', '1em KaTeX_Size1', '1em KaTeX_Size2'];
  for (const face of faces) document.fonts.load(face).catch(() => undefined);
}
