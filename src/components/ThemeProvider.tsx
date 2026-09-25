/** Theme state. `index.html` applies the initial value before first paint; this keeps React in step. */

import { createContext, useCallback, useContext, useEffect, useMemo, useState } from 'react';
import type { ReactNode } from 'react';

import { invalidatePalette, readPalette } from '../lib/viz/palette';
import type { Palette } from '../lib/viz/palette';

export type Theme = 'light' | 'dark';

const STORAGE_KEY = 'mlx-theme';

interface ThemeContextValue {
  theme: Theme;
  toggle(): void;
  /** Increments on every theme change, chart redraw dependency. */
  version: number;
}

const ThemeContext = createContext<ThemeContextValue | null>(null);

function persist(next: Theme) {
  try {
    localStorage.setItem(STORAGE_KEY, next);
  } catch {
    /* private mode */
  }
}

function readInitialTheme(): Theme {
  if (typeof document === 'undefined') return 'light';
  const attr = document.documentElement.getAttribute('data-theme');
  if (attr === 'dark' || attr === 'light') return attr;
  try {
    const stored = localStorage.getItem(STORAGE_KEY);
    if (stored === 'dark' || stored === 'light') return stored;
  } catch {
    /* private mode, fall through to the media query */
  }
  return window.matchMedia?.('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
}

export function ThemeProvider({ children }: { children: ReactNode }) {
  const [theme, setThemeState] = useState<Theme>(readInitialTheme);
  const [version, setVersion] = useState(0);

  // Only an explicit choice is stored.
  const toggle = useCallback(() => {
    setThemeState((prev) => {
      const next = prev === 'dark' ? 'light' : 'dark';
      persist(next);
      return next;
    });
  }, []);

  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme);
    document
      .querySelector('meta[name="theme-color"]')
      ?.setAttribute('content', getComputedStyle(document.documentElement).getPropertyValue('--bg-primary').trim());
    // Drop the palette cache, then wake every chart.
    invalidatePalette();
    // Let the new custom properties land before charts read them.
    const id = requestAnimationFrame(() => setVersion((v) => v + 1));
    return () => cancelAnimationFrame(id);
  }, [theme]);

  // Follow the OS when the reader has not made an explicit choice.
  useEffect(() => {
    const query = window.matchMedia?.('(prefers-color-scheme: dark)');
    if (!query) return undefined;
    const listener = (event: MediaQueryListEvent) => {
      try {
        if (localStorage.getItem(STORAGE_KEY) !== null) return;
      } catch {
        /* ignore */
      }
      setThemeState(event.matches ? 'dark' : 'light');
    };
    query.addEventListener('change', listener);
    return () => query.removeEventListener('change', listener);
  }, []);

  const value = useMemo(() => ({ theme, toggle, version }), [theme, toggle, version]);

  return <ThemeContext.Provider value={value}>{children}</ThemeContext.Provider>;
}

export function useTheme(): ThemeContextValue {
  const ctx = useContext(ThemeContext);
  if (!ctx) throw new Error('useTheme must be used inside <ThemeProvider>');
  return ctx;
}

/** Redraw dependency for canvas code. Safe to call outside a provider (returns 0). */
export function useThemeVersion(): number {
  return useContext(ThemeContext)?.version ?? 0;
}

/** The chart palette for render-time colours, re-read after every theme flip. */
export function usePalette(): Palette {
  useThemeVersion();
  return readPalette();
}
