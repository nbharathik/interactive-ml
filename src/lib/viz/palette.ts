/** Theme-aware chart palette, resolved from the tokens in `styles/tokens.css`. */

export interface Palette {
  /** page + panel surfaces */
  bg: string;
  surface: string;
  surfaceAlt: string;
  /** text */
  text: string;
  textMuted: string;
  textFaint: string;
  /** structure */
  border: string;
  grid: string;
  axis: string;
  /** loss-surface floor colour */
  basin: string;
  /** semantic hues */
  accent: string;
  blue: string;
  green: string;
  orange: string;
  pink: string;
  red: string;
  cyan: string;
  yellow: string;
  violet: string;
  muted: string;
  /** binary class colours (also the two poles of the diverging ramp) */
  classA: string;
  classB: string;
  classC: string;
  classD: string;
  negative: string;
  positive: string;
  /** ordered categorical ramp for clusters / multi-class */
  categorical: string[];
  isDark: boolean;
}

const FALLBACK: Record<string, string> = {
  '--bg-primary': '#ffffff',
  '--bg-secondary': '#f7f7f7',
  '--text-primary': '#2b2b2b',
  '--text-secondary': '#555555',
  '--text-tertiary': '#757575',
  '--border': '#e5e5e5',
  '--accent': '#0f7ea3',
  '--viz-blue': '#0f7ea3',
  '--viz-green': '#1b9e6a',
  '--viz-orange': '#f59322',
  '--viz-pink': '#d63f9f',
  '--viz-red': '#d93a3a',
  '--viz-cyan': '#12b5c9',
  '--viz-yellow': '#c99500',
  '--viz-violet': '#7c3aed',
  '--viz-grid': '#e5e5e5',
  '--viz-muted': '#757575',
  '--viz-surface': '#ffffff',
  '--viz-axis': '#aaaaaa',
  '--viz-basin': '#eef2f6',
  '--viz-class-a': '#0f7ea3',
  '--viz-class-b': '#f59322',
  '--viz-class-c': '#1b9e6a',
  '--viz-class-d': '#d63f9f',
  '--viz-neg': '#f59322',
  '--viz-pos': '#0f7ea3',
};

function readVar(styles: CSSStyleDeclaration, name: string): string {
  const value = styles.getPropertyValue(name).trim();
  return value || FALLBACK[name] || '#888888';
}

function assemble(read: (name: string) => string, isDark: boolean): Palette {
  const p: Palette = {
    bg: read('--bg-primary'),
    surface: read('--viz-surface'),
    surfaceAlt: read('--bg-secondary'),
    text: read('--text-primary'),
    textMuted: read('--text-secondary'),
    textFaint: read('--text-tertiary'),
    border: read('--border'),
    grid: read('--viz-grid'),
    axis: read('--viz-axis'),
    basin: read('--viz-basin'),
    accent: read('--accent'),
    blue: read('--viz-blue'),
    green: read('--viz-green'),
    orange: read('--viz-orange'),
    pink: read('--viz-pink'),
    red: read('--viz-red'),
    cyan: read('--viz-cyan'),
    yellow: read('--viz-yellow'),
    violet: read('--viz-violet'),
    muted: read('--viz-muted'),
    classA: read('--viz-class-a'),
    classB: read('--viz-class-b'),
    classC: read('--viz-class-c'),
    classD: read('--viz-class-d'),
    negative: read('--viz-neg'),
    positive: read('--viz-pos'),
    categorical: [],
    isDark,
  };
  p.categorical = [p.classA, p.classB, p.classC, p.classD, p.cyan, p.violet, p.yellow, p.red];
  return p;
}

let cache: Palette | null = null;
let cacheTheme = '';

/** Resolve the current palette. Cached per theme; `invalidatePalette()` on theme flip. */
export function readPalette(): Palette {
  if (typeof window === 'undefined' || typeof document === 'undefined') {
    return assemble((n) => FALLBACK[n] ?? '#888888', false);
  }

  const root = document.documentElement;
  const theme = root.getAttribute('data-theme') ?? 'light';
  if (cache && cacheTheme === theme) return cache;

  const styles = getComputedStyle(root);
  cache = assemble((n) => readVar(styles, n), theme === 'dark');
  cacheTheme = theme;
  return cache;
}

export function invalidatePalette(): void {
  cache = null;
  cacheTheme = '';
}

/* ---------------- colour maths ---------------- */

export function toRgb(hex: string): [number, number, number] {
  const raw = (hex ?? '').trim();
  if (raw.startsWith('rgb')) {
    const nums = raw.match(/[\d.]+/g);
    if (nums && nums.length >= 3) return [+nums[0], +nums[1], +nums[2]];
    return [136, 136, 136];
  }
  let h = raw.replace('#', '');
  if (h.length === 3) h = h[0] + h[0] + h[1] + h[1] + h[2] + h[2];
  if (h.length < 6) return [136, 136, 136];
  return [parseInt(h.slice(0, 2), 16), parseInt(h.slice(2, 4), 16), parseInt(h.slice(4, 6), 16)];
}

/** `#34548a` + 0.2 becomes `rgba(52,84,138,0.2)`. */
export function rgba(hex: string, alpha: number): string {
  const [r, g, b] = toRgb(hex);
  return 'rgba(' + r + ',' + g + ',' + b + ',' + alpha + ')';
}

/** Linear blend of two colours in sRGB. `t` clamps to [0,1]. */
export function mix(a: string, b: string, t: number): string {
  const k = Math.max(0, Math.min(1, t));
  const [r1, g1, b1] = toRgb(a);
  const [r2, g2, b2] = toRgb(b);
  const r = Math.round(r1 + (r2 - r1) * k);
  const g = Math.round(g1 + (g2 - g1) * k);
  const b3 = Math.round(b1 + (b2 - b1) * k);
  return 'rgb(' + r + ',' + g + ',' + b3 + ')';
}

/** Diverging ramp for signed quantities: -1 is the negative pole, 0 transparent, +1 positive. */
export function diverging(p: Palette, t: number, maxAlpha = 0.85): string {
  const k = Math.max(-1, Math.min(1, t));
  return rgba(k < 0 ? p.negative : p.positive, Math.abs(k) * maxAlpha);
}

/** Pick a categorical colour, wrapping around the ramp. Later laps are tinted. */
export function categorical(p: Palette, index: number): string {
  const ramp = p.categorical;
  const i = ((index % ramp.length) + ramp.length) % ramp.length;
  const lap = Math.floor(index / ramp.length);
  return lap > 0 ? mix(ramp[i], p.surface, 0.4) : ramp[i];
}

/** Perceived luminance in [0,1], used to pick readable label colours over fills. */
export function luminance(hex: string): number {
  const [r, g, b] = toRgb(hex);
  return (0.2126 * r + 0.7152 * g + 0.0722 * b) / 255;
}

/** Near-black or near-white, whichever reads better on `hex`. */
export function readableOn(hex: string): string {
  return luminance(hex) > 0.55 ? '#2b2b2b' : '#ffffff';
}
