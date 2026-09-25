/** Small grayscale glyph images drawn in code: bars, diagonals, crosses, rings and blobs, jittered and noised. */

import { gauss, makeRng, shuffled, uniform } from '../math/rng';

export interface ImageDataset {
  kind: 'images';
  size: number;
  /** Row-major pixels in [0, 1], one array per image. */
  images: Float32Array[];
  labels: number[];
  classNames: string[];
  classSymbols: string[];
  trainIndex: number[];
  testIndex: number[];
}

export interface GlyphConfig {
  size: number;
  classCount: number;
  count: number;
  /** Largest shift of a glyph from the centre, in pixels. */
  jitter: number;
  thickness: number;
  /** Noise standard deviation as a fraction of full brightness. */
  noise: number;
  seed: number;
  trainFraction: number;
}

export const GLYPHS: ReadonlyArray<{ name: string; symbol: string }> = [
  { name: 'horizontal bar', symbol: '─' },
  { name: 'vertical bar', symbol: '│' },
  { name: 'diagonal', symbol: '╱' },
  { name: 'cross', symbol: '┼' },
  { name: 'ring', symbol: '○' },
  { name: 'blob', symbol: '●' },
];

export const DEFAULT_GLYPHS: GlyphConfig = {
  size: 16,
  classCount: 4,
  count: 240,
  jitter: 2,
  thickness: 2,
  noise: 0.1,
  seed: 5,
  trainFraction: 0.75,
};

/** Soft edge: full inside the shape, fading over one pixel outside it. */
function coverage(distance: number): number {
  return Math.max(0, Math.min(1, 1 - distance));
}

/** Paint one glyph of class `label` centred at (cx, cy). */
export function paintGlyph(size: number, label: number, cx: number, cy: number, thickness: number, radius: number): Float32Array {
  const out = new Float32Array(size * size);
  const half = thickness / 2;
  const reach = size * 0.36;
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const dx = x + 0.5 - cx;
      const dy = y + 0.5 - cy;
      let d = Infinity;
      const inSpan = (v: number) => Math.abs(v) <= reach;
      switch (label) {
        case 0:
          if (inSpan(dx)) d = Math.abs(dy) - half;
          break;
        case 1:
          if (inSpan(dy)) d = Math.abs(dx) - half;
          break;
        case 2:
          if (inSpan(dx) && inSpan(dy)) d = Math.abs(dx + dy) / Math.SQRT2 - half;
          break;
        case 3:
          if (inSpan(dx) || inSpan(dy)) d = Math.min(inSpan(dx) ? Math.abs(dy) - half : Infinity, inSpan(dy) ? Math.abs(dx) - half : Infinity);
          break;
        case 4:
          d = Math.abs(Math.hypot(dx, dy) - radius) - half;
          break;
        default:
          d = Math.hypot(dx, dy) - radius;
          break;
      }
      out[y * size + x] = coverage(d);
    }
  }
  return out;
}

export function generateGlyphs(config: GlyphConfig): ImageDataset {
  const size = Math.max(6, Math.round(config.size));
  const classCount = Math.max(2, Math.min(GLYPHS.length, Math.round(config.classCount)));
  const count = Math.max(classCount * 2, Math.round(config.count));
  const rng = makeRng(config.seed);
  const images: Float32Array[] = [];
  const labels: number[] = [];
  for (let i = 0; i < count; i++) {
    const label = i % classCount;
    const cx = size / 2 + uniform(rng, -config.jitter, config.jitter);
    const cy = size / 2 + uniform(rng, -config.jitter, config.jitter);
    const radius = size * 0.22 + uniform(rng, -0.5, 0.5);
    const image = paintGlyph(size, label, cx, cy, config.thickness, radius);
    if (config.noise > 0) {
      for (let p = 0; p < image.length; p++) image[p] = Math.max(0, Math.min(1, image[p] + gauss(rng, 0, config.noise)));
    }
    images.push(image);
    labels.push(label);
  }
  const order = shuffled(rng, Array.from({ length: count }, (_, i) => i));
  const trainCount = Math.max(classCount, Math.min(count - 1, Math.round(count * config.trainFraction)));
  return {
    kind: 'images',
    size,
    images,
    labels,
    classNames: GLYPHS.slice(0, classCount).map((g) => g.name),
    classSymbols: GLYPHS.slice(0, classCount).map((g) => g.symbol),
    trainIndex: order.slice(0, trainCount).sort((a, b) => a - b),
    testIndex: order.slice(trainCount).sort((a, b) => a - b),
  };
}
