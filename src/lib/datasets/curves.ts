/** One-input curves for polynomial regression. Every shape carries the `truth` it was drawn from. */

import { gauss, makeRng, uniform } from '../math/rng';
import type { DatasetConfig, DatasetOption, RegressionData, RegressionPoint } from './types';

export const CURVE_X_RANGE: [number, number] = [0, 5];

/** The y-range every curve is drawn in, so the frame does not jump between degrees. */
const Y_RANGE: [number, number] = [-2.5, 4.5];

interface Shape {
  id: string;
  name: string;
  blurb: string;
  truth: (x: number) => number;
}

const SHAPES: Shape[] = [
  {
    id: 'wave',
    name: 'Wave',
    blurb: 'sin(1.5x) + 0.5x, the curve from the blog post. Degree 3 to 5 fits it; a line cannot.',
    truth: (x) => Math.sin(1.5 * x) + 0.5 * x,
  },
  {
    id: 'line',
    name: 'Straight line',
    blurb: 'A linear truth. Every degree above 1 is extra freedom with nothing to spend it on.',
    truth: (x) => 0.6 * x - 0.5,
  },
  {
    id: 'parabola',
    name: 'Parabola',
    blurb: 'A quadratic truth. Degree 2 is exactly right, and degree 1 must miss somewhere.',
    truth: (x) => 0.5 * (x - 2.5) * (x - 2.5) - 1,
  },
  {
    id: 'step',
    name: 'Step',
    blurb: 'A jump. No polynomial fits a corner; the error floor stays high at every degree.',
    truth: (x) => (x < 2.5 ? 0 : 2),
  },
  {
    id: 'peak',
    name: 'Sharp peak',
    blurb: 'A narrow bump. High degrees reach it, then ring at the edges of the range.',
    truth: (x) => 3 * Math.exp(-2 * (x - 2.5) * (x - 2.5)),
  },
];

function draw(shape: Shape, config: DatasetConfig): RegressionData {
  const rng = makeRng(config.seed);
  const xs: number[] = [];
  for (let i = 0; i < config.count; i++) xs.push(uniform(rng, CURVE_X_RANGE[0], CURVE_X_RANGE[1]));
  xs.sort((a, b) => a - b);
  const points = xs.map((x) => ({ x, y: shape.truth(x) + gauss(rng, 0, config.noise) }));
  return {
    kind: 'regression',
    points,
    xRange: CURVE_X_RANGE,
    yRange: Y_RANGE,
    truth: shape.truth,
    xLabel: 'x',
    yLabel: 'y',
  };
}

export const CURVE_DATASETS: DatasetOption<RegressionData>[] = SHAPES.map((shape) => ({
  id: shape.id,
  name: shape.name,
  blurb: shape.blurb,
  generate: (config) => draw(shape, config),
}));

export function getCurveDataset(id: string): DatasetOption<RegressionData> {
  return CURVE_DATASETS.find((d) => d.id === id) ?? CURVE_DATASETS[0];
}

/** The same x's with fresh noise (one unit of y at noise 1): one training set of the bias-variance loop. */
export function renoise(
  points: readonly RegressionPoint[],
  truth: (x: number) => number,
  noise: number,
  seed: number,
): RegressionPoint[] {
  const rng = makeRng(seed);
  return points.map((p) => ({ x: p.x, y: truth(p.x) + gauss(rng, 0, noise) }));
}

/** The held-out set is a second draw from the same source. */
export function testSeed(seed: number): number {
  return seed + 100_003;
}

/** The k-th fresh training sample for the bias-variance loop. */
export function refitSeed(seed: number, k: number): number {
  return seed + 200_003 + k;
}
