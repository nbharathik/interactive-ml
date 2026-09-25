/** One-input regression datasets. Each carries the `truth` function it was drawn from. */

import { gauss, makeRng, uniform } from '../math/rng';
import type { DatasetConfig, DatasetOption, RegressionData } from './types';

const X_RANGE: [number, number] = [0, 10];

function build(
  points: { x: number; y: number }[],
  truth: (x: number) => number,
  labels: { x: string; y: string },
): RegressionData {
  let lo = Infinity;
  let hi = -Infinity;
  for (const p of points) {
    if (p.y < lo) lo = p.y;
    if (p.y > hi) hi = p.y;
  }
  // Keep the truth curve inside the frame even where no sample landed.
  for (let i = 0; i <= 20; i++) {
    const v = truth(X_RANGE[0] + ((X_RANGE[1] - X_RANGE[0]) * i) / 20);
    if (Number.isFinite(v)) {
      if (v < lo) lo = v;
      if (v > hi) hi = v;
    }
  }
  const pad = Math.max(0.5, (hi - lo) * 0.15);
  return {
    kind: 'regression',
    points,
    xRange: X_RANGE,
    yRange: [lo - pad, hi + pad],
    truth,
    xLabel: labels.x,
    yLabel: labels.y,
  };
}

function sampleXs(config: DatasetConfig, rng: () => number): number[] {
  const xs: number[] = [];
  for (let i = 0; i < config.count; i++) xs.push(uniform(rng, X_RANGE[0], X_RANGE[1]));
  return xs.sort((a, b) => a - b);
}

export const REGRESSION_DATASETS: DatasetOption<RegressionData>[] = [
  {
    id: 'linear',
    name: 'Straight line',
    blurb: 'A clean linear relationship. The best a line can do is very good.',
    generate(config) {
      const rng = makeRng(config.seed);
      const truth = (x: number) => 1.6 * x + 2;
      const sigma = config.noise * 6;
      const points = sampleXs(config, rng).map((x) => ({ x, y: truth(x) + gauss(rng, 0, sigma) }));
      return build(points, truth, { x: 'x', y: 'y' });
    },
  },
  {
    id: 'housing',
    name: 'House prices',
    blurb: 'Floor area against price, the example the blog post opens with.',
    generate(config) {
      const rng = makeRng(config.seed + 7);
      // x is floor area in units of 20 m², y is price in units of 10k EUR.
      const truth = (x: number) => 18 + 9.5 * x;
      const sigma = config.noise * 30;
      const points = sampleXs(config, rng).map((x) => ({
        x,
        y: Math.max(5, truth(x) + gauss(rng, 0, sigma)),
      }));
      return build(points, truth, { x: 'floor area (×20 m²)', y: 'price (×10k €)' });
    },
  },
  {
    id: 'curved',
    name: 'Gentle curve',
    blurb: 'Quadratic truth. A straight line must under-fit somewhere, watch where.',
    generate(config) {
      const rng = makeRng(config.seed + 11);
      const truth = (x: number) => 0.55 * (x - 5) * (x - 5) + 3;
      const sigma = config.noise * 5;
      const points = sampleXs(config, rng).map((x) => ({ x, y: truth(x) + gauss(rng, 0, sigma) }));
      return build(points, truth, { x: 'x', y: 'y' });
    },
  },
  {
    id: 'sinusoid',
    name: 'Wave',
    blurb: 'Periodic truth, the classic case for adding features instead of layers.',
    generate(config) {
      const rng = makeRng(config.seed + 13);
      const truth = (x: number) => 8 * Math.sin(x * 0.9) + 12;
      const sigma = config.noise * 4;
      const points = sampleXs(config, rng).map((x) => ({ x, y: truth(x) + gauss(rng, 0, sigma) }));
      return build(points, truth, { x: 'x', y: 'y' });
    },
  },
  {
    id: 'step',
    name: 'Step',
    blurb: 'A discontinuity. Squared error will split the difference right at the jump.',
    generate(config) {
      const rng = makeRng(config.seed + 17);
      const truth = (x: number) => (x < 5 ? 6 : 18);
      const sigma = config.noise * 4;
      const points = sampleXs(config, rng).map((x) => ({ x, y: truth(x) + gauss(rng, 0, sigma) }));
      return build(points, truth, { x: 'x', y: 'y' });
    },
  },
  {
    id: 'outliers',
    name: 'Line with outliers',
    blurb: 'A few readings that are simply wrong. Squared error hates them, see how far the fit tilts.',
    generate(config) {
      const rng = makeRng(config.seed + 19);
      const truth = (x: number) => 1.6 * x + 2;
      const sigma = config.noise * 4;
      const xs = sampleXs(config, rng);
      const points = xs.map((x) => ({ x, y: truth(x) + gauss(rng, 0, sigma) }));
      const outlierCount = Math.min(6, Math.max(2, Math.round(points.length * 0.05)));
      for (let i = 0; i < outlierCount; i++) {
        const idx = Math.floor(rng() * points.length);
        points[idx] = { x: points[idx].x, y: points[idx].y + (rng() > 0.5 ? 1 : -1) * (18 + rng() * 14) };
      }
      return build(points, truth, { x: 'x', y: 'y' });
    },
  },
  {
    id: 'funnel',
    name: 'Widening spread',
    blurb: 'Noise grows with x. The line is right on average and wrong everywhere.',
    generate(config) {
      const rng = makeRng(config.seed + 23);
      const truth = (x: number) => 1.2 * x + 4;
      const points = sampleXs(config, rng).map((x) => ({
        x,
        y: truth(x) + gauss(rng, 0, config.noise * 1.6 * (0.4 + x)),
      }));
      return build(points, truth, { x: 'x', y: 'y' });
    },
  },
];

export function getRegressionDataset(id: string): DatasetOption<RegressionData> {
  return REGRESSION_DATASETS.find((d) => d.id === id) ?? REGRESSION_DATASETS[0];
}
