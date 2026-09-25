/** The hero playground's three demos, each behind the same small interface. */

import type { DrawArgs } from '../explainer/components/Chart';
import type { Point2D, RegressionPoint } from '../lib/datasets/types';
import { getClassificationDataset, getClusteringDataset } from '../lib/datasets/points';
import { getRegressionDataset } from '../lib/datasets/regression';
import * as linreg from '../lib/ml/linearRegression';
import * as net from '../lib/ml/neuralNetwork';
import * as kmeans from '../lib/ml/kmeans';
import type { Frame } from '../lib/viz/canvas';
import { clamp, drawPath, lerp } from '../lib/viz/canvas';
import { categorical, rgba } from '../lib/viz/palette';
import { drawCategoryField, drawFunction, drawResiduals, drawScatter, drawSignedField } from '../lib/viz/plots';

import { drawCentroid, drawGrid, plainFrame } from './canvas';

export interface DemoRun {
  /** Advance by one frame. `instant` skips any pacing. Returns true once settled. */
  tick(now: number, instant?: boolean): boolean;
  /** True once the model has nothing left to learn. */
  settled(): boolean;
  /** Forget what was learned and keep every point. */
  restart(): void;
  draw(args: DrawArgs): void;
  /** Add a point at canvas pixel coordinates from the last draw; `label` is its class where the demo has classes. */
  addPoint(px: number, py: number, label: number): void;
  status(): string;
}

export type DemoKey = 'line' | 'boundary' | 'clusters';

export interface Demo {
  key: DemoKey;
  title: string;
  hint: string;
  slug: string;
  create(seed: number): DemoRun;
}

/* ---------------- fit a line ---------------- */

function lineDemo(seed: number): DemoRun {
  const data = getRegressionDataset('linear').generate({ count: 40, noise: 0.3, seed: seed * 13 + 1 });
  const points: RegressionPoint[] = data.points.slice();
  const config: linreg.LinRegConfig = {
    learningRate: 0.03,
    degree: 1,
    optimiser: 'gd',
    momentum: 0.9,
    batchMode: 'batch',
    batchSize: 40,
    l2: 0,
    standardise: true,
    seed,
  };
  const norm = linreg.makeNormaliser(points, true);
  let state = linreg.createState(config);
  let frame: Frame | null = null;
  const settled = () => state.epoch > 0 && state.gradientNorm < 0.02;

  return {
    tick() {
      if (settled()) return true;
      state = linreg.step(state, points, config, norm);
      return settled();
    },
    settled,
    restart() {
      state = linreg.createState(config);
    },
    draw({ ctx, width, height, palette }) {
      frame = plainFrame(width, height, data.xRange, data.yRange, false);
      const predict = (x: number) => linreg.predict(state.weights, x, 1, norm);
      drawGrid(ctx, frame, palette);
      drawResiduals(ctx, frame, palette, points, predict, { colour: palette.orange, alpha: 0.35 });
      drawScatter(ctx, frame, palette, points, { radius: 3.6, colourOf: () => palette.blue });
      drawFunction(ctx, frame, predict, palette.orange, { width: 2.6 });
    },
    addPoint(px, py) {
      if (!frame) return;
      const x = clamp(frame.x.invert(px), data.xRange[0], data.xRange[1]);
      const y = clamp(frame.y.invert(py), data.yRange[0], data.yRange[1]);
      points.push({ x, y });
      state = { ...state, gradientNorm: 1 };
    },
    status() {
      const loss = Number.isFinite(state.loss) ? state.loss.toFixed(2) : '...';
      return 'step ' + state.epoch + ' · mean squared error ' + loss;
    },
  };
}

/* ---------------- draw a boundary ---------------- */

const EPOCHS = 320;
// Every added point earns the model this many more epochs to adjust.
const EXTRA_EPOCHS = 200;

function boundaryDemo(seed: number): DemoRun {
  const data = getClassificationDataset('spiral').generate({ count: 180, noise: 0.1, seed: seed * 7 + 3 });
  const points: Point2D[] = data.points.slice();
  let config: net.NetConfig = {
    hiddenLayers: [8, 8],
    activation: 'tanh',
    learningRate: 0.05,
    batchSize: 10,
    inputFeatures: ['x1', 'x2'],
    initialiser: 'xavier',
    regularisation: 'none',
    regRate: 0,
    seed,
    testFraction: 0,
  };
  const training: net.TrainingData = { train: points, test: [] };
  let state = net.createState(config);
  let limit = EPOCHS;
  let frame: Frame | null = null;
  const score = (x: number, y: number) =>
    net.evaluate(state.layers, net.inputVector(config.inputFeatures, x, y), config.activation);
  const settled = () => state.epoch >= limit || state.diverged;

  return {
    tick() {
      if (settled()) return true;
      state = net.step(state, training, config);
      return settled();
    },
    settled,
    restart() {
      // A new seed so each replay starts from different weights.
      config = { ...config, seed: config.seed + 1 };
      state = net.createState(config);
      limit = EPOCHS;
    },
    draw({ ctx, width, height, palette }) {
      frame = plainFrame(width, height, data.xRange, data.yRange);
      drawGrid(ctx, frame, palette);
      drawSignedField(ctx, frame, palette, (x, y) => Math.tanh(score(x, y) * 0.6), {
        cellSize: 5,
        alpha: 0.5,
        contourColour: palette.text,
      });
      drawScatter(ctx, frame, palette, points, { radius: 3.4, shapes: true });
    },
    addPoint(px, py, label) {
      if (!frame) return;
      points.push({ x: frame.x.invert(px), y: frame.y.invert(py), label });
      limit = Math.max(limit, state.epoch + EXTRA_EPOCHS);
    },
    status() {
      return 'epoch ' + state.epoch + ' · accuracy ' + Math.round(state.trainAccuracy * 100) + '%';
    },
  };
}

/* ---------------- find clusters ---------------- */

const HALF_STEP_MS = 560;

// Seeds whose run ends in the good clustering.
const CLUSTER_SEEDS = [5, 3, 10, 19, 24, 11, 22, 23, 29, 31];

function clustersDemo(seed: number): DemoRun {
  const data = getClusteringDataset('blobs').generate({ count: 120, noise: 0.25, seed: 27, classCount: 3 });
  const points: Point2D[] = data.points.slice();
  const bounds = { xRange: data.xRange, yRange: data.yRange };
  let round = seed;
  const configFor = (r: number): kmeans.KMeansConfig => ({
    k: 3,
    init: 'random-coords',
    seed: CLUSTER_SEEDS[r % CLUSTER_SEEDS.length],
    tolerance: 0.01,
  });
  let config = configFor(round);
  let state = kmeans.createState(points, config, bounds);
  let movedAt = -Infinity;
  let nextAt = 0;
  let now = 0;
  let frame: Frame | null = null;
  const settled = () => kmeans.isComplete(state, config.tolerance);

  // Centroids glide from their previous position.
  const shown = (): kmeans.Centroid[] => {
    const t = clamp((now - movedAt) / 320, 0, 1);
    const ease = 1 - (1 - t) * (1 - t);
    return state.centroids.map((c, i) => ({
      x: lerp(state.previousCentroids[i].x, c.x, ease),
      y: lerp(state.previousCentroids[i].y, c.y, ease),
    }));
  };

  return {
    tick(time, instant) {
      now = time;
      if (settled()) return true;
      if (!instant && time < nextAt) return false;
      state = kmeans.step(state, points);
      if (state.phase === 'update') movedAt = time;
      nextAt = time + HALF_STEP_MS;
      return settled();
    },
    settled,
    restart() {
      round += 1;
      config = configFor(round);
      state = kmeans.createState(points, config, bounds);
      movedAt = -Infinity;
      nextAt = 0;
    },
    draw({ ctx, width, height, palette }) {
      const f = plainFrame(width, height, data.xRange, data.yRange);
      frame = f;
      const centroids = shown();
      drawGrid(ctx, f, palette);
      drawCategoryField(ctx, f, palette, (x, y) => kmeans.nearestCentroid(centroids, x, y), {
        cellSize: 3,
        alpha: 0.13,
      });
      drawScatter(ctx, f, palette, points, {
        radius: 3.4,
        colourOf: (_p, i) => {
          const c = state.assignments[i];
          return c >= 0 ? categorical(palette, c) : palette.muted;
        },
      });
      state.trails.forEach((trail, c) => {
        const path = trail.map((p) => ({ x: f.x(p.x), y: f.y(p.y) }));
        drawPath(ctx, path, rgba(categorical(palette, c), 0.55), 1.5, [3, 4]);
      });
      centroids.forEach((c, i) => drawCentroid(ctx, f.x(c.x), f.y(c.y), categorical(palette, i), palette));
    },
    addPoint(px, py) {
      if (!frame) return;
      points.push({ x: frame.x.invert(px), y: frame.y.invert(py), label: 0 });
      state = { ...state, assignments: state.assignments.concat(-1), converged: false, shift: Infinity };
      nextAt = 0;
    },
    status() {
      if (settled()) {
        return 'converged after ' + state.iteration + ' rounds · inertia ' + state.inertia.toFixed(0);
      }
      if (state.phase === 'seeded') return 'centres seeded at random · round 0';
      if (state.phase === 'assign') {
        return 'round ' + (state.iteration + 1) + ' · assign each point to its nearest centre';
      }
      return 'round ' + state.iteration + ' · move each centre to the middle of its points';
    },
  };
}

/* ------------------------------------------------------------------ */

export const DEMOS: Demo[] = [
  {
    key: 'boundary',
    title: 'Draw a boundary',
    hint: 'Left click adds a circle, right click a cross. Train fits the boundary to them.',
    slug: 'neural-network',
    create: boundaryDemo,
  },
  {
    key: 'line',
    title: 'Fit a line',
    hint: 'Click to add a point. Train slides the line towards it.',
    slug: 'linear-regression',
    create: lineDemo,
  },
  {
    key: 'clusters',
    title: 'Find clusters',
    hint: 'Click to add a point. Train assigns points and moves centres in turn.',
    slug: 'k-means',
    create: clustersDemo,
  },
];

export function getDemo(key: DemoKey): Demo {
  return DEMOS.find((d) => d.key === key) ?? DEMOS[0];
}
