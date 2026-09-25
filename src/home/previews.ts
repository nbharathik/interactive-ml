/** Card thumbnails: each explainer's real output, computed once from the model code. */

import type { Point2D } from '../lib/datasets/types';
import { getClassificationDataset, getClusteringDataset, splitData } from '../lib/datasets/points';
import { getRegressionDataset } from '../lib/datasets/regression';
import { getCurveDataset } from '../lib/datasets/curves';
import { fitPolynomial, predictFit } from '../lib/ml/polynomialRegression';
import * as linreg from '../lib/ml/linearRegression';
import * as logreg from '../lib/ml/logisticRegression';
import * as kmeans from '../lib/ml/kmeans';
import * as tree from '../lib/ml/decisionTree';
import * as net from '../lib/ml/neuralNetwork';
import * as knn from '../lib/ml/knn';
import { generateTabular, testRows, trainRows } from '../lib/datasets/tabular';
import { lossSlice, olsSolution, prepare, regularisationPath, ridgeClosedForm } from '../lib/ml/elasticNet';
import { loss as lossOf } from '../lib/ml/losses';
import { layerForward } from '../lib/ml/mlp';
import { createTrainer, dataFromPoints, scoreFromOutput, stepTrainer } from '../lib/ml/mlpTrainer';
import type { TrainerConfig } from '../lib/ml/mlpTrainer';
import { extent2, gridLines, traceLayers } from '../lib/ml/warp';
import { generateGlyphs } from '../lib/datasets/images';
import { convForward, reluForward, tensor } from '../lib/ml/conv';
import { strokeWire } from '../explainer/diagramStyle';
import { drawCross, drawPoint, roundRect } from '../lib/viz/canvas';
import { isoLevels, sampleGrid } from '../lib/viz/contours';
import { categorical, rgba } from '../lib/viz/palette';
import {
  classColour,
  drawCategoryField,
  drawContourMap,
  drawFunction,
  drawIsoLines,
  drawPixelGrid,
  drawResiduals,
  drawScatter,
  drawSignedField,
} from '../lib/viz/plots';

import { drawCentroid, drawGrid, plainFrame } from './canvas';
import type { DrawArgs } from '../explainer/components/Chart';

type Preview = (args: DrawArgs) => void;

const builders: Record<string, () => Preview> = {
  'linear-regression': () => {
    const data = getRegressionDataset('linear').generate({ count: 36, noise: 0.3, seed: 7 });
    const norm = linreg.makeNormaliser(data.points, true);
    const weights = linreg.closedFormSolution(data.points, 1, norm) ?? [0, 0];
    const predict = (x: number) => linreg.predict(weights, x, 1, norm);
    return ({ ctx, width, height, palette }) => {
      const frame = plainFrame(width, height, data.xRange, data.yRange, false);
      drawGrid(ctx, frame, palette);
      drawResiduals(ctx, frame, palette, data.points, predict, { colour: palette.orange, alpha: 0.35 });
      drawScatter(ctx, frame, palette, data.points, { radius: 3.2, colourOf: () => palette.blue });
      drawFunction(ctx, frame, predict, palette.orange, { width: 2.4 });
    };
  },

  'polynomial-regression': () => {
    const data = getCurveDataset('wave').generate({ count: 25, noise: 0.5, seed: 42 });
    const good = fitPolynomial(data.points, { degree: 3, l2: 0, standardise: true });
    const wild = fitPolynomial(data.points, { degree: 12, l2: 0, standardise: true });
    return ({ ctx, width, height, palette }) => {
      const frame = plainFrame(width, height, data.xRange, data.yRange, false);
      drawGrid(ctx, frame, palette);
      drawFunction(ctx, frame, (x) => predictFit(wild, x), rgba(palette.orange, 0.45), { width: 1.6, samples: 240 });
      drawScatter(ctx, frame, palette, data.points, { radius: 3.2, colourOf: () => palette.blue });
      drawFunction(ctx, frame, (x) => predictFit(good, x), palette.orange, { width: 2.4 });
    };
  },

  'logistic-regression': () => {
    const data = getClassificationDataset('gaussians').generate({ count: 70, noise: 0.35, seed: 5 });
    const config: logreg.LogRegConfig = {
      learningRate: 0.3,
      featureMap: 'linear',
      l2: 0,
      batchSize: 70,
      threshold: 0.5,
      standardise: true,
      seed: 2,
    };
    const std = logreg.makeStandardiser(data.points, true);
    let state = logreg.createState(config);
    for (let i = 0; i < 200; i++) state = logreg.step(state, data.points, config, std);
    return ({ ctx, width, height, palette }) => {
      const frame = plainFrame(width, height, data.xRange, data.yRange);
      drawGrid(ctx, frame, palette);
      drawSignedField(
        ctx,
        frame,
        palette,
        (x, y) => logreg.probability(state.weights, x, y, config.featureMap, std) * 2 - 1,
        { cellSize: 5, alpha: 0.45, contourColour: palette.text },
      );
      drawScatter(ctx, frame, palette, data.points, { radius: 3.2, shapes: true });
    };
  },

  'k-means': () => {
    const data = getClusteringDataset('blobs').generate({ count: 90, noise: 0.25, seed: 27, classCount: 3 });
    const config: kmeans.KMeansConfig = { k: 3, init: 'kmeans++', seed: 1, tolerance: 0.01 };
    let state = kmeans.createState(data.points, config, data);
    for (let i = 0; i < 60 && !kmeans.isComplete(state, config.tolerance); i++) {
      state = kmeans.step(state, data.points);
    }
    return ({ ctx, width, height, palette }) => {
      const frame = plainFrame(width, height, data.xRange, data.yRange);
      drawGrid(ctx, frame, palette);
      drawCategoryField(ctx, frame, palette, (x, y) => kmeans.nearestCentroid(state.centroids, x, y), {
        cellSize: 6,
        alpha: 0.13,
      });
      drawScatter(ctx, frame, palette, data.points, {
        radius: 3.2,
        colourOf: (_p, i) => categorical(palette, state.assignments[i]),
      });
      state.centroids.forEach((c, i) =>
        drawCentroid(ctx, frame.x(c.x), frame.y(c.y), categorical(palette, i), palette, 6),
      );
    };
  },

  'decision-tree': () => {
    const data = getClassificationDataset('xor').generate({ count: 90, noise: 0.25, seed: 3 });
    const config: tree.TreeConfig = {
      criterion: 'gini',
      maxDepth: 4,
      minSamplesSplit: 4,
      minSamplesLeaf: 2,
      minImpurityDecrease: 0,
      ccpAlpha: 0,
      classCount: 2,
    };
    const bounds = { x0: data.xRange[0], x1: data.xRange[1], y0: data.yRange[0], y1: data.yRange[1] };
    const state = tree.growAll(tree.createState(data.points, config, bounds), data.points, config);
    return ({ ctx, width, height, palette }) => {
      const frame = plainFrame(width, height, data.xRange, data.yRange);
      drawGrid(ctx, frame, palette);
      for (const { node, bounds: b } of tree.leafRegions(state.root)) {
        ctx.fillStyle = rgba(classColour(palette, node.prediction), 0.22);
        ctx.fillRect(frame.x(b.x0), frame.y(b.y1), frame.x(b.x1) - frame.x(b.x0), frame.y(b.y0) - frame.y(b.y1));
      }
      ctx.strokeStyle = rgba(palette.text, 0.55);
      ctx.lineWidth = 1.2;
      for (const cut of tree.splitLines(state.root)) {
        ctx.beginPath();
        if (cut.feature === 0) {
          ctx.moveTo(frame.x(cut.threshold), frame.y(cut.bounds.y0));
          ctx.lineTo(frame.x(cut.threshold), frame.y(cut.bounds.y1));
        } else {
          ctx.moveTo(frame.x(cut.bounds.x0), frame.y(cut.threshold));
          ctx.lineTo(frame.x(cut.bounds.x1), frame.y(cut.threshold));
        }
        ctx.stroke();
      }
      drawScatter(ctx, frame, palette, data.points, { radius: 3.2, shapes: true });
    };
  },

  'neural-network': () => {
    const data = getClassificationDataset('moons').generate({ count: 100, noise: 0.2, seed: 11 });
    const config: net.NetConfig = {
      hiddenLayers: [6, 6],
      activation: 'tanh',
      learningRate: 0.05,
      batchSize: 10,
      inputFeatures: ['x1', 'x2'],
      initialiser: 'xavier',
      regularisation: 'none',
      regRate: 0,
      seed: 3,
      testFraction: 0,
    };
    const training: net.TrainingData = { train: data.points, test: [] };
    let state = net.createState(config);
    for (let i = 0; i < 150; i++) state = net.step(state, training, config);
    return ({ ctx, width, height, palette }) => {
      const frame = plainFrame(width, height, data.xRange, data.yRange);
      drawGrid(ctx, frame, palette);
      drawSignedField(
        ctx,
        frame,
        palette,
        (x, y) =>
          Math.tanh(net.evaluate(state.layers, net.inputVector(config.inputFeatures, x, y), config.activation) * 0.6),
        { cellSize: 5, alpha: 0.5, contourColour: palette.text },
      );
      drawScatter(ctx, frame, palette, data.points, { radius: 3.2, shapes: true });
    };
  },

  knn: () => {
    const data = getClassificationDataset('three-class').generate({ count: 75, noise: 0.3, seed: 9 });
    const config: knn.KnnConfig = { k: 5, metric: 'euclidean', weighting: 'uniform', classCount: 3, standardise: false };
    const scaling = knn.makeScaling(data.points, false);
    const query: Point2D = { x: 0.6, y: -0.4, label: -1 };
    const neighbours = knn.findNeighbours(data.points, query.x, query.y, config, scaling);
    return ({ ctx, width, height, palette }) => {
      const frame = plainFrame(width, height, data.xRange, data.yRange);
      drawGrid(ctx, frame, palette);
      drawCategoryField(ctx, frame, palette, (x, y) => knn.classify(data.points, x, y, config, scaling).winner, {
        cellSize: 6,
        alpha: 0.16,
      });
      drawScatter(ctx, frame, palette, data.points, { radius: 3.2, shapes: true });
      const qx = frame.x(query.x);
      const qy = frame.y(query.y);
      ctx.strokeStyle = rgba(palette.text, 0.5);
      ctx.lineWidth = 1.2;
      for (const n of neighbours) {
        ctx.beginPath();
        ctx.moveTo(qx, qy);
        ctx.lineTo(frame.x(n.point.x), frame.y(n.point.y));
        ctx.stroke();
      }
      drawPoint(ctx, qx, qy, 5, palette.text, palette.surface, 2);
    };
  },

  regularization: () => {
    // Two correlated features, one of them useless: the lasso parks it at a corner of the diamond.
    const data = generateTabular({
      count: 60,
      features: 2,
      informative: 1,
      correlation: 0.8,
      groupSize: 2,
      noise: 0.4,
      variedScales: false,
      seed: 5,
      trainFraction: 0.75,
    });
    const train = trainRows(data);
    const prep = prepare(train, true);
    const lambda = 1.5;
    const ols = olsSolution(prep);
    const lasso = regularisationPath(train, testRows(data), prep, 1, [lambda]).weights[0];
    const ridge = ridgeClosedForm(prep, lambda);
    const slice = lossSlice(prep, lasso, 0, 1);
    const diamond = Math.abs(lasso[0]) + Math.abs(lasso[1]);
    const circle = Math.hypot(ridge[0], ridge[1]);
    return ({ ctx, width, height, palette }) => {
      const frame = plainFrame(width, height, [-1, 3.2], [-1.75, 1.75]);
      const grid = sampleGrid(frame, slice, 3);
      drawGrid(ctx, frame, palette);
      ctx.save();
      ctx.globalAlpha = 0.55;
      drawContourMap(ctx, frame, palette, slice, { levels: 12, grid });
      ctx.restore();
      drawIsoLines(ctx, frame, palette, slice, isoLevels(grid.min, grid.max, 12, 'log'), { grid, alpha: 0.3, colour: palette.text });
      ctx.save();
      ctx.strokeStyle = rgba(palette.text, 0.35);
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(frame.left, Math.round(frame.y(0)) + 0.5);
      ctx.lineTo(frame.right, Math.round(frame.y(0)) + 0.5);
      ctx.moveTo(Math.round(frame.x(0)) + 0.5, frame.top);
      ctx.lineTo(Math.round(frame.x(0)) + 0.5, frame.bottom);
      ctx.stroke();
      ctx.beginPath();
      ctx.arc(frame.x(0), frame.y(0), frame.x(circle) - frame.x(0), 0, Math.PI * 2);
      ctx.setLineDash([4, 3]);
      ctx.strokeStyle = rgba(palette.blue, 0.8);
      ctx.lineWidth = 1.6;
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.beginPath();
      ctx.moveTo(frame.x(diamond), frame.y(0));
      ctx.lineTo(frame.x(0), frame.y(diamond));
      ctx.lineTo(frame.x(-diamond), frame.y(0));
      ctx.lineTo(frame.x(0), frame.y(-diamond));
      ctx.closePath();
      ctx.fillStyle = rgba(palette.surface, 0.6);
      ctx.fill();
      ctx.fillStyle = rgba(palette.blue, 0.12);
      ctx.fill();
      ctx.strokeStyle = palette.blue;
      ctx.lineWidth = 2;
      ctx.stroke();
      ctx.restore();
      drawCross(ctx, frame.x(ols[0]), frame.y(ols[1]), 5, palette.green, 2.2);
      drawPoint(ctx, frame.x(ridge[0]), frame.y(ridge[1]), 4, palette.surface, palette.blue, 2);
      drawPoint(ctx, frame.x(lasso[0]), frame.y(lasso[1]), 5, palette.orange, palette.surface, 1.6);
    };
  },

  'activation-functions': () => {
    // Two tanh layers of two units on the moons, seen in the first hidden space: the grid bent, the boundary drawn over it.
    const seed = 9;
    const data = getClassificationDataset('moons').generate({ count: 120, noise: 0.25, seed });
    const split = splitData(data.points, 0.75, seed);
    const training = dataFromPoints(split.train, split.test, lossOf('bce', { classCount: 2 }), 2);
    const config: TrainerConfig = {
      task: 'classify',
      depth: 2,
      width: 2,
      activation: 'tanh',
      loss: 'bce',
      classCount: 2,
      init: 'xavier',
      initScale: 1,
      seed,
      optimiser: { name: 'adam' },
      learningRate: 0.05,
      batchSize: 16,
    };
    let state = createTrainer(config, training);
    for (let i = 0; i < 100; i++) state = stepTrainer(state, training, config);
    const lines = gridLines(2);
    const gridPoints = lines.flatMap((l) => l.points);
    const traces = traceLayers(state.net, state.spec, gridPoints.concat(training.train.inputs));
    const stage = 0;
    const space = traces[stage].a as [number, number][];
    // The output's score for a vector of that space, through the layers after it.
    const score = (h: number[]) => {
      let v = h;
      for (let l = stage + 1; l < state.net.layers.length; l++) v = layerForward(state.net.layers[l], v, state.spec.activationParams).a;
      return scoreFromOutput(state.spec.loss, v);
    };
    const bent = space.slice(0, gridPoints.length);
    const labels = training.train.labels ?? [];
    const points = space.slice(gridPoints.length).map(([x, y], i) => ({ x, y, label: labels[i] ?? 0 }));
    const domain = extent2(space);
    return ({ ctx, width, height, palette }) => {
      const frame = plainFrame(width, height, domain.x, domain.y);
      drawSignedField(ctx, frame, palette, (x, y) => score([x, y]), {
        cellSize: 5,
        alpha: 0.45,
        contourColour: palette.text,
      });
      ctx.save();
      let offset = 0;
      for (const line of lines) {
        ctx.strokeStyle = rgba(line.family === 0 ? palette.violet : palette.cyan, line.axis ? 0.9 : 0.4);
        ctx.lineWidth = line.axis ? 1.6 : 1;
        ctx.beginPath();
        line.points.forEach((_, s) => {
          const [x, y] = bent[offset + s];
          if (s === 0) ctx.moveTo(frame.x(x), frame.y(y));
          else ctx.lineTo(frame.x(x), frame.y(y));
        });
        ctx.stroke();
        offset += line.points.length;
      }
      ctx.restore();
      drawScatter(ctx, frame, palette, points, { radius: 3.2, shapes: true });
    };
  },

  cnn: () => {
    const data = generateGlyphs({ size: 16, classCount: 4, count: 8, jitter: 1, thickness: 2, noise: 0.05, seed: 3, trainFraction: 0.5 });
    const image = data.images[3];
    const input = tensor(1, 16, 16, Float64Array.from(image));
    // Three hand-made edge detectors stand in for learned filters.
    const kernels = [
      [-1, -1, -1, 2, 2, 2, -1, -1, -1],
      [-1, 2, -1, -1, 2, -1, -1, 2, -1],
      [2, -1, -1, -1, 2, -1, -1, -1, 2],
    ].map((k) => ({ kernel: 3, inC: 1, outC: 1, pad: 0, W: Float64Array.from(k.map((v) => v / 3)), b: new Float64Array(1) }));
    const maps = kernels.map((k) => reluForward(convForward(input, k)));
    return ({ ctx, width, height, palette }) => {
      const pad = 10;
      const row = (height - 2 * pad) / maps.length;
      const big = Math.min(height - 2 * pad, width * 0.36);
      const map = row - 6;
      const small = map * 0.62;
      const gap = (width - 2 * pad - big - small - map) / 2;
      const kernelX = pad + big + gap;
      const mapX = kernelX + small + gap;
      drawGrid(ctx, plainFrame(width, height, [0, 6], [0, 3], false), palette);
      // A node of the pipeline: a framed box holding its picture.
      const box = (x: number, y: number, side: number, values: ArrayLike<number>, n: number, ramp: 'signed' | 'magnitude') => {
        roundRect(ctx, x, y, side, side, 3);
        ctx.fillStyle = palette.surfaceAlt;
        ctx.fill();
        ctx.strokeStyle = rgba(palette.accent, 0.55);
        ctx.lineWidth = 1.2;
        ctx.stroke();
        const inner = { x: x + 2, y: y + 2, width: side - 4, height: side - 4 };
        drawPixelGrid(ctx, inner, values, n, n, palette, ramp === 'signed' ? { ramp, gap: 1 } : { ramp, gap: 0, scale: n === 16 ? 1 : undefined });
      };
      const wire = (x0: number, y0: number, x1: number, y1: number) => {
        const mid = (x0 + x1) / 2;
        const path = { start: { x: x0, y: y0 }, c1: { x: mid, y: y0 }, c2: { x: mid, y: y1 }, end: { x: x1, y: y1 }, angle: 0, backwards: false, label: { x: mid, y: (y0 + y1) / 2 } };
        strokeWire(ctx, path, { colour: palette.textMuted, alpha: 0.6, width: 1.6, emphasis: 'plain' });
      };
      maps.forEach((m, i) => {
        const cy = pad + row * (i + 0.5);
        wire(pad + big + 2, height / 2, kernelX - 2, cy);
        wire(kernelX + small + 2, cy, mapX - 2, cy);
        box(kernelX, cy - small / 2, small, kernels[i].W, 3, 'signed');
        box(mapX, cy - map / 2, map, m.data, m.w, 'magnitude');
      });
      box(pad, (height - big) / 2, big, image, 16, 'magnitude');
    };
  },
};

const cache = new Map<string, Preview>();

/** The thumbnail for a slug, or null for an explainer without one. */
export function getPreview(slug: string): Preview | null {
  const cached = cache.get(slug);
  if (cached) return cached;
  const build = builders[slug];
  if (!build) return null;
  const preview = build();
  cache.set(slug, preview);
  return preview;
}
