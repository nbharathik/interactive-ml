/** Numerical checks for the algorithm modules: the maths, purity of every `step`, the failure modes. */

import { strict as assert } from 'node:assert';
import { describe, it } from 'node:test';

import { makeRng, gauss, shuffled } from '../src/lib/math/rng.ts';
import {
  confusionMatrix,
  logLoss,
  meanSquaredError,
  precisionRecallF1,
  rSquared,
  rocAuc,
  rocCurve,
  fitStandardizer,
} from '../src/lib/math/stats.ts';
import { REGRESSION_DATASETS } from '../src/lib/datasets/regression.ts';
import { CLASSIFICATION_DATASETS, CLUSTERING_DATASETS, splitData } from '../src/lib/datasets/points.ts';
import type { Point2D } from '../src/lib/datasets/types.ts';

import {
  GraphBuilder,
  edgePath,
  hitTestEdge,
  hitTestNode,
  layoutArchitecture,
  sampleEdge,
} from '../src/explainer/architecture.ts';
import * as lin from '../src/lib/ml/linearRegression.ts';
import * as log from '../src/lib/ml/logisticRegression.ts';
import * as km from '../src/lib/ml/kmeans.ts';
import * as dt from '../src/lib/ml/decisionTree.ts';
import * as nn from '../src/lib/ml/neuralNetwork.ts';
import * as knn from '../src/lib/ml/knn.ts';

const BOUNDS = { xRange: [-6, 6] as [number, number], yRange: [-6, 6] as [number, number] };
const RECT = { x0: -6, x1: 6, y0: -6, y1: 6 };

function close(actual: number, expected: number, tolerance = 1e-6, message?: string) {
  assert.ok(
    Math.abs(actual - expected) <= tolerance,
    (message ?? 'value') + ': expected ' + expected + ', got ' + actual,
  );
}

/* ================================================================== */

describe('rng', () => {
  it('is deterministic for a given seed', () => {
    const a = Array.from({ length: 5 }, makeRng(7));
    const b = Array.from({ length: 5 }, makeRng(7));
    assert.deepEqual(a, b);
  });

  it('produces values in [0, 1)', () => {
    const rng = makeRng(3);
    for (let i = 0; i < 5000; i++) {
      const v = rng();
      assert.ok(v >= 0 && v < 1, 'out of range: ' + v);
    }
  });

  it('gauss has roughly the requested mean and spread', () => {
    const rng = makeRng(11);
    const values = Array.from({ length: 20000 }, () => gauss(rng, 5, 2));
    const mean = values.reduce((a, b) => a + b, 0) / values.length;
    const sd = Math.sqrt(values.reduce((a, b) => a + (b - mean) ** 2, 0) / values.length);
    close(mean, 5, 0.1, 'mean');
    close(sd, 2, 0.1, 'sd');
  });

  it('shuffled keeps every element exactly once', () => {
    const source = Array.from({ length: 50 }, (_, i) => i);
    const out = shuffled(makeRng(5), source);
    assert.equal(out.length, source.length);
    assert.deepEqual(out.slice().sort((a, b) => a - b), source);
  });
});

/* ================================================================== */

describe('stats', () => {
  it('R² is 1 for a perfect fit and 0 for predicting the mean', () => {
    const actual = [1, 2, 3, 4];
    close(rSquared(actual, [1, 2, 3, 4]), 1, 1e-12, 'perfect');
    close(rSquared(actual, [2.5, 2.5, 2.5, 2.5]), 0, 1e-12, 'mean-only');
  });

  it('MSE matches a hand calculation', () => {
    close(meanSquaredError([1, 2], [2, 4]), (1 + 4) / 2);
  });

  it('confusion matrix counts land in the right cells', () => {
    const cm = confusionMatrix([0, 0, 1, 1], [0, 1, 1, 1], 2);
    assert.equal(cm.counts[0][0], 1);
    assert.equal(cm.counts[0][1], 1);
    assert.equal(cm.counts[1][1], 2);
    assert.equal(cm.total, 4);
  });

  it('precision, recall and F1 agree with their definitions', () => {
    // predicted positive: 3, of which 2 correct; actual positives: 2, both found
    const r = precisionRecallF1([0, 1, 1, 0], [1, 1, 1, 0], 1);
    close(r.precision, 2 / 3, 1e-12, 'precision');
    close(r.recall, 1, 1e-12, 'recall');
    close(r.f1, (2 * (2 / 3)) / (2 / 3 + 1), 1e-12, 'f1');
  });

  it('log loss is near zero for confident correct predictions', () => {
    assert.ok(logLoss([1, 0], [0.999, 0.001]) < 0.01);
    assert.ok(logLoss([1, 0], [0.001, 0.999]) > 5);
  });

  it('ROC AUC is 1 for a perfect ranking and 0.5 for a constant score', () => {
    close(rocAuc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]), 1, 1e-12, 'perfect');
    close(rocAuc([0, 1, 0, 1], [0.5, 0.5, 0.5, 0.5]), 0.5, 1e-12, 'ties');
  });

  it('the ROC curve steps over tied scores at once, whatever their order, and never stalls on NaN', () => {
    const last = (points: ReturnType<typeof rocCurve>) => points[points.length - 1];
    assert.deepEqual(rocCurve([1, 0], [0.5, 0.5]), rocCurve([0, 1], [0.5, 0.5]));
    assert.equal(rocCurve([1, 0], [0.5, 0.5]).length, 2);
    const withNaN = rocCurve([1, 0, 1, 0], [Number.NaN, 0.2, Number.NaN, 0.9]);
    assert.equal(last(withNaN).tpr, 1);
    assert.equal(last(withNaN).fpr, 1);
  });

  it('standardiser produces zero mean and unit variance', () => {
    const rows = [[1, 10], [2, 20], [3, 30], [4, 40]];
    const s = fitStandardizer(rows);
    const transformed = rows.map((r) => s.transform(r));
    for (let d = 0; d < 2; d++) {
      const col = transformed.map((r) => r[d]);
      const mean = col.reduce((a, b) => a + b, 0) / col.length;
      close(mean, 0, 1e-9, 'mean of column ' + d);
    }
    // and it round-trips
    assert.deepEqual(
      s.inverse(s.transform(rows[2])).map((v) => Math.round(v * 1e9) / 1e9),
      rows[2],
    );
  });
});

/* ================================================================== */

describe('datasets', () => {
  it('every regression generator is deterministic and in range', () => {
    for (const ds of REGRESSION_DATASETS) {
      const config = { count: 40, noise: 0.3, seed: 5 };
      const a = ds.generate(config);
      const b = ds.generate(config);
      assert.deepEqual(a.points, b.points, ds.id + ' is not deterministic');
      assert.equal(a.points.length, 40, ds.id + ' wrong count');
      for (const p of a.points) {
        assert.ok(Number.isFinite(p.x) && Number.isFinite(p.y), ds.id + ' produced a non-finite point');
        assert.ok(p.x >= a.xRange[0] && p.x <= a.xRange[1], ds.id + ' x out of declared range');
      }
      assert.ok(a.yRange[1] > a.yRange[0], ds.id + ' has an empty y range');
    }
  });

  it('every point generator is deterministic, in range, and correctly labelled', () => {
    for (const ds of [...CLASSIFICATION_DATASETS, ...CLUSTERING_DATASETS]) {
      const config = { count: 60, noise: 0.3, seed: 5, classCount: 3 };
      const a = ds.generate(config);
      const b = ds.generate(config);
      assert.deepEqual(a.points, b.points, ds.id + ' is not deterministic');
      assert.ok(a.points.length > 0, ds.id + ' produced nothing');
      assert.equal(a.classNames.length, a.classCount, ds.id + ' classNames/classCount mismatch');
      for (const p of a.points) {
        assert.ok(Number.isFinite(p.x) && Number.isFinite(p.y), ds.id + ' non-finite point');
        assert.ok(p.x >= -6 && p.x <= 6 && p.y >= -6 && p.y <= 6, ds.id + ' point outside [-6, 6]');
        assert.ok(
          Number.isInteger(p.label) && p.label >= 0 && p.label < a.classCount,
          ds.id + ' label ' + p.label + ' outside 0..' + (a.classCount - 1),
        );
      }
    }
  });

  it('classification sets that claim two classes actually contain both', () => {
    for (const ds of CLASSIFICATION_DATASETS) {
      const data = ds.generate({ count: 120, noise: 0.25, seed: 3, classCount: 3 });
      const present = new Set(data.points.map((p) => p.label));
      assert.ok(present.size >= 2, ds.id + ' produced only one class');
    }
  });

  it('splitData partitions without losing or duplicating points', () => {
    const items = Array.from({ length: 47 }, (_, i) => i);
    const { train, test } = splitData(items, 0.7, 12);
    assert.equal(train.length + test.length, items.length);
    assert.deepEqual([...train, ...test].sort((a, b) => a - b), items);
  });
});

/* ================================================================== */

describe('linear regression', () => {
  const points = [
    { x: 0, y: 1 },
    { x: 1, y: 3 },
    { x: 2, y: 5 },
    { x: 3, y: 7 },
  ]; // exactly y = 2x + 1

  const baseConfig: lin.LinRegConfig = {
    learningRate: 0.1,
    degree: 1,
    optimiser: 'gd',
    momentum: 0.9,
    batchMode: 'batch',
    batchSize: 4,
    l2: 0,
    standardise: false,
    seed: 1,
  };

  it('the closed form recovers an exact linear relationship', () => {
    const norm = lin.makeNormaliser(points, false);
    const solution = lin.closedFormSolution(points, 1, norm, 0);
    assert.ok(solution, 'no solution returned');
    close(solution[0], 1, 1e-8, 'intercept');
    close(solution[1], 2, 1e-8, 'slope');
  });

  it('gradient descent converges towards the closed form', () => {
    const norm = lin.makeNormaliser(points, false);
    let state = lin.createState(baseConfig);
    for (let i = 0; i < 4000; i++) state = lin.step(state, points, baseConfig, norm);
    close(state.weights[0], 1, 1e-3, 'intercept');
    close(state.weights[1], 2, 1e-3, 'slope');
    assert.ok(state.loss < 1e-6, 'loss should be ~0, got ' + state.loss);
  });

  it('step is pure: it never mutates the state it is given', () => {
    const norm = lin.makeNormaliser(points, false);
    const state = lin.createState(baseConfig);
    const before = JSON.stringify(state);
    const next = lin.step(state, points, baseConfig, norm);
    assert.equal(JSON.stringify(state), before, 'step mutated its input');
    assert.notEqual(next, state, 'step returned the same object');
    assert.notEqual(next.weights, state.weights, 'step shared the weights array');
  });

  it('loss decreases monotonically at a sane learning rate', () => {
    const norm = lin.makeNormaliser(points, false);
    let state = lin.createState(baseConfig);
    let previous = Infinity;
    for (let i = 0; i < 200; i++) {
      state = lin.step(state, points, baseConfig, norm);
      assert.ok(state.loss <= previous + 1e-12, 'loss rose at step ' + i);
      previous = state.loss;
    }
  });

  it('a large learning rate diverges, and the flag is set', () => {
    const config = { ...baseConfig, learningRate: 5 };
    const norm = lin.makeNormaliser(points, false);
    let state = lin.createState(config);
    for (let i = 0; i < 400 && !state.diverged; i++) state = lin.step(state, points, config, norm);
    assert.equal(state.diverged, true, 'expected divergence at lr = 5');
    // Once diverged, further steps are a no-op rather than a crash.
    const after = lin.step(state, points, config, norm);
    assert.equal(after, state);
  });

  it('the analytic gradient matches a numerical one', () => {
    const config = { ...baseConfig, degree: 2, l2: 0.01 };
    const norm = lin.makeNormaliser(points, true);
    const weights = [0.3, -0.7, 0.2];
    const batch = points.map((_, i) => i);
    const analytic = lin.computeGradient(weights, points, batch, config, norm);
    const h = 1e-6;
    for (let i = 0; i < weights.length; i++) {
      const up = weights.slice();
      const down = weights.slice();
      up[i] += h;
      down[i] -= h;
      const numeric =
        (lin.computeLoss(up, points, config, norm) - lin.computeLoss(down, points, config, norm)) /
        (2 * h);
      close(analytic[i], numeric, 1e-4, 'gradient component ' + i);
    }
  });

  it('standardising and not standardising reach the same line in original units', () => {
    const data = REGRESSION_DATASETS[0].generate({ count: 80, noise: 0.2, seed: 4 });
    const results = [false, true].map((standardise) => {
      const config = { ...baseConfig, standardise, learningRate: standardise ? 0.1 : 0.01 };
      const norm = lin.makeNormaliser(data.points, standardise);
      let state = lin.createState(config);
      for (let i = 0; i < 8000; i++) state = lin.step(state, data.points, config, norm);
      return lin.toOriginalUnits(state.weights, norm);
    });
    close(results[0].slope, results[1].slope, 1e-2, 'slope');
    close(results[0].intercept, results[1].intercept, 1e-2, 'intercept');
  });

  it('an L2 penalty shrinks the weights', () => {
    const data = REGRESSION_DATASETS[2].generate({ count: 30, noise: 0.5, seed: 8 });
    const norm = lin.makeNormaliser(data.points, true);
    const magnitude = (l2: number) => {
      const solution = lin.closedFormSolution(data.points, 5, norm, l2);
      assert.ok(solution);
      return solution.slice(1).reduce((a, w) => a + Math.abs(w), 0);
    };
    assert.ok(magnitude(0.5) < magnitude(0), 'penalty did not shrink the weights');
  });

  it('solve returns null for a singular system', () => {
    assert.equal(lin.solve([[1, 2], [2, 4]], [1, 2]), null);
  });
});

/* ================================================================== */

describe('logistic regression', () => {
  const config: log.LogRegConfig = {
    learningRate: 0.5,
    featureMap: 'linear',
    l2: 0,
    batchSize: 1000,
    threshold: 0.5,
    standardise: false,
    seed: 1,
  };

  const separable: Point2D[] = [
    { x: -3, y: -3, label: 0 },
    { x: -2, y: -2.5, label: 0 },
    { x: -2.5, y: -3.5, label: 0 },
    { x: 3, y: 3, label: 1 },
    { x: 2, y: 2.5, label: 1 },
    { x: 2.5, y: 3.5, label: 1 },
  ];

  it('sigmoid is correct and numerically stable at the extremes', () => {
    close(log.sigmoid(0), 0.5, 1e-12);
    close(log.sigmoid(1), 1 / (1 + Math.exp(-1)), 1e-12);
    assert.ok(Number.isFinite(log.sigmoid(1000)) && log.sigmoid(1000) === 1);
    assert.ok(Number.isFinite(log.sigmoid(-1000)) && log.sigmoid(-1000) === 0);
  });

  it('separates a separable set', () => {
    const std = log.makeStandardiser(separable, false);
    let state = log.createState(config);
    for (let i = 0; i < 2000; i++) state = log.step(state, separable, config, std);
    assert.equal(log.accuracyOf(state.weights, separable, config, std), 1);
    assert.ok(state.loss < 0.05, 'loss should be small, got ' + state.loss);
  });

  it('step is pure', () => {
    const std = log.makeStandardiser(separable, false);
    const state = log.createState(config);
    const before = JSON.stringify(state);
    const next = log.step(state, separable, config, std);
    assert.equal(JSON.stringify(state), before);
    assert.notEqual(next.weights, state.weights);
  });

  it('cross-entropy decreases monotonically on full batches', () => {
    const std = log.makeStandardiser(separable, false);
    let state = log.createState({ ...config, learningRate: 0.2 });
    let previous = Infinity;
    for (let i = 0; i < 300; i++) {
      state = log.step(state, separable, { ...config, learningRate: 0.2 }, std);
      assert.ok(state.loss <= previous + 1e-9, 'loss rose at step ' + i);
      previous = state.loss;
    }
  });

  it('the analytic gradient matches a numerical one', () => {
    const cfg = { ...config, featureMap: 'quadratic' as const, l2: 0.02 };
    const std = log.makeStandardiser(separable, true);
    const weights = [0.1, -0.4, 0.25, 0.05, -0.15, 0.3];
    const batch = separable.map((_, i) => i);
    const analytic = log.computeGradient(weights, separable, batch, cfg, std);
    const h = 1e-6;
    for (let i = 0; i < weights.length; i++) {
      const up = weights.slice();
      const down = weights.slice();
      up[i] += h;
      down[i] -= h;
      const numeric =
        (log.computeLoss(up, separable, cfg, std) - log.computeLoss(down, separable, cfg, std)) /
        (2 * h);
      close(analytic[i], numeric, 1e-4, 'gradient component ' + i);
    }
  });

  it('a linear map cannot separate the ring, a quadratic map can', () => {
    const data = CLASSIFICATION_DATASETS.find((d) => d.id === 'circle')!.generate({
      count: 160,
      noise: 0.1,
      seed: 2,
    });
    const run = (featureMap: log.FeatureMap) => {
      const cfg = { ...config, featureMap, learningRate: 0.4, standardise: true };
      const std = log.makeStandardiser(data.points, true);
      let state = log.createState(cfg);
      for (let i = 0; i < 3000; i++) state = log.step(state, data.points, cfg, std);
      return log.accuracyOf(state.weights, data.points, cfg, std);
    };
    const linear = run('linear');
    const quadratic = run('quadratic');
    assert.ok(linear < 0.7, 'a line should fail on the ring, got ' + linear);
    assert.ok(quadratic > 0.9, 'a quadratic map should succeed, got ' + quadratic);
  });

  it('the threshold changes predictions without touching the weights', () => {
    const std = log.makeStandardiser(separable, false);
    let state = log.createState(config);
    for (let i = 0; i < 200; i++) state = log.step(state, separable, config, std);
    const weights = state.weights.slice();
    const low = log.predictAll(state.weights, separable, { ...config, threshold: 0.1 }, std);
    const high = log.predictAll(state.weights, separable, { ...config, threshold: 0.9 }, std);
    assert.deepEqual(state.weights, weights, 'prediction mutated the weights');
    assert.deepEqual(low.probabilities, high.probabilities, 'threshold changed the probabilities');
    assert.ok(
      low.labels.filter((l) => l === 1).length >= high.labels.filter((l) => l === 1).length,
      'a lower threshold should predict at least as many positives',
    );
  });

  it('featureNames and features agree on length', () => {
    for (const map of ['linear', 'interaction', 'quadratic'] as const) {
      const std = log.makeStandardiser(separable, false);
      assert.equal(
        log.features(1, 2, map, std).length,
        log.featureNames(map).length,
        map + ' length mismatch',
      );
      assert.equal(log.featureCount(map), log.featureNames(map).length);
    }
  });
});

/* ================================================================== */

describe('k-means', () => {
  const config: km.KMeansConfig = { k: 3, init: 'kmeans++', seed: 7, tolerance: 1e-6 };

  const data = CLUSTERING_DATASETS.find((d) => d.id === 'blobs')!.generate({
    count: 150,
    noise: 0.2,
    seed: 3,
    classCount: 3,
  });

  it('seeds exactly k centroids for every init method', () => {
    for (const init of ['random-points', 'kmeans++', 'random-coords', 'forgy-far'] as const) {
      const centroids = km.initCentroids(data.points, { ...config, init }, BOUNDS);
      assert.equal(centroids.length, 3, init + ' produced the wrong count');
      assert.ok(
        centroids.every((c) => Number.isFinite(c.x) && Number.isFinite(c.y)),
        init + ' produced a non-finite centroid',
      );
    }
  });

  it('handles k greater than the number of points without hanging', () => {
    const tiny = data.points.slice(0, 2);
    const centroids = km.initCentroids(tiny, { ...config, k: 5, init: 'random-points' }, BOUNDS);
    assert.equal(centroids.length, 5);
  });

  it('alternates assign and update', () => {
    let state = km.createState(data.points, config, BOUNDS);
    assert.equal(state.phase, 'seeded');
    state = km.step(state, data.points);
    assert.equal(state.phase, 'assign');
    state = km.step(state, data.points);
    assert.equal(state.phase, 'update');
    state = km.step(state, data.points);
    assert.equal(state.phase, 'assign');
  });

  it('step is pure', () => {
    const seeded = km.createState(data.points, config, BOUNDS);
    const before = JSON.stringify(seeded);

    // assign() may keep the same centroids array; it does not touch it.
    const assigned = km.step(seeded, data.points);
    assert.equal(JSON.stringify(seeded), before, 'assign mutated its input');
    assert.notEqual(assigned, seeded, 'assign returned the same object');
    assert.notEqual(assigned.assignments, seeded.assignments, 'assign shared the assignments array');

    // update() must produce fresh centroids and leave the old ones intact.
    const centroidsBefore = JSON.stringify(assigned.centroids);
    const updated = km.step(assigned, data.points);
    assert.equal(JSON.stringify(assigned.centroids), centroidsBefore, 'update mutated the centroids');
    assert.notEqual(updated.centroids, assigned.centroids, 'update shared the centroids array');
    assert.notEqual(updated.trails, assigned.trails, 'update shared the trails array');
  });

  it('inertia never increases', () => {
    let state = km.createState(data.points, config, BOUNDS);
    let previous = Infinity;
    for (let i = 0; i < 40; i++) {
      const next = km.step(state, data.points);
      if (next === state) break;
      state = next;
      if (state.iteration > 0) {
        assert.ok(state.inertia <= previous + 1e-9, 'inertia rose at step ' + i);
        previous = state.inertia;
      }
    }
  });

  it('converges and stops moving', () => {
    let state = km.createState(data.points, config, BOUNDS);
    for (let i = 0; i < 200; i++) {
      const next = km.step(state, data.points);
      if (next === state) break;
      state = next;
    }
    assert.ok(state.iteration > 0, 'never iterated');
    assert.ok(state.converged || state.shift < 1e-6, 'did not settle');
    // Frozen: another step changes nothing.
    const after = km.step(state, data.points);
    assert.equal(after.inertia, state.inertia);
  });

  it('recovers three well-separated blobs', () => {
    let state = km.createState(data.points, config, BOUNDS);
    for (let i = 0; i < 200; i++) {
      const next = km.step(state, data.points);
      if (next === state) break;
      state = next;
    }
    const sizes = km.clusterSizes(state.assignments, 3);
    assert.ok(sizes.every((s) => s > 20), 'expected balanced clusters, got ' + sizes.join('/'));
  });

  it('k-means++ beats the adversarial init on average inertia', () => {
    const runInit = (init: km.InitMethod) => {
      let total = 0;
      for (let seed = 0; seed < 5; seed++) {
        let state = km.createState(data.points, { ...config, init, seed }, BOUNDS);
        for (let i = 0; i < 120; i++) {
          const next = km.step(state, data.points);
          if (next === state) break;
          state = next;
        }
        total += state.inertia;
      }
      return total / 5;
    };
    assert.ok(
      runInit('kmeans++') <= runInit('forgy-far') + 1e-9,
      'k-means++ should not be worse than the deliberately bad init',
    );
  });

  it('silhouette is high for separated blobs and low for noise', () => {
    const score = (id: string, k: number) => {
      const set = CLUSTERING_DATASETS.find((d) => d.id === id)!.generate({
        count: 150,
        noise: 0.15,
        seed: 3,
        classCount: 3,
      });
      let state = km.createState(set.points, { ...config, k }, BOUNDS);
      for (let i = 0; i < 120; i++) {
        const next = km.step(state, set.points);
        if (next === state) break;
        state = next;
      }
      return km.silhouetteScore(set.points, state.assignments, k);
    };
    const blobs = score('blobs', 3);
    const uniform = score('uniform', 3);
    assert.ok(blobs > uniform, 'blobs (' + blobs + ') should score above noise (' + uniform + ')');
  });

  it('the tolerance actually shortens a run', () => {
    // The tolerance must stop a run on its own, without requiring changedCount === 0.
    const noisy = CLUSTERING_DATASETS.find((d) => d.id === 'uniform')!.generate({
      count: 160,
      noise: 0.4,
      seed: 42,
      classCount: 4,
    });
    const roundsAt = (tolerance: number) => {
      const cfg: km.KMeansConfig = { k: 4, init: 'random-points', seed: 42, tolerance };
      let state = km.createState(noisy.points, cfg, BOUNDS);
      for (let i = 0; i < 200; i++) {
        if (km.isComplete(state, tolerance)) break;
        const next = km.step(state, noisy.points);
        if (next === state) break;
        state = next;
      }
      return state.iteration;
    };
    const loose = roundsAt(0.6);
    const tight = roundsAt(0.001);
    assert.ok(loose < tight, 'a loose tolerance should stop sooner: ' + loose + ' vs ' + tight);
  });

  it('the elbow curve is non-increasing in k', () => {
    const curve = km.elbowCurve(data.points.slice(0, 90), 6, { init: 'kmeans++', seed: 1, tolerance: 1e-6 }, BOUNDS, 2);
    assert.equal(curve.length, 6);
    for (let i = 1; i < curve.length; i++) {
      assert.ok(
        curve[i].inertia <= curve[i - 1].inertia + 1e-6,
        'inertia rose from k=' + i + ' to k=' + (i + 1),
      );
    }
  });

  it('survives a stale assignments array', () => {
    // One render can pair a grown point set with stale assignments. Nothing may throw.
    const grown = [...data.points, ...data.points.slice(0, 40)];
    const short = data.points.slice(0, 30).map((_, i) => i % 3);
    assert.doesNotThrow(() => km.silhouetteScore(grown, short, 3));
    assert.doesNotThrow(() => km.computeInertia(grown, short, [{ x: 0, y: 0 }]));
    assert.doesNotThrow(() => km.clusterSizes(short, 3));
  });

  it('survives labels that exceed k', () => {
    // Lowering k leaves assignments pointing at clusters that no longer exist.
    const labels = data.points.map((_, i) => i % 7);
    assert.doesNotThrow(() => km.silhouetteScore(data.points.slice(0, 100), labels, 3));
    const state = km.createState(data.points, { ...config, k: 3 }, BOUNDS);
    const stale = { ...state, assignments: labels, phase: 'assign' as const };
    assert.doesNotThrow(() => km.update(stale, data.points));
    const updated = km.update(stale, data.points);
    assert.ok(
      updated.centroids.every((c) => Number.isFinite(c.x) && Number.isFinite(c.y)),
      'out-of-range labels produced a NaN centroid',
    );
  });

  it('flags empty clusters instead of crashing', () => {
    const clumped: Point2D[] = Array.from({ length: 20 }, () => ({ x: 0, y: 0, label: 0 }));
    let state = km.createState(clumped, { ...config, k: 4, init: 'random-coords' }, BOUNDS);
    state = km.assign(state, clumped);
    state = km.update(state, clumped);
    assert.ok(state.emptyClusters.length >= 3, 'expected empty clusters to be reported');
    assert.ok(state.centroids.every((c) => Number.isFinite(c.x)), 'an empty cluster produced NaN');
  });
});

/* ================================================================== */

describe('decision tree', () => {
  const config: dt.TreeConfig = {
    criterion: 'gini',
    maxDepth: 5,
    minSamplesSplit: 2,
    minSamplesLeaf: 1,
    minImpurityDecrease: 0,
    ccpAlpha: 0,
    classCount: 2,
  };

  const data = CLASSIFICATION_DATASETS.find((d) => d.id === 'gaussians')!.generate({
    count: 120,
    noise: 0.15,
    seed: 4,
  });

  it('impurity is zero for a pure node and maximal for an even split', () => {
    close(dt.impurityOf([10, 0], 'gini'), 0, 1e-12, 'pure gini');
    close(dt.impurityOf([10, 0], 'entropy'), 0, 1e-12, 'pure entropy');
    close(dt.impurityOf([5, 5], 'gini'), 0.5, 1e-12, 'even gini');
    close(dt.impurityOf([5, 5], 'entropy'), 1, 1e-12, 'even entropy');
    close(dt.impurityOf([1, 1, 1, 1], 'entropy'), 2, 1e-12, 'four even classes');
  });

  it('growOne adds exactly one split per call', () => {
    let state = dt.createState(data.points, config, RECT);
    assert.equal(state.nodeCount, 1);
    for (let i = 1; i <= 4 && !state.finished; i++) {
      state = dt.growOne(state, data.points, config);
      assert.equal(state.nodeCount, 1 + i * 2, 'wrong node count after ' + i + ' splits');
      assert.equal(state.leafCount, i + 1, 'wrong leaf count after ' + i + ' splits');
    }
  });

  it('growOne is pure', () => {
    const state = dt.createState(data.points, config, RECT);
    const before = JSON.stringify(state.root);
    const next = dt.growOne(state, data.points, config);
    assert.equal(JSON.stringify(state.root), before, 'growOne mutated the tree');
    assert.notEqual(next.root, state.root);
  });

  it('respects maxDepth', () => {
    for (const maxDepth of [1, 2, 3]) {
      const state = dt.growAll(
        dt.createState(data.points, { ...config, maxDepth }, RECT),
        data.points,
        { ...config, maxDepth },
      );
      assert.ok(dt.treeDepth(state.root) <= maxDepth, 'depth exceeded maxDepth=' + maxDepth);
    }
  });

  it('respects minSamplesLeaf', () => {
    const cfg = { ...config, maxDepth: 10, minSamplesLeaf: 8 };
    const state = dt.growAll(dt.createState(data.points, cfg, RECT), data.points, cfg);
    dt.walk(state.root, (node) => {
      if (node.isLeaf) {
        assert.ok(
          node.samples.length >= 8 || node.id === 0,
          'leaf with ' + node.samples.length + ' samples violates minSamplesLeaf',
        );
      }
    });
  });

  it('a deep tree reaches perfect training accuracy on separable data', () => {
    const cfg = { ...config, maxDepth: 12 };
    const state = dt.growAll(dt.createState(data.points, cfg, RECT), data.points, cfg);
    close(dt.accuracy(state.root, data.points), 1, 1e-9, 'training accuracy');
  });

  it('leaf regions tile the space without gaps or overlap in area', () => {
    const state = dt.growAll(dt.createState(data.points, config, RECT), data.points, config);
    const regions = dt.leafRegions(state.root);
    const total = regions.reduce(
      (sum, r) => sum + (r.bounds.x1 - r.bounds.x0) * (r.bounds.y1 - r.bounds.y0),
      0,
    );
    close(total, (RECT.x1 - RECT.x0) * (RECT.y1 - RECT.y0), 1e-6, 'total leaf area');
  });

  it('every point lands in the leaf whose bounds contain it', () => {
    const state = dt.growAll(dt.createState(data.points, config, RECT), data.points, config);
    for (const p of data.points) {
      const leaf = dt.decide(state.root, p.x, p.y);
      assert.ok(
        p.x >= leaf.bounds.x0 && p.x <= leaf.bounds.x1 && p.y >= leaf.bounds.y0 && p.y <= leaf.bounds.y1,
        'point (' + p.x + ',' + p.y + ') is outside its own leaf',
      );
    }
  });

  it('decisionPath ends at the leaf decide returns', () => {
    const state = dt.growAll(dt.createState(data.points, config, RECT), data.points, config);
    const path = dt.decisionPath(state.root, 1.2, -0.4);
    assert.equal(path[path.length - 1].id, dt.decide(state.root, 1.2, -0.4).id);
    assert.equal(path[0].id, state.root.id);
  });

  it('the chosen split really is the highest-gain one available', () => {
    const state = dt.createState(data.points, config, RECT);
    const candidates = dt.evaluateSplits(data.points, state.root, config);
    const grown = dt.growOne(state, data.points, config);
    assert.ok(grown.lastSplit, 'no split recorded');
    close(grown.lastSplit.candidate.gain, candidates[0].gain, 1e-12, 'gain of the chosen split');
  });

  it('pruning reduces the leaf count', () => {
    const cfg = { ...config, maxDepth: 12 };
    const state = dt.growAll(dt.createState(data.points, cfg, RECT), data.points, cfg);
    const before = dt.countLeaves(state.root);
    const pruned = dt.prune(state.root, 0.05, data.points.length);
    const after = dt.countLeaves(pruned);
    assert.ok(after < before, 'pruning did nothing: ' + before + ' -> ' + after);
    assert.ok(after >= 1);
  });

  it('reports finished up front when no split clears minImpurityDecrease', () => {
    // A root whose best split misses the floor must be finished from the start.
    const cfg = { ...config, minImpurityDecrease: 10 };
    const state = dt.createState(data.points, cfg, RECT);
    assert.equal(state.finished, true, 'an impossible floor should finish immediately');
    assert.deepEqual(state.frontier, []);
    const after = dt.growOne(state, data.points, cfg);
    assert.equal(after.nodeCount, 1, 'nothing should have been added');
    assert.equal(after.leafCount, 1);
  });

  it('still reports unfinished when a split is available', () => {
    const state = dt.createState(data.points, config, RECT);
    assert.equal(state.finished, false);
    assert.deepEqual(state.frontier, [0]);
  });

  it('survives sample indices left over from a previous dataset', () => {
    const stale = [0, 1, 2, data.points.length + 5, data.points.length + 40];
    assert.doesNotThrow(() => dt.classCounts(data.points, stale, 2));
    const counts = dt.classCounts(data.points, stale, 2);
    assert.equal(
      counts.reduce((a, b) => a + b, 0),
      3,
      'only the in-range indices should be counted',
    );
  });

  it('handles three classes', () => {
    const three = CLASSIFICATION_DATASETS.find((d) => d.id === 'three-class')!.generate({
      count: 120,
      noise: 0.15,
      seed: 6,
      classCount: 3,
    });
    const cfg = { ...config, classCount: 3, maxDepth: 8 };
    const state = dt.growAll(dt.createState(three.points, cfg, RECT), three.points, cfg);
    const predicted = new Set(three.points.map((p) => dt.predict(state.root, p.x, p.y)));
    assert.ok(predicted.size >= 3, 'tree never predicts the third class');
    assert.ok(dt.accuracy(state.root, three.points) > 0.9);
  });
});

/* ================================================================== */

describe('neural network', () => {
  const config: nn.NetConfig = {
    hiddenLayers: [4],
    activation: 'tanh',
    learningRate: 0.1,
    batchSize: 10,
    inputFeatures: ['x1', 'x2'],
    initialiser: 'xavier',
    regularisation: 'none',
    regRate: 0,
    seed: 3,
    testFraction: 0.2,
  };

  const data = CLASSIFICATION_DATASETS.find((d) => d.id === 'gaussians')!.generate({
    count: 120,
    noise: 0.15,
    seed: 2,
  });

  it('builds the requested architecture', () => {
    const state = nn.createState({ ...config, hiddenLayers: [5, 3] });
    assert.equal(state.layers.length, 3, 'two hidden layers plus an output layer');
    assert.equal(state.layers[0].length, 5);
    assert.equal(state.layers[1].length, 3);
    assert.equal(state.layers[2].length, 1);
    assert.equal(state.layers[0][0].weights.length, 2, 'first layer fan-in = feature count');
    assert.equal(state.layers[1][0].weights.length, 5);
  });

  it('with no hidden layers it is a single linear unit', () => {
    const state = nn.createState({ ...config, hiddenLayers: [] });
    assert.equal(state.layers.length, 1);
    assert.equal(state.layers[0].length, 1);
  });

  it('activations and their derivatives are consistent', () => {
    for (const name of ['tanh', 'relu', 'sigmoid', 'linear'] as const) {
      for (const z of [-2, -0.5, 0.5, 2]) {
        const h = 1e-6;
        const numeric = (nn.activate(name, z + h) - nn.activate(name, z - h)) / (2 * h);
        const analytic = nn.activateDerivative(name, nn.activate(name, z));
        close(analytic, numeric, 1e-4, name + " derivative at z=" + z);
      }
    }
  });

  it('backprop gradients match numerical gradients', () => {
    const cfg = { ...config, hiddenLayers: [3], activation: 'tanh' as const };
    const state = nn.createState(cfg);
    const layers = state.layers;
    const point = data.points[0];
    const input = nn.inputVector(cfg.inputFeatures, point.x, point.y);
    const target = nn.targetOf(point.label);

    // Accumulate the analytic gradient for this single sample.
    for (const layer of layers) {
      for (const neuron of layer) {
        neuron.weightGradients.fill(0);
        neuron.biasGradient = 0;
      }
    }
    nn.forward(layers, input, cfg.activation);
    nn.backward(layers, input, target, cfg.activation);

    const lossAt = () => nn.sampleLoss(nn.evaluate(layers, input, cfg.activation), target);
    const h = 1e-5;
    for (let l = 0; l < layers.length; l++) {
      for (let n = 0; n < layers[l].length; n++) {
        const neuron = layers[l][n];
        for (let w = 0; w < neuron.weights.length; w++) {
          const original = neuron.weights[w];
          neuron.weights[w] = original + h;
          const up = lossAt();
          neuron.weights[w] = original - h;
          const down = lossAt();
          neuron.weights[w] = original;
          close(
            neuron.weightGradients[w],
            (up - down) / (2 * h),
            1e-4,
            'layer ' + l + ' neuron ' + n + ' weight ' + w,
          );
        }
        const originalBias = neuron.bias;
        neuron.bias = originalBias + h;
        const up = lossAt();
        neuron.bias = originalBias - h;
        const down = lossAt();
        neuron.bias = originalBias;
        close(neuron.biasGradient, (up - down) / (2 * h), 1e-4, 'layer ' + l + ' neuron ' + n + ' bias');
      }
    }
  });

  it('step is pure', () => {
    const split = nn.splitDataset(data.points, config.testFraction, config.seed);
    const state = nn.createState(config);
    const weightsBefore = JSON.stringify(state.layers.map((l) => l.map((n) => n.weights)));
    const next = nn.step(state, split, config);
    assert.equal(
      JSON.stringify(state.layers.map((l) => l.map((n) => n.weights))),
      weightsBefore,
      'step mutated the input layers',
    );
    assert.notEqual(next.layers, state.layers);
  });

  it('learns a separable dataset', () => {
    const split = nn.splitDataset(data.points, 0.2, 1);
    let state = nn.createState(config);
    for (let i = 0; i < 200; i++) state = nn.step(state, split, config);
    assert.ok(state.trainAccuracy > 0.95, 'train accuracy only ' + state.trainAccuracy);
    assert.ok(state.trainLoss < 0.2, 'train loss ' + state.trainLoss);
  });

  it('a hidden layer solves XOR where a bare linear unit cannot', () => {
    const xor = CLASSIFICATION_DATASETS.find((d) => d.id === 'xor')!.generate({
      count: 200,
      noise: 0.05,
      seed: 5,
    });
    const split = nn.splitDataset(xor.points, 0.2, 1);
    const run = (hiddenLayers: number[], seed: number, steps: number) => {
      const cfg = { ...config, hiddenLayers, seed, learningRate: 0.15, batchSize: 16 };
      let state = nn.createState(cfg);
      for (let i = 0; i < steps; i++) state = nn.step(state, split, cfg);
      return state.trainAccuracy;
    };
    const linear = run([], 1, 400);
    // A few seeds, because a small net on XOR is genuinely seed-sensitive.
    const withHidden = Math.max(...[1, 2, 3].map((s) => run([6], s, 700)));
    assert.ok(linear < 0.75, 'a linear unit should fail XOR, got ' + linear);
    assert.ok(withHidden > 0.9, 'a hidden layer should solve XOR, best was ' + withHidden);
  });

  it('the x1x2 feature makes XOR linearly separable', () => {
    const xor = CLASSIFICATION_DATASETS.find((d) => d.id === 'xor')!.generate({
      count: 200,
      noise: 0.05,
      seed: 5,
    });
    const split = nn.splitDataset(xor.points, 0.2, 1);
    const cfg = {
      ...config,
      hiddenLayers: [],
      inputFeatures: ['x1', 'x2', 'x1x2'] as nn.InputFeature[],
      learningRate: 0.3,
      batchSize: 16,
    };
    let state = nn.createState(cfg);
    for (let i = 0; i < 600; i++) state = nn.step(state, split, cfg);
    assert.ok(state.trainAccuracy > 0.9, 'accuracy with x1x2 was only ' + state.trainAccuracy);
  });

  it('zero initialisation cannot break symmetry', () => {
    const cfg = { ...config, initialiser: 'zeros' as const, hiddenLayers: [4] };
    const split = nn.splitDataset(data.points, 0.2, 1);
    let state = nn.createState(cfg);
    for (let i = 0; i < 60; i++) state = nn.step(state, split, cfg);
    const hidden = state.layers[0];
    // Every hidden neuron should still hold identical weights.
    for (let n = 1; n < hidden.length; n++) {
      assert.deepEqual(
        hidden[n].weights.map((w) => Math.round(w * 1e9)),
        hidden[0].weights.map((w) => Math.round(w * 1e9)),
        'zero init should leave every neuron identical, neuron ' + n + ' differs',
      );
    }
  });

  it('splitDataset partitions cleanly and honours the fraction', () => {
    const split = nn.splitDataset(data.points, 0.25, 9);
    assert.equal(split.train.length + split.test.length, data.points.length);
    assert.equal(split.test.length, Math.round(data.points.length * 0.25));
  });

  it('parameterCount matches the architecture', () => {
    const state = nn.createState({ ...config, hiddenLayers: [3, 2] });
    // (2 in -> 3) 3*(2+1)=9, (3 -> 2) 2*(3+1)=8, (2 -> 1) 1*(2+1)=3
    assert.equal(nn.parameterCount(state.layers), 9 + 8 + 3);
  });

  it('evaluateNeuron agrees with a full forward pass at the output', () => {
    const state = nn.createState({ ...config, hiddenLayers: [3] });
    const input = nn.inputVector(config.inputFeatures, 1.5, -2);
    close(
      nn.evaluateNeuron(state.layers, input, config.activation, state.layers.length - 1, 0),
      nn.evaluate(state.layers, input, config.activation),
      1e-12,
      'output neuron',
    );
  });
});

/* ================================================================== */

describe('k-nearest neighbours', () => {
  const config: knn.KnnConfig = {
    k: 3,
    metric: 'euclidean',
    weighting: 'uniform',
    classCount: 2,
    standardise: false,
  };
  const scaling = { sx: 1, sy: 1 };

  const points: Point2D[] = [
    { x: 0, y: 0, label: 0 },
    { x: 1, y: 0, label: 0 },
    { x: 0, y: 1, label: 0 },
    { x: 5, y: 5, label: 1 },
    { x: 5, y: 6, label: 1 },
    { x: 6, y: 5, label: 1 },
  ];

  it('each metric matches its definition', () => {
    close(knn.distance(0, 0, 3, 4, 'euclidean', scaling), 5);
    close(knn.distance(0, 0, 3, 4, 'manhattan', scaling), 7);
    close(knn.distance(0, 0, 3, 4, 'chebyshev', scaling), 4);
    close(knn.distance(0, 0, 3, 4, 'minkowski3', scaling), Math.cbrt(27 + 64), 1e-9);
  });

  it('neighbours come back sorted, nearest first', () => {
    const ranked = knn.rankNeighbours(points, 0.2, 0.2, config, scaling);
    assert.equal(ranked.length, points.length);
    for (let i = 1; i < ranked.length; i++) {
      assert.ok(ranked[i].distance >= ranked[i - 1].distance, 'not sorted at index ' + i);
    }
    assert.equal(ranked[0].index, 0, 'nearest to (0.2,0.2) should be the origin point');
  });

  it('findNeighbours returns exactly k', () => {
    assert.equal(knn.findNeighbours(points, 0, 0, { ...config, k: 4 }, scaling).length, 4);
    // and never more than the data holds
    assert.equal(knn.findNeighbours(points, 0, 0, { ...config, k: 99 }, scaling).length, points.length);
  });

  it('classifies each cluster correctly', () => {
    assert.equal(knn.classify(points, 0.3, 0.3, config, scaling).winner, 0);
    assert.equal(knn.classify(points, 5.3, 5.3, config, scaling).winner, 1);
  });

  it('an empty neighbour list is not a tie', () => {
    const vote = knn.tally([], 3);
    assert.equal(vote.tied, false, 'no votes is not a deadlock');
    assert.equal(vote.confidence, 0);
    assert.deepEqual(vote.scores, [0, 0, 0]);
  });

  it('detects a tie', () => {
    const even: Point2D[] = [
      { x: -1, y: 0, label: 0 },
      { x: 1, y: 0, label: 1 },
    ];
    const vote = knn.tally(knn.findNeighbours(even, 0, 0, { ...config, k: 2 }, scaling), 2);
    assert.equal(vote.tied, true);
  });

  it('distance weighting favours the closer neighbour', () => {
    const mixed: Point2D[] = [
      { x: 0.1, y: 0, label: 0 },
      { x: 3, y: 0, label: 1 },
      { x: 3.1, y: 0, label: 1 },
    ];
    const uniform = knn.classify(mixed, 0, 0, { ...config, k: 3, weighting: 'uniform' }, scaling);
    const weighted = knn.classify(mixed, 0, 0, { ...config, k: 3, weighting: 'distance' }, scaling);
    assert.equal(uniform.winner, 1, 'uniform voting should follow the majority');
    assert.equal(weighted.winner, 0, 'distance weighting should follow the near neighbour');
  });

  it('training accuracy at k = 1 is always 100%', () => {
    const data = CLASSIFICATION_DATASETS[0].generate({ count: 80, noise: 0.4, seed: 2 });
    const s = knn.makeScaling(data.points, false);
    close(knn.trainingAccuracy(data.points, { ...config, k: 1 }, s), 1, 1e-12);
  });

  it('leave-one-out accuracy at k = 1 is below training accuracy on noisy data', () => {
    const data = CLASSIFICATION_DATASETS[0].generate({ count: 120, noise: 0.75, seed: 2 });
    const s = knn.makeScaling(data.points, false);
    const loo = knn.leaveOneOutAccuracy(data.points, { ...config, k: 1 }, s);
    assert.ok(loo < 1, 'LOO accuracy should not be perfect on noisy data, got ' + loo);
  });

  it('accuracyAcrossK agrees with the direct leave-one-out computation', () => {
    const data = CLASSIFICATION_DATASETS[0].generate({ count: 60, noise: 0.4, seed: 5 });
    const s = knn.makeScaling(data.points, false);
    const curve = knn.accuracyAcrossK(data.points, 9, { metric: 'euclidean', weighting: 'uniform', classCount: 2, standardise: false }, s);
    assert.equal(curve.length, 9);
    for (const entry of [curve[0], curve[4], curve[8]]) {
      close(
        entry.accuracy,
        knn.leaveOneOutAccuracy(data.points, { ...config, k: entry.k }, s),
        1e-12,
        'k=' + entry.k,
      );
    }
  });

  it('standardising changes distances when the axes have different scales', () => {
    const stretched: Point2D[] = points.map((p) => ({ ...p, y: p.y * 100 }));
    const plain = knn.makeScaling(stretched, false);
    const scaled = knn.makeScaling(stretched, true);
    assert.equal(plain.sx, 1);
    assert.ok(scaled.sy > scaled.sx * 10, 'y should be scaled down far more than x');
  });
});

/* ---------------- architecture graph layout ---------------- */

describe('architecture layout', () => {
  /** Roughly linear regression's training graph: a return path and a tall column. */
  function sampleGraph() {
    const g = new GraphBuilder();
    const cIn = g.column('Input');
    const cParams = g.column('Parameters');
    const cModel = g.column('Model');
    const cLoss = g.column('Objective');
    g.node(cIn, { id: 'x', kind: 'input', label: 'x' });
    g.node(cParams, { id: 'w', kind: 'param', label: 'w' });
    g.node(cParams, { id: 'b', kind: 'param', label: 'b' });
    g.node(cModel, { id: 'sum', kind: 'op', label: 'Σ' });
    g.node(cModel, { id: 'yhat', kind: 'output', label: 'ŷ' });
    g.node(cModel, { id: 'sq', kind: 'op', label: '( )²' });
    g.node(cLoss, { id: 'loss', kind: 'loss', label: 'L' });
    g.edge('x', 'w', { weight: 2 });
    g.edge('w', 'sum', { weight: -0.5 });
    g.edge('b', 'sum');
    g.edge('sum', 'yhat');
    g.edge('yhat', 'sq');
    g.edge('sq', 'loss');
    g.edge('loss', 'w', { active: true, symbol: '∂L/∂w' });
    return g.build('summary');
  }

  it('the builder counts rows per column and the largest weight', () => {
    const graph = sampleGraph();
    const rows = Object.fromEntries(graph.nodes.map((n) => [n.id, n.rows]));
    assert.equal(rows.x, 1);
    assert.equal(rows.w, 2);
    assert.equal(rows.sq, 3);
    assert.equal(graph.maxWeight, 2);
    assert.equal(graph.columns.length, 4);
  });

  it('places every node inside the canvas with the headings above the block', () => {
    const graph = sampleGraph();
    const layout = layoutArchitecture(graph, 800, 460);
    for (const node of layout.nodes) {
      assert.ok(node.rect.x >= 0 && node.rect.x + node.rect.w <= 800, node.id + ' fits horizontally');
      assert.ok(node.rect.y >= layout.top - 1e-9, node.id + ' is below the block top');
      assert.ok(node.rect.y + node.rect.h <= layout.bottom + 1e-9, node.id + ' is above the block bottom');
    }
    assert.ok(layout.headerY < layout.top, 'headings sit above the top node');
    assert.ok(!layout.compact, 'a 800x460 canvas has room for labelled boxes');
    // The block is centred rather than hugging the top of the frame.
    const centre = (layout.top + layout.bottom) / 2;
    assert.ok(Math.abs(centre - 460 / 2) < 60, 'block centre ' + centre + ' is near the middle');
  });

  it('a short column centres itself against a tall one and reads as a pair', () => {
    const layout = layoutArchitecture(sampleGraph(), 800, 460);
    const w = layout.byId.get('w')!;
    const b = layout.byId.get('b')!;
    const yhat = layout.byId.get('yhat')!;
    close((w.cy + b.cy) / 2, yhat.cy, 1e-9, 'the pair is centred on the middle of the tall column');
    assert.ok(b.cy - w.cy <= w.rect.h * 1.7 + 1e-9, 'the pair is not spread to the frame edges');
  });

  it('columns never spread wider than the pitch cap', () => {
    const layout = layoutArchitecture(sampleGraph(), 2400, 460, { maxPitch: 200 });
    const gap = layout.columnX[1] - layout.columnX[0];
    close(gap, 200, 1e-9);
    close((layout.columnX[0] + layout.columnX[3]) / 2, 1200, 1e-9, 'and stay centred');
  });

  it('forward edges run left to right and the gradient arcs under the block', () => {
    const graph = sampleGraph();
    const layout = layoutArchitecture(graph, 800, 460, { padBottom: 84 });
    const forward = edgePath(graph.edges.find((e) => e.id === 'x->w')!, layout)!;
    assert.equal(forward.backwards, false);
    assert.ok(forward.start.x < forward.end.x);

    const back = edgePath(graph.edges.find((e) => e.id === 'loss->w')!, layout)!;
    assert.equal(back.backwards, true);
    const b = layout.byId.get('b')!;
    close(back.end.x, b.cx, 1e-9, 'arrives under the lowest node of the parameter column');
    assert.ok(back.end.y > b.rect.y + b.rect.h, 'from underneath');
    const lowest = Math.max(...sampleEdge(back, 40).map((p) => p.y));
    assert.ok(lowest > layout.bottom + 10, 'the arc clears every node');
    assert.ok(lowest < 460, 'and stays inside the canvas');
  });

  it('hit tests find nodes and the curved edges', () => {
    const graph = sampleGraph();
    const layout = layoutArchitecture(graph, 800, 460, { padBottom: 84 });
    const w = layout.byId.get('w')!;
    assert.equal(hitTestNode(layout, w.cx, w.cy)?.id, 'w');
    assert.equal(hitTestNode(layout, 2, 2), null);

    const back = edgePath(graph.edges.find((e) => e.id === 'loss->w')!, layout)!;
    const mid = sampleEdge(back, 2)[1];
    assert.equal(hitTestEdge(graph, layout, mid.x, mid.y)?.id, 'loss->w');
    assert.equal(hitTestEdge(graph, layout, back.label.x, layout.headerY - 40), null);
  });
});
