/** Ridge, lasso and elastic net: the solvers agree, the path behaves, the data generator is honest. */

import { strict as assert } from 'node:assert';
import { describe, it } from 'node:test';

import { DEFAULT_TABULAR, generateTabular, testRows, trainRows } from '../src/lib/datasets/tabular.ts';
import { pearson } from '../src/lib/math/stats.ts';
import {
  coordinateSweep,
  createState,
  entryLambda,
  istaEpoch,
  lambdaGrid,
  lambdaMax,
  meanSquaredErrorOf,
  objective,
  olsSolution,
  penaltyAt,
  predictRaw,
  prepare,
  regularisationPath,
  ridgeClosedForm,
  smoothGradient,
  smoothLoss,
  softThreshold,
  step,
  toRawUnits,
} from '../src/lib/ml/elasticNet.ts';
import type { ElasticNetConfig } from '../src/lib/ml/elasticNet.ts';

function close(actual: number, expected: number, tolerance = 1e-6, message?: string) {
  assert.ok(
    Math.abs(actual - expected) <= tolerance,
    (message ?? 'value') + ': expected ' + expected + ', got ' + actual,
  );
}

function solveCd(prep: ReturnType<typeof prepare>, config: ElasticNetConfig, sweeps = 500): number[] {
  let w = new Array<number>(prep.p).fill(0);
  for (let i = 0; i < sweeps; i++) {
    const next = coordinateSweep(w, prep, config);
    w = next.w;
    if (next.maxChange < 1e-12) break;
  }
  return w;
}

const data = generateTabular({ ...DEFAULT_TABULAR, features: 10, informative: 3, count: 80, seed: 5 });
const train = trainRows(data);
const test = testRows(data);
const prep = prepare(train, true);

describe('elastic net', () => {
  it('soft threshold shrinks towards zero and clips', () => {
    close(softThreshold(2, 0.5), 1.5);
    close(softThreshold(-2, 0.5), -1.5);
    close(softThreshold(0.3, 0.5), 0);
  });

  it('coordinate descent at alpha 0 matches the ridge closed form, and at lambda 0 least squares', () => {
    const ridge = ridgeClosedForm(prep, 0.3);
    const cd = solveCd(prep, { lambda: 0.3, alpha: 0, solver: 'cd', learningRate: 0 });
    cd.forEach((v, j) => close(v, ridge[j], 1e-6, 'ridge coord ' + j));
    const ols = olsSolution(prep);
    const cd0 = solveCd(prep, { lambda: 0, alpha: 1, solver: 'cd', learningRate: 0 });
    cd0.forEach((v, j) => close(v, ols[j], 1e-5, 'ols coord ' + j));
  });

  it('proximal gradient and coordinate descent agree on a lasso solution', () => {
    const config: ElasticNetConfig = { lambda: 0.1, alpha: 1, solver: 'ista', learningRate: 0.9 / prep.lipschitz };
    let w = new Array<number>(prep.p).fill(0);
    for (let i = 0; i < 3000; i++) w = istaEpoch(w, prep, config);
    const cd = solveCd(prep, { ...config, solver: 'cd' });
    w.forEach((v, j) => close(v, cd[j], 1e-4, 'coord ' + j));
  });

  it('the objective never rises along a sweep or a proximal epoch', () => {
    const config: ElasticNetConfig = { lambda: 0.05, alpha: 0.5, solver: 'cd', learningRate: 0.5 / prep.lipschitz };
    let w = new Array<number>(prep.p).fill(0.5);
    let last = objective(w, prep, config);
    for (let i = 0; i < 20; i++) {
      w = coordinateSweep(w, prep, config).w;
      const value = objective(w, prep, config);
      assert.ok(value <= last + 1e-12, 'sweep ' + i);
      last = value;
    }
    w = new Array<number>(prep.p).fill(0.5);
    last = objective(w, prep, config);
    for (let i = 0; i < 20; i++) {
      w = istaEpoch(w, prep, config);
      const value = objective(w, prep, config);
      assert.ok(value <= last + 1e-12, 'epoch ' + i);
      last = value;
    }
  });

  it('the smooth gradient matches finite differences of the smooth loss', () => {
    const config: ElasticNetConfig = { lambda: 0.2, alpha: 0.3, solver: 'cd', learningRate: 0 };
    const w = prep.xty.map((v) => v * 0.7);
    const grad = smoothGradient(w, prep, config);
    const ridge = config.lambda * (1 - config.alpha);
    const f = (v: readonly number[]) => smoothLoss(v, prep) + (ridge / 2) * v.reduce((s, x) => s + x * x, 0);
    w.forEach((_, j) => {
      const h = 1e-5;
      const up = w.slice();
      const down = w.slice();
      up[j] += h;
      down[j] -= h;
      close(grad[j], (f(up) - f(down)) / (2 * h), 1e-5, 'coord ' + j);
    });
  });

  it('lambda max zeroes every lasso coefficient and the path grows its support coming down', () => {
    const top = lambdaMax(prep, 1);
    const atTop = solveCd(prep, { lambda: top, alpha: 1, solver: 'cd', learningRate: 0 });
    assert.ok(atTop.every((v) => v === 0));
    const path = regularisationPath(train, test, prep, 1, lambdaGrid(top, 40));
    for (let i = 1; i < path.activeCounts.length; i++) {
      assert.ok(path.activeCounts[i] >= path.activeCounts[i - 1], 'support at ' + i);
    }
    assert.ok(path.testMse[path.bestIndex] <= path.testMse[0], 'best beats the empty model');
    const informative = data.trueWeights.map((w, j) => (w !== 0 ? j : -1)).filter((j) => j >= 0);
    // A quarter of the way down the grid the support is exactly the informative features.
    const quarter = path.weights[Math.floor(path.lambdas.length / 4)];
    const kept = informative.filter((j) => quarter[j] !== 0).length;
    const spurious = quarter.filter((v, j) => v !== 0 && data.trueWeights[j] === 0).length;
    assert.equal(kept, informative.length, 'keeps the true support: ' + kept);
    assert.equal(spurious, 0, 'no spurious: ' + spurious);
    for (const j of informative) assert.ok(entryLambda(path, j) !== null);
  });

  it('a ridge path shrinks every coefficient towards zero as lambda grows', () => {
    const path = regularisationPath(train, test, prep, 0, lambdaGrid(50, 30));
    const norms = path.weights.map((w) => Math.sqrt(w.reduce((s, v) => s + v * v, 0)));
    for (let i = 1; i < norms.length; i++) assert.ok(norms[i] >= norms[i - 1] - 1e-9, 'norm at ' + i);
    assert.ok(path.activeCounts.every((c) => c === prep.p), 'ridge never zeroes');
  });

  it('predictions in raw units equal predictions from the raw weights', () => {
    const w = solveCd(prep, { lambda: 0.05, alpha: 1, solver: 'cd', learningRate: 0 });
    const raw = toRawUnits(w, prep);
    for (const row of test.X.slice(0, 5)) {
      const direct = raw.bias + row.reduce((s, v, j) => s + v * raw.weights[j], 0);
      close(predictRaw(w, prep, row), direct, 1e-9);
    }
    assert.ok(meanSquaredErrorOf(w, prep, test) < meanSquaredErrorOf(new Array<number>(prep.p).fill(0), prep, test));
  });

  it('step is pure and converges', () => {
    const config: ElasticNetConfig = { lambda: 0.1, alpha: 1, solver: 'cd', learningRate: 0 };
    let state = createState(prep, config);
    const before = JSON.stringify(state);
    const next = step(state, prep, config);
    assert.equal(JSON.stringify(state), before);
    assert.notEqual(next.w, state.w);
    for (let i = 0; i < 500 && !state.converged; i++) state = step(state, prep, config);
    assert.ok(state.converged, 'converged in ' + state.epoch);
    assert.ok(state.activeCount < prep.p);
    const wild: ElasticNetConfig = { lambda: 0.1, alpha: 1, solver: 'ista', learningRate: 5 / prep.lipschitz };
    let bad = createState(prep, wild);
    for (let i = 0; i < 300 && !bad.diverged; i++) bad = step(bad, prep, wild);
    assert.ok(bad.diverged, 'a step above 2 over the curvature diverges');
  });

  it('the two-weight penalty is a diamond for lasso and a disc for ridge', () => {
    close(penaltyAt(1)(1, 1), 2);
    close(penaltyAt(0)(1, 1), 1);
    close(penaltyAt(0.5)(2, 0), 1 + 1);
  });
});

describe('tabular data', () => {
  it('is deterministic, partitions its rows and honours the informative count', () => {
    const a = generateTabular({ ...DEFAULT_TABULAR, seed: 3 });
    const b = generateTabular({ ...DEFAULT_TABULAR, seed: 3 });
    assert.deepEqual(a.y, b.y);
    assert.equal(a.X.length, DEFAULT_TABULAR.count);
    assert.equal(a.X[0].length, DEFAULT_TABULAR.features);
    assert.equal(a.trainIndex.length + a.testIndex.length, a.X.length);
    assert.equal(new Set([...a.trainIndex, ...a.testIndex]).size, a.X.length);
    assert.equal(a.trueWeights.filter((w) => w !== 0).length, DEFAULT_TABULAR.informative);
  });

  it('correlates features inside a group and not across groups', () => {
    const d = generateTabular({ ...DEFAULT_TABULAR, features: 6, correlation: 0.9, count: 400, seed: 9 });
    const col = (j: number) => d.X.map((row) => row[j]);
    assert.ok(pearson(col(0), col(1)) > 0.6, 'within group');
    assert.ok(Math.abs(pearson(col(0), col(3))) < 0.3, 'across groups');
  });

  it('varied scales give every feature its own spread', () => {
    const d = generateTabular({ ...DEFAULT_TABULAR, features: 6, variedScales: true, count: 200, seed: 2 });
    const spread = (j: number) => {
      const xs = d.X.map((row) => row[j]);
      const m = xs.reduce((a, b) => a + b, 0) / xs.length;
      return Math.sqrt(xs.reduce((a, x) => a + (x - m) ** 2, 0) / xs.length);
    };
    const spreads = Array.from({ length: 6 }, (_, j) => spread(j));
    assert.ok(Math.max(...spreads) / Math.min(...spreads) > 3);
  });
});
