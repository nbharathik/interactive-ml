/** Polynomial regression: the exact solve, the refit loop, the curves over degree. */

import { strict as assert } from 'node:assert';
import { describe, it } from 'node:test';

import katex from 'katex';

import { texSymbol } from '../src/explainer/tex.ts';
import { MAX_DEGREE } from '../src/explainers/polynomial-regression/config.ts';
import { powerLabel, weightLabel } from '../src/explainers/polynomial-regression/graph.ts';
import { CURVE_DATASETS, getCurveDataset, renoise, testSeed } from '../src/lib/datasets/curves.ts';
import {
  biasVariance,
  biasVarianceSweep,
  createRefitState,
  fitPolynomial,
  gridOver,
  meanSquaredError,
  predictFit,
  refitStep,
  validationCurve,
} from '../src/lib/ml/polynomialRegression.ts';

const wave = getCurveDataset('wave');
const shared = { l2: 0, standardise: true };

describe('polynomial regression', () => {
  it('recovers a quadratic exactly from noiseless points', () => {
    const points = Array.from({ length: 12 }, (_, i) => ({ x: i * 0.4, y: 2 - 0.5 * i * 0.4 + 0.3 * (i * 0.4) ** 2 }));
    for (const standardise of [true, false]) {
      const fit = fitPolynomial(points, { degree: 2, l2: 0, standardise });
      assert.ok(!fit.failed);
      assert.ok(meanSquaredError(fit, points) < 1e-18);
      assert.ok(Math.abs(predictFit(fit, 7) - (2 - 3.5 + 0.3 * 49)) < 1e-9);
    }
  });

  it('degree 0 is the mean and degree n − 1 passes through every point', () => {
    const data = wave.generate({ count: 10, noise: 0.5, seed: 3 });
    const mean = data.points.reduce((a, p) => a + p.y, 0) / data.points.length;
    const flat = fitPolynomial(data.points, { degree: 0, ...shared });
    assert.ok(Math.abs(predictFit(flat, 2) - mean) < 1e-9);
    const through = fitPolynomial(data.points, { degree: 9, ...shared });
    assert.ok(meanSquaredError(through, data.points) < 1e-9);
  });

  it('ridge shrinks the weights and standardising leaves the fit unchanged at low degree', () => {
    const data = wave.generate({ count: 25, noise: 0.5, seed: 42 });
    const plain = fitPolynomial(data.points, { degree: 8, ...shared });
    const ridge = fitPolynomial(data.points, { degree: 8, l2: 0.1, standardise: true });
    const norm = (w: number[]) => Math.sqrt(w.slice(1).reduce((a, v) => a + v * v, 0));
    assert.ok(norm(ridge.weights) < norm(plain.weights) / 2);
    const raw = fitPolynomial(data.points, { degree: 4, l2: 0, standardise: false });
    const std = fitPolynomial(data.points, { degree: 4, l2: 0, standardise: true });
    assert.ok(Math.abs(meanSquaredError(raw, data.points) - meanSquaredError(std, data.points)) < 1e-9);
  });

  it('the validation curve has a monotone training error and a best degree in the middle', () => {
    const train = wave.generate({ count: 25, noise: 0.5, seed: 42 }).points;
    const test = wave.generate({ count: 15, noise: 0.5, seed: testSeed(42) }).points;
    const curve = validationCurve(train, test, 15, shared);
    for (let d = 1; d <= 15; d++) assert.ok(curve.train[d] <= curve.train[d - 1] + 1e-9, 'degree ' + d);
    assert.ok(curve.best >= 3 && curve.best <= 6);
    assert.ok(curve.test[15] > curve.test[curve.best]);
  });

  it('refitStep is pure and keeps a running mean and spread', () => {
    const data = wave.generate({ count: 25, noise: 0.5, seed: 42 });
    const grid = gridOver([0.25, 4.75], 41);
    const sample = (k: number) => renoise(data.points, data.truth!, 0.5, 1000 + k);
    const config = { degree: 3, ...shared };
    const before = createRefitState(grid.length);
    const snapshot = JSON.stringify(before);
    let state = before;
    for (let i = 0; i < 20; i++) state = refitStep(state, sample, config, grid);
    assert.equal(JSON.stringify(before), snapshot);
    assert.equal(state.curves.length, 20);
    const mean = state.curves.reduce((a, c) => a + c[10], 0) / 20;
    assert.ok(Math.abs(state.mean[10] - mean) < 1e-9);
    const truth = grid.map(data.truth!);
    const bv = biasVariance(state, truth);
    assert.ok(bv.bias2 >= 0 && bv.variance > 0);
    assert.ok(Number.isNaN(biasVariance(createRefitState(grid.length), truth).bias2));
  });

  it('the sweep shows bias falling and variance rising with the degree, and zero variance without noise', () => {
    const data = wave.generate({ count: 25, noise: 0.5, seed: 42 });
    const grid = gridOver([0.25, 4.75], 41);
    const truth = grid.map(data.truth!);
    const sweep = biasVarianceSweep((k) => renoise(data.points, data.truth!, 0.5, 500 + k), truth, grid, 12, 40, shared, 0.5);
    assert.ok(sweep.bias2[0] > 5 * sweep.bias2[5]);
    assert.ok(sweep.variance[12] > 100 * sweep.variance[1]);
    assert.equal(sweep.noise, 0.25);
    const quiet = biasVarianceSweep((k) => renoise(data.points, data.truth!, 0, 500 + k), truth, grid, 6, 10, shared, 0);
    for (const v of quiet.variance) assert.ok(v < 1e-12);
  });

  it('every curve dataset is deterministic and carries its truth', () => {
    for (const d of CURVE_DATASETS) {
      const a = d.generate({ count: 20, noise: 0.3, seed: 9 });
      const b = d.generate({ count: 20, noise: 0.3, seed: 9 });
      assert.deepEqual(a.points, b.points);
      assert.equal(a.points.length, 20);
      assert.ok(typeof a.truth === 'function');
      for (const p of a.points) assert.ok(p.x >= a.xRange[0] && p.x <= a.xRange[1]);
    }
  });
});

describe('card symbols', () => {
  it('every power and weight up to the top degree typesets, x¹² as one superscript', () => {
    assert.equal(texSymbol('x̃¹²'), '\\tilde{x}^{12}');
    assert.equal(texSymbol('x̃₂²'), '\\tilde{x}_{2}^{2}');
    for (let k = 1; k <= MAX_DEGREE; k++) {
      for (const standardised of [false, true]) {
        const tex = texSymbol(powerLabel(k, standardised)) + ' + ' + texSymbol(weightLabel(k, MAX_DEGREE));
        assert.doesNotThrow(() => katex.renderToString(tex, { throwOnError: true }), tex);
      }
    }
  });
});
