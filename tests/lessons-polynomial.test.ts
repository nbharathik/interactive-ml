/**
 * The polynomial regression lessons, driven against the real model: every lesson
 * opens the way the page would, every run plays to its outcome, then the copy
 * must read cleanly and the claims the presets and the sandbox make must hold.
 */

import { strict as assert } from 'node:assert';
import { describe, it } from 'node:test';

import { DEFAULT_PARAMS, MAX_DEGREE, PRESETS } from '../src/explainers/polynomial-regression/config.ts';
import type { PolyParams } from '../src/explainers/polynomial-regression/config.ts';
import { LESSONS, makePolyContext } from '../src/explainers/polynomial-regression/lessons.ts';
import type { PolyLessonContext } from '../src/explainers/polynomial-regression/lessons.ts';
import { BV_GRID, CURVE_GRID, innerBiasVariance, samplerFor } from '../src/explainers/polynomial-regression/model.ts';
import { getCurveDataset, testSeed } from '../src/lib/datasets/curves.ts';
import type { RegressionData, RegressionPoint } from '../src/lib/datasets/types.ts';
import {
  biasVarianceSweep,
  createRefitState,
  fitPolynomial,
  refitStep,
  validationCurve,
} from '../src/lib/ml/polynomialRegression.ts';
import type { BiasVarianceSweep, PolyConfig, PolyFit, RefitState, ValidationCurve } from '../src/lib/ml/polynomialRegression.ts';
import { LessonRig, checkStructure, exerciseRun, exerciseScreen, openLesson, reads } from './lessonRig.ts';

interface Model {
  data: RegressionData;
  test: RegressionPoint[];
  config: PolyConfig;
  fit: PolyFit;
  curve: ValidationCurve;
  sweep: BiasVarianceSweep;
  truthOnGrid: number[];
  sampler: (k: number) => RegressionPoint[];
  refits: number;
}

type Ui = PolyLessonContext['ui'];
type Rig = LessonRig<PolyParams, RefitState, Model, Ui>;

const DISPLAY = ['showResiduals', 'showTruth', 'showTest', 'showRefits'];

const make = (): Rig =>
  new LessonRig({
    defaults: DEFAULT_PARAMS,
    presets: PRESETS,
    build: (p) => {
      const dataset = getCurveDataset(p.dataset);
      const data = dataset.generate({ count: p.sampleCount, noise: p.noise, seed: p.seed });
      const test = dataset.generate({ count: p.testCount, noise: p.noise, seed: testSeed(p.seed) }).points;
      const config: PolyConfig = { degree: p.degree, l2: p.l2, standardise: p.standardise };
      const shared = { l2: p.l2, standardise: p.standardise };
      const truth = data.truth!;
      const truthOnGrid = BV_GRID.map(truth);
      const sampler = samplerFor(data.points, truth, p.noise, p.seed);
      return {
        data,
        test,
        config,
        fit: fitPolynomial(data.points, config),
        curve: validationCurve(data.points, test, MAX_DEGREE, shared),
        sweep: biasVarianceSweep(sampler, truthOnGrid, BV_GRID, MAX_DEGREE, p.refits, shared, p.noise),
        truthOnGrid,
        sampler,
        refits: p.refits,
      };
    },
    create: () => createRefitState(CURVE_GRID.length),
    step: (s, m) => refitStep(s, m.sampler, m.config, CURVE_GRID),
    complete: (s, m) => s.epoch >= m.refits,
    ui: () => ({ view: 'curve', logScale: true, open: null, pointEdits: 0 }),
    resets: (key) => !DISPLAY.includes(key),
  });

const context = (rig: Rig): PolyLessonContext => {
  const { data, test, fit, curve, sweep, truthOnGrid } = rig.model();
  return {
    ...makePolyContext({
      params: rig.params,
      state: rig.state,
      data,
      test,
      fit,
      curve,
      sweep,
      bv: innerBiasVariance(rig.state, truthOnGrid),
      grid: BV_GRID,
      truthOnGrid,
      ui: rig.ui,
    }),
    ...rig.base(),
  };
};

describe('polynomial regression lessons: structure', () => {
  it('follows the shared rules, one screen per lesson', () => checkStructure(LESSONS, PRESETS, DEFAULT_PARAMS, { oneScreen: true, instant: true }));
});

const reveals: string[] = [];

describe('polynomial regression lessons: every lesson and every run against the model', () => {
  for (const lesson of LESSONS) {
    it(lesson.id + ' opens on a picture that reads', () => exerciseScreen(openLesson(make, lesson, context), lesson, context, reveals));
    const s = lesson.steps[0];
    for (const e of s.experiments) {
      it(lesson.id + ' / ' + e.label, () => exerciseRun(openLesson(make, lesson, context), e, lesson.id + ' / ' + e.label, context, reveals));
    }
  }

  it('prints the live numbers', () => {
    for (const line of reveals) console.log('  ' + line);
  });
});

describe('polynomial regression lessons: the numbers the copy quotes', () => {
  const say = (id: string) => {
    const lesson = LESSONS.find((l) => l.id === id)!;
    const rig = openLesson(make, lesson, context);
    return { text: String(lesson.steps[0].say), d: context(rig).derived };
  };
  const three = (v: number) => v.toFixed(3);

  it('the introduction and overfitting quote the line and the degree-12 errors', () => {
    const intro = say('polynomial');
    reads('polynomial', intro.text, three(intro.d.trainError));
    reads('polynomial', intro.text, three(intro.d.testError));
    const over = say('overfitting');
    reads('overfitting', over.text, three(over.d.trainError));
    reads('overfitting', over.text, three(over.d.testError));
    reads('overfitting', over.text, Math.round(over.d.testError / over.d.trainError) + ' times');
  });

  it('the validation curve quotes its ends and its bottom', () => {
    const v = say('errors');
    reads('errors', v.text, three(v.d.errorAt(0).train));
    reads('errors', v.text, three(v.d.errorAt(MAX_DEGREE).train));
    reads('errors', v.text, three(v.d.bestTestError) + ' at degree ' + v.d.best);
    reads('errors', v.text, three(v.d.errorAt(MAX_DEGREE).test));
  });

  it('bias and variance quote the fifty line refits, interpolation its zero', () => {
    const bv = say('bias-variance');
    assert.equal(bv.d.refits, DEFAULT_PARAMS.refits);
    reads('bias-variance', bv.text, 'variance ' + three(bv.d.variance));
    reads('bias-variance', bv.text, 'bias² ' + three(bv.d.bias2));
    const inter = say('interpolation');
    reads('interpolation', inter.text, three(inter.d.trainError));
    reads('interpolation', inter.text, inter.d.testError.toExponential(1));
  });
});

describe('polynomial regression lessons: the claims hold', () => {
  const at = (over: Partial<PolyParams>, refits = 0) => {
    const rig = make();
    rig.merge({ ...PRESETS[0].params, ...over });
    rig.stepOnce(refits);
    return context(rig).derived;
  };
  const preset = (id: string) => PRESETS.find((p) => p.id === id)!.params;

  it('the sweep and the refit loop agree, because they draw the same samples', () => {
    const d = at(preset('variance'), DEFAULT_PARAMS.refits);
    assert.equal(d.refits, DEFAULT_PARAMS.refits);
    assert.ok(Math.abs(d.bias2 - d.sweepAt(10).bias2) < 1e-9, d.bias2 + ' vs ' + d.sweepAt(10).bias2);
    assert.ok(Math.abs(d.variance - d.sweepAt(10).variance) < 1e-9, d.variance + ' vs ' + d.sweepAt(10).variance);
  });

  it('degree 1 underfits, degree 3 fits, degree 12 overfits the wave', () => {
    const line = at(preset('line'));
    const cubic = at(preset('cubic'));
    const wild = at(preset('overfit'));
    assert.ok(line.trainError > 1.5 * cubic.trainError, 'line ' + line.trainError + ' cubic ' + cubic.trainError);
    assert.ok(line.testError > 1.5 * cubic.testError);
    assert.ok(wild.trainError < cubic.trainError, 'train should keep falling');
    assert.ok(wild.testError > 4 * cubic.testError, 'wild test ' + wild.testError);
    assert.ok(cubic.testError <= 1.05 * cubic.bestTestError, 'degree 3 is within 5% of the best');
  });

  it('training error never rises with the degree; the test error turns back up', () => {
    const d = at({});
    for (let k = 1; k <= MAX_DEGREE; k++) {
      assert.ok(d.errorAt(k).train <= d.errorAt(k - 1).train + 1e-9, 'train rose at degree ' + k);
    }
    assert.ok(d.best >= 3 && d.best <= 6, 'best ' + d.best);
    assert.ok(d.errorAt(MAX_DEGREE).test > 3 * d.bestTestError);
  });

  it('bias falls and variance rises with the degree', () => {
    const d = at({});
    assert.ok(d.sweepAt(1).bias2 > 3 * d.sweepAt(10).bias2, 'bias ' + d.sweepAt(1).bias2 + ' -> ' + d.sweepAt(10).bias2);
    assert.ok(d.sweepAt(10).variance > 100 * d.sweepAt(1).variance, 'variance ' + d.sweepAt(1).variance + ' -> ' + d.sweepAt(10).variance);
    for (let k = 1; k <= MAX_DEGREE; k++) {
      assert.ok(d.sweepAt(k).variance >= d.sweepAt(k - 1).variance, 'variance fell at degree ' + k);
    }
    assert.ok(d.bvBest >= 3 && d.bvBest <= 6, 'bvBest ' + d.bvBest);
  });

  it('the ridge preset shrinks the weights and the test error at degree 12', () => {
    const plain = at(preset('overfit'));
    const ridge = at(preset('ridge'));
    assert.ok(ridge.maxWeight < plain.maxWeight / 5, plain.maxWeight + ' -> ' + ridge.maxWeight);
    assert.ok(ridge.testError < plain.testError / 4, plain.testError + ' -> ' + ridge.testError);
    assert.ok(Math.abs(ridge.maxWeightPlain - plain.maxWeight) < 1e-9);
  });

  it('two hundred points tame degree 12', () => {
    const many = at(preset('many'));
    assert.ok(many.testError < 0.5, 'test ' + many.testError);
    assert.ok(many.testError < 2 * many.trainError);
  });

  it('ten points and ten weights interpolate', () => {
    const d = at(preset('interpolate'));
    assert.ok(d.trainError < 1e-6, 'train ' + d.trainError);
    assert.ok(d.testError > 100, 'test ' + d.testError);
    assert.ok(d.truthGap > 5);
    assert.ok(at({ ...preset('interpolate'), degree: 1 }).testError < 1);
  });

  it('raw x⁸ runs into the hundreds of thousands and standardising keeps the column in the hundreds', () => {
    const raw = at({ degree: 8, standardise: false });
    const std = at({ degree: 8, standardise: true });
    assert.ok(raw.maxFeature > 50_000 && raw.maxFeature <= 390_625, 'raw ' + raw.maxFeature);
    assert.ok(std.maxFeature < 500, 'std ' + std.maxFeature);
    assert.ok(Math.abs(raw.weightAt(8)) < 0.1, 'w8 ' + raw.weightAt(8));
    assert.ok(Math.abs(raw.trainError - std.trainError) < 1e-6, 'the exact solve gives the same fit either way');
  });

  it('the sandbox claims', () => {
    const mean = at({ degree: 0 });
    assert.ok(Math.abs(mean.r2Train) < 1e-9, 'degree 0 explains nothing');
    const line = at({ dataset: 'line', degree: 10 });
    for (let k = 1; k <= 8; k++) assert.ok(line.sweepAt(k).bias2 < 0.05, 'line bias at ' + k + ': ' + line.sweepAt(k).bias2);
    assert.ok(line.sweepAt(0).bias2 > 0.3);
    const parabola = at({ dataset: 'parabola', degree: 2 });
    assert.ok(Math.abs(parabola.errorAt(1).test - 1) < 0.2, 'parabola degree 1 ' + parabola.errorAt(1).test);
    assert.ok(Math.abs(parabola.errorAt(2).test - 0.3) < 0.1, 'parabola degree 2 ' + parabola.errorAt(2).test);
    const step = at({ dataset: 'step', degree: 7 });
    assert.ok(step.sweepAt(7).bias2 > 0.05 && step.sweepAt(7).bias2 < 0.2, 'step bias ' + step.sweepAt(7).bias2);
    const quiet = at({ noise: 0, degree: 5 });
    assert.ok(quiet.trainError < 5e-4 && quiet.testError < 5e-4, quiet.trainError + ' ' + quiet.testError);
    for (let k = 0; k <= MAX_DEGREE; k++) assert.ok(quiet.sweepAt(k).variance < 1e-9);
    const loud = at({ noise: 1, degree: 3 });
    assert.ok(Math.abs(loud.noise - 1) < 1e-9);
    assert.ok(Math.abs(loud.sweepAt(3).variance / at({}).sweepAt(3).variance - 4) < 1e-6, 'variance scales with σ²');
    assert.ok(loud.bvBest <= at({}).bvBest, 'noisier data favours a lower degree');
  });

  it('a dragged point changes the fit but not the refits', () => {
    const rig = make();
    rig.merge(preset('overfit'));
    rig.stepOnce(5);
    const before = context(rig);
    const points = rig.model().data.points.slice();
    points[3] = { ...points[3], y: points[3].y + 2 };
    rig.model().data.points = points;
    rig.model().fit = fitPolynomial(points, rig.model().config);
    const after = context(rig);
    assert.notEqual(before.derived.testError, after.derived.testError);
    assert.equal(before.derived.refits, after.derived.refits);
    assert.equal(before.derived.variance, after.derived.variance);
  });
});
