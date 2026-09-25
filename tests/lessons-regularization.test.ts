/**
 * The ridge and lasso lessons, driven against the real model: every step opens
 * the way the page would, every run under it converges quickly, and every
 * number the copy quotes must hold.
 */

import { strict as assert } from 'node:assert';
import { describe, it } from 'node:test';

import { DEFAULT_PARAMS, PRESETS } from '../src/explainers/regularization/config.ts';
import type { RegParams } from '../src/explainers/regularization/config.ts';
import { LESSONS, makeRegContext, pairOf, truthOf } from '../src/explainers/regularization/lessons.ts';
import type { RegLessonContext } from '../src/explainers/regularization/lessons.ts';
import { generatePolynomial } from '../src/lib/datasets/polynomial.ts';
import { generateTabular, testRows, trainRows } from '../src/lib/datasets/tabular.ts';
import type { Design, TabularData } from '../src/lib/datasets/tabular.ts';
import {
  createState,
  lambdaGrid,
  lambdaMax,
  olsSolution,
  prepare,
  regularisationPath,
  step,
} from '../src/lib/ml/elasticNet.ts';
import type { ElasticNetConfig, ElasticNetState, PathResult, Prepared, Solver } from '../src/lib/ml/elasticNet.ts';
import { axisRadius, constrainedMinimum, levelOf, shapeValue, sliceQuadratic } from '../src/lib/ml/penaltyGeometry.ts';
import { LessonRig, checkStructure, exerciseRun, exerciseScreen, openLesson } from './lessonRig.ts';

interface Model {
  data: TabularData;
  train: Design;
  test: Design;
  prep: Prepared;
  config: ElasticNetConfig;
  path: PathResult;
  ols: number[];
  alpha: number;
  pair: [number, number];
}

type Ui = RegLessonContext['ui'];
type Rig = LessonRig<RegParams, ElasticNetState, Model, Ui>;

/** Mirrors the page: ridge gets its closed form unless a solver is named. */
function solverOf(p: RegParams, alpha: number): Solver {
  if (p.solver === 'ista') return 'ista';
  if (p.solver === 'cd') return 'cd';
  return alpha === 0 ? 'closed' : 'cd';
}

function build(p: RegParams): Model {
  const data =
    p.dataset === 'two'
      ? generateTabular({ count: p.count, features: 2, informative: 2, correlation: 0, groupSize: 3, noise: p.noise, variedScales: false, seed: p.seed, trainFraction: 0.75 })
      : generatePolynomial({ count: p.count, degree: p.features, noise: p.noise, seed: p.seed, trainFraction: 0.75 });
  const train = trainRows(data);
  const test = testRows(data);
  const prep = prepare(train, p.standardise);
  const alpha = p.penalty === 'ridge' ? 0 : p.penalty === 'lasso' ? 1 : p.alpha;
  const config: ElasticNetConfig = { lambda: p.lambda, alpha, solver: solverOf(p, alpha), learningRate: p.learningRate };
  const path = regularisationPath(train, test, prep, alpha, lambdaGrid(lambdaMax(prep, alpha), 60));
  const ols = olsSolution(prep);
  const pair = pairOf(prep.p, p.plane, truthOf(data, prep));
  return { data, train, test, prep, config, path, ols, alpha, pair };
}

const make = (): Rig =>
  new LessonRig({
    defaults: DEFAULT_PARAMS,
    presets: PRESETS,
    build,
    create: (m) => createState(m.prep, m.config),
    step: (s, m) => step(s, m.prep, m.config),
    complete: (s) => s.converged || s.diverged,
    ui: () => ({ open: null }),
    resets: (key) => !['plane', 'showOls', 'showTruth', 'showTest'].includes(key),
  });

const context = (rig: Rig): RegLessonContext => {
  const { data, train, test, prep, path, ols, alpha, pair } = rig.model();
  return {
    ...makeRegContext({ params: rig.params, state: rig.state, data, train, test, prep, path, ols, alpha, pair, ui: rig.ui }),
    ...rig.base(),
  };
};

/** A lesson's one-click runs animate at 20 sweeps a second, so each must finish inside this. */
const RUN_SWEEPS = 200;

describe('regularisation lessons: structure', () => {
  it('follows the shared rules, one screen per lesson', () => checkStructure(LESSONS, PRESETS, DEFAULT_PARAMS, { oneScreen: true }));

  it('is one screen per lesson: six lessons of one step, each a set of runs', () => {
    assert.equal(LESSONS.length, 6);
    for (const lesson of LESSONS) {
      const [only] = lesson.steps;
      assert.ok(only.experiments.length >= 2 && only.experiments.length <= 4, lesson.id + ' has ' + only.experiments.length + ' runs');
      assert.ok(only.enter?.some((a) => a.type === 'runTo'), lesson.id + ' must open solved');
    }
  });

  it('every run names its λ, so runs never stack on each other', () => {
    for (const lesson of LESSONS) {
      for (const run of lesson.steps[0].experiments) {
        assert.ok(run.patch && 'lambda' in run.patch, lesson.id + ' / ' + run.label + ' leaves λ as it was');
      }
    }
  });
});

const reveals: string[] = [];

describe('regularisation lessons: every lesson and every run against the model', () => {
  for (const lesson of LESSONS) {
    it(lesson.id + ' opens on a picture that reads', () => exerciseScreen(openLesson(make, lesson, context), lesson, context, reveals));
    for (const e of lesson.steps[0].experiments) {
      it(lesson.id + ' / ' + e.label, () => exerciseRun(openLesson(make, lesson, context), e, lesson.id + ' / ' + e.label, context, reveals));
    }
  }

  it('every run converges within the animation budget from its lesson’s preset', () => {
    for (const lesson of LESSONS) {
      const [only] = lesson.steps;
      for (const run of only.experiments) {
        const rig = make();
        rig.applyPreset(lesson.presetId);
        rig.merge(run.patch ?? {});
        rig.stepOnce(5000);
        assert.ok(rig.complete && !rig.state.diverged, lesson.id + ' / ' + run.label + ' did not converge');
        assert.ok(rig.state.epoch <= RUN_SWEEPS, lesson.id + ' / ' + run.label + ' took ' + rig.state.epoch + ' sweeps');
      }
    }
  });

  it('prints the live numbers', () => {
    for (const line of reveals) console.log('  ' + line);
  });
});

/** The converged model for a preset, with a patch on top. */
function solved(presetId: string, over: Partial<RegParams> = {}) {
  const rig = make();
  rig.applyPreset(presetId);
  if (Object.keys(over).length > 0) rig.merge(over);
  rig.stepOnce(5000);
  assert.ok(rig.complete, presetId + ': did not converge');
  return { rig, d: context(rig).derived, m: rig.model() };
}

/** `value` reads as `text` to the decimals the copy uses. */
function reads(value: number, text: string, what: string) {
  const decimals = (text.split('.')[1] ?? '').length;
  assert.equal(value.toFixed(decimals), text, what + ': ' + value + ' does not read as ' + text);
}

describe('regularisation lessons: the numbers in the copy hold', () => {
  it('overfitting: no penalty fits the noise, λ = 0.1 fixes it, ten times the data fixes it too', () => {
    const none = solved('overfit').d;
    reads(none.trainMse, '0.007', 'fitted error');
    reads(none.testMse, '0.026', 'held-out error');
    assert.ok(none.trainMse < (none.truthTrainMse ?? 0), 'below the wave: ' + none.trainMse + ' vs ' + none.truthTrainMse);
    reads(none.maxAbs, '0.89', 'largest weight');
    assert.equal(none.n + solved('overfit').m.test.X.length, 24);
    assert.equal(none.n, 18);
    const some = solved('overfit', { lambda: 0.1, count: 24 }).d;
    reads(some.maxAbs, '0.39', 'largest weight at λ = 0.1');
    reads(some.testMse, '0.015', 'held-out error at λ = 0.1');
    const many = solved('overfit', { lambda: 0.0001, count: 200 }).d;
    reads(many.trainMse, '0.037', 'fitted error with 200 points');
    reads(many.truthTrainMse ?? 0, '0.040', 'the wave on 200 points');
  });

  it('ridge: shrinks with λ, never to zero', () => {
    for (const [lambda, max, test] of [
      [0.1, '0.39', '0.015'],
      [0.0001, '0.89', '0.026'],
      [1, '0.12', '0.052'],
      [10, '0.024', '0.104'],
    ] as const) {
      const d = solved('ridge', { lambda }).d;
      assert.equal(d.activeCount, 10, 'ridge λ = ' + lambda + ' keeps ' + d.activeCount);
      reads(d.maxAbs, max, 'largest weight at λ = ' + lambda);
      reads(d.testMse, test, 'held-out error at λ = ' + lambda);
    }
    assert.equal(solved('ridge').rig.state.epoch, 1, 'the closed form takes one step');
  });

  it('lasso: the terms drop out one by one', () => {
    const few = solved('lasso').d;
    assert.deepEqual(few.activeNames, ['x', 'x²', 'x⁵']);
    reads(few.testMse, '0.019', 'held-out error at λ = 0.03');
    const two = solved('lasso', { lambda: 0.1 });
    assert.deepEqual(two.d.activeNames, ['x²', 'x⁷']);
    assert.equal(two.rig.state.epoch, 14);
    reads(two.d.testMse, '0.065', 'held-out error at λ = 0.1');
    const one = solved('lasso', { lambda: 0.2 }).d;
    assert.deepEqual(one.activeNames, ['x²']);
    reads(one.testMse, '0.092', 'held-out error at λ = 0.2');
    const none = solved('lasso', { lambda: 0.5 }).d;
    assert.equal(none.activeCount, 0);
    reads(none.lambdaMax, '0.29', 'λ max');
    reads(none.testMse, '0.127', 'held-out error above λ max');
  });

  it('two weights: ridge lands off the axes, the lasso and the elastic net on one', () => {
    const ridge = solved('two-weights', { penalty: 'ridge', alpha: 0.5, lambda: 1 });
    reads(ridge.d.pairWeights[0], '1.40', 'ridge w₁');
    reads(ridge.d.pairWeights[1], '0.48', 'ridge w₂');
    reads(ridge.m.ols[0], '2.81', 'least squares w₁');
    reads(ridge.m.ols[1], '1.00', 'least squares w₂');
    const lasso = solved('two-weights', { penalty: 'lasso', alpha: 0.5, lambda: 1 });
    reads(lasso.d.pairWeights[0], '1.78', 'lasso w₁');
    assert.equal(lasso.d.pairWeights[1], 0);
    // The hollow dot: ridge with the diamond's axis radius stays off the axis.
    const q = sliceQuadratic(lasso.m.prep, lasso.rig.state.w, 0, 1);
    const radius = axisRadius(1, shapeValue(1, lasso.d.pairWeights));
    const other = constrainedMinimum(q, 0, levelOf(0, radius));
    assert.ok(Math.abs(other[1]) > 0.05, 'ridge at the same budget ' + other);
    const elastic = solved('two-weights', { penalty: 'elastic', alpha: 0.5, lambda: 2 }).d;
    reads(elastic.pairWeights[0], '0.89', 'elastic w₁');
    assert.equal(elastic.pairWeights[1], 0);
  });

  it('the constrained minimum agrees with the solver on the two-weight problem', () => {
    for (const penalty of ['ridge', 'lasso', 'elastic'] as const) {
      const { rig, m } = solved('two-weights', { penalty, alpha: 0.5 });
      const alpha = m.alpha;
      const q = sliceQuadratic(m.prep, rig.state.w, 0, 1);
      const level = shapeValue(alpha, [rig.state.w[0], rig.state.w[1]]);
      const point = constrainedMinimum(q, alpha, level);
      assert.ok(Math.hypot(point[0] - rig.state.w[0], point[1] - rig.state.w[1]) < 1e-3, penalty + ': ' + point + ' vs ' + rig.state.w.slice(0, 2));
    }
  });

  it('choosing λ: the cross sits at 0.078 and moves with the noise', () => {
    const base = solved('ridge').d;
    reads(base.bestLambda, '0.078', 'best λ');
    reads(base.bestTestMse, '0.015', 'best held-out error');
    reads(solved('ridge', { lambda: 0.078, noise: 0.3 }).d.testMse, '0.015', 'held-out error at the best λ');
    reads(solved('ridge', { lambda: 0.1, noise: 0.6 }).d.bestLambda, '0.37', 'best λ with noise 0.6');
    const clean = solved('ridge', { lambda: 0.1, noise: 0.1 });
    const grid = clean.m.path.lambdas;
    assert.equal(clean.d.bestLambda, Math.min(...grid), 'best λ with noise 0.1 is the smallest on the chart');
  });

  it('experiments: three powers underfit, wide data fits exactly, the elastic net keeps five, raw powers dwarf x¹⁰', () => {
    const three = solved('overfit', { features: 3 }).d;
    reads(three.testMse, '0.027', 'held-out error with three powers');
    assert.ok(three.testMse > solved('overfit').d.testMse, 'no better than ten powers');
    const wide = solved('overfit', { features: 12, count: 12 }).d;
    assert.equal(wide.n, 9);
    assert.equal(wide.p, 12);
    reads(wide.trainMse, '0.002', 'fitted error on wide data');
    reads(wide.testMse, '0.034', 'held-out error on wide data');
    const elastic = solved('overfit', { penalty: 'elastic', alpha: 0.5, lambda: 0.1 }).d;
    assert.deepEqual(elastic.activeNames, ['x', 'x²', 'x⁵', 'x⁷', 'x⁹']);
    reads(elastic.testMse, '0.038', 'held-out error for the elastic net');
    assert.equal(solved('overfit', { penalty: 'lasso', lambda: 0.1 }).d.activeCount, 2);
    const raw = solved('overfit', { lambda: 0.1, standardise: false });
    const xTen = raw.rig.state.w[9];
    assert.ok(xTen > 0 && xTen < 2e-6, 'w(x¹⁰) on raw powers ' + xTen);
    reads(raw.d.testMse, '0.019', 'held-out error unstandardised');
    const xMax = Math.max(...raw.m.train.X.map((row) => Math.abs(row[0])));
    const xTenMax = Math.max(...raw.m.train.X.map((row) => Math.abs(row[9])));
    assert.ok(xMax < 3 && xTenMax > 1000, 'x spans ' + xMax + ', x¹⁰ ' + xTenMax);
  });
});

describe('regularisation: the true weights on the plane', () => {
  it('a noise-free least-squares fit lands exactly on the true weights the page draws', () => {
    for (const standardise of [true, false]) {
      for (const variedScales of [false, true]) {
        const data = generateTabular({ count: 60, features: 4, informative: 3, correlation: 0, groupSize: 3, noise: 0, variedScales, seed: 31, trainFraction: 0.75 });
        const prep = prepare(trainRows(data), standardise);
        const ols = olsSolution(prep);
        truthOf(data, prep)!.forEach((w, j) => {
          assert.ok(Math.abs(ols[j] - w) < 1e-6 * Math.max(1, Math.abs(w)), 'w' + (j + 1) + ' ' + ols[j] + ' against ' + w);
        });
      }
    }
  });
});
