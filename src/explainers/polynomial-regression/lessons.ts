/** Polynomial regression lessons, one screen each. Pure data, driven against the real model in tests/lessons-polynomial.test.ts. */

import type { Lesson, LessonAction, LessonContextBase, LiveNumber } from '../../explainer/lessons';
import type { RegressionData, RegressionPoint } from '../../lib/datasets/types';
import { fmtKnob, rSquared } from '../../lib/math/stats';
import { fitPolynomial, meanSquaredError, predictFit } from '../../lib/ml/polynomialRegression';
import type { BiasVariance, BiasVarianceSweep, PolyFit, RefitState, ValidationCurve } from '../../lib/ml/polynomialRegression';
import type { PolyParams } from './config';
import { PR_NODES, inputId } from './graph';

/* ---------------- context ---------------- */

export type SecondView = 'curve' | 'bv';

export interface PolyLessonContext extends LessonContextBase {
  params: PolyParams;
  state: RefitState;
  derived: {
    n: number;
    nTest: number;
    degree: number;
    weightCount: number;
    trainError: number;
    testError: number;
    r2Train: number;
    r2Test: number;
    failed: boolean;
    /** Largest |w| among the non-bias weights. */
    maxWeight: number;
    weightAt: (index: number) => number;
    /** The same at λ = 0, for the ridge step. */
    maxWeightPlain: number;
    /** Largest |x̃ᵏ| over the training points. */
    maxFeature: number;
    /** Degree with the lowest test error, and that error. */
    best: number;
    bestTestError: number;
    errorAt: (degree: number) => { train: number; test: number };
    /** What the true function scores on the test points: the noise. */
    truthTestError: number;
    /** σ² of the noise. */
    noise: number;
    /** Refits so far, and the target count. */
    refits: number;
    total: number;
    /** From the refits so far; NaN under two. */
    bias2: number;
    variance: number;
    /** The decomposition at any degree, from the same refits. */
    sweepAt: (degree: number) => { bias2: number; variance: number; total: number };
    /** Degree where bias² + variance + σ² is lowest. */
    bvBest: number;
    /** Furthest the fitted curve strays from the truth over the inner x range. */
    truthGap: number;
  };
  ui: {
    view: SecondView;
    logScale: boolean;
    open: string | null;
    pointEdits: number;
  };
}

export interface PolyLessonInput {
  params: PolyParams;
  state: RefitState;
  data: RegressionData;
  test: readonly RegressionPoint[];
  fit: PolyFit;
  curve: ValidationCurve;
  sweep: BiasVarianceSweep;
  /** The refits so far, scored on the inner grid. */
  bv: BiasVariance;
  /** The bias-variance grid and the truth on it. */
  grid: readonly number[];
  truthOnGrid: readonly number[];
  ui: PolyLessonContext['ui'];
}

/** The page's share of the context; the lesson hook adds `sim` and `lesson`. */
export function makePolyContext(input: PolyLessonInput): Omit<PolyLessonContext, keyof LessonContextBase> {
  const { params, state, data, test, fit, curve, sweep, bv, grid, truthOnGrid, ui } = input;
  const points = data.points;
  const predictions = points.map((p) => predictFit(fit, p.x));
  const actuals = points.map((p) => p.y);
  const testPredictions = test.map((p) => predictFit(fit, p.x));
  const testActuals = test.map((p) => p.y);
  const truth = data.truth ?? (() => 0);
  const plain = params.l2 > 0 ? fitPolynomial(points, { degree: fit.degree, l2: 0, standardise: params.standardise }) : fit;

  let maxFeature = 0;
  for (const p of points) {
    const row = fitRow(fit, p.x);
    for (const v of row) maxFeature = Math.max(maxFeature, Math.abs(v));
  }
  let truthGap = 0;
  for (let i = 0; i < grid.length; i++) {
    const gap = Math.abs(predictFit(fit, grid[i]) - truthOnGrid[i]);
    if (gap > truthGap) truthGap = gap;
  }
  const totals = sweep.bias2.map((b, i) => b + sweep.variance[i] + sweep.noise);
  let bvBest = 0;
  totals.forEach((t, i) => {
    if (Number.isFinite(t) && (!Number.isFinite(totals[bvBest]) || t < totals[bvBest])) bvBest = i;
  });

  return {
    params,
    state,
    derived: {
      n: points.length,
      nTest: test.length,
      degree: fit.degree,
      weightCount: fit.degree + 1,
      trainError: fit.failed ? Number.NaN : meanSquaredError(fit, points),
      testError: fit.failed ? Number.NaN : meanSquaredError(fit, test),
      r2Train: rSquared(actuals, predictions),
      r2Test: rSquared(testActuals, testPredictions),
      failed: fit.failed,
      maxWeight: largest(fit.weights.slice(1)),
      weightAt: (index) => fit.weights[index] ?? Number.NaN,
      maxWeightPlain: largest(plain.weights.slice(1)),
      maxFeature,
      best: curve.best,
      bestTestError: curve.test[curve.best],
      errorAt: (degree) => ({ train: curve.train[degree] ?? Number.NaN, test: curve.test[degree] ?? Number.NaN }),
      truthTestError: test.reduce((a, p) => a + (truth(p.x) - p.y) ** 2, 0) / Math.max(1, test.length),
      noise: sweep.noise,
      refits: state.curves.length,
      total: params.refits,
      bias2: bv.bias2,
      variance: bv.variance,
      sweepAt: (degree) => ({
        bias2: sweep.bias2[degree] ?? Number.NaN,
        variance: sweep.variance[degree] ?? Number.NaN,
        total: totals[degree] ?? Number.NaN,
      }),
      bvBest,
      truthGap,
    },
    ui,
  };
}

function fitRow(fit: PolyFit, x: number): number[] {
  const z = (x - fit.norm.mean) / fit.norm.std;
  const out: number[] = [];
  let power = 1;
  for (let i = 0; i <= fit.degree; i++) {
    out.push(power);
    power *= z;
  }
  return out;
}

function largest(values: readonly number[]): number {
  return values.reduce((best, w) => (Number.isFinite(w) ? Math.max(best, Math.abs(w)) : best), 0);
}

/* ---------------- copy helpers ---------------- */

type Ctx = PolyLessonContext;
type L = Lesson<Ctx, PolyParams>;

const steps = (n: number) => n.toLocaleString('en-US');

/** Errors: three decimals under 10, compact above, scientific once they run away. */
function num(v: number): string {
  if (!Number.isFinite(v)) return 'infinity';
  if (Math.abs(v) >= 1e6) return v.toExponential(1);
  if (Math.abs(v) >= 1000) return steps(Math.round(v));
  if (Math.abs(v) >= 10) return v.toFixed(1);
  return v.toFixed(3);
}

/** A ratio for prose: 3, 12, 760. */
function times(v: number): string {
  if (!Number.isFinite(v)) return 'many';
  if (v >= 100) return steps(Math.round(v / 10) * 10);
  if (v >= 10) return steps(Math.round(v));
  return v.toFixed(1).replace(/\.0$/, '');
}

const refitsDone = (c: Ctx) => c.state.epoch >= c.derived.total;

/* ---------------- the lessons ---------------- */

type Act = LessonAction<PolyParams>;

const CURVE: Act = { type: 'view', id: 'second', value: 'curve' };
const BV: Act = { type: 'view', id: 'second', value: 'bv' };
const LOG: Act = { type: 'view', id: 'scale', value: 'log' };
/** The refit loop, run to its end. */
const REFITS: Act = { type: 'runTo', steps: 500 };
/** An instant run: the exact solve needs no steps. */
const NOW = () => true;

/** A run restates every knob it depends on: the preset, then its own changes. */
const on = (preset: string, patch: Partial<PolyParams> | null, ...rest: Act[]): Act[] => [
  { type: 'preset', id: preset },
  ...(patch ? [{ type: 'params' as const, patch }] : []),
  ...rest,
];

const errors = (c: Ctx): LiveNumber[] => [
  { label: 'degree', value: steps(c.derived.degree) },
  { label: 'train', value: num(c.derived.trainError) },
  { label: 'test', value: num(c.derived.testError) },
];

const trainTest = (c: Ctx) => 'training error ' + num(c.derived.trainError) + ', test error ' + num(c.derived.testError);

const introduction: L = {
  id: 'polynomial',
  title: 'Polynomial regression',
  hook: 'One knob, the degree, from 1 to 12.',
  section: 'hook',
  presetId: 'line',
  knobs: ['degree'],
  steps: [
    {
      id: 'degree',
      kind: 'sandbox',
      say: '25 dots from a wave; the 15 rings are test points the fit never sees. Degree 1 is a straight line with two weights. It cannot bend, so it misses the wave on both sides: training error 0.457, test error 0.766. Try it: Degree 3.',
      takeaway: 'Polynomial regression is linear regression on the powers of x. Another weight can never raise the training error; the test rings tell the truth.',
      focus: { kind: 'control', key: 'degree', label: 'Degree' },
      enter: [{ type: 'preset', id: 'line' }, { type: 'reset' }, CURVE, LOG],
      numbers: errors,
      experiments: [
        {
          label: 'Degree 3',
          say: 'x, x² and x³ as three inputs to one weighted sum.',
          enter: on('line', { degree: 3 }),
          until: NOW,
          then: (c) => 'Degree 3, ' + steps(c.derived.weightCount) + ' weights: ' + trainTest(c) + '. The curve follows the wave.',
          focus: { kind: 'node', id: PR_NODES.sum, label: 'Σ' },
        },
        {
          label: 'Degree 12',
          say: 'Thirteen weights for twenty-five points.',
          enter: on('line', { degree: 12 }),
          until: NOW,
          then: (c) => 'Degree 12: ' + trainTest(c) + '. The curve passes close to every dot and swings between them; the rings pay for it.',
          focus: { kind: 'metric', key: 'test', label: 'Test error' },
        },
        {
          label: 'Degree 1 again',
          say: 'Back to the line.',
          enter: on('line', null),
          until: NOW,
          then: (c) => 'Degree 1: ' + trainTest(c) + '. Underfitting: the model is too simple for the wave.',
          focus: { kind: 'control', key: 'degree', label: 'Degree' },
        },
      ],
    },
  ],
};

const features: L = {
  id: 'features',
  title: 'Polynomial features',
  hook: 'x̃, x̃², x̃³ as inputs, and their scales.',
  section: 'core',
  presetId: 'cubic',
  knobs: ['degree', 'standardise'],
  steps: [
    {
      id: 'inputs',
      kind: 'sandbox',
      say: 'Degree 3: the inputs are x̃, x̃² and x̃³, each a column of 25 values, plus the constant 1 for the bias. The Σ card shows the equation with its live weights and, under it, the normal equations that solved it in one go: no steps. Try it: Raw powers.',
      takeaway: 'Least squares has a closed form, w = (ΦᵀΦ)⁻¹Φᵀy. Standardise before raising to powers, so every column stays on a scale the arithmetic can handle.',
      focus: { kind: 'node', id: PR_NODES.sum, label: 'Σ' },
      enter: [{ type: 'preset', id: 'cubic' }, { type: 'reset' }, CURVE, LOG, { type: 'open', target: 'node:' + PR_NODES.sum }],
      numbers: (c) => [
        { label: 'degree', value: steps(c.derived.degree) },
        { label: 'largest |xᵏ|', value: num(c.derived.maxFeature) },
        { label: 'largest |w|', value: num(c.derived.maxWeight) },
      ],
      experiments: [
        {
          label: 'Raw powers, degree 8',
          say: 'Standardise x off.',
          enter: on('cubic', { degree: 8, standardise: false }, { type: 'open', target: null }),
          until: NOW,
          then: (c) =>
            'x runs from 0 to 5, so raw x⁸ reaches ' + num(c.derived.maxFeature) + ' while the constant stays at 1, and w₈ shrinks to ' +
            num(Math.abs(c.derived.weightAt(8))) + ' to compensate. The exact solve survives; gradient descent would not.',
          focus: { kind: 'node', id: inputId(8), label: 'x⁸' },
        },
        {
          label: 'Standardised, degree 8',
          say: 'Standardise x on.',
          enter: on('cubic', { degree: 8, standardise: true }, { type: 'open', target: null }),
          until: NOW,
          then: (c) =>
            'x̃ = (x − μ) / s runs from about −1.7 to 1.7, so x̃⁸ tops out at ' + num(c.derived.maxFeature) + ' and the largest weight is ' +
            num(c.derived.maxWeight) + '. The fit is the same curve.',
          focus: { kind: 'control', key: 'standardise', label: 'Standardise x' },
        },
      ],
    },
  ],
};

const validation: L = {
  id: 'errors',
  title: 'Training and test error',
  hook: 'One falls forever, the other turns back up.',
  section: 'core',
  presetId: 'line',
  knobs: ['degree'],
  steps: [
    {
      id: 'curve',
      kind: 'sandbox',
      say: 'The chart on the right refits every degree from 0 to 15 on the same 25 dots and scores each on the 15 rings. Training error only falls, 0.530 to 0.118. Test error bottoms out at 0.364 at degree 5, then climbs to 5.344. Try it: Degree 5.',
      takeaway: 'Training error measures memory, test error measures prediction. Choosing the degree by held-out error is model selection.',
      focus: { kind: 'panel', id: 'second', label: 'Validation curve' },
      enter: [{ type: 'preset', id: 'line' }, { type: 'reset' }, CURVE, LOG],
      numbers: errors,
      experiments: [
        {
          label: 'Degree 5, the best',
          say: 'The lowest test error of all.',
          enter: on('line', { degree: 5 }),
          until: NOW,
          then: (c) =>
            'Degree 5: test error ' + num(c.derived.testError) + ', the lowest. The true wave itself scores ' + num(c.derived.truthTestError) +
            ' on the rings: that is the noise alone, and no fit can honestly beat it.',
          focus: { kind: 'metric', key: 'test', label: 'Test error' },
        },
        {
          label: 'Degree 3',
          say: 'Two weights fewer.',
          enter: on('line', { degree: 3 }),
          until: NOW,
          then: (c) =>
            'Degree 3: test error ' + num(c.derived.testError) + ', ' + (c.derived.testError <= 1.05 * c.derived.bestTestError ? 'within 5% of' : 'close to') +
            ' the best. Degrees 3 to 6 are all fine choices here.',
          focus: { kind: 'metric', key: 'test', label: 'Test error' },
        },
        {
          label: 'Degree 12',
          say: 'Far up the right arm of the U.',
          enter: on('line', { degree: 12 }),
          until: NOW,
          then: (c) => 'Degree 12: ' + trainTest(c) + '. The training error keeps falling; the test error has turned back up.',
          focus: { kind: 'panel', id: 'second', label: 'Validation curve' },
        },
      ],
    },
  ],
};

const biasVarianceLesson: L = {
  id: 'bias-variance',
  title: 'Bias and variance',
  hook: 'Fifty refits on fresh noise, at three degrees.',
  section: 'core',
  presetId: 'bias',
  knobs: ['degree'],
  steps: [
    {
      id: 'refits',
      kind: 'sandbox',
      say: 'Each refit keeps the x’s, draws fresh noise and fits again: fifty lines. They sit in a narrow band, variance 0.020, but their average, dashed, misses the wave: bias² 0.377. The chart adds bias², variance and σ² for every degree. Try it: Degree 10.',
      takeaway: 'Expected test error = bias² + variance + σ². Bias is the average fit’s error; variance is how much one fit depends on its noise.',
      focus: { kind: 'panel', id: 'fit', label: 'The fit' },
      enter: [{ type: 'preset', id: 'bias' }, { type: 'reset' }, BV, REFITS],
      numbers: (c) => [
        { label: 'degree', value: steps(c.derived.degree) },
        { label: 'refits', value: steps(c.derived.refits) },
        { label: 'bias²', value: num(c.derived.bias2) },
        { label: 'variance', value: num(c.derived.variance) },
      ],
      experiments: [
        {
          label: 'Degree 10 refits',
          say: 'Same x’s, same noise level, ten weights more.',
          enter: on('variance', null, BV, { type: 'reset' }, { type: 'play' }),
          speed: 'fast',
          until: refitsDone,
          then: (c) =>
            steps(c.derived.refits) + ' refits at degree 10: variance ' + num(c.derived.variance) + ', about ' +
            times(c.derived.sweepAt(10).variance / c.derived.sweepAt(1).variance) + ' times the line’s. Bias² ' + num(c.derived.bias2) +
            ': the average is close to the wave, each single fit is not.',
          focus: { kind: 'metric', key: 'variance', label: 'Variance' },
        },
        {
          label: 'Degree 3 refits',
          say: 'The degree where neither term dominates.',
          enter: on('balanced', null, BV, { type: 'reset' }, { type: 'play' }),
          speed: 'fast',
          until: refitsDone,
          then: (c) =>
            'Degree 3: bias² ' + num(c.derived.bias2) + ', variance ' + num(c.derived.variance) + '. The refits agree with each other and with the wave; over all degrees their sum is lowest at degree ' +
            steps(c.derived.bvBest) + '.',
          focus: { kind: 'panel', id: 'second', label: 'Bias and variance' },
        },
        {
          label: 'Degree 1 refits',
          say: 'The lines again.',
          enter: on('bias', null, BV, { type: 'reset' }, { type: 'play' }),
          speed: 'fast',
          until: refitsDone,
          then: (c) => steps(c.derived.refits) + ' refits at degree 1: variance ' + num(c.derived.variance) + ', bias² ' + num(c.derived.bias2) + '. A line is biased on a wave however much data it sees.',
          focus: { kind: 'panel', id: 'fit', label: 'The fit' },
        },
      ],
    },
  ],
};

const overfitting: L = {
  id: 'overfitting',
  title: 'Overfitting',
  hook: 'Thirteen weights for twenty-five points.',
  section: 'failure',
  presetId: 'overfit',
  knobs: ['degree', 'sampleCount', 'l2'],
  steps: [
    {
      id: 'degree-12',
      kind: 'sandbox',
      say: 'Degree 12: thirteen weights for twenty-five dots. Training error 0.135, test error 8.603, 64 times higher. The curve has enough freedom to visit the noise; move one dot and the whole curve would move with it. Try it: 200 points.',
      takeaway: 'Low training error with high test error is overfitting. More data or a penalty tames the same degree.',
      focus: { kind: 'metric', key: 'test', label: 'Test error' },
      enter: [{ type: 'preset', id: 'overfit' }, { type: 'reset' }, CURVE, LOG],
      numbers: errors,
      experiments: [
        {
          label: '200 points',
          say: 'Same degree, eight times the data.',
          enter: on('many', null),
          until: NOW,
          then: (c) => 'Degree 12 on ' + steps(c.derived.n) + ' points: ' + trainTest(c) + '. Thirteen weights cannot chase the noise of two hundred points.',
          focus: { kind: 'control', key: 'sampleCount', label: 'Points' },
        },
        {
          label: 'Ridge penalty λ = 0.01',
          say: 'Back to 25 points, with a charge on large weights.',
          enter: on('ridge', null),
          until: NOW,
          then: (c) =>
            'λ = ' + fmtKnob(c.params.l2) + ': the largest weight goes from ' + num(c.derived.maxWeightPlain) + ' to ' + num(c.derived.maxWeight) +
            ', and the test error to ' + num(c.derived.testError) + '. Same degree, less variance.',
          focus: { kind: 'control', key: 'l2', label: 'Ridge (λ)' },
        },
        {
          label: 'Degree 3',
          say: 'Fewer weights instead.',
          enter: on('overfit', { degree: 3 }),
          until: NOW,
          then: (c) => 'Degree 3 on the same 25 dots: ' + trainTest(c) + '. The cheapest cure is a model that cannot bend that much.',
          focus: { kind: 'control', key: 'degree', label: 'Degree' },
        },
      ],
    },
  ],
};

const interpolation: L = {
  id: 'interpolation',
  title: 'Interpolation',
  hook: 'Ten points, ten weights, zero training error.',
  section: 'failure',
  presetId: 'interpolate',
  knobs: ['degree'],
  steps: [
    {
      id: 'exact',
      kind: 'sandbox',
      say: 'Degree 9 has ten weights and there are ten dots: one solution, through every one of them. Training error 0.000, test error 6.4e+8. Fifty refits on fresh noise gave fifty entirely different curves through their ten dots. Try it: Degree 1.',
      takeaway: 'With as many weights as points, least squares stops fitting and starts interpolating: the curve is made of nothing but noise.',
      focus: { kind: 'panel', id: 'fit', label: 'The fit' },
      enter: [{ type: 'preset', id: 'interpolate' }, { type: 'reset' }, CURVE, LOG, REFITS],
      numbers: (c) => [...errors(c), { label: 'variance', value: num(c.derived.variance) }],
      experiments: [
        {
          label: 'Degree 1',
          say: 'A line through the same ten dots.',
          enter: on('interpolate', { degree: 1 }, { type: 'reset' }, REFITS),
          until: refitsDone,
          then: (c) => 'Degree 1: ' + trainTest(c) + ', variance ' + num(c.derived.variance) + ' over ' + steps(c.derived.refits) + ' refits. With ten points only a low degree can be trusted.',
          focus: { kind: 'control', key: 'degree', label: 'Degree' },
        },
        {
          label: 'Degree 4',
          say: 'Five weights for ten dots.',
          enter: on('interpolate', { degree: 4 }, { type: 'reset' }, REFITS),
          until: refitsDone,
          then: (c) => 'Degree 4: ' + trainTest(c) + ', variance ' + num(c.derived.variance) + '. Half the weights, and the curve is no longer free between the dots.',
          focus: { kind: 'control', key: 'degree', label: 'Degree' },
        },
      ],
    },
  ],
};

const experiments: L = {
  id: 'experiments',
  title: 'Experiments',
  hook: 'Six experiments with known outcomes.',
  section: 'sandbox',
  presetId: 'cubic',
  steps: [
    {
      id: 'sandbox',
      kind: 'sandbox',
      say: 'Pick an experiment: the settings change and the outcome opens under it.',
      enter: [{ type: 'preset', id: 'cubic' }, { type: 'reset' }, { type: 'view', id: 'second', value: 'curve' }],
      experiments: [
        {
          label: 'Degree 0',
          say: 'Degree 0 predicts the mean of y. Its training error is the variance of y, the number every other degree is measured against.',
          patch: { degree: 0 },
          focus: { kind: 'control', key: 'degree', label: 'Degree' },
        },
        {
          label: 'Straight-line truth',
          say: 'A linear truth at degree 10. From degree 1 up the bias is close to zero, so everything the validation curve shows above degree 1 is variance.',
          patch: { dataset: 'line', degree: 10 },
          focus: { kind: 'panel', id: 'second', label: 'Validation curve' },
        },
        {
          label: 'Parabola at degree 2',
          say: 'A quadratic truth: degree 1 misses with test error near 1, degree 2 is exactly right at about 0.3, and nothing above it helps.',
          patch: { dataset: 'parabola', degree: 2 },
          focus: { kind: 'panel', id: 'fit', label: 'The fit' },
        },
        {
          label: 'Step',
          say: 'A jump is not a polynomial. The bias never goes away, bias² is still about 0.1 at degree 7, and the curve rings on both sides of the step.',
          patch: { dataset: 'step', degree: 7 },
          focus: { kind: 'panel', id: 'fit', label: 'The fit' },
        },
        {
          label: 'Zero noise',
          say: 'With no noise the variance is zero at every degree, and the wave is a degree-5 polynomial to three decimals: both errors reach 0.000.',
          patch: { noise: 0, degree: 5 },
          focus: { kind: 'control', key: 'noise', label: 'Noise' },
        },
        {
          label: 'Double the noise',
          say: 'Noise 1.0 quadruples σ² to 1. Every variance quadruples with it, so the U bottoms out higher and its best degree moves down.',
          patch: { noise: 1, degree: 3 },
          focus: { kind: 'panel', id: 'second', label: 'Validation curve' },
        },
      ],
    },
  ],
};

export const LESSONS: L[] = [introduction, features, validation, biasVarianceLesson, overfitting, interpolation, experiments];
