/** Ridge and lasso lessons. Pure data, driven against the real model in tests/lessons-regularization.test.ts. */

import type { Lesson, LessonContextBase } from '../../explainer/lessons';
import type { PolynomialData } from '../../lib/datasets/polynomial';
import { isPolynomial } from '../../lib/datasets/polynomial';
import type { Design, TabularData } from '../../lib/datasets/tabular';
import { lambdaMax, meanSquaredErrorOf } from '../../lib/ml/elasticNet';
import type { ElasticNetState, PathResult, Prepared } from '../../lib/ml/elasticNet';
import type { RegParams } from './config';

/* ---------------- context ---------------- */

export interface RegLessonContext extends LessonContextBase {
  params: RegParams;
  state: ElasticNetState;
  derived: {
    n: number;
    p: number;
    names: string[];
    /** The penalty's mix: 1 lasso, 0 ridge. */
    alpha: number;
    stopped: boolean;
    activeCount: number;
    activeNames: string[];
    maxAbs: number;
    olsMaxAbs: number;
    trainMse: number;
    testMse: number;
    olsTrainMse: number;
    olsTestMse: number;
    /** What the wave itself scores on the fitted points, the floor no curve should beat honestly. */
    truthTrainMse: number | null;
    lambdaMax: number;
    bestLambda: number;
    bestTestMse: number;
    /** The two weights the geometry draws. */
    pair: [number, number];
    pairWeights: [number, number];
  };
  ui: {
    /** The feature whose card is open, if any. */
    open: number | null;
  };
}

export interface RegLessonInput {
  params: RegParams;
  state: ElasticNetState;
  data: TabularData | PolynomialData;
  train: Design;
  test: Design;
  prep: Prepared;
  path: PathResult;
  ols: readonly number[];
  alpha: number;
  pair: [number, number];
  ui: RegLessonContext['ui'];
}

const ZERO = 1e-12;

/** The table's true weights in the solver's space, so bars and ticks share one scale. The curve has none. */
export function truthOf(data: TabularData | PolynomialData, prep: Prepared): number[] | null {
  return isPolynomial(data) ? null : data.trueWeights.map((w, j) => (w * prep.stds[j]) / data.scales[j]);
}

/** The two weights the plane draws: the picked consecutive pair of the curve, or the table's two strongest true weights. */
export function pairOf(p: number, plane: number, truth: readonly number[] | null): [number, number] {
  if (truth) {
    const order = truth.map((v, j) => ({ j, v: Math.abs(v) })).sort((a, b) => b.v - a.v || a.j - b.j);
    return [order[0]?.j ?? 0, order[1]?.j ?? 1];
  }
  const first = Math.min(Math.max(0, Math.round(plane)), Math.max(0, p - 2));
  return [first, Math.min(first + 1, Math.max(0, p - 1))];
}

/** The page's share of the context; the lesson hook adds `sim` and `lesson`. */
export function makeRegContext(input: RegLessonInput): Omit<RegLessonContext, keyof LessonContextBase> {
  const { params, state, data, train, test, prep, path, ols, alpha, pair, ui } = input;
  const w = state.w;
  const finite = !state.diverged && w.every((v) => Number.isFinite(v));
  const active = w.map((v, j) => (Math.abs(v) > ZERO ? j : -1)).filter((j) => j >= 0);
  const curve = isPolynomial(data) ? data : null;
  return {
    params,
    state,
    derived: {
      n: prep.n,
      p: prep.p,
      names: data.featureNames.slice(),
      alpha,
      stopped: state.converged || state.diverged,
      activeCount: active.length,
      activeNames: active.map((j) => data.featureNames[j]),
      maxAbs: finite ? w.reduce((m, v) => Math.max(m, Math.abs(v)), 0) : Infinity,
      olsMaxAbs: ols.reduce((m, v) => Math.max(m, Math.abs(v)), 0),
      trainMse: finite ? meanSquaredErrorOf(w, prep, train) : Infinity,
      testMse: finite ? meanSquaredErrorOf(w, prep, test) : Infinity,
      olsTrainMse: meanSquaredErrorOf(ols, prep, train),
      olsTestMse: meanSquaredErrorOf(ols, prep, test),
      truthTrainMse: curve
        ? data.trainIndex.reduce((a, i) => a + (curve.y[i] - curve.truth(curve.x[i])) ** 2, 0) / Math.max(1, data.trainIndex.length)
        : null,
      lambdaMax: lambdaMax(prep, Math.max(alpha, 1e-9)),
      bestLambda: path.lambdas[path.bestIndex] ?? 0,
      bestTestMse: path.testMse[path.bestIndex] ?? 0,
      pair,
      pairWeights: [Number.isFinite(w[pair[0]]) ? w[pair[0]] : 0, Number.isFinite(w[pair[1]]) ? w[pair[1]] : 0],
    },
    ui,
  };
}

/* ---------------- the lessons ---------------- */

// One screen per lesson: the picture opens solved, the text says what is on it, the runs
// under it are one click each, and every number quoted is checked by the test suite.

type L = Lesson<RegLessonContext, RegParams>;

const SOLVE = [{ type: 'reset' as const }, { type: 'runTo' as const, steps: 5000 }];
const LAMBDA = { kind: 'control' as const, key: 'lambda', label: 'Strength (λ)' };
const PENALTY = { kind: 'custom' as const, id: 'model:penalty', label: 'The charge' };

const overfitting: L = {
  id: 'overfitting',
  title: 'Overfitting',
  hook: 'The problem a penalty solves.',
  section: 'hook',
  presetId: 'overfit',
  steps: [
    {
      id: 'no-penalty',
      title: 'No penalty',
      kind: 'sandbox',
      enter: [{ type: 'preset', id: 'overfit' }, ...SOLVE],
      say: 'Blue dots are fitted, orange rings held back to test the fit; the dashed wave is the truth. The curve is the model, ten terms w₁x + … + w₁₀x¹⁰. At λ = 0.0001 nothing holds the weights back: the curve chases noise and misses the rings. Try it: Add a penalty.',
      takeaway: 'Overfitting: error 0.007 on the fitted dots but 0.026 on the held-out rings. The curve learnt the noise; only the held-out error shows it.',
      focus: LAMBDA,
      experiments: [
        {
          label: 'Add a penalty (λ = 0.1)',
          patch: { lambda: 0.1, count: 24 },
          say: 'Ridge at λ = 0.1: the curve settles onto the wave, the largest weight falls from 0.89 to 0.39 and the held-out error from 0.026 to 0.015. The next lesson explains the charge.',
          focus: LAMBDA,
        },
        {
          label: 'Ten times the data',
          patch: { lambda: 0.0001, count: 200 },
          say: '200 points and still no penalty: the curve stops chasing the noise on its own. Its error on the fitted dots, 0.037, is about what the wave itself scores, 0.040. Overfitting is a shortage of data as much as of weights.',
          focus: { kind: 'control', key: 'count', label: 'Points' },
        },
      ],
    },
  ],
};

const ridge: L = {
  id: 'ridge',
  title: 'Ridge regression (L2)',
  hook: 'Charge the squared weights.',
  section: 'core',
  presetId: 'ridge',
  steps: [
    {
      id: 'l2',
      title: 'λ = 0.1',
      kind: 'sandbox',
      enter: [{ type: 'preset', id: 'ridge' }, ...SOLVE],
      say: 'Ridge adds a charge to the error: λ × (w₁² + … + w₁₀²), so the biggest weights are cut hardest. At λ = 0.1 every bar is small, the curve follows the wave, and the held-out error is 0.015 against 0.026 without the charge. Try it: Ten times more.',
      takeaway: 'Ridge shrinks every weight but never to zero. Small λ changes little, large λ flattens the curve. Its closed form solves in one step.',
      focus: PENALTY,
      experiments: [
        {
          label: 'No penalty (λ = 0.0001)',
          patch: { lambda: 0.0001 },
          say: 'λ = 0.0001: the charge is gone, the largest weight grows from 0.39 to 0.89 and the curve wiggles again. Held-out error 0.026.',
          focus: LAMBDA,
        },
        {
          label: 'Ten times more (λ = 1)',
          patch: { lambda: 1 },
          say: 'λ = 1: the largest weight is 0.12 and the curve is too smooth to reach the wave’s peaks. Held-out error 0.052, and all ten weights are still non-zero.',
          focus: LAMBDA,
        },
        {
          label: 'Far too much (λ = 10)',
          patch: { lambda: 10 },
          say: 'λ = 10: the largest weight is 0.024 and the curve is almost the flat mean of y. Held-out error 0.104. Ten non-zero weights even now: ridge only shrinks.',
          focus: { kind: 'metric', key: 'nonzero', label: 'Non-zero weights' },
        },
      ],
    },
  ],
};

const lasso: L = {
  id: 'lasso',
  title: 'Lasso regression (L1)',
  hook: 'Charge the absolute weights.',
  section: 'core',
  presetId: 'lasso',
  steps: [
    {
      id: 'l1',
      title: 'λ = 0.03',
      kind: 'sandbox',
      enter: [{ type: 'preset', id: 'lasso' }, ...SOLVE],
      say: 'The lasso charges the absolute weights: error + λ × (|w₁| + … + |w₁₀|). At λ = 0.03 seven of the ten weights are exactly zero: only x, x² and x⁵ carry weight, and the curve still follows the wave, held-out error 0.019. Try it: Stronger.',
      takeaway: 'Under the L1 charge a weak term costs more than it saves, so its weight lands on exactly zero: the lasso selects features, ridge never does.',
      focus: LAMBDA,
      experiments: [
        {
          label: 'Stronger (λ = 0.1)',
          patch: { lambda: 0.1 },
          say: 'λ = 0.1: only x² and x⁷ survive, found in 14 sweeps. The curve is too simple now; held-out error 0.065.',
          focus: LAMBDA,
        },
        {
          label: 'One term left (λ = 0.2)',
          patch: { lambda: 0.2 },
          say: 'λ = 0.2: one term left, x², so the curve is a parabola. Held-out error 0.092.',
          focus: { kind: 'metric', key: 'nonzero', label: 'Non-zero weights' },
        },
        {
          label: 'Above λ max (λ = 0.5)',
          patch: { lambda: 0.5 },
          say: 'λ = 0.5 is above λ max = 0.29, the strongest pull any term has on this data. Every weight is zero and the curve is the flat mean of y. Held-out error 0.127.',
          focus: { kind: 'metric', key: 'nonzero', label: 'Non-zero weights' },
        },
      ],
    },
  ],
};

const geometry: L = {
  id: 'geometry',
  title: 'L1 and L2 geometry',
  hook: 'Corners make zeros, discs do not.',
  section: 'core',
  presetId: 'two-weights',
  steps: [
    {
      id: 'plane',
      title: 'Two weights',
      kind: 'sandbox',
      enter: [{ type: 'preset', id: 'two-weights' }, ...SOLVE],
      say: 'Every pair (w₁, w₂) is a point on the plane. Grey rings join pairs with equal error, smaller towards the ×. A penalty is a budget: the answer must stay inside the shape. The lasso’s diamond meets the smallest ring at a corner, so w₂ = 0. Try it: drag the ×.',
      takeaway: 'Corners sit on the axes, and a ring meets a corner over a whole range of × positions: that is why the lasso gives exact zeros and the cornerless disc never does.',
      focus: { kind: 'custom', id: 'pane:geometry', label: 'Drag the ×' },
      experiments: [
        {
          label: 'Ridge (the disc)',
          patch: { penalty: 'ridge', alpha: 0.5, lambda: 1 },
          say: 'Ridge with the same λ: the budget is w₁² + w₂², a disc. The ring touches it at w₁ = 1.40, w₂ = 0.48. Both shrank from the × at (2.81, 1.00), neither is zero. The dashed diamond shows where the lasso would land.',
          focus: { kind: 'control', key: 'penalty', label: 'Penalty' },
        },
        {
          label: 'Lasso (the diamond)',
          patch: { penalty: 'lasso', alpha: 0.5, lambda: 1 },
          say: 'The lasso with λ = 1: w₁ = 1.78, w₂ = 0. The ring meets the diamond at its corner on the w₁ axis, and the hollow dot shows ridge off the axis at the same budget.',
          focus: { kind: 'control', key: 'penalty', label: 'Penalty' },
        },
        {
          label: 'Elastic net (α = 0.5)',
          patch: { penalty: 'elastic', alpha: 0.5, lambda: 2 },
          say: 'Elastic net, half L1 and half L2: a rounded diamond between the two dashed shapes. The corners are blunter but still there: w₁ = 0.89, w₂ = 0. α sets how sharp they are, 1 the diamond, 0 the disc.',
          focus: { kind: 'control', key: 'alpha', label: 'Mix (α)' },
        },
      ],
    },
  ],
};

const choosing: L = {
  id: 'choosing-lambda',
  title: 'Choosing λ',
  hook: 'Too little overfits, too much underfits.',
  section: 'failure',
  presetId: 'ridge',
  steps: [
    {
      id: 'curve',
      title: 'Error against λ',
      kind: 'sandbox',
      enter: [{ type: 'preset', id: 'ridge' }, ...SOLVE],
      say: 'The chart tries every λ. Blue: error on the fitted dots, rising as the charge pulls the curve off them. Orange: error on the held-out rings, falling then rising. The cross is the best: λ = 0.078, error 0.015. Left overfits, right underfits. Try it: The best λ.',
      takeaway: 'Pick λ by the held-out error, never the fitted one: sweep λ on a log grid, score each on held-out points, keep the best. That is cross-validation.',
      focus: { kind: 'panel', id: 'second', label: 'Error against λ' },
      experiments: [
        {
          label: 'The best λ (0.078)',
          patch: { lambda: 0.078, noise: 0.3 },
          say: 'λ = 0.078, the cross: held-out error 0.015, the lowest ridge reaches on this data.',
          focus: LAMBDA,
        },
        {
          label: 'Noisier data (0.6)',
          patch: { lambda: 0.1, noise: 0.6 },
          say: 'Noise doubled to 0.6 and the points redrawn: the cross moves right, to λ = 0.37. Noisier dots need a stronger charge to keep the curve from chasing them.',
          focus: { kind: 'control', key: 'noise', label: 'Noise' },
        },
        {
          label: 'Cleaner data (0.1)',
          patch: { lambda: 0.1, noise: 0.1 },
          say: 'Noise 0.1: the cross moves to the left end, the smallest λ the chart tries. With little noise there is little to overfit, and the charge only gets in the way.',
          focus: { kind: 'control', key: 'noise', label: 'Noise' },
        },
      ],
    },
  ],
};

/** The overfit preset's knobs, restated so the experiments do not stack. */
const BASE: Partial<RegParams> = { features: 10, count: 24, penalty: 'ridge', lambda: 0.0001, alpha: 0.5, standardise: true };

const experiments: L = {
  id: 'experiments',
  title: 'Experiments',
  hook: 'Four experiments with known outcomes.',
  section: 'sandbox',
  presetId: 'overfit',
  steps: [
    {
      id: 'sandbox',
      kind: 'sandbox',
      enter: [{ type: 'preset', id: 'overfit' }, ...SOLVE],
      say: 'Pick an experiment: the settings change, the solver replays, and the outcome opens under it. Every knob stays live, so change anything afterwards.',
      experiments: [
        {
          label: 'Three powers of x',
          patch: { ...BASE, features: 3 },
          say: 'Three powers of x and no penalty: a cubic cannot bend enough to follow the wave, so it underfits. Held-out error 0.027, no better than ten powers overfitting. The terms set what the curve can do.',
          focus: { kind: 'control', key: 'features', label: 'Powers of x' },
        },
        {
          label: 'More weights than points',
          patch: { ...BASE, features: 12, count: 12 },
          say: 'Twelve powers of x and nine fitted dots, almost no penalty: error 0.002 on the dots, 0.034 on the rings. With more weights than points many curves fit exactly, and a little λ picks the smallest one.',
          focus: { kind: 'control', key: 'features', label: 'Powers of x' },
        },
        {
          label: 'Elastic net on the wave',
          patch: { ...BASE, penalty: 'elastic', alpha: 0.5, lambda: 0.1 },
          say: 'Half L1, half L2 at λ = 0.1: five terms kept (x, x², x⁵, x⁷, x⁹) where the lasso at the same λ keeps two, since only half the charge makes zeros. Held-out error 0.038.',
          focus: { kind: 'control', key: 'penalty', label: 'Penalty' },
        },
        {
          label: 'Standardise off',
          patch: { ...BASE, lambda: 0.1, standardise: false },
          say: 'Ridge at λ = 0.1 on raw powers: x stays below 3 while x¹⁰ runs into the thousands, so x¹⁰ needs a weight near 0.000001 and the charge barely notices it. Held-out error 0.019 against 0.015 standardised.',
          focus: { kind: 'control', key: 'standardise', label: 'Standardise' },
        },
      ],
    },
  ],
};

export const LESSONS: L[] = [overfitting, ridge, lasso, geometry, choosing, experiments];
