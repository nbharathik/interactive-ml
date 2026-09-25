/** Linear regression lessons, one screen each. Pure data, driven against the real model in tests/lessons-linear.test.ts. */

import type { Lesson, LessonAction, LessonContextBase, LiveNumber } from '../../explainer/lessons';
import type { RegressionData } from '../../lib/datasets/types';
import { fmtCompact, fmtKnob, fmtPercent, meanAbsoluteError, rSquared } from '../../lib/math/stats';
import { predict, toOriginalUnits } from '../../lib/ml/linearRegression';
import type { LinRegConfig, LinRegState, Normaliser } from '../../lib/ml/linearRegression';
import type { LinRegParams } from './config';
import { LR_EDGES } from './graph';

/* ---------------- context ---------------- */

export interface LinRegLessonContext extends LessonContextBase {
  params: LinRegParams;
  state: LinRegState;
  derived: {
    n: number;
    initialLoss: number;
    closedFormLoss: number | null;
    gapToOptimum: number | null;
    r2: number;
    mae: number;
    line: { slope: number; intercept: number };
    /** Share of the current loss carried by the k largest squared errors. */
    lossShareOfTop: (k: number) => number;
    /** Largest |w| among the non-bias weights. */
    largestWeight: number;
    /** Points further than 10 from the relationship the data was drawn from. */
    outlierCount: number;
    truthSlope: number;
    /** Mean squared error the true relationship itself scores on these points. */
    truthLoss: number;
    /** Furthest the fitted curve strays from the true relationship over the x range. */
    truthGap: number;
    /** Root mean square gap between the curve and the true relationship at the points. */
    truthRms: number;
  };
  ui: {
    view: 'model' | 'training';
    lossView: 'curve' | 'surface';
    logScale: boolean;
    open: string | null;
    pointEdits: number;
  };
}

export interface LinRegLessonInput {
  params: LinRegParams;
  state: LinRegState;
  data: RegressionData;
  config: LinRegConfig;
  norm: Normaliser;
  closedFormLoss: number | null;
  initialLoss: number;
  ui: LinRegLessonContext['ui'];
}

const OUTLIER_DISTANCE = 10;

/** The page's share of the context; the lesson hook adds `sim` and `lesson`. */
export function makeLinRegContext(input: LinRegLessonInput): Omit<LinRegLessonContext, keyof LessonContextBase> {
  const { params, state, data, config, norm, closedFormLoss, initialLoss, ui } = input;
  const points = data.points;
  const predictions = points.map((p) => predict(state.weights, p.x, config.degree, norm));
  const actuals = points.map((p) => p.y);
  const squared = predictions.map((p, i) => (p - actuals[i]) ** 2).sort((a, b) => b - a);
  const total = squared.reduce((a, b) => a + b, 0);
  const gapToOptimum =
    closedFormLoss !== null && Number.isFinite(state.loss) && state.epoch > 0 ? state.loss - closedFormLoss : null;

  const [x0, x1] = data.xRange;
  const truth = data.truth ?? (() => 0);
  let truthGap = 0;
  for (let i = 0; i <= 100; i++) {
    const x = x0 + ((x1 - x0) * i) / 100;
    const gap = Math.abs(predict(state.weights, x, config.degree, norm) - truth(x));
    if (gap > truthGap) truthGap = gap;
  }

  const truthRms = Math.sqrt(points.reduce((a, p) => a + (predict(state.weights, p.x, config.degree, norm) - truth(p.x)) ** 2, 0) / Math.max(1, points.length));

  return {
    params,
    state,
    derived: {
      n: points.length,
      initialLoss,
      closedFormLoss,
      gapToOptimum,
      r2: rSquared(actuals, predictions),
      mae: meanAbsoluteError(actuals, predictions),
      line: toOriginalUnits(state.weights, norm),
      lossShareOfTop: (k) => (total > 0 ? squared.slice(0, k).reduce((a, b) => a + b, 0) / total : 0),
      largestWeight: state.weights.slice(1).reduce((best, w) => Math.max(best, Math.abs(w)), 0),
      outlierCount: points.filter((p) => Math.abs(p.y - truth(p.x)) > OUTLIER_DISTANCE).length,
      truthSlope: (truth(x1) - truth(x0)) / (x1 - x0),
      truthLoss: points.reduce((a, p) => a + (truth(p.x) - p.y) ** 2, 0) / Math.max(1, points.length),
      truthGap,
      truthRms,
    },
    ui,
  };
}

/* ---------------- copy helpers ---------------- */

type Ctx = LinRegLessonContext;
type L = Lesson<Ctx, LinRegParams>;

const GOOD_RATE = 0.03;

const steps = (n: number) => n.toLocaleString('en-US');

/** Loss values: thousands with a separator, huge ones in scientific form. */
function num(v: number): string {
  if (!Number.isFinite(v)) return '∞';
  if (Math.abs(v) >= 1e9) return v.toExponential(1);
  if (Math.abs(v) >= 1000) return steps(Math.round(v));
  return fmtCompact(v);
}

const stopped = (c: Ctx) => c.state.converged || c.state.diverged;

const exactClause = (c: Ctx) =>
  c.derived.gapToOptimum !== null && c.derived.gapToOptimum < 0.01 ? ', the exact answer' : '';

/** How much of the drop from the starting loss to the exact one is done. */
function progress(c: Ctx): number {
  const { initialLoss, closedFormLoss } = c.derived;
  if (closedFormLoss === null || !Number.isFinite(c.state.loss)) return 0;
  const drop = initialLoss - closedFormLoss;
  return drop > 0 ? Math.max(0, Math.min(1, (initialLoss - c.state.loss) / drop)) : 0;
}

const rateRatio = (c: Ctx) => Math.round(Math.max(GOOD_RATE / c.params.learningRate, c.params.learningRate / GOOD_RATE));

const outlierShare = (c: Ctx) => c.derived.lossShareOfTop(Math.max(1, c.derived.outlierCount));

const lossRatio = (c: Ctx) =>
  c.derived.closedFormLoss !== null && c.derived.closedFormLoss > 0 ? c.state.loss / c.derived.closedFormLoss : Infinity;


/* ---------------- the lessons ---------------- */

type Act = LessonAction<LinRegParams>;

const VIEW_MODEL: Act = { type: 'view', id: 'graph', value: 'auto' };
const VIEW_TRAINING: Act = { type: 'view', id: 'graph', value: 'training' };
const CURVE: Act = { type: 'view', id: 'second', value: 'curve' };
const SURFACE: Act = { type: 'view', id: 'second', value: 'surface' };
const LINEAR: Act = { type: 'view', id: 'scale', value: 'linear' };
const LOG: Act = { type: 'view', id: 'scale', value: 'log' };
const PLAY: Act = { type: 'play' };
/** Opens the lesson on its finished picture. */
const SOLVE: Act = { type: 'runTo', steps: 5000 };

/** A run restates every knob it depends on: the preset, then its own changes, then a fresh start. */
const on = (preset: string, patch: Partial<LinRegParams> | null, ...rest: Act[]): Act[] => [
  { type: 'preset', id: preset },
  ...(patch ? [{ type: 'params' as const, patch }] : []),
  { type: 'reset' },
  ...rest,
];

/** α as the knob shows it: 0.0005, not 5.0e-4. */
const alpha = (v: number) => (v >= 0.001 ? fmtKnob(v) : v.toFixed(4));
const two = (v: number) => v.toFixed(2);

const stepLoss = (c: Ctx): LiveNumber[] => [
  { label: 'step', value: steps(c.state.epoch) },
  { label: 'loss', value: num(c.state.loss) },
];
const stepLossSlope = (c: Ctx): LiveNumber[] => [...stepLoss(c), { label: 'slope', value: fmtCompact(c.derived.line.slope) }];

const converged = (c: Ctx) =>
  c.state.converged
    ? 'Converged at step ' + steps(c.state.epoch) + ' with loss ' + num(c.state.loss) + exactClause(c)
    : 'Diverged at step ' + steps(c.state.epoch) + ': the loss overflowed';

const fittingALine: L = {
  id: 'fitting-a-line',
  title: 'Fitting a line',
  hook: 'Sixty house prices, one line, one loop.',
  section: 'hook',
  presetId: 'goldilocks',
  steps: [
    {
      id: 'fit',
      kind: 'sandbox',
      say: 'Sixty houses: floor area across, price up. The orange line is the model, w·x + b, already fitted; the red sticks are its errors. The loss is the mean of the squared errors: 46.56 here, down from 4,983 at the flat start. Try it: Watch it train.',
      takeaway: 'Predict, measure, update, repeat. Every model on this site learns with this same loop.',
      focus: { kind: 'panel', id: 'fit', label: 'The fit' },
      enter: [{ type: 'preset', id: 'goldilocks' }, { type: 'reset' }, VIEW_MODEL, CURVE, LINEAR, SOLVE],
      numbers: stepLossSlope,
      experiments: [
        {
          label: 'Watch it train',
          say: 'Replays the fit from the flat line.',
          enter: on('goldilocks', null, VIEW_TRAINING, CURVE, LINEAR, PLAY),
          speed: 'normal',
          until: stopped,
          then: (c) => converged(c) + '. Big moves while the loss was high, tiny ones as the line settled into place.',
          focus: { kind: 'metric', key: 'loss', label: 'Loss' },
        },
        {
          label: 'One step',
          say: 'One update from the flat line.',
          enter: on('goldilocks', null, VIEW_TRAINING, CURVE, LINEAR, { type: 'step', count: 1 }),
          until: (c) => c.state.epoch >= 1,
          then: (c) =>
            'Step 1: loss ' + num(c.derived.initialLoss) + ' to ' + num(c.state.loss) +
            '. Predict, measure the loss, nudge w and b downhill by the gradient times α = ' + alpha(c.params.learningRate) + '. Repeat.',
          focus: { kind: 'edge', id: LR_EDGES.gradient, label: 'The gradient' },
        },
      ],
    },
  ],
};

const gradientDescent: L = {
  id: 'gradient-descent',
  title: 'Gradient descent',
  hook: 'The gradient, α and the loss landscape.',
  section: 'core',
  presetId: 'goldilocks',
  knobs: ['learningRate', 'batchMode'],
  steps: [
    {
      id: 'descent',
      kind: 'sandbox',
      say: 'Training view: the dashed arc is the gradient, flowing from the loss L back to the weights. The landscape on the right is the loss at every (b, w); the trail is the path so far, 20 steps in. Each step moves a weight by its gradient times α. Try it: One step.',
      takeaway: 'The loss surface is steepest far from the minimum, so the first steps make the most progress.',
      focus: { kind: 'edge', id: LR_EDGES.gradient, label: 'The gradient' },
      enter: [{ type: 'preset', id: 'goldilocks' }, { type: 'reset' }, VIEW_TRAINING, SURFACE, { type: 'runTo', steps: 20 }],
      numbers: (c) => [...stepLoss(c), { label: 'of the drop', value: fmtPercent(progress(c), 0) }],
      experiments: [
        {
          label: 'One step',
          say: 'One update from the flat line.',
          enter: on('goldilocks', null, VIEW_TRAINING, SURFACE, { type: 'step', count: 1 }),
          until: (c) => c.state.epoch >= 1,
          then: (c) =>
            'Step 1: loss ' + num(c.derived.initialLoss) + ' to ' + num(c.state.loss) + '. The gradient said which way is downhill; α = ' +
            alpha(c.params.learningRate) + ' said how far.',
          focus: { kind: 'edge', id: LR_EDGES.gradient, label: 'The gradient' },
        },
        {
          label: 'Twenty steps',
          say: 'Twenty updates from the flat line.',
          enter: on('goldilocks', null, VIEW_TRAINING, SURFACE, { type: 'step', count: 20 }),
          until: (c) => c.state.epoch >= 20,
          then: (c) =>
            'Step 20: loss ' + num(c.state.loss) + ', ' + fmtPercent(progress(c), 0) +
            ' of the way from the start to the exact answer. Long strides down the wall, then small ones along the valley floor.',
          focus: { kind: 'panel', id: 'second', label: 'The loss landscape' },
        },
        {
          label: 'Run to the bottom',
          say: 'The whole descent.',
          enter: on('goldilocks', null, VIEW_TRAINING, SURFACE, PLAY),
          speed: 'fast',
          until: stopped,
          then: (c) => converged(c) + '. Most of the drop came in the first twenty steps; the rest is the slow walk along the floor.',
          focus: { kind: 'panel', id: 'second', label: 'The loss landscape' },
        },
        {
          label: 'One point per step',
          say: 'Stochastic gradient descent: each update reads one point instead of all sixty.',
          enter: on('stochastic', null, VIEW_TRAINING, CURVE, LINEAR, PLAY),
          speed: 'normal',
          until: (c) => c.state.diverged || c.state.epoch >= 300,
          then: (c) =>
            'One point per step, ' + steps(c.state.epoch) + ' steps: loss ' + num(c.state.loss) + ', ' + fmtPercent(lossRatio(c) - 1, 0) + ' above the exact ' +
            num(c.derived.closedFormLoss ?? NaN) + '. Jagged and never quite settled: each point pulls the line its own way, at 1/' + steps(c.derived.n) + ' of the cost.',
          focus: { kind: 'panel', id: 'second', label: 'Loss over time' },
        },
      ],
    },
  ],
};

const learningRate: L = {
  id: 'learning-rate',
  title: 'Learning rate α',
  hook: 'High α diverges, low α crawls.',
  section: 'core',
  presetId: 'goldilocks',
  knobs: ['learningRate'],
  steps: [
    {
      id: 'alpha',
      kind: 'sandbox',
      say: 'α scales every step. At α = 0.03 the line converged in 267 steps: the loss curve drops fast, then flattens along the floor. Too large and each step overshoots the valley; too small and it crawls. Try it: α = 2.2, then α = 0.0005.',
      takeaway: 'Every step is scaled by α. Sixty times smaller, sixty times as many steps; too large, and the steps jump over the valley.',
      focus: { kind: 'control', key: 'learningRate', label: 'Learning rate (α)' },
      enter: [{ type: 'preset', id: 'goldilocks' }, { type: 'reset' }, VIEW_MODEL, CURVE, LOG, SOLVE],
      numbers: stepLoss,
      experiments: [
        {
          label: 'α = 2.2',
          say: 'Seventy times the rate that worked, on the landscape.',
          enter: on('divergence', null, SURFACE, PLAY),
          speed: 'normal',
          until: stopped,
          then: (c) =>
            'α = ' + alpha(c.params.learningRate) + ': step 1 took the loss from ' + num(c.derived.initialLoss) + ' to ' + num(c.state.lossHistory[0] ?? Infinity) +
            '; step ' + steps(c.state.epoch) + ' overflowed and the run diverged. Each step crossed the valley and landed higher up the far side.',
          focus: { kind: 'panel', id: 'second', label: 'The loss landscape' },
        },
        {
          label: 'α = 0.0005',
          say: 'Sixty times smaller than the rate that worked, at turbo speed.',
          enter: on('too-small', null, CURVE, LOG, PLAY),
          speed: 'turbo',
          until: (c) => stopped(c) || c.state.epoch >= 20_000,
          then: (c) =>
            'α = ' + alpha(c.params.learningRate) + ', ' + steps(rateRatio(c)) + ' times smaller: ' + converged(c).toLowerCase() +
            '. One quick drop, then thousands of tiny steps along the flat floor.',
          focus: { kind: 'panel', id: 'second', label: 'Loss over time' },
        },
        {
          label: 'α = 0.03',
          say: 'The rate that works, again.',
          enter: on('goldilocks', null, CURVE, LOG, PLAY),
          speed: 'normal',
          until: stopped,
          then: (c) => 'α = ' + alpha(c.params.learningRate) + ': ' + converged(c).toLowerCase() + '.',
          focus: { kind: 'panel', id: 'second', label: 'Loss over time' },
        },
      ],
    },
  ],
};

const outliers: L = {
  id: 'outliers',
  title: 'Outliers',
  hook: 'Three outliers and the loss they carry.',
  section: 'core',
  presetId: 'outliers',
  knobs: ['dataset'],
  steps: [
    {
      id: 'squares',
      kind: 'sandbox',
      say: 'Three points sit far from the rest. Each red square is one error squared, and the loss is the mean of their areas, so the three outliers carry 96% of it. They tilt the line towards them: slope 1.49 against the true 1.60, the dashed line. Try it: Without them.',
      takeaway: 'Squaring amplifies large errors: an error of 20 costs 400, an error of 2 costs 4. A few far points can steer the whole line.',
      focus: { kind: 'panel', id: 'fit', label: 'The fit' },
      enter: [{ type: 'preset', id: 'outliers' }, { type: 'reset' }, VIEW_MODEL, CURVE, LINEAR, SOLVE],
      numbers: (c) => [...stepLossSlope(c), { label: 'from outliers', value: fmtPercent(outlierShare(c), 0) }],
      experiments: [
        {
          label: 'Without them',
          say: 'The same straight-line truth, drawn without the bad readings.',
          enter: on('outliers', { dataset: 'linear' }, CURVE, LINEAR, PLAY),
          speed: 'fast',
          until: (c) => stopped(c) || c.state.epoch >= 2000,
          then: (c) =>
            'Without the bad readings: slope ' + fmtCompact(c.derived.line.slope) + ' against the true ' + fmtCompact(c.derived.truthSlope) + ', loss ' +
            num(c.state.loss) + '. Every point pulls on the line; three far ones can steer it.',
          focus: { kind: 'panel', id: 'fit', label: 'The fit' },
        },
        {
          label: 'With them again',
          say: 'The three bad readings back in.',
          enter: on('outliers', null, CURVE, LINEAR, PLAY),
          speed: 'fast',
          until: (c) => stopped(c) || c.state.epoch >= 2000,
          then: (c) =>
            'With the ' + steps(c.derived.outlierCount) + ' outliers: slope ' + fmtCompact(c.derived.line.slope) + ', loss ' + num(c.state.loss) + ', ' +
            fmtPercent(outlierShare(c), 0) + ' of it from three points.',
          focus: { kind: 'metric', key: 'mae', label: 'Mean abs. error' },
        },
      ],
    },
  ],
};

const featureScaling: L = {
  id: 'feature-scaling',
  title: 'Feature scaling',
  hook: 'Raw x⁵ overflows; standardise first.',
  section: 'failure',
  presetId: 'unscaled',
  knobs: ['standardise', 'learningRate', 'degree'],
  steps: [
    {
      id: 'scale',
      kind: 'sandbox',
      say: 'Degree 5 on raw x, which runs 0 to 10: x⁵ reaches 100,000 while x stays under 10, so the gradient on w₅ is 100,000 times larger than the others. Step 1 took the loss from 96.68 to 3.5e+13; step 3 overflowed. Try it: Standardise x.',
      takeaway: 'Standardising puts every input on one scale, so one α suits every weight. It is the cheapest fix in machine learning.',
      focus: { kind: 'control', key: 'standardise', label: 'Standardise x' },
      enter: [{ type: 'preset', id: 'unscaled' }, { type: 'reset' }, VIEW_MODEL, CURVE, LINEAR, SOLVE],
      numbers: stepLoss,
      experiments: [
        {
          label: 'Standardise x',
          say: 'Every input with mean 0 and spread 1, and the ordinary α = 0.03.',
          enter: on('unscaled', { standardise: true, learningRate: 0.03 }, CURVE, LINEAR, PLAY),
          speed: 'fast',
          until: (c) => stopped(c) || c.state.epoch >= 300,
          then: (c) =>
            'Standardised, α = ' + alpha(c.params.learningRate) + ': one α suits all ' + steps(c.params.degree + 1) + ' weights. Loss ' + num(c.state.loss) +
            ' at step ' + steps(c.state.epoch) + (c.state.converged ? ', converged.' : ' and still falling.'),
          focus: { kind: 'control', key: 'standardise', label: 'Standardise x' },
        },
        {
          label: 'Raw x again',
          say: 'The same run on raw powers of x.',
          enter: on('unscaled', null, CURVE, LINEAR, PLAY),
          speed: 'normal',
          until: stopped,
          then: (c) =>
            'Raw x⁵: step 1 took the loss from ' + num(c.derived.initialLoss) + ' to ' + num(c.state.lossHistory[0] ?? Infinity) + '; ' +
            converged(c).toLowerCase() + '.',
          focus: { kind: 'control', key: 'standardise', label: 'Standardise x' },
        },
      ],
    },
  ],
};

const overfitting: L = {
  id: 'overfitting',
  title: 'Overfitting',
  hook: 'Nine weights for twenty noisy points.',
  section: 'failure',
  presetId: 'overfit',
  knobs: ['degree', 'sampleCount', 'l2'],
  steps: [
    {
      id: 'degree-8',
      kind: 'sandbox',
      say: 'Degree 8: nine weights for twenty noisy points, enough to bend through every dot. The training loss, 3.54, beats what the true line itself scores, 5.40, so the curve has fitted the noise: on average it sits 1.38 from the dashed truth. Try it: Degree 1.',
      takeaway: 'Lower training loss is not a better model. Judge a fit on points it has not seen.',
      focus: { kind: 'control', key: 'degree', label: 'Polynomial degree' },
      enter: [{ type: 'preset', id: 'overfit' }, { type: 'reset' }, VIEW_MODEL, CURVE, LINEAR, { type: 'runTo', steps: 2000 }],
      numbers: (c) => [...stepLoss(c), { label: 'from truth', value: two(c.derived.truthRms) }],
      experiments: [
        {
          label: 'Degree 1',
          say: 'A straight line through the same twenty points.',
          enter: on('overfit', { degree: 1 }, CURVE, LINEAR, PLAY),
          speed: 'turbo',
          until: (c) => stopped(c) || c.state.epoch >= 2000,
          then: (c) =>
            'Degree 1: loss ' + num(c.state.loss) + ', worse on the training points, yet only ' + two(c.derived.truthRms) +
            ' from the truth. A straight line cannot follow the noise.',
          focus: { kind: 'control', key: 'degree', label: 'Polynomial degree' },
        },
        {
          label: '200 points',
          say: 'Degree 8 again, with ten times the data.',
          enter: on('overfit', { sampleCount: 200 }, CURVE, LINEAR, PLAY),
          speed: 'turbo',
          until: (c) => stopped(c) || c.state.epoch >= 2000,
          then: (c) =>
            'Degree 8 with ' + steps(c.derived.n) + ' points: loss ' + num(c.state.loss) + ', about what the truth itself scores, and the curve is ' +
            two(c.derived.truthRms) + ' from it. More data leaves less room to fit the noise.',
          focus: { kind: 'control', key: 'sampleCount', label: 'Points' },
        },
        {
          label: 'L2 penalty λ = 0.05',
          say: 'Degree 8 with a charge on large weights.',
          enter: on('regularised', null, CURVE, LINEAR, PLAY),
          speed: 'turbo',
          until: (c) => stopped(c) || c.state.epoch >= 2000,
          then: (c) =>
            'λ = ' + fmtKnob(c.params.l2) + ': the largest weight is ' + fmtCompact(c.derived.largestWeight) + ' instead of 1.84 and the curve stops swinging past the last point; loss ' +
            num(c.state.loss) + ', a little higher. Ridge and lasso is the chapter on this.',
          focus: { kind: 'control', key: 'l2', label: 'L2 penalty (λ)' },
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
  presetId: 'goldilocks',
  steps: [
    {
      id: 'sandbox',
      kind: 'sandbox',
      say: 'Pick an experiment: the settings change, the run replays, and the outcome opens under it.',
      enter: [{ type: 'preset', id: 'goldilocks' }, { type: 'reset' }, VIEW_MODEL, CURVE, LINEAR, { type: 'speed', speed: 'turbo' }],
      experiments: [
        {
          label: 'α = 0.5, one step',
          say: 'With x standardised the loss surface is round, and α = 0.5 lands on the exact answer in one step.',
          patch: { learningRate: 0.5 },
          focus: { kind: 'control', key: 'learningRate', label: 'Learning rate (α)' },
        },
        {
          label: 'α = 1, oscillation',
          say: 'α = 1 jumps between the same two points forever, never diverging.',
          patch: { learningRate: 1 },
          focus: { kind: 'control', key: 'learningRate', label: 'Learning rate (α)' },
        },
        {
          label: 'Momentum at low α',
          say: 'The slow run, with momentum: it converges about eleven times sooner.',
          patch: { learningRate: 0.0005, optimiser: 'momentum' },
          focus: { kind: 'control', key: 'optimiser', label: 'Update rule' },
        },
        {
          label: 'Adam at high α',
          say: 'The rate that diverged, tamed: Adam scales each step by recent gradients and converges near step 300.',
          patch: { learningRate: 2.2, optimiser: 'adam' },
          focus: { kind: 'control', key: 'optimiser', label: 'Update rule' },
        },
        {
          label: 'Polynomial degree',
          say: 'On the curved data, degree 1 stops at loss 17.3, degree 2 at 1.85, degree 3 adds nothing. Watch where the loss stops falling.',
          patch: { dataset: 'curved', degree: 1 },
          focus: { kind: 'control', key: 'degree', label: 'Polynomial degree' },
        },
        {
          label: 'Ten points',
          say: 'Shuffle the seed: with ten points the slope swings between about 6 and 12; with sixty it stays near 9 to 10.',
          patch: { sampleCount: 10 },
          focus: { kind: 'control', key: 'sampleCount', label: 'Points' },
        },
      ],
    },
  ],
};

export const LESSONS: L[] = [fittingALine, gradientDescent, learningRate, outliers, featureScaling, overfitting, experiments];
