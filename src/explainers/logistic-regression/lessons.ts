/** Logistic regression lessons, one screen each. Pure data, driven against the real model in tests/lessons-logistic.test.ts. */

import type { Lesson, LessonAction, LessonContextBase, LiveNumber } from '../../explainer/lessons';
import type { PointData } from '../../lib/datasets/types';
import { accuracy as accuracyOf, fmt, fmtPercent, logLoss, precisionRecallF1, rocAuc } from '../../lib/math/stats';
import { computeLoss, createState, predictAll } from '../../lib/ml/logisticRegression';
import type { LogRegConfig, LogRegState, Standardiser } from '../../lib/ml/logisticRegression';
import type { LogRegParams } from './config';
import { LOG_NODES } from './graph';

/* ---------------- context ---------------- */

export type SecondView = 'sigmoid' | 'curves' | 'confusion' | 'roc';

export interface LogRegLessonContext extends LessonContextBase {
  params: LogRegParams;
  state: LogRegState;
  derived: {
    n: number;
    positives: number;
    accuracy: number;
    logLoss: number;
    /** What always calling the bigger class scores. */
    baseline: number;
    precision: number;
    recall: number;
    f1: number;
    tp: number;
    fp: number;
    fn: number;
    tn: number;
    auc: number;
    /** Loss at the starting weights, so the first step has a before. */
    initialLoss: number;
    /** Largest |w| among the non-bias weights. */
    largestWeight: number;
    featureCount: number;
    /** Share of points whose probability sits within 0.1 of one half. */
    unsureShare: number;
  };
  ui: {
    view: 'model' | 'training';
    secondView: SecondView;
    open: string | null;
    /** Points added or removed since the step was entered. */
    pointEdits: number;
  };
}

export interface LogRegLessonInput {
  params: LogRegParams;
  state: LogRegState;
  data: PointData;
  config: LogRegConfig;
  std: Standardiser;
  ui: LogRegLessonContext['ui'];
}

/** The page's share of the context; the lesson hook adds `sim` and `lesson`. */
export function makeLogRegContext(input: LogRegLessonInput): Omit<LogRegLessonContext, keyof LessonContextBase> {
  const { params, state, data, config, std, ui } = input;
  const points = data.points;
  const prediction = predictAll(state.weights, points, config, std);
  const actual = points.map((p) => p.label);
  const positives = actual.filter((v) => v === 1).length;
  const prf = precisionRecallF1(actual, prediction.labels, 1);
  const unsure = prediction.probabilities.filter((p) => Math.abs(p - 0.5) < 0.1).length;
  return {
    params,
    state,
    derived: {
      n: points.length,
      positives,
      accuracy: accuracyOf(actual, prediction.labels),
      logLoss: logLoss(actual, prediction.probabilities),
      baseline: points.length === 0 ? 0 : Math.max(positives, points.length - positives) / points.length,
      precision: prf.precision,
      recall: prf.recall,
      f1: prf.f1,
      tp: prf.tp,
      fp: prf.fp,
      fn: prf.fn,
      tn: prf.tn,
      auc: rocAuc(actual, prediction.probabilities),
      initialLoss: computeLoss(createState(config).weights, points, config, std),
      largestWeight: state.weights.slice(1).reduce((best, w) => Math.max(best, Math.abs(w)), 0),
      featureCount: state.weights.length,
      unsureShare: points.length === 0 ? 0 : unsure / points.length,
    },
    ui,
  };
}

/* ---------------- copy helpers ---------------- */

type Ctx = LogRegLessonContext;
type L = Lesson<Ctx, LogRegParams>;

const steps = (n: number) => n.toLocaleString('en-US');
const pc = (v: number) => fmtPercent(v, 0);
const loss = (v: number) => (Number.isFinite(v) ? fmt(v, 3) : 'infinity');
const stopped = (c: Ctx) => c.state.converged || c.state.diverged;

/* ---------------- the lessons ---------------- */

type Act = LessonAction<LogRegParams>;

const VIEW_MODEL: Act = { type: 'view', id: 'graph', value: 'auto' };
const SIGMOID: Act = { type: 'view', id: 'second', value: 'sigmoid' };
const CURVES: Act = { type: 'view', id: 'second', value: 'curves' };
const CONFUSION: Act = { type: 'view', id: 'second', value: 'confusion' };
const PLAY: Act = { type: 'play' };
/** Opens the lesson on its trained picture. */
const SOLVE: Act = { type: 'runTo', steps: 2000 };
const NOW = () => true;

/** A run restates every knob it depends on: the preset, then its own changes, then a fresh start. */
const on = (preset: string, patch: Partial<LogRegParams> | null, ...rest: Act[]): Act[] => [
  { type: 'preset', id: preset },
  ...(patch ? [{ type: 'params' as const, patch }] : []),
  { type: 'reset' },
  ...rest,
];

const stepLossAcc = (c: Ctx): LiveNumber[] => [
  { label: 'step', value: steps(c.state.epoch) },
  { label: 'log loss', value: loss(c.derived.logLoss) },
  { label: 'accuracy', value: pc(c.derived.accuracy) },
];

const ended = (c: Ctx) =>
  c.state.diverged
    ? 'Diverged at step ' + steps(c.state.epoch)
    : (c.state.converged ? 'Converged at step ' : 'Step ') + steps(c.state.epoch) + ': log loss ' + loss(c.derived.logLoss) + ', accuracy ' + pc(c.derived.accuracy);

const decisionBoundary: L = {
  id: 'sigmoid',
  title: 'Sigmoid and decision boundary',
  hook: 'A score, a sigmoid, a boundary at 0.5.',
  section: 'hook',
  presetId: 'clean-blobs',
  steps: [
    {
      id: 'boundary',
      kind: 'sandbox',
      say: '120 points in two classes. The unit scores each one, z = w·x̃ + b; the sigmoid squashes z into a probability p between 0 and 1, and the boundary is the line where p = 0.5. Trained, it sits in the gap: accuracy 97%, log loss 0.052. Try it: Watch it train.',
      takeaway: 'Up to z this is linear regression. The sigmoid and the cross-entropy loss are the only changes; the gradient is (p − y)·x̃, the same shape as least squares.',
      focus: { kind: 'node', id: LOG_NODES.sigmoid, label: 'The sigmoid σ' },
      enter: [{ type: 'preset', id: 'clean-blobs' }, { type: 'reset' }, VIEW_MODEL, SIGMOID, SOLVE],
      numbers: stepLossAcc,
      experiments: [
        {
          label: 'Watch it train',
          say: 'The boundary swings into the gap.',
          enter: on('clean-blobs', null, VIEW_MODEL, CURVES, PLAY),
          speed: 'normal',
          until: (c) => stopped(c) || c.derived.logLoss < 0.1 || c.state.epoch >= 400,
          then: (c) => ended(c) + '. Large updates while many points were misclassified, small ones once few were.',
          focus: { kind: 'panel', id: 'field', label: 'Probability across the plane' },
        },
        {
          label: 'The sigmoid chart',
          say: 'Every point at its own score z.',
          enter: [SIGMOID],
          until: NOW,
          then: (c) =>
            pc(c.derived.unsureShare) + ' of the points sit within 0.1 of p = 0.5: the uncertain ones. Cross-entropy charges a confident mistake far more than an uncertain one; a coin flip scores 0.693.',
          focus: { kind: 'panel', id: 'second', label: 'The sigmoid' },
        },
        {
          label: 'Loss and accuracy',
          say: 'Both curves over the run.',
          enter: [CURVES],
          until: NOW,
          then: (c) => 'Log loss fell from ' + loss(c.derived.initialLoss) + ' at the start to ' + loss(c.derived.logLoss) + '; accuracy reached ' + pc(c.derived.accuracy) + '.',
          focus: { kind: 'panel', id: 'second', label: 'Loss and accuracy' },
        },
      ],
    },
  ],
};

const featureEngineering: L = {
  id: 'features',
  title: 'Feature engineering',
  hook: 'Quadratic features bend the boundary.',
  section: 'core',
  presetId: 'ring-linear',
  knobs: ['featureMap'],
  steps: [
    {
      id: 'ring',
      kind: 'sandbox',
      say: 'A ring inside a ring, and a model that can only draw a straight line: the boundary is where w·x̃ + b = 0, a line whatever the weights. It converged at step 35 with accuracy 54%; always calling the larger class would score 50%. Try it: Quadratic features.',
      takeaway: 'The model stays linear in its weights. New inputs bend the boundary; the unit, the sigmoid and the loss never change.',
      focus: { kind: 'panel', id: 'field', label: 'Probability across the plane' },
      enter: [{ type: 'preset', id: 'ring-linear' }, { type: 'reset' }, VIEW_MODEL, CURVES, SOLVE],
      numbers: (c) => [...stepLossAcc(c), { label: 'weights', value: steps(c.derived.featureCount) }],
      experiments: [
        {
          label: 'Quadratic features',
          say: 'The unit also receives x̃₁², x̃₂² and x̃₁x̃₂.',
          enter: on('ring-quadratic', null, CURVES, PLAY),
          speed: 'normal',
          until: (c) => stopped(c) || c.derived.logLoss < 0.08 || c.state.epoch >= 600,
          then: (c) => 'Quadratic features, ' + steps(c.derived.featureCount) + ' weights instead of 3. ' + ended(c) + '. The boundary closed into a circle.',
          focus: { kind: 'control', key: 'featureMap', label: 'Features' },
        },
        {
          label: 'Linear features again',
          say: 'Back to x̃₁ and x̃₂ alone.',
          enter: on('ring-linear', null, CURVES, PLAY),
          speed: 'normal',
          until: (c) => stopped(c) || c.state.epoch >= 300,
          then: (c) => ended(c) + '; the larger class alone would score ' + pc(c.derived.baseline) + '. A line cannot separate a ring from its centre.',
          focus: { kind: 'panel', id: 'field', label: 'Probability across the plane' },
        },
      ],
    },
  ],
};

const decisionThreshold: L = {
  id: 'threshold',
  title: 'Decision threshold',
  hook: 'Precision, recall and the threshold t.',
  section: 'core',
  presetId: 'imbalanced',
  knobs: ['threshold'],
  steps: [
    {
      id: 'imbalanced',
      kind: 'sandbox',
      say: 'Only 12% of the points are positive. Trained: accuracy 92%, but always calling the larger class scores 88%. The confusion matrix counts hits and misses, rows the truth, columns the call: recall 0.38, 15 of 24 positives missed. Try it: Threshold 0.20.',
      takeaway: 'On imbalanced data accuracy mostly measures the imbalance. The threshold is chosen after training, by what a missed positive costs against a false alarm.',
      focus: { kind: 'panel', id: 'second', label: 'Confusion matrix' },
      enter: [{ type: 'preset', id: 'imbalanced' }, { type: 'reset' }, VIEW_MODEL, CONFUSION, SOLVE],
      numbers: (c) => [
        { label: 'threshold', value: fmt(c.params.threshold, 2) },
        { label: 'recall', value: fmt(c.derived.recall, 2) },
        { label: 'precision', value: fmt(c.derived.precision, 2) },
        { label: 'false alarms', value: steps(c.derived.fp) },
      ],
      experiments: [
        {
          label: 'Threshold 0.20',
          say: 'A point is called positive at p ≥ 0.20; the weights stay put.',
          enter: on('imbalanced', { threshold: 0.2 }, CONFUSION, SOLVE),
          until: (c) => stopped(c) || c.state.epoch >= 2000,
          then: (c) =>
            'Threshold ' + fmt(c.params.threshold, 2) + ': recall ' + fmt(c.derived.recall, 2) + ', precision ' + fmt(c.derived.precision, 2) + ', false alarms ' +
            steps(c.derived.fp) + '. Not one weight moved.',
          focus: { kind: 'control', key: 'threshold', label: 'Threshold (t)' },
        },
        {
          label: 'Threshold 0.90',
          say: 'Only the surest calls count as positive.',
          enter: on('imbalanced', { threshold: 0.9 }, CONFUSION, SOLVE),
          until: (c) => stopped(c) || c.state.epoch >= 2000,
          then: (c) =>
            'Threshold ' + fmt(c.params.threshold, 2) + ': recall ' + fmt(c.derived.recall, 2) + ', precision ' + fmt(c.derived.precision, 2) + ', false alarms ' +
            steps(c.derived.fp) + '. Few calls, almost all of them right.',
          focus: { kind: 'control', key: 'threshold', label: 'Threshold (t)' },
        },
        {
          label: 'Threshold 0.50',
          say: 'The default cut.',
          enter: on('imbalanced', null, CONFUSION, SOLVE),
          until: (c) => stopped(c) || c.state.epoch >= 2000,
          then: (c) =>
            'Threshold ' + fmt(c.params.threshold, 2) + ': accuracy ' + pc(c.derived.accuracy) + ', recall ' + fmt(c.derived.recall, 2) + ', ' + steps(c.derived.fn) +
            ' of ' + steps(c.derived.positives) + ' positives missed.',
          focus: { kind: 'panel', id: 'second', label: 'Confusion matrix' },
        },
      ],
    },
  ],
};

const learningRate: L = {
  id: 'learning-rate',
  title: 'Learning rate α',
  hook: 'High α is hard to overshoot; low α crawls.',
  section: 'failure',
  presetId: 'reckless-rate',
  knobs: ['learningRate', 'l2'],
  steps: [
    {
      id: 'alpha',
      kind: 'sandbox',
      say: 'α = 100, a hundred times the rate that worked, and still the run converged: step 512, log loss 0.051, accuracy 97%. A confidently correct point adds almost nothing to the gradient, so cross-entropy is hard to overshoot. Try it: α = 0.003.',
      takeaway: 'Too small is safe and slow; too large is usually safe here too. Watch the loss curve, not the rate.',
      focus: { kind: 'control', key: 'learningRate', label: 'Learning rate (α)' },
      enter: [{ type: 'preset', id: 'reckless-rate' }, { type: 'reset' }, VIEW_MODEL, CURVES, SOLVE],
      numbers: stepLossAcc,
      experiments: [
        {
          label: 'α = 0.003',
          say: 'Nothing is broken; the steps are too small to watch.',
          enter: on('crawl', null, CURVES, PLAY),
          speed: 'turbo',
          until: (c) => stopped(c) || c.state.epoch >= 2000,
          then: (c) => ended(c) + ', still falling slowly.',
          focus: { kind: 'panel', id: 'second', label: 'Loss and accuracy' },
        },
        {
          label: 'α = 100 again',
          say: 'The reckless rate, live.',
          enter: on('reckless-rate', null, CURVES, PLAY),
          speed: 'fast',
          until: (c) => stopped(c) || c.state.epoch >= 1000,
          then: (c) => ended(c) + '. Each correct point at p near 1 or 0 pushes almost nothing back.',
          focus: { kind: 'metric', key: 'loss', label: 'Log loss' },
        },
        {
          label: 'α = 100 with λ = 1',
          say: 'An L2 penalty added: its gradient 2λw never saturates.',
          enter: on('overflow', null, CURVES, PLAY),
          speed: 'normal',
          until: (c) => stopped(c) || c.state.epoch >= 1000,
          then: (c) => ended(c) + '. With α·2λ above 1 each step overshoots zero; the weights flip sign and grow until they overflow.',
          focus: { kind: 'control', key: 'l2', label: 'L2 penalty (λ)' },
        },
      ],
    },
  ],
};

const experiments: L = {
  id: 'experiments',
  title: 'Experiments',
  hook: 'Ten experiments with known outcomes.',
  section: 'sandbox',
  presetId: 'clean-blobs',
  steps: [
    {
      id: 'sandbox',
      kind: 'sandbox',
      say: 'Pick an experiment: the settings change, the run replays, and the outcome opens under it.',
      enter: [{ type: 'preset', id: 'clean-blobs' }, { type: 'reset' }, VIEW_MODEL, { type: 'speed', speed: 'turbo' }],
      experiments: [
        {
          label: 'XOR with x₁x₂',
          say: 'Opposite quadrants share a label. One product term, x̃₁x̃₂, is positive in two quadrants and negative in the other two, so XOR needs nothing more: 100% by step 74.',
          patch: { dataset: 'xor', featureMap: 'interaction', sampleCount: 160 },
          focus: { kind: 'control', key: 'featureMap', label: 'Features' },
        },
        {
          label: 'Separable data',
          say: 'Two blobs with almost no overlap: once every point is on the correct side, scaling the weights up only raises confidence, so the largest weight passes 7 by step 3,000 and nothing stops it.',
          patch: { noise: 0.02, l2: 0, learningRate: 2, sampleCount: 60 },
          focus: { kind: 'control', key: 'noise', label: 'Noise' },
        },
        {
          label: 'L2 penalty on separable data',
          say: 'The same blobs with λ = 0.02: the penalty charges for weight size, so the weights stop at 1.58 by step 51 and p changes gradually across the gap instead of flipping.',
          patch: { noise: 0.02, l2: 0.02, learningRate: 2, sampleCount: 60 },
          focus: { kind: 'control', key: 'l2', label: 'L2 penalty (λ)' },
        },
        {
          label: 'High α with L2',
          say: 'α = 100 with an L2 penalty of 1: the penalty gradient 2λw grows with w and never saturates, so with α·2λ above 1 each step overshoots zero. The weights flip sign and grow until the run diverges.',
          patch: { learningRate: 100, l2: 1 },
          focus: { kind: 'control', key: 'l2', label: 'L2 penalty (λ)' },
        },
        {
          label: 'Overlapping classes',
          say: 'Noise 0.8: the loss flattens at 0.38 and accuracy at 80%. No straight boundary can do better on these points.',
          patch: { noise: 0.8 },
          focus: { kind: 'control', key: 'noise', label: 'Noise' },
        },
        {
          label: 'Batch size 1',
          say: 'Batch size 1 at α = 0.3: the loss curve turns noisy but still trends down, 120 times cheaper per step.',
          patch: { batchSize: 1, learningRate: 0.3 },
          focus: { kind: 'control', key: 'batchSize', label: 'Batch size' },
        },
        {
          label: 'Moons, quadratic features',
          say: 'Two moons with the quadratic map: one curved boundary gets to about 88% and cannot follow the tips.',
          patch: { dataset: 'moons', featureMap: 'quadratic' },
          focus: { kind: 'control', key: 'featureMap', label: 'Features' },
        },
        {
          label: 'Spiral, quadratic features',
          say: 'Even a quadratic boundary stalls under 60% on the spiral. This is where a neural network is needed.',
          patch: { dataset: 'spiral', featureMap: 'quadratic' },
          focus: { kind: 'control', key: 'dataset', label: 'Dataset' },
        },
        {
          label: 'No standardisation',
          say: 'The quadratic moons on raw inputs: x² reaches 36 while x reaches 6, so their weights want different step sizes. The run starts higher and ends less accurate.',
          patch: { dataset: 'moons', featureMap: 'quadratic', standardise: false },
          focus: { kind: 'control', key: 'standardise', label: 'Standardise features' },
        },
        {
          label: 'Threshold 0.9',
          say: 'The overlapping classes cut at t = 0.9: precision 1.0, recall 0.45, no false positives. The weights are unchanged.',
          patch: { noise: 0.8, threshold: 0.9 },
          focus: { kind: 'control', key: 'threshold', label: 'Threshold (t)' },
        },
      ],
    },
  ],
};

export const LESSONS: L[] = [decisionBoundary, featureEngineering, decisionThreshold, learningRate, experiments];
