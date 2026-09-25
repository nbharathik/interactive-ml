/** Linear regression: params, controls and presets. */

import type { ControlGroup, Preset } from '../../explainer/types';
import { REGRESSION_DATASETS } from '../../lib/datasets/regression';
import { fmtKnob } from '../../lib/math/stats';

export interface LinRegParams extends Record<string, number | string | boolean> {
  dataset: string;
  sampleCount: number;
  noise: number;
  seed: number;

  learningRate: number;
  degree: number;
  optimiser: string;
  momentum: number;
  batchMode: string;
  batchSize: number;
  l2: number;
  standardise: boolean;

  showResiduals: boolean;
  showSquares: boolean;
  showTruth: boolean;
  showClosedForm: boolean;
}

export const DEFAULT_PARAMS: LinRegParams = {
  dataset: 'housing',
  sampleCount: 60,
  noise: 0.25,
  seed: 42,

  learningRate: 0.03,
  degree: 1,
  optimiser: 'gd',
  momentum: 0.9,
  batchMode: 'batch',
  batchSize: 8,
  l2: 0,
  standardise: true,

  showResiduals: true,
  showSquares: false,
  showTruth: false,
  showClosedForm: true,
};

export const CONTROL_GROUPS: ControlGroup<keyof LinRegParams & string>[] = [
  {
    title: 'Data',
    blurb: 'The model can only be as good as what it is shown.',
    controls: [
      {
        kind: 'select',
        key: 'dataset',
        label: 'Dataset',
        help: 'Each shape stresses a different assumption. "Straight line" is the case linear regression was built for; the others are not.',
        options: REGRESSION_DATASETS.map((d) => ({
          value: d.id,
          label: d.name,
          hint: d.blurb,
        })),
      },
      {
        kind: 'slider',
        key: 'sampleCount',
        label: 'Points',
        min: 10,
        max: 300,
        step: 10,
        help: 'More data steadies the fit. Drop to 10 and watch how much the line swings when you reshuffle.',
      },
      {
        kind: 'slider',
        key: 'noise',
        label: 'Noise',
        min: 0,
        max: 1,
        step: 0.05,
        format: (v) => v.toFixed(2),
        help: 'How far the observations scatter from the true relationship. At zero the fit can be perfect.',
      },
      { kind: 'seed', key: 'seed', label: 'Seed', help: 'Same seed, same data, every time.' },
    ],
  },
  {
    title: 'Model',
    zone: 'top',
    controls: [
      {
        kind: 'stepper',
        key: 'degree',
        label: 'Polynomial degree',
        min: 1,
        max: 8,
        step: 1,
        help: 'Degree 1 is a straight line. Higher degrees add x², x³, … as extra features, still "linear" in the weights, which is all the algorithm cares about.',
      },
      {
        kind: 'stepper',
        key: 'l2',
        advanced: true,
        label: 'L2 penalty (λ)',
        min: 0,
        max: 1,
        step: 0.001,
        scale: 'log',
        format: (v) => (v < 0.001 ? '0' : fmtKnob(v)),
        help: 'Adds λ·Σw² to the loss, pulling weights towards zero. At high degree this is what stops the curve from chasing every point.',
      },
      {
        kind: 'toggle',
        key: 'standardise',
        label: 'Standardise x',
        help: 'Rescales x to zero mean and unit variance before fitting. Turn it off at degree 4+ and watch gradient descent fall apart; on x up to 10, x⁸ is ten million times larger than x.',
      },
    ],
  },
  {
    title: 'Optimiser',
    zone: 'top',
    controls: [
      {
        kind: 'stepper',
        key: 'learningRate',
        label: 'Learning rate (α)',
        min: 0.0001,
        max: 3,
        step: 0.0001,
        scale: 'log',
        format: fmtKnob,
        help: 'How far to move along the gradient each step. Too small and nothing happens; too large and the loss explodes.',
      },
      {
        kind: 'segmented',
        key: 'optimiser',
        label: 'Update rule',
        help: 'How each step turns the gradient into a move. Plain descent follows it directly; the others remember past steps.',
        options: [
          { value: 'gd', label: 'Plain', hint: 'w ← w − α·∇L. The definition of gradient descent.' },
          { value: 'momentum', label: 'Momentum', hint: 'Accumulates velocity, so it rolls through shallow valleys instead of crawling.' },
          { value: 'adam', label: 'Adam', hint: 'Per-weight adaptive step sizes. Forgiving of a badly chosen learning rate.' },
        ],
      },
      {
        kind: 'stepper',
        key: 'momentum',
        label: 'Momentum (β)',
        min: 0,
        max: 0.99,
        step: 0.01,
        format: (v) => v.toFixed(2),
        visibleWhen: (p) => p.optimiser === 'momentum' || p.optimiser === 'adam',
        help: 'How much of the previous step carries over. Above 0.95 the run can overshoot and oscillate.',
      },
    ],
  },
  {
    title: 'Batching',
    blurb: 'How many points each step reads before it moves.',
    controls: [
      {
        kind: 'segmented',
        key: 'batchMode',
        label: 'Batch',
        help: 'How many points each step looks at before moving. Fewer points means noisier, cheaper steps.',
        options: [
          { value: 'batch', label: 'Full', hint: 'Every point contributes to every step. Smooth, and slow on large data.' },
          { value: 'mini', label: 'Mini', hint: 'A slice at a time, the noisy compromise everyone actually uses.' },
          { value: 'stochastic', label: 'Single', hint: 'One point per step. The loss curve becomes a scribble.' },
        ],
      },
      {
        kind: 'slider',
        key: 'batchSize',
        label: 'Batch size',
        help: 'Points per step in mini-batch mode. Used only when Batch is set to Mini.',
        min: 2,
        max: 64,
        step: 1,
        visibleWhen: (p) => p.batchMode === 'mini',
      },
    ],
  },
  {
    title: 'Display',
    collapsedByDefault: true,
    controls: [
      {
        kind: 'toggle',
        key: 'showResiduals',
        label: 'Residual lines',
        help: 'The vertical gap between each point and the line, the quantity being squared and summed.',
      },
      {
        kind: 'toggle',
        key: 'showSquares',
        label: 'Squared errors',
        help: 'Draws the literal square whose area each residual contributes. This is why one far-away point matters so much.',
      },
      {
        kind: 'toggle',
        key: 'showTruth',
        label: 'True relationship',
        help: 'The function the data was generated from. Real data does not come with this.',
      },
      {
        kind: 'toggle',
        key: 'showClosedForm',
        label: 'Exact solution',
        help: 'The answer from the normal equations, where descent is heading.',
      },
    ],
  },
];

export const PRESETS: Preset<LinRegParams>[] = [
  {
    id: 'goldilocks',
    name: 'A learning rate that works',
    blurb: 'Steady descent, converges in a few hundred steps.',
    params: { dataset: 'housing', learningRate: 0.03, degree: 1, optimiser: 'gd', standardise: true },
    autoRun: true,
  },
  {
    id: 'too-small',
    name: 'Learning rate too small',
    blurb: 'It is working, just far too slowly to watch.',
    params: { dataset: 'housing', learningRate: 0.0005, degree: 1, optimiser: 'gd', standardise: true },
    autoRun: true,
  },
  {
    id: 'divergence',
    name: 'Learning rate too large',
    blurb: 'Each step overshoots further than the last. The loss goes to infinity.',
    params: { dataset: 'housing', learningRate: 2.2, degree: 1, optimiser: 'gd', standardise: true },
    autoRun: true,
  },
  {
    id: 'unscaled',
    name: 'Forgot to standardise',
    blurb: 'Degree 5 on raw x, where x⁵ reaches 100,000. The first step overshoots and the run explodes in three steps.',
    params: { dataset: 'curved', degree: 5, standardise: false, learningRate: 0.0003, optimiser: 'gd' },
    autoRun: true,
  },
  {
    id: 'overfit',
    name: 'Too much flexibility',
    blurb: 'Degree 8 through 20 noisy points. The curve fits the noise, not the trend.',
    params: { dataset: 'linear', degree: 8, sampleCount: 20, noise: 0.4, l2: 0, standardise: true, learningRate: 0.03, optimiser: 'adam', showTruth: true },
  },
  {
    id: 'regularised',
    name: 'Same curve, with a penalty',
    blurb: 'Degree 8 again, now with λ = 0.05. Compare the wiggle.',
    params: { dataset: 'linear', degree: 8, sampleCount: 20, noise: 0.4, l2: 0.05, standardise: true, learningRate: 0.03, optimiser: 'adam', showTruth: true },
  },
  {
    id: 'outliers',
    name: 'A few bad readings',
    blurb: 'Squared error weights big mistakes heavily. Watch the line tilt towards the outliers.',
    params: { dataset: 'outliers', degree: 1, showSquares: true, showTruth: true, learningRate: 0.03 },
  },
  {
    id: 'stochastic',
    name: 'One point at a time',
    blurb: 'Single-sample updates. The loss curve becomes a scribble that still trends down.',
    params: { dataset: 'housing', batchMode: 'stochastic', learningRate: 0.02, degree: 1 },
    autoRun: true,
  },
];
