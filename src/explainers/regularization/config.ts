/** Ridge and lasso: params, controls and presets. `plane` picks the two weights the plane draws. */

import type { ControlGroup, Preset } from '../../explainer/types';
import { fmtKnob } from '../../lib/math/stats';

export interface RegParams extends Record<string, number | string | boolean> {
  /** The wave with its powers of x, or two plain columns for the weight plane. */
  dataset: string;
  /** Highest power of x for the curve. */
  features: number;
  count: number;
  noise: number;
  seed: number;

  penalty: string;
  lambda: number;
  alpha: number;
  solver: string;
  learningRate: number;
  standardise: boolean;

  /** The first of the two consecutive weights the plane draws. */
  plane: number;
  showOls: boolean;
  showTruth: boolean;
  showTest: boolean;
}

export const DEFAULT_PARAMS: RegParams = {
  dataset: 'curve',
  features: 10,
  count: 24,
  noise: 0.3,
  seed: 11,

  penalty: 'lasso',
  lambda: 0.03,
  alpha: 0.5,
  solver: 'auto',
  learningRate: 0.2,
  standardise: true,

  plane: 0,
  showOls: false,
  showTruth: true,
  showTest: true,
};

export const CONTROL_GROUPS: ControlGroup<keyof RegParams & string>[] = [
  {
    title: 'Data',
    zone: 'top',
    controls: [
      {
        kind: 'select',
        key: 'dataset',
        label: 'Dataset',
        help: 'What the model fits.',
        options: [
          { value: 'curve', label: 'Curve', hint: 'One input x and its powers x, x², x³, … fitted to a noisy wave. Overfitting you can see.' },
          { value: 'two', label: 'Two columns', hint: 'Two plain columns and two weights, so the whole weight space is a plane you can draw.' },
        ],
      },
      {
        kind: 'stepper',
        key: 'features',
        label: 'Powers of x',
        min: 1,
        max: 12,
        step: 1,
        visibleWhen: (p) => p.dataset === 'curve',
        help: 'The highest power of x the curve may use: x, x², … up to this one. More powers, more freedom to bend.',
      },
    ],
  },
  {
    title: 'Penalty',
    zone: 'top',
    controls: [
      {
        kind: 'select',
        key: 'penalty',
        label: 'Penalty',
        help: 'What the model pays for large weights.',
        options: [
          { value: 'ridge', label: 'Ridge (L2)', hint: 'Charges the squared weights. Shrinks everything, zeroes nothing.' },
          { value: 'lasso', label: 'Lasso (L1)', hint: 'Charges the absolute weights. Small ones are pushed to exactly zero.' },
          { value: 'elastic', label: 'Elastic net', hint: 'A mix of the two, set by α.' },
        ],
      },
      {
        kind: 'stepper',
        key: 'lambda',
        label: 'Strength (λ)',
        min: 0.0001,
        max: 30,
        step: 0.0001,
        scale: 'log',
        format: fmtKnob,
        help: 'How much the penalty weighs against the error. The smallest value is as good as least squares; above λ max the lasso keeps nothing.',
      },
      {
        kind: 'stepper',
        key: 'alpha',
        label: 'Mix (α)',
        min: 0,
        max: 1,
        step: 0.1,
        format: (v) => v.toFixed(1),
        visibleWhen: (p) => p.penalty === 'elastic',
        help: 'Share of the penalty that is L1. One is the lasso, zero is ridge.',
      },
    ],
  },
  {
    title: 'Solver',
    zone: 'top',
    controls: [
      {
        kind: 'select',
        key: 'solver',
        label: 'Solver',
        advanced: true,
        help: 'How the weights are found. Every solver reaches the same answer when it converges.',
        options: [
          { value: 'auto', label: 'Automatic', hint: 'Ridge by its closed form in one step; anything with an L1 term by coordinate descent.' },
          { value: 'cd', label: 'Coordinate descent', hint: 'One weight at a time, solved exactly with the others fixed. A step is one sweep over every weight.' },
          { value: 'ista', label: 'Proximal gradient', hint: 'A gradient step on the error, then a soft threshold. A step is one epoch.' },
        ],
      },
      {
        kind: 'stepper',
        key: 'learningRate',
        label: 'Step size (η)',
        advanced: true,
        min: 0.01,
        max: 2,
        step: 0.01,
        scale: 'log',
        format: fmtKnob,
        visibleWhen: (p) => p.solver === 'ista',
        help: 'The gradient step of the proximal solver. Above 2 over the largest curvature it diverges.',
      },
    ],
  },
  {
    title: 'Data',
    blurb: 'The rows the model sees.',
    controls: [
      {
        kind: 'slider',
        key: 'count',
        label: 'Points',
        min: 12,
        max: 200,
        step: 4,
        help: 'Rows in the dataset; a quarter are held out as the test set.',
      },
      {
        kind: 'slider',
        key: 'noise',
        label: 'Noise',
        min: 0,
        max: 1,
        step: 0.05,
        format: (v) => v.toFixed(2),
        help: 'Noise on y as a fraction of the signal’s spread.',
      },
      {
        kind: 'toggle',
        key: 'standardise',
        label: 'Standardise',
        help: 'Divide every feature by its spread before fitting, so one λ means the same thing for every weight.',
      },
      { kind: 'seed', key: 'seed', label: 'Seed', help: 'Same seed, same data, every time.' },
    ],
  },
  {
    title: 'Display',
    collapsedByDefault: true,
    controls: [
      { kind: 'toggle', key: 'showOls', label: 'Least squares', help: 'The unpenalised fit for comparison: a dotted curve, ghost bars, the × on the weight plane.' },
      { kind: 'toggle', key: 'showTruth', label: 'Truth', help: 'The wave the points were drawn from, or the weights the two columns were made with.' },
      { kind: 'toggle', key: 'showTest', label: 'Test error', help: 'The held-out error on the error curve.' },
    ],
  },
];

export const PRESETS: Preset<RegParams>[] = [
  {
    id: 'overfit',
    name: 'Overfitting',
    blurb: 'Eighteen points, ten powers of x and almost no penalty. The curve bends through the noise instead of following the wave.',
    params: { penalty: 'ridge', lambda: 0.0001 },
    autoRun: true,
  },
  {
    id: 'ridge',
    name: 'Ridge',
    blurb: 'The same ten powers with an L2 penalty. Every weight shrinks, the curve settles onto the wave, and no weight reaches zero.',
    params: { penalty: 'ridge', lambda: 0.1 },
    autoRun: true,
  },
  {
    id: 'lasso',
    name: 'Lasso',
    blurb: 'The same ten powers with an L1 penalty. Most weights are exactly zero: a handful of terms carry the whole curve.',
    params: {},
    autoRun: true,
  },
  {
    id: 'two-weights',
    name: 'Two weights',
    blurb: 'Two columns, so the whole picture fits on the page: the error as rings, the penalty as a disc or a diamond, and the answer where they touch.',
    params: { dataset: 'two', count: 60, seed: 31, lambda: 1, showOls: true },
    autoRun: true,
  },
  {
    id: 'wide',
    name: 'Wide data',
    blurb: 'Twelve powers of x and nine fitted points: with almost no penalty many curves fit the points exactly, and the test error shows which one you got.',
    params: { features: 12, count: 12, penalty: 'ridge', lambda: 0.0001 },
    autoRun: true,
  },
];
