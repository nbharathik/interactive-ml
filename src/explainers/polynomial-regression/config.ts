/** Polynomial regression: params, controls and presets. */

import type { ControlGroup, Preset } from '../../explainer/types';
import { CURVE_DATASETS } from '../../lib/datasets/curves';
import { fmtKnob } from '../../lib/math/stats';

export const MAX_DEGREE = 15;

export interface PolyParams extends Record<string, number | string | boolean> {
  dataset: string;
  sampleCount: number;
  testCount: number;
  noise: number;
  seed: number;

  degree: number;
  l2: number;
  standardise: boolean;
  refits: number;

  showResiduals: boolean;
  showTruth: boolean;
  showTest: boolean;
  showRefits: boolean;
}

export const DEFAULT_PARAMS: PolyParams = {
  dataset: 'wave',
  sampleCount: 25,
  testCount: 15,
  noise: 0.5,
  seed: 42,

  degree: 3,
  l2: 0,
  standardise: true,
  refits: 50,

  showResiduals: false,
  showTruth: true,
  showTest: true,
  showRefits: true,
};

export const CONTROL_GROUPS: ControlGroup<keyof PolyParams & string>[] = [
  {
    title: 'Data',
    blurb: 'Every shape comes with the function it was drawn from.',
    controls: [
      {
        kind: 'select',
        key: 'dataset',
        label: 'Dataset',
        help: 'The true relationship under the noise. Each shape has a degree that suits it, or none at all.',
        options: CURVE_DATASETS.map((d) => ({ value: d.id, label: d.name, hint: d.blurb })),
      },
      {
        kind: 'slider',
        key: 'sampleCount',
        label: 'Points',
        min: 5,
        max: 200,
        step: 5,
        help: 'Training points. The fit only ever sees these. Fewer points and a high degree is a curve through every one of them.',
      },
      {
        kind: 'slider',
        key: 'testCount',
        label: 'Test points',
        min: 5,
        max: 100,
        step: 5,
        help: 'A second draw from the same source, never used for fitting. The error on these is the honest one.',
      },
      {
        kind: 'slider',
        key: 'noise',
        label: 'Noise',
        min: 0,
        max: 1,
        step: 0.05,
        format: (v) => v.toFixed(2),
        help: 'Standard deviation of the noise on y. At zero the right degree fits perfectly and the variance is zero.',
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
        label: 'Degree',
        min: 0,
        max: MAX_DEGREE,
        step: 1,
        help: 'Highest power of x. Degree 0 predicts the mean, 1 is a straight line, and each step up adds one more weight to bend with.',
      },
      {
        kind: 'stepper',
        key: 'l2',
        label: 'Ridge (λ)',
        min: 0,
        max: 1,
        step: 0.001,
        scale: 'log',
        format: (v) => (v < 0.001 ? '0' : fmtKnob(v)),
        help: 'Adds λ·Σw² to the loss. Large weights are what a wiggly curve is made of, so charging for them smooths it.',
      },
      {
        kind: 'toggle',
        key: 'standardise',
        advanced: true,
        label: 'Standardise x',
        help: 'Rescale x to mean 0 and unit variance before raising it to powers. Raw x¹⁵ runs into the billions on this range; x̃¹⁵ stays in the thousands.',
      },
      {
        kind: 'stepper',
        key: 'refits',
        advanced: true,
        label: 'Refits',
        min: 10,
        max: 200,
        step: 10,
        help: 'How many fresh samples the play button fits, one per step, to estimate bias and variance.',
      },
    ],
  },
  {
    title: 'Display',
    collapsedByDefault: true,
    controls: [
      {
        kind: 'toggle',
        key: 'showTruth',
        label: 'True function',
        help: 'The function the data was generated from. Bias is the gap between the average fit and this.',
      },
      {
        kind: 'toggle',
        key: 'showTest',
        label: 'Test points',
        help: 'The held-out points, drawn as rings.',
      },
      {
        kind: 'toggle',
        key: 'showRefits',
        label: 'Refits',
        help: 'Every refit as a faint curve, with their average dashed. The spread is the variance.',
      },
      {
        kind: 'toggle',
        key: 'showResiduals',
        label: 'Residual lines',
        help: 'The vertical gap between each training point and the curve, the quantity squared and averaged into the training error.',
      },
    ],
  },
];

export const PRESETS: Preset<PolyParams>[] = [
  {
    id: 'cubic',
    name: 'Degree 3, a good fit',
    blurb: 'Four weights follow the wave. Training and test error land close together.',
    params: { degree: 3 },
  },
  {
    id: 'line',
    name: 'Degree 1, a line through a wave',
    blurb: 'The straight line misses the bends. Both errors are high: underfitting.',
    params: { degree: 1 },
  },
  {
    id: 'overfit',
    name: 'Degree 12, the curve chases the noise',
    blurb: 'Thirteen weights for twenty-five points. Training error falls, test error climbs.',
    params: { degree: 12 },
  },
  {
    id: 'bias',
    name: 'Refits at degree 1',
    blurb: 'Fifty fresh samples, fifty lines, all in nearly the same place and all missing the wave: high bias, low variance.',
    params: { degree: 1 },
    autoRun: true,
  },
  {
    id: 'variance',
    name: 'Refits at degree 10',
    blurb: 'Fifty fresh samples, fifty different curves. Their average is close to the truth: low bias, high variance.',
    params: { degree: 10 },
    autoRun: true,
  },
  {
    id: 'balanced',
    name: 'Refits at degree 3',
    blurb: 'The refits agree with each other and with the wave: both bias and variance small.',
    params: { degree: 3 },
    autoRun: true,
  },
  {
    id: 'ridge',
    name: 'Degree 12 with a ridge penalty',
    blurb: 'Same thirteen weights, now charged for their size. The wiggle goes and the test error comes back down.',
    params: { degree: 12, l2: 0.01 },
  },
  {
    id: 'many',
    name: 'Degree 12 on two hundred points',
    blurb: 'The other fix for variance: more data. Thirteen weights cannot chase the noise of two hundred points.',
    params: { degree: 12, sampleCount: 200 },
  },
  {
    id: 'interpolate',
    name: 'Ten points, ten weights',
    blurb: 'Degree 9 through ten points passes through every one: training error zero, test error enormous.',
    params: { degree: 9, sampleCount: 10 },
  },
];
