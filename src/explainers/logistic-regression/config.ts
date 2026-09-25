/** Logistic regression: params, controls and presets. */

import type { ControlGroup, Preset } from '../../explainer/types';
import { CLASSIFICATION_DATASETS } from '../../lib/datasets/points';
import { fmtKnob } from '../../lib/math/stats';

/** This page fits one boundary between two classes, so the 3-class shape is left out. */
const BINARY_DATASETS = CLASSIFICATION_DATASETS.filter((d) => d.id !== 'three-class');

/** Batch sizes at or above this mean "use every point", whatever the sample count. */
export const FULL_BATCH = 240;

export interface LogRegParams extends Record<string, number | string | boolean> {
  dataset: string;
  sampleCount: number;
  noise: number;
  seed: number;
  /** Which label a click on empty space adds. */
  brushClass: string;

  featureMap: string;
  l2: number;
  standardise: boolean;

  learningRate: number;
  batchSize: number;

  threshold: number;

  showField: boolean;
  showMisclassified: boolean;
  showProjection: boolean;
}

export const DEFAULT_PARAMS: LogRegParams = {
  dataset: 'gaussians',
  sampleCount: 120,
  noise: 0.25,
  seed: 42,
  brushClass: '1',

  featureMap: 'linear',
  l2: 0,
  standardise: true,

  learningRate: 1,
  batchSize: FULL_BATCH,

  threshold: 0.5,

  showField: true,
  showMisclassified: true,
  showProjection: true,
};

export const CONTROL_GROUPS: ControlGroup<keyof LogRegParams & string>[] = [
  {
    title: 'Data',
    blurb: 'Click the main chart to add a point; right-click adds the other class, shift-click removes the nearest.',
    controls: [
      {
        kind: 'select',
        key: 'dataset',
        label: 'Dataset',
        help: 'Two blobs is the case a straight boundary was built for. The ring, XOR and the spiral are not; they are here so you can watch a linear model fail honestly.',
        options: BINARY_DATASETS.map((d) => ({ value: d.id, label: d.name, hint: d.blurb })),
      },
      {
        kind: 'slider',
        key: 'sampleCount',
        label: 'Points',
        min: 20,
        max: FULL_BATCH,
        step: 10,
        help: 'More points steady the boundary. At 20 the line swings noticeably every time you reshuffle the seed.',
      },
      {
        kind: 'slider',
        key: 'noise',
        label: 'Noise',
        min: 0,
        max: 1,
        step: 0.01,
        format: (v) => v.toFixed(2),
        help: 'How far the classes bleed into each other. Cleanly separated classes let the weights grow without limit; an L2 penalty stops them.',
      },
      { kind: 'seed', key: 'seed', label: 'Seed', help: 'Same seed, same points, every time.' },
      {
        kind: 'segmented',
        key: 'brushClass',
        label: 'Left click adds',
        options: [
          { value: '0', label: 'Negative', hint: 'Circles. The class the model pushes probability towards 0.' },
          { value: '1', label: 'Positive', hint: 'Crosses. The class the model pushes probability towards 1.' },
        ],
        help: 'The class a left click adds; a right click adds the other one. One point deep in the wrong region shows how far a single observation moves the boundary.',
      },
    ],
  },
  {
    title: 'Model',
    zone: 'top',
    blurb: 'What the model is allowed to compute before the sigmoid sees it.',
    controls: [
      {
        kind: 'segmented',
        key: 'featureMap',
        label: 'Features',
        options: [
          { value: 'linear', label: 'Linear', hint: '1, x₁, x₂. The boundary is a straight line.' },
          { value: 'interaction', label: '+ x₁x₂', hint: 'Adds the product term. The boundary becomes a hyperbola, which is exactly what XOR needs.' },
          { value: 'quadratic', label: 'Quadratic', hint: 'Adds x₁², x₂² and x₁x₂. The boundary can now be an ellipse, try it on the ring.' },
        ],
        help: 'Extra columns computed from x₁ and x₂ before training. The model stays linear in its weights, which is the only thing the gradient cares about; the boundary stops being straight.',
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
        help: 'Adds λ·Σw² to the loss, pulling every non-bias weight towards zero. On separable data this is the thing that stops the weights running away to infinity.',
      },
      {
        kind: 'toggle',
        key: 'standardise',
        advanced: true,
        label: 'Standardise features',
        help: 'Centres and rescales x₁ and x₂ before training. With the quadratic map the raw squares reach 36 while x₁ reaches 6, so one step size cannot suit every weight.',
      },
    ],
  },
  {
    title: 'Training',
    zone: 'top',
    controls: [
      {
        kind: 'stepper',
        key: 'learningRate',
        label: 'Learning rate (α)',
        min: 0.001,
        max: 100,
        step: 0.001,
        scale: 'log',
        format: fmtKnob,
        help: 'How far each step moves along the gradient. Push it as high as it goes: cross-entropy is remarkably hard to blow up, because a confidently-correct point contributes nothing to the gradient rather than more.',
      },
      {
        kind: 'stepper',
        key: 'batchSize',
        advanced: true,
        label: 'Batch size',
        min: 1,
        max: FULL_BATCH,
        step: 1,
        format: (v) => (v >= FULL_BATCH ? 'full batch' : String(v)),
        help: 'How many points each step averages over. One point per step makes the loss curve a scribble that still trends down, at a fraction of the cost per step.',
      },
    ],
  },
  {
    title: 'The decision rule',
    zone: 'top',
    blurb: 'Chosen after training. Nothing here changes a weight.',
    controls: [
      {
        kind: 'stepper',
        key: 'threshold',
        label: 'Threshold (t)',
        min: 0.05,
        max: 0.95,
        step: 0.01,
        format: (v) => v.toFixed(2),
        help: 'The probability above which a point is called positive. Drag it while the model is trained: the boundary slides, precision and recall trade against each other, and not one weight moves.',
      },
    ],
  },
  {
    title: 'Display',
    // Display knobs start collapsed.
    collapsedByDefault: true,
    controls: [
      {
        kind: 'toggle',
        key: 'showField',
        label: 'Probability field',
        help: 'Shades every position in the plane by the probability the model assigns it. Turn it off to see the boundary line on its own.',
      },
      {
        kind: 'toggle',
        key: 'showMisclassified',
        label: 'Ring the mistakes',
        help: 'Draws a ring around every point the model currently puts on the wrong side of the threshold.',
      },
      {
        kind: 'toggle',
        key: 'showProjection',
        label: 'Project onto sigmoid',
        help: 'Places each point on the sigmoid at its own linear score, with a line up or down to the label it should have had. That gap is what the gradient is made of.',
      },
    ],
  },
];

export const PRESETS: Preset<LogRegParams>[] = [
  {
    id: 'clean-blobs',
    name: 'Two clean blobs',
    blurb: 'The boundary swings into the gap between the classes and settles there.',
    params: { dataset: 'gaussians', featureMap: 'linear', learningRate: 1 },
    autoRun: true,
  },
  {
    id: 'overlap',
    name: 'Blobs that overlap',
    blurb: 'Loss flattens out well above zero. No straight boundary can do better than this.',
    params: { dataset: 'gaussians', noise: 0.8, featureMap: 'linear', learningRate: 1 },
    autoRun: true,
  },
  {
    id: 'ring-linear',
    name: 'A line against a ring',
    blurb: 'Accuracy sticks near 50% forever. Train it as long as you like, the model cannot bend.',
    params: { dataset: 'circle', featureMap: 'linear', learningRate: 1, sampleCount: 160 },
    autoRun: true,
  },
  {
    id: 'ring-quadratic',
    name: 'The same ring, quadratic',
    blurb: 'Two squared features turn the boundary into a circle. Same loss, same gradient, same loop.',
    params: { dataset: 'circle', featureMap: 'quadratic', learningRate: 1, sampleCount: 160 },
    autoRun: true,
  },
  {
    id: 'xor-interaction',
    name: 'XOR needs one product',
    blurb: 'Adding x₁x₂ alone is enough. The boundary becomes a pair of crossed branches.',
    params: { dataset: 'xor', featureMap: 'interaction', learningRate: 1, sampleCount: 160 },
    autoRun: true,
  },
  {
    id: 'imbalanced',
    name: 'Accuracy is lying',
    blurb: 'Only 12% of the points are positive. Accuracy reaches 92%; recall never gets past 0.4.',
    params: { dataset: 'imbalanced', sampleCount: 200, noise: 0.9, learningRate: 1, threshold: 0.5 },
    autoRun: true,
  },
  {
    id: 'recall-threshold',
    name: 'Buying recall with precision',
    blurb: 'Identical weights, cut at t = 0.20. Recall rises past 0.5 and the false alarms go from 2 to 23.',
    params: { dataset: 'imbalanced', sampleCount: 200, noise: 0.9, learningRate: 1, threshold: 0.2 },
    autoRun: true,
  },
  {
    id: 'reckless-rate',
    name: 'A rate 100 times too big',
    blurb: 'α = 100 and the loss still falls. Cross-entropy is very hard to make explode.',
    params: { dataset: 'gaussians', learningRate: 100, featureMap: 'linear' },
    autoRun: true,
  },
  {
    id: 'overflow',
    name: 'The one way to overflow it',
    blurb: 'α = 100 with λ = 1. The penalty term does not saturate, and the weights run to infinity.',
    params: { dataset: 'gaussians', learningRate: 100, l2: 1, featureMap: 'linear' },
    autoRun: true,
  },
  {
    id: 'crawl',
    name: 'A learning rate that crawls',
    blurb: 'Nothing is broken. The steps are too small to watch.',
    params: { dataset: 'gaussians', learningRate: 0.003, featureMap: 'linear' },
    autoRun: true,
  },
  {
    id: 'single-point',
    name: 'One point per step',
    blurb: 'A noisy estimate of the gradient, computed 120 times more cheaply.',
    params: { dataset: 'gaussians', batchSize: 1, learningRate: 0.3, featureMap: 'linear' },
    autoRun: true,
  },
  {
    id: 'runaway',
    name: 'Weights with nothing to stop them',
    blurb: 'Perfectly separable data and no penalty. The boundary stops moving; the weights never do.',
    params: { dataset: 'gaussians', noise: 0.02, l2: 0, learningRate: 2, sampleCount: 60 },
    autoRun: true,
  },
  {
    id: 'runaway-penalised',
    name: 'The same run, penalised',
    blurb: 'λ = 0.02 on identical data. The weights stop growing and the field softens.',
    params: { dataset: 'gaussians', noise: 0.02, l2: 0.02, learningRate: 2, sampleCount: 60 },
    autoRun: true,
  },
];
