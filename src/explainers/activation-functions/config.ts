/** Activations and losses: params, controls and the presets the lessons start from. Two loss knobs, one per task, so the row never reflows. `view` picks the one picture on show: the live network, the activation, the loss, the bent plane or the surface. */

import type { ControlGroup, Preset } from '../../explainer/types';
import { CLASSIFICATION_DATASETS } from '../../lib/datasets/points';
import { REGRESSION_DATASETS } from '../../lib/datasets/regression';
import { fmtKnob } from '../../lib/math/stats';
import { ACTIVATION_LABELS, ACTIVATION_NAMES } from '../../lib/ml/activations';
import type { ActivationName } from '../../lib/ml/activations';

export interface ActParams extends Record<string, number | string | boolean> {
  task: string;
  pointsDataset: string;
  curveDataset: string;
  count: number;
  noise: number;
  seed: number;

  activation: string;
  leak: number;
  classLoss: string;
  regLoss: string;
  huberDelta: number;

  depth: number;
  width: number;
  learningRate: number;
  initScale: number;
  batchSize: number;

  view: string;
  surfaceOf: string;
  showDerivative: boolean;
  showAll: boolean;
  showRug: boolean;
  logScale: boolean;
}

export const DEFAULT_PARAMS: ActParams = {
  task: 'classify',
  pointsDataset: 'moons',
  curveDataset: 'sinusoid',
  count: 120,
  noise: 0.25,
  seed: 9,

  activation: 'relu',
  leak: 0.1,
  classLoss: 'bce',
  regLoss: 'mse',
  huberDelta: 1,

  depth: 4,
  width: 6,
  learningRate: 0.02,
  initScale: 1,
  batchSize: 16,

  view: 'network',
  surfaceOf: 'output',
  showDerivative: true,
  showAll: false,
  showRug: false,
  logScale: true,
};

const ACTIVATION_HINTS: Record<ActivationName, string> = {
  sigmoid: 'Squashes to (0, 1). Its slope peaks at 0.25, so every layer shrinks the gradient.',
  tanh: 'Squashes to (−1, 1), centred on zero. Slope up to 1, still flat far out.',
  relu: 'Zero below zero, identity above. Cheap, no saturation on the right, but a unit can die.',
  leakyRelu: 'ReLU with a small slope on the left, so a negative unit still learns.',
  elu: 'Smooth ReLU that goes slightly negative on the left.',
  gelu: 'A smooth gate on the input, the transformer default.',
  softplus: 'A smooth ReLU: log(1 + eᶻ).',
  swish: 'z times its own sigmoid: dips below zero before rising.',
  linear: 'No non-linearity at all: a stack of these is still one linear map, so the network is logistic regression.',
};

export const CONTROL_GROUPS: ControlGroup<keyof ActParams & string>[] = [
  {
    title: 'Data',
    controls: [
      {
        kind: 'select',
        key: 'pointsDataset',
        label: 'Dataset',
        visibleWhen: (p) => p.task === 'classify',
        help: 'The classification set.',
        options: CLASSIFICATION_DATASETS.map((d) => ({ value: d.id, label: d.name, hint: d.blurb })),
      },
      {
        kind: 'select',
        key: 'curveDataset',
        label: 'Dataset',
        visibleWhen: (p) => p.task === 'regress',
        help: 'The regression set.',
        options: REGRESSION_DATASETS.map((d) => ({ value: d.id, label: d.name, hint: d.blurb })),
      },
      { kind: 'slider', key: 'count', label: 'Points', min: 20, max: 300, step: 10, help: 'Samples in the dataset; a quarter are held out.' },
      { kind: 'slider', key: 'noise', label: 'Noise', min: 0, max: 1, step: 0.05, format: (v) => v.toFixed(2), help: 'How scattered the samples are.' },
      { kind: 'seed', key: 'seed', label: 'Seed', help: 'Fixes the data and the initial weights.' },
    ],
  },
  {
    title: 'Functions',
    zone: 'top',
    controls: [
      {
        kind: 'select',
        key: 'activation',
        label: 'Activation',
        help: 'The non-linearity on every hidden unit.',
        options: ACTIVATION_NAMES.map((name) => ({ value: name, label: ACTIVATION_LABELS[name], hint: ACTIVATION_HINTS[name] })),
      },
      {
        kind: 'select',
        key: 'classLoss',
        label: 'Class loss',
        visibleWhen: (p) => p.task === 'classify',
        disabledWhen: (p) => p.pointsDataset === 'three-class',
        help: 'How a class prediction is scored. Three classes need one output each, so they always use softmax.',
        options: [
          { value: 'bce', label: 'Cross-entropy', hint: 'On the logit. Gradient p − y: large when confidently wrong.' },
          { value: 'hinge', label: 'Hinge', hint: 'Zero once the margin is met, so easy points stop mattering.' },
          { value: 'mse', label: 'Squared error', hint: 'On a sigmoid output. The gradient vanishes exactly where the answer is confidently wrong.' },
          { value: 'softmaxCE', label: 'Softmax cross-entropy', hint: 'One logit per class; gradient p − y per class.' },
        ],
      },
      {
        kind: 'select',
        key: 'regLoss',
        label: 'Value loss',
        visibleWhen: (p) => p.task === 'regress',
        help: 'How a predicted number is scored.',
        options: [
          { value: 'mse', label: 'Squared error', hint: 'Gradient grows with the error, so outliers pull hard.' },
          { value: 'mae', label: 'Absolute error', hint: 'Gradient is ±1 whatever the error: robust, but never settles.' },
          { value: 'huber', label: 'Huber', hint: 'Squared near zero, absolute beyond δ: the best of both.' },
        ],
      },
    ],
  },
  {
    title: 'Model',
    zone: 'top',
    controls: [
      {
        kind: 'select',
        key: 'task',
        label: 'Task',
        help: 'Classify two-dimensional points, or fit a curve.',
        options: [
          { value: 'classify', label: 'Classify' },
          { value: 'regress', label: 'Regress' },
        ],
      },
      { kind: 'stepper', key: 'depth', label: 'Depth', min: 0, max: 8, step: 1, help: 'Hidden layers. Zero is one output unit on the inputs, logistic regression; deep is where gradients vanish.' },
      {
        kind: 'stepper',
        key: 'learningRate',
        label: 'Learning rate (η)',
        min: 0.0001,
        max: 1,
        step: 0.0001,
        scale: 'log',
        format: fmtKnob,
        help: 'Adam’s step size.',
      },
    ],
  },
  {
    title: 'Functions',
    controls: [
      {
        kind: 'slider',
        key: 'leak',
        label: 'Leak (α)',
        min: 0.01,
        max: 0.5,
        step: 0.01,
        format: (v) => v.toFixed(2),
        visibleWhen: (p) => p.activation === 'leakyRelu',
        help: 'The slope of leaky ReLU left of zero.',
      },
      {
        kind: 'slider',
        key: 'huberDelta',
        label: 'Huber δ',
        min: 0.1,
        max: 3,
        step: 0.1,
        format: (v) => v.toFixed(1),
        visibleWhen: (p) => p.task === 'regress' && p.regLoss === 'huber',
        help: 'Where Huber switches from squared to absolute.',
      },
    ],
  },
  {
    title: 'Network',
    controls: [
      { kind: 'slider', key: 'width', label: 'Width', min: 2, max: 8, step: 1, help: 'Units per hidden layer.' },
      {
        kind: 'slider',
        key: 'initScale',
        label: 'Init scale',
        min: 0.1,
        max: 5,
        step: 0.1,
        format: (v) => v.toFixed(1),
        help: 'Multiplies the initial weights. Big values saturate sigmoids and kill ReLUs.',
      },
      { kind: 'slider', key: 'batchSize', label: 'Batch', min: 4, max: 64, step: 4, help: 'Samples per update.' },
    ],
  },
  {
    title: 'Display',
    collapsedByDefault: true,
    controls: [
      { kind: 'toggle', key: 'showDerivative', label: 'Derivatives', help: 'Draw f′ and dL beside the functions.' },
      { kind: 'toggle', key: 'showAll', label: 'Compare all', help: 'Draw every other activation faintly behind the chosen one.' },
      { kind: 'toggle', key: 'showRug', label: 'Live values', help: 'Mark where the network’s own units and outputs sit on the curve.' },
      { kind: 'toggle', key: 'logScale', label: 'Log scale', help: 'Loss and gradient charts on a log scale, so a vanishing layer and a slow tail stay visible.' },
    ],
  },
];

export const PRESETS: Preset<ActParams>[] = [
  {
    id: 'relu-deep',
    name: 'ReLU, four layers',
    blurb: 'Four ReLU layers on the moons, healthy: the gradient reaches every layer and the boundary bends within a few dozen epochs.',
    params: {},
    autoRun: true,
  },
  {
    id: 'bend',
    name: 'Bend the plane',
    blurb: 'Two tanh layers of two units each, so every hidden space is drawn as it is: the grid bends until a straight line separates the moons.',
    params: { view: 'space', activation: 'tanh', depth: 2, width: 2, learningRate: 0.05 },
    autoRun: true,
  },
  {
    id: 'fold',
    name: 'ReLU folds',
    blurb: 'Four ReLU units on a ring: the dashed grid is the linear part, ReLU clips it at zero and folds the sheet.',
    params: { view: 'space', activation: 'relu', depth: 1, width: 4, pointsDataset: 'circle' },
    autoRun: true,
  },
  {
    id: 'no-activation',
    name: 'No activation',
    blurb: 'Three linear layers: a grid stays a grid, the three matrices multiply out to one, and the boundary is a straight line.',
    params: { view: 'space', activation: 'linear', depth: 3, width: 2 },
    autoRun: true,
  },
  {
    id: 'logistic',
    name: 'Logistic regression',
    blurb: 'Depth zero: one sigmoid unit on the inputs, a single ramp with a straight cut, enough for two blobs.',
    params: { view: 'surface', depth: 0, pointsDataset: 'gaussians' },
    autoRun: true,
  },
  {
    id: 'ridges',
    name: 'Units as ridges',
    blurb: 'Four ReLU units drawn over the input plane, each a crease along w·x + b = 0; the output adds them up.',
    params: { view: 'surface', surfaceOf: 'units', activation: 'relu', depth: 1, width: 4 },
    autoRun: true,
  },
  {
    id: 'sigmoid-deep',
    name: 'Sigmoid, six layers',
    blurb: 'Six sigmoid layers: each one multiplies the gradient by at most 0.25 on the way back, so the first layer barely learns.',
    params: { activation: 'sigmoid', depth: 6, learningRate: 0.01 },
    autoRun: true,
  },
  {
    id: 'tanh-deep',
    name: 'Tanh, six layers',
    blurb: 'The same six layers with tanh: a slope of up to one keeps the gradient alive and the boundary appears within twenty epochs.',
    params: { activation: 'tanh', depth: 6, learningRate: 0.01 },
    autoRun: true,
  },
  {
    id: 'dead-relu',
    name: 'Dead ReLUs',
    blurb: 'Oversized weights and a big rate push most units to negative z for every input, where ReLU is flat: they never recover.',
    params: { initScale: 3, learningRate: 0.3 },
    autoRun: true,
  },
  {
    id: 'leaky-fix',
    name: 'Leaky ReLU',
    blurb: 'The same oversized start with a small slope on the left: negative units keep a gradient, so nothing dies.',
    params: { activation: 'leakyRelu', initScale: 3, learningRate: 0.3 },
    autoRun: true,
  },
  {
    id: 'mse-for-classes',
    name: 'Squared error on classes',
    blurb: 'Squared error scoring a sigmoid output from oversized weights: its gradient vanishes at confidently wrong outputs, so the loss stalls.',
    params: { classLoss: 'mse', depth: 2, initScale: 5 },
    autoRun: true,
  },
  {
    id: 'hinge',
    name: 'Hinge loss',
    blurb: 'The hinge is zero once a point clears the margin: outputs beyond ±1 send nothing back, and training stops when every point is clear.',
    params: { classLoss: 'hinge', depth: 2 },
    autoRun: true,
  },
  {
    id: 'softmax',
    name: 'Softmax, three classes',
    blurb: 'Three classes, one logit each: softmax cross-entropy turns the logits into probabilities and its gradient is p − y per class.',
    params: { pointsDataset: 'three-class', classLoss: 'softmaxCE', depth: 2 },
    autoRun: true,
  },
  {
    id: 'huber-outliers',
    name: 'Huber on outliers',
    blurb: 'A curve through a few wild points: beyond δ Huber caps the pull, so the outliers cannot drag the curve the way squared error lets them.',
    params: { task: 'regress', curveDataset: 'outliers', regLoss: 'huber', depth: 2, activation: 'tanh', learningRate: 0.01 },
    autoRun: true,
  },
  {
    id: 'mae-flat',
    name: 'Absolute error',
    blurb: 'Absolute error pulls every point with the same force: the curve gets near, then jitters, because the gradient never shrinks.',
    params: { task: 'regress', curveDataset: 'sinusoid', regLoss: 'mae', depth: 2, activation: 'tanh', learningRate: 0.01 },
    autoRun: true,
  },
];
