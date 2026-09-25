/** Neural network: params, controls and presets. `hiddenLayers` is a string ("4,3") for the URL. */

import type { ControlGroup, Preset } from '../../explainer/types';
import { CLASSIFICATION_DATASETS } from '../../lib/datasets/points';
import { fmtKnob } from '../../lib/math/stats';

export interface NeuralNetParams extends Record<string, number | string | boolean> {
  dataset: string;
  sampleCount: number;
  noise: number;
  testFraction: number;
  seed: number;

  /** Hidden layer widths, comma separated. "" means no hidden layer at all. */
  hiddenLayers: string;

  featX1: boolean;
  featX2: boolean;
  featX1sq: boolean;
  featX2sq: boolean;
  featX1x2: boolean;
  featSinX1: boolean;
  featSinX2: boolean;

  activation: string;
  learningRate: number;
  batchSize: number;
  initialiser: string;
  regularisation: string;
  regRate: number;

  showTest: boolean;
  showThumbnails: boolean;
  bands: boolean;
}

/** Hard limits, mirrored by the layer editor's disabled states. */
export const MAX_LAYERS = 4;
export const MAX_UNITS = 8;

export const DEFAULT_PARAMS: NeuralNetParams = {
  dataset: 'circle',
  sampleCount: 200,
  noise: 0.2,
  testFraction: 0.3,
  seed: 42,

  hiddenLayers: '4',

  featX1: true,
  featX2: true,
  featX1sq: false,
  featX2sq: false,
  featX1x2: false,
  featSinX1: false,
  featSinX2: false,

  activation: 'tanh',
  learningRate: 0.03,
  batchSize: 10,
  initialiser: 'xavier',
  regularisation: 'none',
  regRate: 0.1,

  showTest: true,
  showThumbnails: true,
  bands: false,
};

/* ---------------- encoding helpers ---------------- */

/** "4,3" to [4, 3], repairing anything a hand-edited URL might contain. */
export function parseLayers(encoded: string): number[] {
  if (!encoded) return [];
  return encoded
    .split(',')
    .map((part) => Math.round(Number(part.trim())))
    .filter((n) => Number.isFinite(n) && n > 0)
    .map((n) => Math.min(MAX_UNITS, n))
    .slice(0, MAX_LAYERS);
}

export function formatLayers(layers: readonly number[]): string {
  return layers.join(',');
}

/** "2 → 4 → 1", the notation used throughout the page. */
export function architectureLabel(inputCount: number, hidden: readonly number[]): string {
  return [inputCount, ...hidden, 1].join(' → ');
}

/* ---------------- controls ---------------- */

// A single ±1 output cannot express three classes.
const BINARY_DATASETS = CLASSIFICATION_DATASETS.filter((d) => d.id !== 'three-class');

export const CONTROL_GROUPS: ControlGroup<keyof NeuralNetParams & string>[] = [
  {
    title: 'Data',
    blurb: 'How hard a shape the network is being asked to draw.',
    controls: [
      {
        kind: 'select',
        key: 'dataset',
        label: 'Dataset',
        help: 'Two blobs needs no hidden layer at all. The ring needs a few units. The spiral needs real capacity and patience.',
        options: BINARY_DATASETS.map((d) => ({ value: d.id, label: d.name, hint: d.blurb })),
      },
      {
        kind: 'slider',
        key: 'sampleCount',
        label: 'Points',
        min: 30,
        max: 400,
        step: 10,
        help: 'More points make the held-out score trustworthy and make memorising the training set harder.',
      },
      {
        kind: 'slider',
        key: 'noise',
        label: 'Noise',
        min: 0,
        max: 1,
        step: 0.05,
        format: (v) => v.toFixed(2),
        help: 'How far points stray from their true region. Noise is what a large network memorises when you let it.',
      },
      {
        kind: 'slider',
        key: 'testFraction',
        label: 'Held out',
        min: 0,
        max: 0.6,
        step: 0.05,
        format: (v) => Math.round(v * 100) + '%',
        help: 'These points are drawn ringed and never appear in a gradient. The gap between the two loss curves is the whole reason they exist.',
      },
      {
        kind: 'seed',
        key: 'seed',
        label: 'Seed',
        help: 'Sets both the data and the starting weights. Two seeds on the same architecture can land in different solutions, try it on the spiral.',
      },
    ],
  },
  {
    title: 'Input features',
    blurb: 'What each input node is fed. Everything else is discovered.',
    controls: [
      {
        kind: 'toggle',
        key: 'featX1',
        label: 'x₁',
        help: 'The horizontal coordinate. With x₁ and x₂ alone the network must invent every curve it needs.',
      },
      { kind: 'toggle', key: 'featX2', label: 'x₂', help: 'The vertical coordinate.' },
      {
        kind: 'toggle',
        key: 'featX1sq',
        label: 'x₁²',
        help: 'A hand-built curve. Turning this on lets even a hidden-layer-free model bend vertically.',
      },
      { kind: 'toggle', key: 'featX2sq', label: 'x₂²', help: 'The same trick on the other axis. x₁² and x₂² together give you circles.' },
      {
        kind: 'toggle',
        key: 'featX1x2',
        label: 'x₁x₂',
        help: 'The product. It is positive in two opposite quadrants and negative in the other two, which is exactly XOR, handed over for free.',
      },
      { kind: 'toggle', key: 'featSinX1', label: 'sin x₁', help: 'A periodic feature. Useful on the stripes; misleading almost everywhere else.' },
      { kind: 'toggle', key: 'featSinX2', label: 'sin x₂', help: 'The periodic feature on the vertical axis.' },
    ],
  },
  {
    title: 'Model',
    zone: 'top',
    blurb: 'Layer widths are edited above the diagram.',
    controls: [
      {
        kind: 'segmented',
        key: 'activation',
        label: 'Activation',
        help: 'The squash applied to every hidden unit. This is the only non-linear step in the whole network.',
        options: [
          { value: 'tanh', label: 'Tanh', hint: 'Smooth, zero-centred, output in (−1, 1). The safe default here.' },
          { value: 'relu', label: 'ReLU', hint: 'max(0, z). Fast and non-saturating, but a unit pushed fully negative stops learning.' },
          { value: 'sigmoid', label: 'Sigmoid', hint: 'Output in (0, 1). Saturates hard, so gradients through deep stacks get tiny.' },
          { value: 'linear', label: 'None', hint: 'No squash at all. Stacking layers now buys nothing, the whole network collapses to a line.' },
        ],
      },
      {
        kind: 'select',
        key: 'initialiser',
        advanced: true,
        label: 'Initial weights',
        help: 'Where training starts. Symmetry has to be broken by the initialiser, because gradient descent cannot break it on its own.',
        options: [
          { value: 'xavier', label: 'Xavier', hint: 'Uniform, scaled by √(6/(fan-in + fan-out)). Keeps signal size steady across layers.' },
          { value: 'he', label: 'He', hint: 'Gaussian, scaled by √(2/fan-in). Built for ReLU, which throws half the signal away.' },
          { value: 'small', label: 'Tiny', hint: 'Uniform in ±0.05. Barely breaks symmetry, so early progress is slow.' },
          { value: 'large', label: 'Huge', hint: 'Uniform in ±3. Saturates tanh and sigmoid immediately; gradients vanish.' },
          { value: 'zeros', label: 'All zeros', hint: 'Every unit identical forever. The clearest possible demonstration of why symmetry breaking matters.' },
        ],
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
        label: 'Learning rate (η)',
        min: 0.0001,
        max: 3,
        step: 0.0001,
        scale: 'log',
        format: fmtKnob,
        help: 'How far each weight moves per batch. Networks tolerate a much narrower band than a single line does.',
      },
      {
        kind: 'stepper',
        key: 'batchSize',
        advanced: true,
        label: 'Batch size',
        min: 1,
        max: 50,
        step: 1,
        help: 'Points per update. One epoch is a full sweep, so a small batch means many small updates per epoch.',
      },
      {
        kind: 'segmented',
        key: 'regularisation',
        label: 'Regularisation',
        help: 'A penalty added to every gradient, pulling weights towards zero. It is the main lever against memorising the training points.',
        options: [
          { value: 'none', label: 'None', hint: 'Nothing stops a weight from growing as large as the data allows.' },
          { value: 'l2', label: 'L2', hint: 'Adds λw to the gradient. Shrinks every weight in proportion to its size.' },
          { value: 'l1', label: 'L1', hint: 'Adds λ·sign(w). Drives small weights to hover near zero, pruning connections.' },
        ],
      },
      {
        kind: 'stepper',
        key: 'regRate',
        label: 'L2/L1 penalty (λ)',
        min: 0.001,
        max: 1,
        step: 0.0001,
        scale: 'log',
        format: fmtKnob,
        visibleWhen: (p) => p.regularisation !== 'none',
        help: 'Too small and nothing changes; too large and every weight is dragged to zero and the boundary flattens out. Useful values here are larger than you might expect, because the penalty competes with a gradient summed over the whole batch.',
      },
    ],
  },
  {
    title: 'Display',
    collapsedByDefault: true,
    controls: [
      {
        kind: 'toggle',
        key: 'showThumbnails',
        label: 'Node thumbnails',
        help: 'Paints each unit’s own output over the input space inside its node. Turn it off to see the raw graph.',
      },
      {
        kind: 'toggle',
        key: 'showTest',
        label: 'Show held-out points',
        help: 'The ringed points. They are scored every epoch but never trained on.',
      },
      {
        kind: 'toggle',
        key: 'bands',
        label: 'Banded boundary',
        help: 'Quantises the output field into steps, which makes the contours of confidence easier to count.',
      },
    ],
  },
];

/* ---------------- presets ---------------- */

export const PRESETS: Preset<NeuralNetParams>[] = [
  {
    id: 'ring-four',
    name: 'Four units wrap the ring',
    blurb: 'Watch four half-planes close in around the inner disc, one thumbnail each.',
    params: { dataset: 'circle', hiddenLayers: '4', activation: 'tanh' },
    autoRun: true,
  },
  {
    id: 'no-hidden',
    name: 'No hidden layer',
    blurb: 'Take the hidden layer away: the boundary is a line, and the ring defeats it.',
    params: { dataset: 'circle', hiddenLayers: '', learningRate: 0.03 },
    autoRun: true,
  },
  {
    id: 'xor-linear',
    name: 'XOR without a hidden layer',
    blurb: 'Accuracy sticks near half. No straight line separates alternating quadrants.',
    params: { dataset: 'xor', hiddenLayers: '', featX1x2: false, learningRate: 0.03 },
    autoRun: true,
  },
  {
    id: 'xor-feature',
    name: 'XOR, handed the x₁x₂ feature',
    blurb: 'Same model, one extra input. Solved almost immediately; the feature did the work.',
    params: { dataset: 'xor', hiddenLayers: '', featX1x2: true, learningRate: 0.03 },
    autoRun: true,
  },
  {
    id: 'xor-hidden',
    name: 'XOR, discovered by a hidden layer',
    blurb: 'Only x₁ and x₂ go in. Four units find something equivalent to the product on their own.',
    params: { dataset: 'xor', hiddenLayers: '4', featX1x2: false, learningRate: 0.03 },
    autoRun: true,
  },
  {
    id: 'spiral-small',
    name: 'Spiral with two units',
    blurb: 'Two half-planes cannot follow a spiral. It plateaus well short of the answer.',
    params: { dataset: 'spiral', hiddenLayers: '2', sampleCount: 300, noise: 0.1, learningRate: 0.03 },
    autoRun: true,
  },
  {
    id: 'spiral-deep',
    name: 'Spiral with room to work',
    blurb: 'Two layers of eight. The arms come apart, but it takes several hundred epochs.',
    params: { dataset: 'spiral', hiddenLayers: '8,8', sampleCount: 300, noise: 0.1, learningRate: 0.03 },
    autoRun: true,
  },
  {
    id: 'dead-relu',
    name: 'ReLU units that die',
    blurb: 'Sixteen ReLU units at a step size of 0.5. About a third of the thumbnails go blank and never recover.',
    params: {
      dataset: 'spiral',
      sampleCount: 300,
      noise: 0.1,
      hiddenLayers: '8,8',
      activation: 'relu',
      learningRate: 0.5,
    },
    autoRun: true,
  },
  {
    id: 'zeros',
    name: 'Every weight starts at zero',
    blurb: 'All four units compute the same thing forever, because they all get the same gradient.',
    params: { dataset: 'circle', hiddenLayers: '4', initialiser: 'zeros', learningRate: 0.05 },
    autoRun: true,
  },
  {
    id: 'too-fast',
    name: 'A step size that is too large',
    blurb: 'The loss climbs about as often as it falls, then a weight overflows around epoch seventy.',
    params: { dataset: 'circle', hiddenLayers: '4', learningRate: 1 },
    autoRun: true,
  },
  {
    id: 'overfit',
    name: 'Memorising fifty noisy points',
    blurb: 'Twenty-four units, fifty training points, heavy noise. Train loss keeps falling; test loss turns back up.',
    params: {
      dataset: 'circle',
      hiddenLayers: '8,8,8',
      sampleCount: 100,
      noise: 0.55,
      testFraction: 0.5,
      learningRate: 0.03,
      regularisation: 'none',
    },
  },
  {
    id: 'overfit-regularised',
    name: 'The same net, with L2',
    blurb: 'Identical data and architecture, λ = 0.3. The band between the two loss curves closes to a third of its width.',
    params: {
      dataset: 'circle',
      hiddenLayers: '8,8,8',
      sampleCount: 100,
      noise: 0.55,
      testFraction: 0.5,
      learningRate: 0.03,
      regularisation: 'l2',
      regRate: 0.3,
    },
  },
];
