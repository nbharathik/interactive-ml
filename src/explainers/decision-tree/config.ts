/** Decision trees: params, controls and presets. */

import type { ControlGroup, Preset } from '../../explainer/types';
import { CLASSIFICATION_DATASETS } from '../../lib/datasets/points';

export interface TreeParams extends Record<string, number | string | boolean> {
  dataset: string;
  sampleCount: number;
  noise: number;
  classCount: number;
  seed: number;
  trainFraction: number;

  criterion: string;
  maxDepth: number;
  minSamplesSplit: number;
  minSamplesLeaf: number;
  minImpurityDecrease: number;
  ccpAlpha: number;

  showRegions: boolean;
  showSplitLines: boolean;
  showTestPoints: boolean;
  markMistakes: boolean;
}

export const DEFAULT_PARAMS: TreeParams = {
  dataset: 'gaussians',
  sampleCount: 200,
  noise: 0.5,
  classCount: 3,
  seed: 42,
  trainFraction: 0.7,

  criterion: 'gini',
  maxDepth: 4,
  minSamplesSplit: 4,
  minSamplesLeaf: 1,
  minImpurityDecrease: 0,
  ccpAlpha: 0,

  showRegions: true,
  showSplitLines: true,
  showTestPoints: true,
  markMistakes: false,
};

export const CONTROL_GROUPS: ControlGroup<keyof TreeParams & string>[] = [
  {
    title: 'Data',
    blurb: 'The two features are the two axes. Every question the tree asks is about one of them.',
    controls: [
      {
        kind: 'select',
        key: 'dataset',
        label: 'Dataset',
        help: 'The blobs overlap on purpose. A cleanly separated pair would be finished after one split, and then there would be nothing to watch.',
        options: CLASSIFICATION_DATASETS.map((d) => ({
          value: d.id,
          label: d.name,
          hint: d.blurb,
        })),
      },
      {
        kind: 'slider',
        key: 'sampleCount',
        label: 'Points',
        min: 40,
        max: 400,
        step: 10,
        help: 'More points means more candidate thresholds to score at every node, training cost grows with n log n per split.',
      },
      {
        kind: 'slider',
        key: 'noise',
        label: 'Noise',
        min: 0,
        max: 1,
        step: 0.05,
        format: (v) => v.toFixed(2),
        help: 'How far the classes bleed into each other. Noise is what a deep tree ends up memorising.',
      },
      {
        kind: 'segmented',
        key: 'classCount',
        label: 'Classes',
        options: [
          { value: '3', label: '3' },
          { value: '4', label: '4' },
        ],
        visibleWhen: (p) => p.dataset === 'three-class',
        help: 'Trees handle more than two classes without any change to the algorithm: a leaf just votes for whichever class it holds most of.',
      },
      {
        kind: 'slider',
        key: 'trainFraction',
        label: 'Train split',
        min: 0.3,
        max: 1,
        step: 0.05,
        format: (v) => Math.round(v * 100) + '%',
        help: 'The rest is held out. Training accuracy alone cannot tell you a tree has overfitted; the held-out number is the one that falls.',
      },
      { kind: 'seed', key: 'seed', label: 'Seed', help: 'Same seed, same points, same tree.' },
    ],
  },
  {
    title: 'Growth',
    zone: 'top',
    blurb: 'When may the tree ask another question?',
    controls: [
      {
        kind: 'segmented',
        key: 'criterion',
        label: 'Impurity',
        help: 'How a node measures "how mixed am I". Both are zero for a pure node and largest for an even mix.',
        options: [
          { value: 'gini', label: 'Gini', hint: '1 − Σp². No logarithm, so it is marginally cheaper. Scikit-learn’s default.' },
          { value: 'entropy', label: 'Entropy', hint: '−Σp log₂p, measured in bits. Almost always picks the same split.' },
        ],
      },
      {
        kind: 'stepper',
        key: 'maxDepth',
        label: 'Max depth',
        min: 1,
        max: 10,
        step: 1,
        help: 'The hardest stop of the four. Depth 1 is a single question; depth 10 can carve out a rectangle around one point.',
      },
      {
        kind: 'stepper',
        key: 'minSamplesSplit',
        advanced: true,
        label: 'Min samples to split',
        min: 2,
        max: 40,
        step: 1,
        help: 'A node holding fewer points than this becomes a leaf, however mixed it still is.',
      },
      {
        kind: 'stepper',
        key: 'minSamplesLeaf',
        advanced: true,
        label: 'Min samples per leaf',
        min: 1,
        max: 20,
        step: 1,
        help: 'Rejects any split that would leave a child smaller than this. At 1 the tree is free to fence off single points.',
      },
      {
        kind: 'stepper',
        key: 'minImpurityDecrease',
        advanced: true,
        label: 'Min gain',
        min: 0,
        max: 0.1,
        step: 0.005,
        format: (v) => v.toFixed(3),
        help: 'Refuses a split that buys less impurity than this. Set it above the best available gain and the tree stops immediately.',
      },
    ],
  },
  {
    title: 'Pruning',
    zone: 'top',
    blurb: 'Grow first, then take back the questions that did not pay for themselves.',
    controls: [
      {
        kind: 'stepper',
        key: 'ccpAlpha',
        label: 'Pruning strength (α)',
        min: 0,
        max: 0.05,
        step: 0.001,
        format: (v) => (v === 0 ? 'off' : v.toFixed(3)),
        help: 'Charges every leaf a rent of α. A subtree that does not reduce impurity by more than the rent it costs is collapsed back into one leaf.',
      },
    ],
  },
  {
    title: 'Display',
    collapsedByDefault: true,
    controls: [
      {
        kind: 'toggle',
        key: 'showRegions',
        label: 'Leaf regions',
        help: 'Paints each leaf’s rectangle in the colour it predicts. The whole model is those rectangles.',
      },
      {
        kind: 'toggle',
        key: 'showSplitLines',
        label: 'Split lines',
        help: 'Every cut the tree has made, each one clipped to the region its node owns.',
      },
      {
        kind: 'toggle',
        key: 'showTestPoints',
        label: 'Held-out points',
        help: 'Drawn at half opacity. The tree has never seen them; they only score it.',
      },
      {
        kind: 'toggle',
        key: 'markMistakes',
        label: 'Mark mistakes',
        help: 'Rings every point whose leaf votes for the wrong class. On a deep tree the training mistakes disappear and the held-out ones do not.',
      },
    ],
  },
];

export const PRESETS: Preset<TreeParams>[] = [
  {
    id: 'watch-it-grow',
    name: 'Watch it grow',
    blurb: 'Two overlapping blobs, depth 4. Each step takes the biggest bite left.',
    params: { dataset: 'gaussians', maxDepth: 4, criterion: 'gini', noise: 0.5, sampleCount: 200 },
    autoRun: true,
  },
  {
    id: 'stump',
    name: 'A single question',
    blurb: 'Depth 1. Two leaves, and a held-out score the depth-4 tree does not beat.',
    params: { dataset: 'gaussians', maxDepth: 1, noise: 0.5, sampleCount: 200 },
    autoRun: true,
  },
  {
    id: 'staircase',
    name: 'The staircase',
    blurb: 'Two moons, depth 8. Thirteen rectangles tracing one curve.',
    params: { dataset: 'moons', maxDepth: 8, noise: 0.35, sampleCount: 240 },
    autoRun: true,
  },
  {
    id: 'diagonal',
    name: 'Diagonal stripes',
    blurb: 'The worst case for a tree: 27 splits to approximate lines it cannot draw.',
    params: { dataset: 'stripes', maxDepth: 8, noise: 0.15, sampleCount: 260 },
    autoRun: true,
  },
  {
    id: 'memorise',
    name: 'Memorising the noise',
    blurb: 'Depth 10, leaves of one. Training accuracy reaches 100% and held-out falls to 85%.',
    params: {
      dataset: 'gaussians',
      maxDepth: 10,
      noise: 0.6,
      sampleCount: 260,
      minSamplesLeaf: 1,
      minSamplesSplit: 2,
      ccpAlpha: 0,
      markMistakes: true,
    },
    autoRun: true,
  },
  {
    id: 'xor',
    name: 'XOR needs patience',
    blurb: 'The first cut makes held-out accuracy worse. Five splits later the tree is perfect.',
    params: { dataset: 'xor', maxDepth: 3, noise: 0.1, sampleCount: 220 },
    autoRun: true,
  },
  {
    id: 'starved',
    name: 'Starved of samples',
    blurb: 'Min 20 per leaf. Growth stops at depth 3 with the leaves still visibly mixed.',
    params: { dataset: 'gaussians', maxDepth: 10, minSamplesLeaf: 20, minSamplesSplit: 40, noise: 0.5 },
    autoRun: true,
  },
];
