/** Convolutional networks: params, controls and presets. */

import type { ControlGroup, Preset } from '../../explainer/types';
import { fmtKnob } from '../../lib/math/stats';

export interface CnnParams extends Record<string, number | string | boolean> {
  classCount: number;
  count: number;
  jitter: number;
  thickness: number;
  noise: number;
  seed: number;

  model: string;
  filters: number;
  kernel: number;
  pool: string;
  padding: string;
  secondConv: boolean;
  dense: number;

  learningRate: number;
  optimiser: string;
  batchSize: number;

  probe: number;
}

export const DEFAULT_PARAMS: CnnParams = {
  classCount: 4,
  count: 240,
  jitter: 2,
  thickness: 2,
  noise: 0.1,
  seed: 5,

  model: 'cnn',
  filters: 4,
  kernel: 3,
  pool: 'max',
  padding: 'valid',
  secondConv: false,
  dense: 16,

  learningRate: 0.01,
  optimiser: 'adam',
  batchSize: 16,

  probe: 0,
};

export const CONTROL_GROUPS: ControlGroup<keyof CnnParams & string>[] = [
  {
    title: 'Data',
    blurb: 'Sixteen by sixteen glyphs drawn in code.',
    controls: [
      { kind: 'slider', key: 'classCount', label: 'Classes', min: 2, max: 6, step: 1, help: 'Bars, then a diagonal, a cross, a ring and a blob.' },
      { kind: 'slider', key: 'count', label: 'Images', min: 40, max: 400, step: 40, help: 'How many images are drawn; a quarter are held out.' },
      { kind: 'slider', key: 'jitter', label: 'Jitter', min: 0, max: 4, step: 1, help: 'How far a glyph can sit from the centre, in pixels. Shift is what pooling is for.' },
      { kind: 'slider', key: 'thickness', label: 'Thickness', min: 1, max: 3, step: 1, help: 'Stroke width of the glyphs.' },
      { kind: 'slider', key: 'noise', label: 'Noise', min: 0, max: 0.5, step: 0.05, format: (v) => v.toFixed(2), help: 'Pixel noise as a fraction of full brightness.' },
      { kind: 'seed', key: 'seed', label: 'Seed', help: 'Fixes the images and the initial kernels.' },
    ],
  },
  {
    title: 'Model',
    zone: 'top',
    controls: [
      {
        kind: 'select',
        key: 'model',
        label: 'Model',
        help: 'A convolutional stack, or a dense layer straight on the pixels for comparison.',
        options: [
          { value: 'cnn', label: 'Convolutional', hint: 'Kernels slide over the image, then pool, then a dense head.' },
          { value: 'dense', label: 'Dense only', hint: 'Every pixel wired to every hidden unit: more parameters, no notion of nearby.' },
        ],
      },
      {
        kind: 'stepper',
        key: 'filters',
        label: 'Kernels',
        min: 1,
        max: 8,
        step: 1,
        visibleWhen: (p) => p.model === 'cnn',
        help: 'How many kernels the first layer learns. Each one becomes a feature map.',
      },
      {
        kind: 'select',
        key: 'kernel',
        label: 'Kernel',
        visibleWhen: (p) => p.model === 'cnn',
        help: 'Side of the square window each kernel looks through.',
        options: [
          { value: '3', label: '3 × 3' },
          { value: '5', label: '5 × 5' },
        ],
      },
      {
        kind: 'select',
        key: 'pool',
        label: 'Pooling',
        visibleWhen: (p) => p.model === 'cnn',
        help: 'How each 2 × 2 block of a feature map is summarised.',
        options: [
          { value: 'max', label: 'Max', hint: 'Keeps the strongest response: where exactly no longer matters.' },
          { value: 'avg', label: 'Average', hint: 'Blurs the block into its mean.' },
          { value: 'none', label: 'None', hint: 'Keeps every position, and every shift looks like a new pattern.' },
        ],
      },
      {
        kind: 'select',
        key: 'padding',
        advanced: true,
        label: 'Padding',
        visibleWhen: (p) => p.model === 'cnn',
        help: 'Valid keeps the kernel inside the image, so a map is smaller than its input. Same pads the border with zeros and keeps the size.',
        options: [
          { value: 'valid', label: 'Valid' },
          { value: 'same', label: 'Same' },
        ],
      },
      {
        kind: 'toggle',
        key: 'secondConv',
        advanced: true,
        label: 'Second conv',
        visibleWhen: (p) => p.model === 'cnn',
        help: 'Stack a second convolution on the pooled maps, so kernels see combinations of strokes.',
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
        max: 1,
        step: 0.0001,
        scale: 'log',
        format: fmtKnob,
        help: 'The step size of every update.',
      },
    ],
  },
  {
    title: 'Head and updates',
    controls: [
      { kind: 'slider', key: 'dense', label: 'Dense units', min: 0, max: 32, step: 4, help: 'Hidden units between the maps and the class scores. Zero goes straight to the scores.' },
      { kind: 'slider', key: 'batchSize', label: 'Batch', min: 4, max: 64, step: 4, help: 'Images per update.' },
      {
        kind: 'select',
        key: 'optimiser',
        label: 'Optimiser',
        help: 'How the gradient becomes a step.',
        options: [
          { value: 'adam', label: 'Adam' },
          { value: 'momentum', label: 'Momentum' },
          { value: 'gd', label: 'Plain SGD' },
        ],
      },
    ],
  },
  {
    title: 'Probe',
    blurb: 'The held-out image the diagram follows. Click one in the gallery instead.',
    controls: [
      {
        kind: 'slider',
        key: 'probe',
        label: 'Test image',
        min: 0,
        max: 99,
        step: 1,
        format: (v) => String(v + 1),
        help: 'Which held-out image the diagram follows, counted from 1; past the last one it stays on the last.',
      },
    ],
  },
];

export const PRESETS: Preset<CnnParams>[] = [
  {
    id: 'edges',
    name: 'Four kernels',
    blurb: 'Four 3 × 3 kernels on four glyph classes. Watch the kernels settle into stroke detectors, the feature maps light up along the strokes, and the held-out accuracy climb within a few dozen batches.',
    params: {},
    autoRun: true,
  },
  {
    id: 'one-filter',
    name: 'One kernel',
    blurb: 'A single kernel has to serve four shapes from one feature map. It gets there, but takes more batches than four kernels, and the kernel it learns is a compromise rather than a clean stroke detector.',
    params: { filters: 1 },
    autoRun: true,
  },
  {
    id: 'max-pool',
    name: 'Max pooling, shifted glyphs',
    blurb: 'Glyphs shifted by up to three pixels with max pooling. Each pooled cell keeps the strongest of four, so a stroke one pixel over gives the same answer.',
    params: { jitter: 3 },
    autoRun: true,
  },
  {
    id: 'no-pool',
    name: 'No pooling',
    blurb: 'Glyphs shifted by up to three pixels and no pooling. The maps keep every position, so the head has four times the inputs and every shift of a stroke is a pattern of its own. Compare the parameter count and the early batches with max pooling.',
    params: { pool: 'none', jitter: 3 },
    autoRun: true,
  },
  {
    id: 'dense-only',
    name: 'Dense only',
    blurb: 'A dense layer straight on the 256 pixels. Every pixel has its own weight, so a bar two pixels to the left is a different input: with shifted glyphs and few images it memorises the training set while the held-out accuracy lags. Switch the model back to compare.',
    params: { model: 'dense', jitter: 4, count: 120 },
    autoRun: true,
  },
  {
    id: 'two-conv',
    name: 'Two convolutions',
    blurb: 'A second convolution on the pooled maps sees combinations of first-layer strokes: a cross is a horizontal and a vertical response in the same place.',
    params: { secondConv: true, classCount: 6 },
    autoRun: true,
  },
  {
    id: 'noisy',
    name: 'Noisy pixels',
    blurb: 'Heavy pixel noise. The kernels average it out, because a 3 × 3 sum of noise is small next to a stroke, and pooling drops what is left. Push the noise higher still and watch where it breaks.',
    params: { noise: 0.3 },
    autoRun: true,
  },
];
