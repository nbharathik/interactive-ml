/** Convolutional network lessons, one screen each. Pure data, driven against the real model in tests/lessons-cnn.test.ts. */

import type { Lesson, LessonAction, LessonContextBase, LiveNumber } from '../../explainer/lessons';
import type { ImageDataset } from '../../lib/datasets/images';
import { fmt, fmtPercent } from '../../lib/math/stats';
import { createCnn, parameterCount } from '../../lib/ml/conv';
import type { CnnCache, CnnSpec, CnnState } from '../../lib/ml/conv';
import type { CnnParams } from './config';
import { NODE } from './scene';

/* ---------------- context ---------------- */

export type ArchView = '2d' | '3d' | 'window';
export type SecondView = 'curves' | 'accuracy' | 'confusion' | 'filters';

export interface CnnLessonContext extends LessonContextBase {
  params: CnnParams;
  state: CnnState;
  derived: {
    paramCount: number;
    /** Weights and biases in the convolutional layers alone. */
    convParams: number;
    /** Weights and biases from the flattened vector to the scores. */
    headParams: number;
    /** What the convolutional model would hold with the same knobs, whatever the model setting. */
    cnnParamCount: number;
    kernels: number;
    kernel: number;
    /** Side of a first-layer map, and of its pooled map. */
    mapSize: number;
    pooledSize: number;
    positions: number;
    flatSize: number;
    denseUnits: number;
    classCount: number;
    trainCount: number;
    testCount: number;
    gap: number;
    fitted: boolean;
    probe: { className: string; predictedName: string; p: number; correct: boolean };
    /** First-layer maps that are zero everywhere for the probe, and dense units at zero. */
    deadMaps: number;
    deadUnits: number;
  };
  ui: { view: ArchView; secondView: SecondView; open: string | null };
}

export interface CnnContextInput {
  params: CnnParams;
  state: CnnState;
  data: ImageDataset;
  spec: CnnSpec;
  cache: CnnCache;
  label: number;
  ui: CnnLessonContext['ui'];
}

export const FITTED = 0.98;

export function isFitted(state: CnnState): boolean {
  return state.step > 20 && state.testAccuracy >= FITTED && state.trainAccuracy >= FITTED;
}

/** The page's share of the lesson context; the hook adds `sim` and `lesson`. */
export function makeCnnContext({ params, state, data, spec, cache, label, ui }: CnnContextInput): Omit<CnnLessonContext, keyof LessonContextBase> {
  const model = state.params;
  let convParams = 0;
  for (const c of model.convs) convParams += c.W.length + c.b.length;
  const paramCount = parameterCount(model);
  const first = cache.stages[0];
  let deadMaps = 0;
  if (first) {
    for (let c = 0; c < first.a.c; c++) {
      let any = false;
      for (let i = 0; i < first.a.h * first.a.w; i++) if (first.a.data[c * first.a.h * first.a.w + i] > 0) any = true;
      if (!any) deadMaps++;
    }
  }
  let deadUnits = 0;
  if (cache.hiddenA) for (let u = 0; u < cache.hiddenA.length; u++) if (cache.hiddenA[u] === 0) deadUnits++;
  return {
    params,
    state,
    derived: {
      paramCount,
      convParams,
      headParams: paramCount - convParams,
      cnnParamCount: spec.model === 'cnn' ? paramCount : parameterCount(createCnn({ ...spec, model: 'cnn' })),
      kernels: model.convs[0]?.outC ?? 0,
      kernel: model.convs[0]?.kernel ?? 0,
      mapSize: first?.a.h ?? 0,
      pooledSize: first?.pooled.h ?? 0,
      positions: first ? first.a.h * first.a.w : 0,
      flatSize: cache.flat.length,
      denseUnits: model.hidden?.outSize ?? 0,
      classCount: data.classNames.length,
      trainCount: data.trainIndex.length,
      testCount: data.testIndex.length,
      gap: state.trainAccuracy - state.testAccuracy,
      fitted: isFitted(state),
      probe: {
        className: data.classNames[label] ?? 'glyph',
        predictedName: data.classNames[cache.predicted] ?? '',
        p: cache.probs[label] ?? 0,
        correct: cache.predicted === label,
      },
      deadMaps,
      deadUnits,
    },
    ui,
  };
}

/* ---------------- copy helpers ---------------- */

type Ctx = CnnLessonContext;
type L = Lesson<Ctx, CnnParams>;

const pc = (v: number) => fmtPercent(v, 0);
const n = (v: number) => v.toLocaleString('en-US');
const batch = (c: Ctx) => 'Batch ' + n(c.state.step);
const done = (c: Ctx) => c.derived.fitted || c.state.diverged;
const at90 = (c: Ctx) => c.state.testAccuracy >= 0.9 || c.state.diverged;

/** Batches the reference runs take, checked by the tests. */
export const REFERENCE = {
  /** Four 3 × 3 kernels on the default glyphs: fitted at this batch. */
  fourKernelsFit: 35,
  /** One kernel: 90% held-out at this batch. */
  oneKernelTo90: 35,
  /** Max pooling on glyphs shifted by 3: 90% at this batch. */
  maxPoolTo90: 24,
  /** Two convolutions on six classes after 300 batches, and one convolution on the same classes. */
  twoConvSixClasses: { params: 562, test: 0.97 },
  oneConvSixClasses: { params: 3294, test: 0.93 },
};

/* ---------------- the lessons ---------------- */

type Act = LessonAction<CnnParams>;

const WINDOW: Act = { type: 'view', id: 'arch', value: 'window' };
const FLAT: Act = { type: 'view', id: 'arch', value: '2d' };
const PLAY: Act = { type: 'play' };
/** Opens the lesson on the trained network. */
const TRAIN: Act = { type: 'runTo', steps: 300 };
const NOW = () => true;

/** A run restates every knob it depends on: the preset, then its own changes, then a fresh start. */
const on = (preset: string, patch: Partial<CnnParams> | null, ...rest: Act[]): Act[] => [
  { type: 'preset', id: preset },
  ...(patch ? [{ type: 'params' as const, patch }] : []),
  { type: 'reset' },
  ...rest,
];

const numbers = (c: Ctx): LiveNumber[] => [
  { label: 'batch', value: n(c.state.step) },
  { label: 'held out', value: pc(c.state.testAccuracy) },
  { label: 'parameters', value: n(c.derived.paramCount) },
];

const convolution: L = {
  id: 'convolution',
  title: 'Convolution',
  hook: 'Nine weights, slid over every pixel.',
  section: 'hook',
  presetId: 'edges',
  steps: [
    {
      id: 'kernel',
      kind: 'sandbox',
      say: 'The image is 16 × 16 numbers, 0 dark and 1 bright. A kernel is 9 weights on a 3 × 3 patch: multiply each pixel by the weight above it, add the products and a bias, and that is one pixel of the map m₁. It slides and repeats, 196 times. Try it: Watch it train.',
      takeaway: 'Convolution = the same 9 weights, applied at every position. Training learns the kernels; the sliding never changes.',
      focus: { kind: 'panel', id: 'architecture', label: 'The window' },
      enter: [{ type: 'preset', id: 'edges' }, { type: 'reset' }, WINDOW, TRAIN],
      numbers,
      experiments: [
        {
          label: 'Watch it train',
          say: 'The kernels start random; the maps light up along the strokes.',
          enter: on('edges', null, FLAT, PLAY),
          speed: 'normal',
          until: (c) => done(c) || c.state.step >= 300,
          then: (c) =>
            batch(c) + ': held-out accuracy ' + pc(c.state.testAccuracy) + '. Only numbers changed: the ' + n(c.derived.convParams) + ' in the kernels and the ' + n(c.derived.headParams) +
            ' in the head. Each kernel became a small stroke detector.',
          focus: { kind: 'node', id: NODE.filter(0, 0), label: 'Kernel k₁' },
        },
        {
          label: 'Slide the window',
          say: 'The trained kernel, position by position.',
          enter: [WINDOW],
          until: NOW,
          then: () => 'Patch ⊙ kernel = products, summed, plus the bias, through ReLU: one map pixel. Hover the map to move the window.',
          focus: { kind: 'panel', id: 'architecture', label: 'The window' },
        },
      ],
    },
  ],
};

const receptiveField: L = {
  id: 'receptive-field',
  title: 'Receptive field and parameter sharing',
  hook: 'What one map pixel can see.',
  section: 'core',
  presetId: 'edges',
  knobs: ['kernel', 'padding'],
  steps: [
    {
      id: 'field',
      kind: 'sandbox',
      say: 'Each map pixel is computed from one 3 × 3 patch of the image, the outlined square: its receptive field, all it can see. When the pixel moves a step its patch moves too, and the kernel never changes: 196 pixels share 9 weights and 1 bias. Try it: Kernel 5 × 5.',
      takeaway: 'Sharing keeps the layer small and finds the same stroke anywhere. Kernel size sets the field; padding decides what happens at the border.',
      focus: { kind: 'panel', id: 'architecture', label: 'The patch and its pixel' },
      enter: [{ type: 'preset', id: 'edges' }, { type: 'reset' }, WINDOW, TRAIN],
      numbers: (c) => [{ label: 'kernel', value: c.params.kernel + ' × ' + c.params.kernel }, { label: 'map', value: c.derived.mapSize + ' × ' + c.derived.mapSize }, { label: 'positions', value: n(c.derived.positions) }],
      experiments: [
        {
          label: 'Kernel 5 × 5, same padding',
          say: 'A wider patch, and a ring of zeros around the image.',
          enter: on('edges', { kernel: 5, padding: 'same' }, WINDOW, TRAIN),
          until: NOW,
          then: (c) =>
            'Each pixel now sees a 25 pixel patch, and the zeros (the dashed border) let the window sit on the edge pixels too, so the map stays ' + c.derived.mapSize + ' × ' + c.derived.mapSize + ' instead of shrinking.',
          focus: { kind: 'panel', id: 'architecture', label: 'A 5 × 5 patch on the padded image' },
        },
        {
          label: 'Kernel 3 × 3 again',
          say: 'Nine weights, no padding.',
          enter: on('edges', null, WINDOW, TRAIN),
          until: NOW,
          then: (c) => 'A 3 × 3 patch and a ' + c.derived.mapSize + ' × ' + c.derived.mapSize + ' map: ' + n(c.derived.positions) + ' positions share 10 parameters. A dense unit reading the same image would need 257.',
          focus: { kind: 'panel', id: 'architecture', label: 'The kernel' },
        },
      ],
    },
  ],
};

const pooling: L = {
  id: 'pooling',
  title: 'Pooling',
  hook: 'Keep what was found, forget exactly where.',
  section: 'core',
  presetId: 'max-pool',
  knobs: ['pool'],
  steps: [
    {
      id: 'max',
      kind: 'sandbox',
      say: 'Max pooling keeps only the largest of each 2 × 2 block, so the 14 × 14 map becomes 7 × 7. Where inside the block the stroke was is forgotten, on purpose: a stroke one pixel over lands in the same block and gives the same cell. Try it: No pooling.',
      takeaway: 'Pooling keeps what was found and forgets exactly where. Without it, every shift of a stroke is a new pattern for the head to learn.',
      focus: { kind: 'panel', id: 'architecture', label: 'The block and its cell' },
      enter: [{ type: 'preset', id: 'max-pool' }, { type: 'reset' }, WINDOW, { type: 'open', target: 'node:' + NODE.pool(0, 0) }, TRAIN],
      numbers,
      experiments: [
        {
          label: 'No pooling',
          say: 'The head reads every map value.',
          enter: on('no-pool', null, FLAT, PLAY),
          speed: 'normal',
          until: (c) => at90(c) || c.state.step >= 300,
          then: (c) =>
            batch(c) + ': held-out accuracy ' + pc(c.state.testAccuracy) + ' with ' + n(c.derived.paramCount) + ' parameters, against batch ' + REFERENCE.maxPoolTo90 +
            ' with max pooling and a quarter of them.',
          focus: { kind: 'node', id: NODE.flat, label: 'Flatten f' },
        },
        {
          label: 'Max pooling again',
          say: 'The largest of each block, live.',
          enter: on('max-pool', null, FLAT, PLAY),
          speed: 'normal',
          until: (c) => at90(c) || c.state.step >= 300,
          then: (c) => batch(c) + ': held-out accuracy ' + pc(c.state.testAccuracy) + ' with ' + n(c.derived.paramCount) + ' parameters.',
          focus: { kind: 'control', key: 'pool', label: 'Pooling' },
        },
        {
          label: 'Average pooling',
          say: 'The mean of each block instead of its largest.',
          enter: on('max-pool', { pool: 'avg' }, FLAT, PLAY),
          speed: 'normal',
          until: (c) => at90(c) || c.state.step >= 300,
          then: (c) => batch(c) + ': held-out accuracy ' + pc(c.state.testAccuracy) + '. The pooled maps blur, and the head copes about as well.',
          focus: { kind: 'control', key: 'pool', label: 'Pooling' },
        },
      ],
    },
  ],
};

const stacking: L = {
  id: 'stacking',
  title: 'Stacking convolutions',
  hook: 'A second layer reads the first one’s maps.',
  section: 'core',
  presetId: 'two-conv',
  knobs: ['secondConv'],
  steps: [
    {
      id: 'second',
      kind: 'sandbox',
      say: 'A second convolution reads the four pooled maps instead of the image. Its kernel has one 3 × 3 slice per map, so it can combine a horizontal stroke from m₁ and a vertical one from m₂ into a cross; one pixel of m′₁ sees an 8 × 8 patch. Try it: One convolution.',
      takeaway: 'Each layer sees a bigger patch, so deeper layers see whole shapes. Depth composes strokes into shapes, so fewer parameters do better.',
      focus: { kind: 'panel', id: 'architecture', label: 'On the image: 8 × 8' },
      enter: [{ type: 'preset', id: 'two-conv' }, { type: 'reset' }, WINDOW, { type: 'open', target: 'node:' + NODE.map(1, 0) }, TRAIN],
      numbers,
      experiments: [
        {
          label: 'One convolution',
          say: 'The same six classes with the second layer removed.',
          enter: on('two-conv', { secondConv: false }, FLAT, PLAY),
          speed: 'fast',
          until: (c) => c.state.testAccuracy >= 0.95 || c.state.step >= 300 || c.state.diverged,
          then: (c) => batch(c) + ': held-out accuracy ' + pc(c.state.testAccuracy) + ' with ' + n(c.derived.paramCount) + ' parameters, six times as many as the two-layer network.',
          focus: { kind: 'control', key: 'secondConv', label: 'Second convolution' },
        },
        {
          label: 'Two convolutions, live',
          say: 'Watch the held-out accuracy.',
          enter: on('two-conv', null, FLAT, PLAY),
          speed: 'fast',
          until: (c) => c.state.testAccuracy >= 0.95 || c.state.step >= 300 || c.state.diverged,
          then: (c) => batch(c) + ': held-out accuracy ' + pc(c.state.testAccuracy) + ' with ' + n(c.derived.paramCount) + ' parameters, because the second layer leaves the head a tiny 2 × 2 × 4 vector.',
          focus: { kind: 'metric', key: 'test-acc', label: 'Held-out accuracy' },
        },
      ],
    },
  ],
};

const denseHead: L = {
  id: 'dense-head',
  title: 'Dense head and softmax',
  hook: 'From maps to a vector to probabilities.',
  section: 'core',
  presetId: 'edges',
  knobs: ['dense'],
  steps: [
    {
      id: 'head',
      kind: 'sandbox',
      say: 'After pooling the 4 maps are laid out as one vector of 196 numbers, the tall strip. Each of the 16 dense units takes a weighted sum of all of them plus a bias, then ReLU: a brighter circle is a larger output. Softmax turns the scores into p. Try it: Softmax.',
      takeaway: 'The dense head mixes all the features; the kernels only look locally. Training pushes the true class’s p towards 1, and every weight gets a share of the blame.',
      focus: { kind: 'node', id: NODE.unit(1), label: 'Unit h₂' },
      enter: [{ type: 'preset', id: 'edges' }, { type: 'reset' }, FLAT, { type: 'open', target: 'node:' + NODE.unit(1) }, TRAIN],
      numbers: (c) => [{ label: 'units', value: String(c.derived.denseUnits) }, { label: 'p for the probe', value: fmt(c.derived.probe.p, 2) }, { label: 'held out', value: pc(c.state.testAccuracy) }],
      experiments: [
        {
          label: 'Softmax',
          say: 'The class card: z, eᶻ and p for every class.',
          enter: [{ type: 'open', target: 'node:' + NODE.out(0) }],
          until: NOW,
          then: (c) => 'The probe, a ' + c.derived.probe.className + ', gets p = ' + fmt(c.derived.probe.p, 2) + ' for its own class. The four p add to 1; the largest is the call.',
          focus: { kind: 'node', id: NODE.out(0), label: 'Class ─' },
        },
        {
          label: 'Watch it train',
          say: 'The true class fills with blue as its p rises.',
          enter: on('edges', null, FLAT, { type: 'open', target: 'node:' + NODE.out(0) }, PLAY),
          speed: 'normal',
          until: (c) => done(c) || c.state.step >= 300,
          then: (c) => batch(c) + ': the probe, a ' + c.derived.probe.className + ', gets p = ' + fmt(c.derived.probe.p, 2) + ' for its own class; held-out accuracy ' + pc(c.state.testAccuracy) + '.',
          focus: { kind: 'node', id: NODE.out(0), label: 'Class ─' },
        },
        {
          label: 'No dense layer',
          say: 'The scores read the pooled values directly.',
          enter: on('edges', { dense: 0 }, FLAT, PLAY),
          speed: 'normal',
          until: (c) => done(c) || c.state.step >= 300,
          then: (c) => batch(c) + ': held-out accuracy ' + pc(c.state.testAccuracy) + ' with ' + n(c.derived.paramCount) + ' parameters. The kernels and the pooling do the work; the head only weighs them.',
          focus: { kind: 'control', key: 'dense', label: 'Dense units' },
        },
      ],
    },
  ],
};

const rawPixels: L = {
  id: 'raw-pixels',
  title: 'Dense on raw pixels',
  hook: 'Every pixel its own weight, and no shift tolerance.',
  section: 'failure',
  presetId: 'dense-only',
  knobs: ['model'],
  steps: [
    {
      id: 'dense',
      kind: 'sandbox',
      say: 'No convolution: the 256 pixels go straight into 16 dense units, one weight per pixel per unit, for only 90 training images that sit up to 4 pixels off centre. After 200 batches: training accuracy 100%, held-out 87%. Memorised. Try it: Convolutional model.',
      takeaway: 'To a per-pixel weight, a bar two pixels to the left is a different input. Sharing one kernel across positions is shift tolerance built into the model.',
      focus: { kind: 'metric', key: 'test-acc', label: 'Held-out accuracy' },
      enter: [{ type: 'preset', id: 'dense-only' }, { type: 'reset' }, FLAT, { type: 'runTo', steps: 200 }],
      numbers: (c) => [{ label: 'batch', value: n(c.state.step) }, { label: 'train', value: pc(c.state.trainAccuracy) }, { label: 'held out', value: pc(c.state.testAccuracy) }, { label: 'parameters', value: n(c.derived.paramCount) }],
      experiments: [
        {
          label: 'Convolutional model',
          say: 'The same images; the kernels find a stroke wherever it sits.',
          enter: on('dense-only', { model: 'cnn' }, FLAT, PLAY),
          speed: 'fast',
          until: (c) => c.state.step >= 200 || c.state.diverged,
          then: (c) => batch(c) + ': training accuracy ' + pc(c.state.trainAccuracy) + ', held-out ' + pc(c.state.testAccuracy) + ', with ' + n(c.derived.paramCount) + ' parameters.',
          focus: { kind: 'metric', key: 'test-acc', label: 'Held-out accuracy' },
        },
        {
          label: 'Dense again',
          say: 'Watch the two accuracies drift apart.',
          enter: on('dense-only', null, FLAT, PLAY),
          speed: 'fast',
          until: (c) => c.state.step >= 200 || c.state.diverged,
          then: (c) => batch(c) + ': training accuracy ' + pc(c.state.trainAccuracy) + ', held-out ' + pc(c.state.testAccuracy) + '. It memorised the training images.',
          focus: { kind: 'metric', key: 'test-acc', label: 'Held-out accuracy' },
        },
      ],
    },
  ],
};

const fewKernels: L = {
  id: 'few-kernels',
  title: 'Too few kernels',
  hook: 'One map for four shapes.',
  section: 'failure',
  presetId: 'one-filter',
  knobs: ['filters'],
  steps: [
    {
      id: 'one',
      kind: 'sandbox',
      say: 'One kernel, one map, four shapes to tell apart. A single 3 × 3 pattern cannot detect horizontal and vertical strokes at once, so it settles for a compromise and the head does the work: 90% held-out at batch 35 with 878 parameters. Try it: Four kernels.',
      takeaway: 'One map must carry every shape. With more kernels each can specialise on one kind of stroke.',
      focus: { kind: 'node', id: NODE.filter(0, 0), label: 'Kernel k₁' },
      enter: [{ type: 'preset', id: 'one-filter' }, { type: 'reset' }, FLAT, { type: 'runTo', steps: REFERENCE.oneKernelTo90 }],
      numbers,
      experiments: [
        {
          label: 'Four kernels',
          say: 'Each kernel can settle on one kind of stroke.',
          enter: on('one-filter', { filters: 4 }, FLAT, PLAY),
          speed: 'normal',
          until: (c) => at90(c) || c.state.step >= 400,
          then: (c) => batch(c) + ': held-out accuracy ' + pc(c.state.testAccuracy) + ', against batch ' + REFERENCE.oneKernelTo90 + ' with one kernel.',
          focus: { kind: 'control', key: 'filters', label: 'Kernels' },
        },
        {
          label: 'Eight kernels',
          say: 'More kernels than strokes.',
          enter: on('one-filter', { filters: 8 }, FLAT, PLAY),
          speed: 'normal',
          until: (c) => at90(c) || c.state.step >= 400,
          then: (c) => batch(c) + ': held-out accuracy ' + pc(c.state.testAccuracy) + ' with ' + n(c.derived.paramCount) + ' parameters. More kernels help until the strokes are covered; after that they only add parameters.',
          focus: { kind: 'control', key: 'filters', label: 'Kernels' },
        },
        {
          label: 'One kernel again',
          say: 'Watch what one pattern settles for.',
          enter: on('one-filter', null, FLAT, PLAY),
          speed: 'normal',
          until: (c) => at90(c) || c.state.step >= 400,
          then: (c) => batch(c) + ': held-out accuracy ' + pc(c.state.testAccuracy) + ' with ' + n(c.derived.paramCount) + ' parameters.',
          focus: { kind: 'node', id: NODE.filter(0, 0), label: 'Kernel k₁' },
        },
      ],
    },
  ],
};

const noise: L = {
  id: 'noise',
  title: 'Pixel noise',
  hook: 'Averaging beats noise, up to a point.',
  section: 'failure',
  presetId: 'noisy',
  knobs: ['noise'],
  steps: [
    {
      id: 'speckle',
      kind: 'sandbox',
      say: 'Every pixel carries random noise with a spread of 0.3, a third of a stroke: the image is speckled. A kernel adds 9 pixels, and 9 random values mostly cancel while a stroke adds up, so the maps stay clean: held-out accuracy 98% at batch 100. Try it: Noise 0.5.',
      takeaway: 'A 3 × 3 sum averages noise away, and max pooling drops what is left. Past a point the kernels fit the noise of the training images instead of the strokes.',
      focus: { kind: 'node', id: NODE.map(0, 0), label: 'Map m₁' },
      enter: [{ type: 'preset', id: 'noisy' }, { type: 'reset' }, FLAT, TRAIN],
      numbers: (c) => [{ label: 'batch', value: n(c.state.step) }, { label: 'noise', value: String(c.params.noise) }, { label: 'train', value: pc(c.state.trainAccuracy) }, { label: 'held out', value: pc(c.state.testAccuracy) }],
      experiments: [
        {
          label: 'Noise 0.5',
          say: 'Half a stroke on every pixel.',
          enter: on('noisy', { noise: 0.5 }, FLAT, PLAY),
          speed: 'fast',
          until: (c) => c.state.step >= 200 || c.state.diverged,
          then: (c) => batch(c) + ': training accuracy ' + pc(c.state.trainAccuracy) + ', held-out ' + pc(c.state.testAccuracy) + '. The gap is the sign of memorising.',
          focus: { kind: 'control', key: 'noise', label: 'Noise' },
        },
        {
          label: 'Noise 0.3 again',
          say: 'Watch the maps stay clean.',
          enter: on('noisy', null, FLAT, PLAY),
          speed: 'normal',
          until: (c) => done(c) || c.state.step >= 300,
          then: (c) => batch(c) + ': held-out accuracy ' + pc(c.state.testAccuracy) + ' with noise at ' + c.params.noise + '.',
          focus: { kind: 'node', id: NODE.map(0, 0), label: 'Map m₁' },
        },
      ],
    },
  ],
};

const experiments: L = {
  id: 'experiments',
  title: 'Experiments',
  hook: 'Nine experiments with known outcomes.',
  section: 'sandbox',
  presetId: 'edges',
  steps: [
    {
      id: 'sandbox',
      kind: 'sandbox',
      enter: [{ type: 'preset', id: 'edges' }, { type: 'reset' }, FLAT, { type: 'speed', speed: 'normal' }],
      say: 'Pick an experiment: the settings change, the run replays, and the outcome opens under it.',
      experiments: [
        {
          label: 'Average pooling',
          patch: { pool: 'avg', jitter: 3 },
          say: 'Glyphs shifted by 3 pixels with average pooling: each cell is the mean of its block, so the pooled maps blur, yet it reaches 90% at batch 24 and fits at 48, as fast as max pooling.',
          focus: { kind: 'control', key: 'pool', label: 'Pooling' },
        },
        {
          label: 'Eight kernels',
          patch: { filters: 8 },
          say: 'Eight kernels and eight maps: 6,436 parameters, fitted at batch 24 against 35 with four. More kernels help until the strokes are covered; after that they only add parameters.',
          focus: { kind: 'control', key: 'filters', label: 'Kernels' },
        },
        {
          label: 'Kernel 5 × 5',
          patch: { kernel: 5 },
          say: '25 weights per kernel and 12 × 12 maps: 2,492 parameters, fitted at batch 45 against 35 for 3 × 3. A wider window sees more of a stroke at once but has fewer positions to learn from.',
          focus: { kind: 'control', key: 'kernel', label: 'Kernel' },
        },
        {
          label: 'Same padding',
          patch: { padding: 'same' },
          say: 'Zeros around the border keep the maps at 16 × 16, so the head reads 256 values instead of 196: 4,220 parameters, fitted at batch 45.',
          focus: { kind: 'control', key: 'padding', label: 'Padding' },
        },
        {
          label: 'No dense layer',
          patch: { dense: 0 },
          say: 'The scores read the 196 pooled values directly: 828 parameters, and still 100% held-out by batch 45. The kernels and the pooling do the work; the head only has to weigh them.',
          focus: { kind: 'control', key: 'dense', label: 'Dense units' },
        },
        {
          label: 'Batch 64',
          patch: { batchSize: 64 },
          say: 'Four times the images per update: fitted after 21 batches instead of 35, but each batch is four times the work, so it is about as many images seen, not fewer.',
          focus: { kind: 'control', key: 'batchSize', label: 'Batch' },
        },
        {
          label: 'Plain SGD',
          patch: { optimiser: 'gd' },
          say: 'Without Adam scaling each weight’s step, the same η = 0.01 reaches 90% at batch 50 and fits at 95, nearly three times slower.',
          focus: { kind: 'control', key: 'optimiser', label: 'Optimiser' },
        },
        {
          label: 'η = 0.1',
          patch: { learningRate: 0.1 },
          say: 'Ten times the rate: the first updates push every map and every dense unit to zero, ReLU passes no gradient back, and accuracy sits at chance for 400 batches.',
          focus: { kind: 'control', key: 'learningRate', label: 'Learning rate (η)' },
        },
        {
          label: 'Seed 3',
          patch: { seed: 3 },
          say: 'Different starting kernels and different glyphs: fitted at batch 21 instead of 35. How fast a run goes depends on where the kernels start.',
          focus: { kind: 'control', key: 'seed', label: 'Seed' },
        },
      ],
    },
  ],
};

export const LESSONS: L[] = [convolution, receptiveField, pooling, stacking, denseHead, rawPixels, fewKernels, noise, experiments];
