/** Neural network lessons, one screen each. Pure data, driven against the real model in tests/lessons-network.test.ts. */

import type { Lesson, LessonAction, LessonContextBase, LiveNumber } from '../../explainer/lessons';
import { fmt, fmtKnob, fmtPercent } from '../../lib/math/stats';
import { maxAbsWeight, parameterCount } from '../../lib/ml/neuralNetwork';
import type { InputFeature, NetState, TrainingData } from '../../lib/ml/neuralNetwork';
import { parseLayers } from './config';
import type { NeuralNetParams } from './config';
import { countFlatUnits, flatUnitRefs } from './grids';
import type { NeuronGrids } from './grids';

/* ---------------- context ---------------- */

export type SecondView = 'loss' | 'units' | 'weights';

export interface NetLessonContext extends LessonContextBase {
  params: NeuralNetParams;
  state: NetState;
  derived: {
    hidden: number[];
    hiddenUnits: number;
    featureCount: number;
    paramCount: number;
    trainCount: number;
    testCount: number;
    gap: number | null;
    flatUnits: number;
    /** Encoded targets of the flat units, e.g. 'unit:0:3'. */
    flatUnitIds: string[];
    largestWeight: number;
    /** The lowest held-out loss so far and the epoch it came at. */
    bestTest: { loss: number; epoch: number } | null;
    /** How many of the last ten epochs raised the train loss. */
    recentRises: number;
  };
  ui: { secondView: SecondView; open: string | null; layerEdits: number };
}

export interface NetContextInput {
  params: NeuralNetParams;
  state: NetState;
  split: TrainingData;
  features: readonly InputFeature[];
  grids: NeuronGrids;
  ui: NetLessonContext['ui'];
}

/** The diagram target id of a hidden unit. */
export function unitTarget(layer: number, neuron: number): string {
  return 'unit:' + layer + ':' + neuron;
}

/** The page's share of the lesson context; the hook adds `sim` and `lesson`. */
export function makeNetContext({
  params,
  state,
  split,
  features,
  grids,
  ui,
}: NetContextInput): Omit<NetLessonContext, keyof LessonContextBase> {
  const hidden = parseLayers(params.hiddenLayers);
  const history = state.lossHistory;
  const hasTest = split.test.length > 0;
  let bestTest: { loss: number; epoch: number } | null = null;
  if (hasTest) {
    history.forEach((row, i) => {
      if (Number.isFinite(row.test) && (bestTest === null || row.test < bestTest.loss)) {
        bestTest = { loss: row.test, epoch: i + 1 };
      }
    });
  }
  let recentRises = 0;
  for (let i = Math.max(1, history.length - 10); i < history.length; i++) {
    if (history[i].train > history[i - 1].train) recentRises += 1;
  }
  return {
    params,
    state,
    derived: {
      hidden,
      hiddenUnits: hidden.reduce((a, b) => a + b, 0),
      featureCount: features.length,
      paramCount: parameterCount(state.layers),
      trainCount: split.train.length,
      testCount: split.test.length,
      gap: state.epoch > 0 && hasTest ? state.testLoss - state.trainLoss : null,
      flatUnits: countFlatUnits(grids),
      flatUnitIds: flatUnitRefs(grids).map((ref) => unitTarget(ref.layer, ref.neuron)),
      largestWeight: maxAbsWeight(state.layers),
      bestTest,
      recentRises,
    },
    ui,
  };
}

/* ---------------- copy helpers ---------------- */

type Ctx = NetLessonContext;
type NetLesson = Lesson<Ctx, NeuralNetParams>;

const pc = (v: number) => fmtPercent(v, 0);
const epoch = (c: Ctx) => c.state.epoch.toLocaleString('en-US');

/** Epochs in the whole run where the train loss went up, and how many epochs there were. */
function lossRises(c: Ctx): { rises: number; epochs: number } {
  const history = c.state.lossHistory;
  let rises = 0;
  let epochs = 0;
  for (let i = 0; i < history.length; i++) {
    if (Number.isFinite(history[i].train)) epochs += 1;
    if (i > 0 && history[i].train > history[i - 1].train) rises += 1;
  }
  return { rises, epochs };
}

/** The gap the unregularised run in the overfitting lesson reaches at epoch 500. */
const OVERFIT_GAP = 0.141;

/* ---------------- the lessons ---------------- */

type Act = LessonAction<NeuralNetParams>;

const UNITS: Act = { type: 'view', id: 'second', value: 'units' };
const LOSS: Act = { type: 'view', id: 'second', value: 'loss' };
const PLAY: Act = { type: 'play' };

/** A run restates every knob it depends on: the preset, then its own changes, then a fresh start. */
const on = (preset: string, patch: Partial<NeuralNetParams> | null, ...rest: Act[]): Act[] => [
  { type: 'preset', id: preset },
  ...(patch ? [{ type: 'params' as const, patch }] : []),
  { type: 'reset' },
  ...rest,
];

const scores = (c: Ctx): LiveNumber[] => [
  { label: 'epoch', value: epoch(c) },
  { label: 'train', value: pc(c.state.trainAccuracy) },
  { label: 'held out', value: pc(c.state.testAccuracy) },
];

const both = (c: Ctx) => 'train accuracy ' + pc(c.state.trainAccuracy) + ', held-out ' + pc(c.state.testAccuracy);

const hiddenUnits: NetLesson = {
  id: 'hidden-units',
  title: 'Hidden units',
  hook: 'Four straight cuts add up to a circle.',
  section: 'hook',
  presetId: 'ring-four',
  steps: [
    {
      id: 'ring',
      kind: 'sandbox',
      say: 'Two inputs, four hidden units, one output: 17 parameters, trained on the ring. Each hidden node shows the straight cut it draws on the plane; the output adds the four cuts into a bowl around the ring: train accuracy 100%, held-out 100%. Try it: Watch it train.',
      takeaway: 'A hidden unit is a weighted sum of the inputs, squashed: one tilted cut on the plane. Held-out points never touch the weights, so their score is the one to trust.',
      focus: { kind: 'node', id: unitTarget(0, 0), label: 'Hidden unit a₁' },
      enter: [{ type: 'preset', id: 'ring-four' }, { type: 'reset' }, UNITS, { type: 'runTo', steps: 300 }],
      numbers: scores,
      experiments: [
        {
          label: 'Watch it train',
          say: 'The four lines turn and the boundary closes around the ring.',
          enter: on('ring-four', null, UNITS, PLAY),
          speed: 'normal',
          until: (c) => (c.state.trainAccuracy >= 0.99 && c.state.testAccuracy >= 0.99) || c.state.epoch >= 300 || c.state.diverged,
          then: (c) => 'Epoch ' + epoch(c) + ': ' + both(c) + '. Four cuts facing four directions, added up into a bowl.',
          focus: { kind: 'metric', key: 'test-acc', label: 'Held-out accuracy' },
        },
        {
          label: 'One unit',
          say: 'A hidden layer of one: a single cut.',
          enter: on('ring-four', { hiddenLayers: '1' }, UNITS, PLAY),
          speed: 'fast',
          until: (c) => c.state.epoch >= 200 || c.state.diverged,
          then: (c) => 'One hidden unit, epoch ' + epoch(c) + ': ' + both(c) + '. One cut cannot close around a ring.',
          focus: { kind: 'custom', id: 'layers', label: 'Hidden layers' },
        },
      ],
    },
  ],
};

const noHiddenLayer: NetLesson = {
  id: 'no-hidden',
  title: 'No hidden layer',
  hook: 'One unit, one straight boundary.',
  section: 'core',
  presetId: 'no-hidden',
  steps: [
    {
      id: 'linear',
      kind: 'sandbox',
      say: 'No hidden layer: the output reads the inputs directly, 3 parameters in all. One unit draws one straight line, and no straight line separates a ring from its centre: after 200 epochs train accuracy is stuck at 70%. Try it: Add a hidden layer.',
      takeaway: 'Each hidden unit draws a line; the output adds the lines up, and the sum can curve. That is what a hidden layer is for.',
      focus: { kind: 'node', id: 'output', label: 'The output ŷ' },
      enter: [{ type: 'preset', id: 'no-hidden' }, { type: 'reset' }, UNITS, { type: 'runTo', steps: 200 }],
      numbers: (c) => [...scores(c), { label: 'parameters', value: String(c.derived.paramCount) }],
      experiments: [
        {
          label: 'Add a hidden layer',
          say: 'Four units between the inputs and the output.',
          enter: on('no-hidden', { hiddenLayers: '4' }, UNITS, PLAY),
          speed: 'normal',
          until: (c) => c.state.trainAccuracy >= 0.99 || c.state.epoch >= 200 || c.state.diverged,
          then: (c) => 'With a hidden layer of four units, ' + c.derived.paramCount + ' parameters, epoch ' + epoch(c) + ': train accuracy ' + pc(c.state.trainAccuracy) + '. The boundary bends.',
          focus: { kind: 'custom', id: 'layers', label: 'Hidden layers' },
        },
        {
          label: 'No hidden layer again',
          say: 'One line, live.',
          enter: on('no-hidden', null, UNITS, PLAY),
          speed: 'fast',
          until: (c) => c.state.epoch >= 200 || c.state.diverged,
          then: (c) => 'No hidden layer, ' + c.derived.paramCount + ' parameters, epoch ' + epoch(c) + ': train accuracy ' + pc(c.state.trainAccuracy) + '. It cannot wrap the ring.',
          focus: { kind: 'node', id: 'output', label: 'The output ŷ' },
        },
      ],
    },
  ],
};

const featureLearning: NetLesson = {
  id: 'feature-learning',
  title: 'Feature learning',
  hook: 'XOR by feature, then by hidden layer.',
  section: 'core',
  presetId: 'xor-linear',
  steps: [
    {
      id: 'xor',
      kind: 'sandbox',
      say: 'XOR: opposite quadrants share a label. No straight line puts two diagonal corners on one side and the other two on the other, so after 100 epochs train accuracy is 55% and the weights shrink towards zero. Try it: Hand in x₁x₂.',
      takeaway: 'x₁x₂ is positive in two corners and negative in the other two, so XOR becomes one line. Hidden units learn the features you did not think to hand in.',
      focus: { kind: 'metric', key: 'train-acc', label: 'Train accuracy' },
      enter: [{ type: 'preset', id: 'xor-linear' }, { type: 'reset' }, UNITS, { type: 'runTo', steps: 100 }],
      numbers: scores,
      experiments: [
        {
          label: 'Hand in x₁x₂',
          say: 'The product as a third input.',
          enter: on('xor-feature', null, UNITS, PLAY),
          speed: 'normal',
          until: (c) => c.state.trainAccuracy >= 0.99 || c.state.epoch >= 300 || c.state.diverged,
          then: (c) => 'With x₁x₂ as a third input: train accuracy ' + pc(c.state.trainAccuracy) + ' at epoch ' + epoch(c) + '. One line is enough.',
          focus: { kind: 'node', id: 'input:2', label: 'Input x₁x₂' },
        },
        {
          label: 'Four hidden units',
          say: 'x₁x₂ taken away; the network must find the feature itself.',
          enter: on('xor-hidden', null, UNITS, PLAY),
          speed: 'fast',
          until: (c) => c.state.trainAccuracy >= 0.99 || c.state.epoch >= 400 || c.state.diverged,
          then: (c) => 'Without x₁x₂, four hidden units: train accuracy ' + pc(c.state.trainAccuracy) + ' at epoch ' + epoch(c) + ', slower than the hand-made feature. The four units found diagonal cuts, much like x₁x₂.',
          focus: { kind: 'panel', id: 'second', label: 'Every hidden unit' },
        },
      ],
    },
  ],
};

const modelCapacity: NetLesson = {
  id: 'capacity',
  title: 'Model capacity',
  hook: 'A spiral needs more width and depth.',
  section: 'core',
  presetId: 'spiral-small',
  steps: [
    {
      id: 'spiral',
      kind: 'sandbox',
      say: 'Two hidden units on a spiral: after 300 epochs train accuracy 68% and the loss flat at 0.43. Two straight cuts cannot trace a spiral however long they train. Try it: Two layers of eight.',
      takeaway: 'The second layer combines curves the first layer made. Depth buys shapes width cannot.',
      focus: { kind: 'metric', key: 'train-acc', label: 'Train accuracy' },
      enter: [{ type: 'preset', id: 'spiral-small' }, { type: 'reset' }, LOSS, { type: 'runTo', steps: 300 }],
      numbers: (c) => [...scores(c), { label: 'parameters', value: String(c.derived.paramCount) }],
      experiments: [
        {
          label: 'Two layers of eight',
          say: 'Nothing for a while, then the spiral appears.',
          enter: on('spiral-deep', null, UNITS, PLAY),
          speed: 'turbo',
          until: (c) => c.state.trainAccuracy >= 0.9 || c.state.epoch >= 800 || c.state.diverged,
          then: (c) =>
            'Two layers of eight, ' + c.derived.paramCount + ' parameters: ' +
            (c.state.trainAccuracy >= 0.9 ? 'train accuracy passed 90% at epoch ' + epoch(c) : 'epoch ' + epoch(c) + ', train accuracy ' + pc(c.state.trainAccuracy)) +
            '. First layer: straight cuts. Second layer: curved shapes built from them.',
          focus: { kind: 'panel', id: 'second', label: 'Every hidden unit' },
        },
        {
          label: 'Two units again',
          say: 'The plateau, live.',
          enter: on('spiral-small', null, LOSS, PLAY),
          speed: 'fast',
          until: (c) => c.state.epoch >= 300 || c.state.diverged,
          then: (c) => 'Two hidden units, epoch ' + epoch(c) + ': train accuracy ' + pc(c.state.trainAccuracy) + ', loss ' + fmt(c.state.trainLoss, 2) + ' and flat.',
          focus: { kind: 'metric', key: 'train-acc', label: 'Train accuracy' },
        },
      ],
    },
  ],
};

const learningRate: NetLesson = {
  id: 'learning-rate',
  title: 'Learning rate η',
  hook: 'High η overflows, low η crawls.',
  section: 'failure',
  presetId: 'too-fast',
  knobs: ['learningRate'],
  steps: [
    {
      id: 'eta',
      kind: 'sandbox',
      say: 'η = 0.1 on the ring: a smooth fall, train accuracy 100% by epoch 30 and the loss never rose. Too large a rate overshoots the minimum, and on a steep slope the next step is larger still; too small a rate crawls. Try it: η = 1.',
      takeaway: 'A good rate gives a steady fall. Watch the loss curve, not the number.',
      focus: { kind: 'control', key: 'learningRate', label: 'Learning rate (η)' },
      enter: [{ type: 'preset', id: 'too-fast' }, { type: 'params', patch: { learningRate: 0.1 } }, { type: 'reset' }, LOSS, { type: 'runTo', steps: 30 }],
      numbers: (c) => [...scores(c), { label: 'loss', value: fmt(c.state.trainLoss, 3) }],
      experiments: [
        {
          label: 'η = 1',
          say: 'About thirty times the rate that works.',
          enter: on('too-fast', null, LOSS, PLAY),
          speed: 'normal',
          until: (c) => c.state.diverged || c.state.epoch >= 300,
          then: (c) => {
            const { rises, epochs } = lossRises(c);
            return (
              'η = ' + fmtKnob(c.params.learningRate) + ': train loss rose in ' + rises + ' of ' + epochs + ' epochs, a saw-tooth' +
              (c.state.diverged ? '. A weight passed 1e8 at epoch ' + epoch(c) + ': diverged.' : ', still oscillating at epoch ' + epoch(c) + '.')
            );
          },
          focus: { kind: 'metric', key: 'max-weight', label: 'Largest weight' },
        },
        {
          label: 'η = 0.001',
          say: 'The opposite, for three hundred epochs.',
          enter: on('too-fast', { learningRate: 0.001 }, LOSS, PLAY),
          speed: 'fast',
          until: (c) => c.state.epoch >= 300 || c.state.diverged,
          then: (c) => 'η = ' + fmtKnob(c.params.learningRate) + ', epoch ' + epoch(c) + ': train accuracy ' + pc(c.state.trainAccuracy) + ', loss ' + fmt(c.state.trainLoss, 3) + ' and still falling. Safe and slow.',
          focus: { kind: 'metric', key: 'train-loss', label: 'Train loss' },
        },
        {
          label: 'η = 0.1 again',
          say: 'The rate in between, live.',
          enter: on('too-fast', { learningRate: 0.1 }, LOSS, PLAY),
          speed: 'normal',
          until: (c) => c.state.epoch >= 30 || c.state.diverged,
          then: (c) => {
            const { rises, epochs } = lossRises(c);
            return 'η = ' + fmtKnob(c.params.learningRate) + ', epoch ' + epoch(c) + ': train accuracy ' + pc(c.state.trainAccuracy) + '; the loss rose in ' + rises + ' of ' + epochs + ' epochs.';
          },
          focus: { kind: 'control', key: 'learningRate', label: 'Learning rate (η)' },
        },
      ],
    },
  ],
};

const deadRelu: NetLesson = {
  id: 'dead-relu',
  title: 'Dead ReLU units',
  hook: 'A high η kills ReLU units for good.',
  section: 'failure',
  presetId: 'dead-relu',
  knobs: ['learningRate', 'activation'],
  steps: [
    {
      id: 'relu',
      kind: 'sandbox',
      say: 'ReLU at η = 0.5. ReLU outputs 0 for any negative input, and 0 has no gradient: one large step pushed 5 of 16 units negative for every point and they never came back, the blank pictures in the unit panel. Epoch 200: train 91%, held-out 86%. Try it: η = 0.03.',
      takeaway: 'Dead units are capacity you paid for and cannot use. Smaller steps keep them alive.',
      focus: { kind: 'panel', id: 'second', label: 'Every hidden unit' },
      enter: [{ type: 'preset', id: 'dead-relu' }, { type: 'reset' }, UNITS, { type: 'runTo', steps: 200 }],
      numbers: (c) => [...scores(c), { label: 'dead units', value: c.derived.flatUnits + ' / ' + c.derived.hiddenUnits }],
      experiments: [
        {
          label: 'η = 0.03',
          say: 'Same network, same ReLU, smaller steps.',
          enter: on('dead-relu', { learningRate: 0.03 }, UNITS, PLAY),
          speed: 'fast',
          until: (c) => c.state.epoch >= 100 || c.state.diverged,
          then: (c) => 'At η = ' + fmtKnob(c.params.learningRate) + ': ' + c.derived.flatUnits + ' of ' + c.derived.hiddenUnits + ' units blank at epoch ' + epoch(c) + ', train accuracy ' + pc(c.state.trainAccuracy) + '.',
          focus: { kind: 'control', key: 'learningRate', label: 'Learning rate (η)' },
        },
        {
          label: 'η = 0.5 again',
          say: 'Watch the unit pictures: some go blank.',
          enter: on('dead-relu', null, UNITS, PLAY),
          speed: 'fast',
          until: (c) => c.state.epoch >= 50 || c.state.diverged,
          then: (c) => 'ReLU at η = ' + fmtKnob(c.params.learningRate) + ', epoch ' + epoch(c) + ': ' + c.derived.flatUnits + ' of ' + c.derived.hiddenUnits + ' units are blank, output 0 everywhere. Open one: its output column reads 0 for every point.',
          focus: { kind: 'panel', id: 'second', label: 'Every hidden unit' },
        },
      ],
    },
  ],
};

const initialisation: NetLesson = {
  id: 'initialisation',
  title: 'Weight initialisation',
  hook: 'From zero, every unit stays identical.',
  section: 'failure',
  presetId: 'zeros',
  knobs: ['initialiser'],
  steps: [
    {
      id: 'zeros',
      kind: 'sandbox',
      say: 'Every weight started at exactly 0, so every unit picture is the same blank. Equal weights get equal gradients, so equal weights stay equal: after 100 epochs the four units are still identical and train accuracy is 51%. Try it: Xavier initialisation.',
      takeaway: 'Random starting weights are what let units specialise.',
      focus: { kind: 'panel', id: 'second', label: 'Every hidden unit' },
      enter: [{ type: 'preset', id: 'zeros' }, { type: 'reset' }, UNITS, { type: 'runTo', steps: 100 }],
      numbers: (c) => [...scores(c), { label: 'identical', value: c.derived.flatUnits + ' / ' + c.derived.hiddenUnits }],
      experiments: [
        {
          label: 'Xavier initialisation',
          say: 'Small random weights, scaled by the layer size.',
          enter: on('zeros', { initialiser: 'xavier' }, UNITS, PLAY),
          speed: 'normal',
          until: (c) => c.state.epoch >= 30 || c.state.trainAccuracy >= 0.99 || c.state.diverged,
          then: (c) => 'From a Xavier start, epoch ' + epoch(c) + ': four different pictures, train accuracy ' + pc(c.state.trainAccuracy) + '.',
          focus: { kind: 'control', key: 'initialiser', label: 'Initial weights' },
        },
        {
          label: 'All zeros again',
          say: 'Watch whether the four units ever differ.',
          enter: on('zeros', null, UNITS, PLAY),
          speed: 'fast',
          until: (c) => c.state.epoch >= 100 || c.state.diverged,
          then: (c) => 'From all-zero weights, epoch ' + epoch(c) + ': ' + c.derived.flatUnits + ' of ' + c.derived.hiddenUnits + ' units still identical, train accuracy ' + pc(c.state.trainAccuracy) + '.',
          focus: { kind: 'panel', id: 'second', label: 'Every hidden unit' },
        },
      ],
    },
  ],
};

const overfitting: NetLesson = {
  id: 'overfitting',
  title: 'Overfitting and regularisation',
  hook: 'Learning the noise, then an L2 penalty.',
  section: 'failure',
  presetId: 'overfit',
  knobs: ['regularisation', 'regRate'],
  steps: [
    {
      id: 'noise',
      kind: 'sandbox',
      say: 'Three layers of eight on 50 noisy points; the rings are held out. Train loss 0.037 keeps falling, but held-out loss bottomed at 0.128 near epoch 118 and is 0.178 at epoch 500: the network is fitting the noise. Train 100%, held-out 94%. Try it: L2 penalty.',
      takeaway: 'Only the held-out number says how a network does on new data. A penalty keeps the weights small, so no unit can carve a pocket around one noisy point.',
      focus: { kind: 'metric', key: 'gap', label: 'Generalisation gap' },
      enter: [{ type: 'preset', id: 'overfit' }, { type: 'reset' }, LOSS, { type: 'runTo', steps: 500 }],
      numbers: (c) => [...scores(c), { label: 'gap', value: fmt(c.derived.gap ?? Number.NaN, 3) }],
      experiments: [
        {
          label: 'L2 penalty',
          say: 'The same run with a charge on large weights.',
          enter: on('overfit-regularised', null, LOSS, PLAY),
          speed: 'turbo',
          until: (c) => c.state.epoch >= 500 || c.state.diverged,
          then: (c) => 'With the L2 penalty: gap ' + fmt(OVERFIT_GAP, 3) + ' before, ' + fmt(c.derived.gap ?? Number.NaN, 3) + ' now. Held-out accuracy ' + pc(c.state.testAccuracy) + '.',
          focus: { kind: 'metric', key: 'gap', label: 'Generalisation gap' },
        },
        {
          label: 'Four units instead',
          say: 'A network too small to memorise.',
          enter: on('overfit', { hiddenLayers: '4' }, LOSS, PLAY),
          speed: 'turbo',
          until: (c) => c.state.epoch >= 500 || c.state.diverged,
          then: (c) => 'Four hidden units: gap ' + fmt(c.derived.gap ?? Number.NaN, 3) + ', held-out accuracy ' + pc(c.state.testAccuracy) + '. Less capacity, less to fit the noise with.',
          focus: { kind: 'custom', id: 'layers', label: 'Hidden layers' },
        },
        {
          label: 'No penalty again',
          say: 'Watch the two loss curves drift apart.',
          enter: on('overfit', null, LOSS, PLAY),
          speed: 'turbo',
          until: (c) => c.state.epoch >= 500 || c.state.diverged,
          then: (c) => {
            const best = c.derived.bestTest;
            if (!best) return 'No held-out points, so there is no held-out loss to watch.';
            return 'Held-out loss was lowest at epoch ' + best.epoch.toLocaleString('en-US') + ' (' + fmt(best.loss, 3) + '); at epoch ' + epoch(c) + ' it is ' + fmt(c.state.testLoss, 3) + ' while train loss is ' + fmt(c.state.trainLoss, 3) + '.';
          },
          focus: { kind: 'metric', key: 'gap', label: 'Generalisation gap' },
        },
      ],
    },
  ],
};

const experiments: NetLesson = {
  id: 'experiments',
  title: 'Experiments',
  hook: 'Six experiments with known outcomes.',
  section: 'sandbox',
  presetId: 'spiral-deep',
  steps: [
    {
      id: 'sandbox',
      kind: 'sandbox',
      enter: [{ type: 'preset', id: 'spiral-deep' }, { type: 'speed', speed: 'normal' }],
      say: 'Pick an experiment: the settings change, the run replays, and the outcome opens under it.',
      experiments: [
        {
          label: 'Sigmoid activation',
          patch: { activation: 'sigmoid' },
          say: 'After 1,500 epochs still near 54%: the sigmoid slope is tiny far from zero, so the gradients vanish.',
          focus: { kind: 'control', key: 'activation', label: 'Activation' },
        },
        {
          label: 'Linear activation',
          patch: { activation: 'linear' },
          say: 'Without non-linear activations the layers collapse into a single linear model: near 54% and flat.',
          focus: { kind: 'control', key: 'activation', label: 'Activation' },
        },
        {
          label: 'Different seed',
          patch: { seed: 3 },
          say: 'Same network, different random start: reaches 90% at epoch 116 instead of 341.',
          focus: { kind: 'control', key: 'seed', label: 'Random seed' },
        },
        {
          label: 'Tanh, η = 0.5',
          patch: { activation: 'tanh', learningRate: 0.5 },
          say: 'With tanh at η = 0.5 every unit stays alive: 0 flat units, 90% by epoch 118 and 100% by 125.',
          focus: { kind: 'control', key: 'learningRate', label: 'Learning rate (η)' },
        },
        {
          label: 'L1 penalty 0.3',
          patch: {
            dataset: 'circle',
            hiddenLayers: '8,8,8',
            sampleCount: 100,
            noise: 0.55,
            testFraction: 0.5,
            regularisation: 'l1',
            regRate: 0.3,
          },
          say: 'L1 penalty at 0.3: 21 of 24 units go flat and accuracy drops to 78%. Too much penalty costs capacity.',
          focus: { kind: 'control', key: 'regRate', label: 'L2/L1 penalty (λ)' },
        },
        {
          label: 'Smaller network',
          patch: { dataset: 'circle', hiddenLayers: '4', sampleCount: 100, noise: 0.55, testFraction: 0.5 },
          say: 'Four units on the overfitting data: gap 0.06 and 96% held-out, matching the penalised large network.',
          focus: { kind: 'custom', id: 'layers', label: 'Hidden layers' },
        },
      ],
    },
  ],
};

export const LESSONS: NetLesson[] = [
  hiddenUnits,
  noHiddenLayer,
  featureLearning,
  modelCapacity,
  learningRate,
  deadRelu,
  initialisation,
  overfitting,
  experiments,
];
