/** Activations and losses lessons, one screen each. Pure data, driven against the real model in tests/lessons-activation.test.ts. */

import type { Lesson, LessonAction, LessonContextBase, LiveNumber } from '../../explainer/lessons';
import { fmtCompact, fmtPercent, rSquared } from '../../lib/math/stats';
import type { ActivationFn } from '../../lib/ml/activations';
import { predict } from '../../lib/ml/mlp';
import { lossAxisValue } from '../../lib/ml/mlpTrainer';
import type { TrainerData, TrainerState } from '../../lib/ml/mlpTrainer';
import type { ActParams } from './config';

/* ---------------- context ---------------- */

export type GraphView = 'network' | 'activation' | 'loss' | 'space' | 'surface';
export type SurfaceOf = 'output' | 'units';
export type SecondView = 'gradients' | 'loss';

export interface ActNumbers {
  classify: boolean;
  epoch: number;
  trainLoss: number;
  testLoss: number;
  /** Held-out accuracy, or null for a curve fit. */
  accuracy: number | null;
  r2: number | null;
  /** First hidden layer's gradient norm over the output layer's, or null before training. */
  gradRatio: number | null;
  /** Share of hidden units whose slope was below 1e-3 on every sample of the last batch. */
  deadShare: number;
  /** Share of hidden pre-activations sitting where |f′| < 0.05. */
  flatShare: number;
  /** Share of the last batch's outputs past the hinge margin, where the loss is zero. */
  pastMargin: number;
  fitted: boolean;
  diverged: boolean;
  hiddenUnits: number;
}

export interface ActLessonContext extends LessonContextBase {
  params: ActParams;
  state: TrainerState;
  derived: ActNumbers;
  ui: { view: GraphView; surfaceOf: SurfaceOf; secondView: SecondView; open: string | null };
}

/** The numbers the page, the lessons and the tests all read off a state. */
export function deriveNumbers(
  state: TrainerState,
  data: TrainerData,
  act: ActivationFn,
  lossName: string,
  fitted: number,
): ActNumbers {
  const classify = data.train.labels !== null;
  const hidden = state.layers.slice(0, -1);
  const zs = hidden.flatMap((l) => l.zSamples);
  const flatShare = zs.length ? zs.filter((z) => Math.abs(act.df(z)) < 0.05).length / zs.length : 0;
  const deadShare = hidden.length ? hidden.reduce((s, l) => s + l.smallGradFraction, 0) / hidden.length : 0;
  const firstGrad = hidden[0]?.gradNorm ?? 0;
  const lastGrad = state.layers[state.layers.length - 1]?.gradNorm ?? 0;
  const gradRatio = state.epoch > 0 && hidden.length > 0 && lastGrad > 0 ? firstGrad / lastGrad : null;
  let pastMargin = 0;
  if (lossName === 'hinge' && state.outputs.length > 0) {
    const clear = state.outputs.filter((o, i) => {
      const target = state.batchTargets[i] ?? [];
      const s = lossAxisValue('hinge', o, target);
      return (target[0] === 1 ? s : -s) >= 1;
    });
    pastMargin = clear.length / state.outputs.length;
  }
  let r2: number | null = null;
  if (!classify && state.epoch > 0 && !state.diverged) {
    const actual = data.test.targets.map((t) => t[0]);
    const predicted = data.test.inputs.map((x) => predict(state.net, x, state.spec)[0]);
    r2 = rSquared(actual, predicted);
  }
  return {
    classify,
    epoch: state.epoch,
    trainLoss: state.trainLoss,
    testLoss: state.testLoss,
    accuracy: state.testAccuracy,
    r2,
    gradRatio,
    deadShare,
    flatShare,
    pastMargin,
    fitted: state.epoch > 10 && state.trainLoss < fitted,
    diverged: state.diverged,
    hiddenUnits: hidden.reduce((s, _l, i) => s + state.net.layers[i].outSize, 0),
  };
}

/* ---------------- copy helpers ---------------- */

type Ctx = ActLessonContext;
type ActLesson = Lesson<Ctx, ActParams>;

const pc = (v: number | null) => fmtPercent(v ?? 0, 0);
const epoch = (c: Ctx) => 'Epoch ' + c.derived.epoch.toLocaleString('en-US');
const acc = (c: Ctx) => 'test accuracy ' + pc(c.derived.accuracy);
const ratio = (c: Ctx) => (c.derived.gradRatio === null ? 'not yet measured' : fmtCompact(c.derived.gradRatio));
const settled = (c: Ctx) => c.derived.fitted || c.derived.diverged;
/** Every experiment names all the knobs it depends on, so runs never stack. */
const MOONS: Partial<ActParams> = { task: 'classify', pointsDataset: 'moons', classLoss: 'bce', activation: 'relu', depth: 4, width: 6, learningRate: 0.02, initScale: 1, view: 'network' };
const CURVE: Partial<ActParams> = { task: 'regress', curveDataset: 'outliers', regLoss: 'huber', activation: 'tanh', depth: 2, width: 6, learningRate: 0.01, initScale: 1, view: 'network' };

/* ---------------- Introduction ---------------- */

type Act = LessonAction<ActParams>;

const PLAY: Act = { type: 'play' };
/** Opens the lesson on its trained picture. */
const FIT: Act = { type: 'runTo', steps: 300 };

/** A run restates every knob it depends on: the preset, then its own changes, then a fresh start. */
const on = (preset: string, patch: Partial<ActParams> | null, ...rest: Act[]): Act[] => [
  { type: 'preset', id: preset },
  ...(patch ? [{ type: 'params' as const, patch }] : []),
  { type: 'reset' },
  ...rest,
];

const numbers = (c: Ctx): LiveNumber[] => [
  { label: 'epoch', value: c.derived.epoch.toLocaleString('en-US') },
  { label: 'loss', value: fmtCompact(c.derived.trainLoss) },
  { label: 'held out', value: pc(c.derived.accuracy) },
];

const bend: ActLesson = {
  id: 'bend',
  title: 'Non-linearity',
  hook: 'Without it, a network can only draw straight lines.',
  section: 'hook',
  presetId: 'bend',
  knobs: ['activation'],
  steps: [
    {
      id: 'plane',
      kind: 'sandbox',
      say: 'Two hidden layers of two tanh units, trained on the moons. Each layer takes weighted sums Wh + b (the grid turns and stretches, still a grid), then bends it through tanh; in h₂ one straight line splits the moons. ▶ replays the flow. Try it: Linear.',
      takeaway: 'The hidden layers bend the plane until one straight line can separate the classes. Stacked linear layers multiply into one linear layer: the activation is the only source of curves.',
      focus: { kind: 'custom', id: 'layer:2', label: 'The last layer' },
      enter: [{ type: 'preset', id: 'bend' }, { type: 'reset' }, FIT],
      numbers,
      experiments: [
        {
          label: 'Linear',
          say: 'f(z) = z, the same network retrained: no arrow bends anything.',
          enter: on('bend', { activation: 'linear' }, PLAY),
          speed: 'normal',
          until: (c) => c.derived.epoch >= 40 || settled(c),
          then: (c) => 'Linear, f(z) = z, ' + epoch(c).toLowerCase() + ': nothing bends. Every layer’s grid stays a flat grid, the boundary is a straight line, and the loss stalls at ' + fmtCompact(c.derived.trainLoss) + '.',
          focus: { kind: 'control', key: 'activation', label: 'Activation' },
        },
        {
          label: 'Watch tanh train',
          say: 'The bend, epoch by epoch.',
          enter: on('bend', null, PLAY),
          speed: 'fast',
          until: (c) => c.derived.trainLoss < 0.15 || c.derived.epoch >= 300 || settled(c),
          then: (c) => epoch(c) + ': in h₂ the two moons sit on opposite sides of one straight line, the output unit; ' + acc(c) + '.',
          focus: { kind: 'custom', id: 'layer:2', label: 'The last layer' },
        },
      ],
    },
  ],
};

/* ---------------- Core concepts ---------------- */

const ridges: ActLesson = {
  id: 'ridges',
  title: 'Ridge functions',
  hook: 'Every hidden unit is one ridge; the output adds them up.',
  section: 'core',
  presetId: 'ridges',
  knobs: ['width', 'activation'],
  steps: [
    {
      id: 'units',
      kind: 'sandbox',
      say: 'Four ReLU units after 80 epochs, each a sheet over the input plane: a = f(w·x + b) is flat on one side of the line w·x + b = 0 and rises on the other, a ridge. Next to them their sum z, the output before the sigmoid; ▶ replays the bend. Try it: Two units.',
      takeaway: 'A network is a sum of ridges: the activation sets their shape, w their direction and sharpness, v how much each lifts or lowers the output.',
      focus: { kind: 'custom', id: 'unit:0', label: 'Unit 1' },
      enter: [{ type: 'preset', id: 'ridges' }, { type: 'reset' }, { type: 'runTo', steps: 80 }],
      numbers,
      experiments: [
        {
          label: 'Two units',
          say: 'Two ridges to add up.',
          enter: on('ridges', { width: 2 }, { type: 'runTo', steps: 80 }),
          until: (c) => c.derived.epoch >= 80 || settled(c),
          then: (c) => 'Two ridges, ' + epoch(c).toLowerCase() + ': ' + acc(c) + ', loss ' + fmtCompact(c.derived.trainLoss) + '. Two creases can only make a wedge.',
          focus: { kind: 'control', key: 'width', label: 'Width' },
        },
        {
          label: 'Eight units',
          say: 'Eight ridges.',
          enter: on('ridges', { width: 8 }, { type: 'runTo', steps: 80 }),
          until: (c) => c.derived.epoch >= 80 || settled(c),
          then: (c) => 'Eight ridges, ' + epoch(c).toLowerCase() + ': ' + acc(c) + ', loss ' + fmtCompact(c.derived.trainLoss) + '. More creases, a rounder sum.',
          focus: { kind: 'control', key: 'width', label: 'Width' },
        },
        {
          label: 'Tanh units',
          say: 'The same four units with a smooth activation.',
          enter: on('ridges', { activation: 'tanh' }, { type: 'runTo', steps: 80 }),
          until: (c) => c.derived.epoch >= 80 || settled(c),
          then: (c) => 'Tanh, ' + epoch(c).toLowerCase() + ': each sheet is an S-shaped slope instead of a crease; ' + acc(c) + '.',
          focus: { kind: 'control', key: 'activation', label: 'Activation' },
        },
        {
          label: 'Flat, on the plane',
          say: 'The same four units as flat tiles over the input plane.',
          enter: on('ridges', { view: 'space' }, { type: 'runTo', steps: 80 }, { type: 'open', target: 'unit:0' }),
          until: (c) => c.derived.epoch >= 80 || settled(c),
          then: (c) => 'On the plane, ' + epoch(c).toLowerCase() + ': z₁ is a ramp over x₁, x₂, orange below 0 and blue above. ReLU turns the orange half white: 0, the flat side of the ridge. Any unit in the map opens its tiles.',
          focus: { kind: 'custom', id: 'unit:0', label: 'Unit 1' },
        },
      ],
    },
  ],
};

/* ---------------- Failure modes ---------------- */

const vanishing: ActLesson = {
  id: 'vanishing',
  title: 'Vanishing gradients',
  hook: 'Sigmoid’s slope is at most 0.25; six layers of it stop learning.',
  section: 'failure',
  presetId: 'sigmoid-deep',
  knobs: ['activation', 'depth'],
  steps: [
    {
      id: 'gradients',
      kind: 'sandbox',
      say: 'Six sigmoid layers after 20 epochs. Backprop multiplies one slope f′ per layer on the way back, and sigmoid’s slope is at most 0.25: each row ends with the gradient its weights get, and h₁ gets 3.8e−4 of the output layer’s. Nothing learnt yet. Try it: Tanh.',
      takeaway: 'Small slopes multiply into tiny gradients. Use ReLU or tanh inside a deep network; keep sigmoid for the output.',
      focus: { kind: 'custom', id: 'layer:1', label: 'First layer' },
      enter: [{ type: 'preset', id: 'sigmoid-deep' }, { type: 'reset' }, { type: 'view', id: 'second', value: 'gradients' }, { type: 'runTo', steps: 20 }],
      numbers: (c) => [...numbers(c), { label: 'h₁ / output gradient', value: ratio(c) }],
      experiments: [
        {
          label: 'Tanh',
          say: 'Its slope reaches 1 at zero, so far less is lost per layer.',
          enter: on('sigmoid-deep', { activation: 'tanh' }, { type: 'view', id: 'second', value: 'gradients' }, PLAY),
          speed: 'normal',
          until: (c) => c.derived.epoch >= 20 || settled(c),
          then: (c) => 'Tanh, ' + epoch(c).toLowerCase() + ': h₁ now receives ' + ratio(c) + ' times the output layer’s gradient, ' + acc(c) + '.',
          focus: { kind: 'control', key: 'activation', label: 'Activation' },
        },
        {
          label: 'ReLU',
          say: 'Slope exactly 1 wherever a unit is on.',
          enter: on('sigmoid-deep', { activation: 'relu' }, { type: 'view', id: 'second', value: 'gradients' }, PLAY),
          speed: 'normal',
          until: (c) => c.derived.epoch >= 20 || settled(c),
          then: (c) => 'ReLU, ' + epoch(c).toLowerCase() + ': h₁ receives ' + ratio(c) + ' times the output layer’s gradient, ' + acc(c) + '. Nothing is lost through a unit that is on.',
          focus: { kind: 'control', key: 'activation', label: 'Activation' },
        },
        {
          label: 'Sigmoid again',
          say: 'Watch the gradient shrink from ŷ up to h₁.',
          enter: on('sigmoid-deep', null, { type: 'view', id: 'second', value: 'gradients' }, PLAY),
          speed: 'normal',
          until: (c) => c.derived.epoch >= 20 || c.derived.diverged,
          then: (c) => epoch(c) + ': h₁ receives ' + ratio(c) + ' of the gradient the output layer does, about 0.25 multiplied six times. Loss ' + fmtCompact(c.derived.trainLoss) + ', accuracy ' + pc(c.derived.accuracy) + '.',
          focus: { kind: 'custom', id: 'layer:1', label: 'First layer' },
        },
      ],
    },
  ],
};

/* ---------------- Practice ---------------- */

const experiments: ActLesson = {
  id: 'experiments',
  title: 'Experiments',
  hook: 'Eleven experiments with known outcomes.',
  section: 'sandbox',
  presetId: 'relu-deep',
  steps: [
    {
      id: 'sandbox',
      kind: 'sandbox',
      enter: [{ type: 'preset', id: 'relu-deep' }, { type: 'play' }],
      say: 'Four ReLU layers on the moons: each unit is its curve with the values it sees, the output is the loss with the points on it. Pick an experiment: the run replays and the outcome opens under it.',
      experiments: [
        {
          label: 'Dead ReLUs',
          patch: { ...MOONS, initScale: 3, learningRate: 0.3 },
          say: 'Weights three times bigger and a rate of 0.3: by epoch 3, 71% of the units turn red, their values piled left of zero where ReLU is flat. A unit with z < 0 everywhere outputs 0, gets no gradient and never recovers.',
          focus: { kind: 'control', key: 'initScale', label: 'Init scale' },
        },
        {
          label: 'Leaky ReLU',
          patch: { ...MOONS, activation: 'leakyRelu', initScale: 3, learningRate: 0.3 },
          say: 'The same start with a small slope on the left: a unit sitting left of zero still passes α of the gradient back, so none turns red and the moons are fitted by epoch 16.',
          focus: { kind: 'control', key: 'activation', label: 'Activation' },
        },
        {
          label: 'Squared error on classes',
          patch: { ...MOONS, classLoss: 'mse', depth: 2, initScale: 5 },
          say: 'Squared error through a sigmoid, from weights five times too big. The loss against the margin is flat at both ends, and most points sit on the far plateaus, confidently wrong with no slope: accuracy 43%, loss 0.200.',
          focus: { kind: 'custom', id: 'function:loss', label: 'The output' },
        },
        {
          label: 'Cross-entropy instead',
          patch: { ...MOONS, classLoss: 'bce', depth: 2, initScale: 5 },
          say: 'The same oversized start scored by cross-entropy: the curve keeps climbing on the wrong side, so a confidently wrong point has the largest gradient of all and the pile slides right. 87% by epoch 60, against 43%.',
          focus: { kind: 'control', key: 'classLoss', label: 'Class loss' },
        },
        {
          label: 'Hinge loss',
          patch: { ...MOONS, classLoss: 'hinge', depth: 2 },
          say: 'The hinge is zero once a point clears the margin at 1: the output curve is flat right of it. By epoch 8, 90% of the batch sits on the flat part and sends nothing back; only the points near the boundary still train.',
          focus: { kind: 'custom', id: 'function:loss', label: 'The output' },
        },
        {
          label: 'Softmax, three classes',
          patch: { ...MOONS, pointsDataset: 'three-class', classLoss: 'softmaxCE', depth: 2 },
          say: 'Three classes, three output units, one logit each. Softmax turns the logits into probabilities, the gradient on each is p − y, and every class’s points slide right on its own curve: held-out accuracy 100% by epoch 20.',
          focus: { kind: 'custom', id: 'function:loss', label: 'Three outputs' },
        },
        {
          label: 'Logistic regression',
          patch: { ...MOONS, pointsDataset: 'gaussians', depth: 0, view: 'surface' },
          say: 'Depth 0 is one sigmoid unit reading the inputs directly: a single ramp with a straight cut. Enough for two blobs, 97% by epoch 45; on the moons the same ramp stalls at loss 0.45.',
          focus: { kind: 'control', key: 'depth', label: 'Depth' },
        },
        {
          label: 'Eight linear layers',
          patch: { ...MOONS, pointsDataset: 'circle', activation: 'linear', depth: 8, width: 2, view: 'space' },
          say: 'Eight layers with f(z) = z on a ring: every layer’s grid stays a grid, the boundary stays straight, and accuracy stays near 60%. Eight matrices multiply out to one.',
          focus: { kind: 'control', key: 'activation', label: 'Activation' },
        },
        {
          label: 'Huber on outliers',
          patch: { ...CURVE },
          say: 'A curve through a few wild points. On the output box the outliers sit far out on the error axis, where Huber turns from a parabola into a straight line: their pull is capped, and held-out R² reaches 0.9.',
          focus: { kind: 'custom', id: 'function:loss', label: 'The output' },
        },
        {
          label: 'Squared error, outliers',
          patch: { ...CURVE, regLoss: 'mse' },
          say: 'Squared error on the same wild points: the parabola keeps steepening, so the outliers far out pull hardest of all and drag the curve towards them. Held-out R² drops to about 0.7 from Huber’s 0.9.',
          focus: { kind: 'control', key: 'regLoss', label: 'Value loss' },
        },
        {
          label: 'Absolute error',
          patch: { ...CURVE, curveDataset: 'sinusoid', regLoss: 'mae' },
          say: 'Absolute error is a V: the slope is ±1 everywhere, so every point pulls with the same force however close it is. The curve gets near, then the loss jitters twice as much per epoch as squared error does.',
          focus: { kind: 'control', key: 'regLoss', label: 'Value loss' },
        },
      ],
    },
  ],
};

export const LESSONS: ActLesson[] = [bend, ridges, vanishing, experiments];
