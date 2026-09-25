/** The neural network lessons, driven against the real model. */

import { strict as assert } from 'node:assert';
import { describe, it } from 'node:test';

import type { FocusTarget, LessonAction, LessonStep } from '../src/explainer/lessons.ts';
import { focusKey, resolveSay } from '../src/explainer/lessons.ts';
import { getClassificationDataset } from '../src/lib/datasets/points.ts';
import { createState, splitDataset, step } from '../src/lib/ml/neuralNetwork.ts';
import type { Activation, InputFeature, Initialiser, NetConfig, NetState, Regularisation, TrainingData } from '../src/lib/ml/neuralNetwork.ts';
import { CONTROL_GROUPS, DEFAULT_PARAMS, PRESETS, parseLayers } from '../src/explainers/neural-network/config.ts';
import type { NeuralNetParams } from '../src/explainers/neural-network/config.ts';
import { computeNeuronGrids, countFlatUnits } from '../src/explainers/neural-network/grids.ts';
import { LESSONS, makeNetContext } from '../src/explainers/neural-network/lessons.ts';
import type { NetLessonContext } from '../src/explainers/neural-network/lessons.ts';

type Step = LessonStep<NetLessonContext, NeuralNetParams>;
type Ui = NetLessonContext['ui'];

const METRIC_KEYS = ['train-loss', 'test-loss', 'train-acc', 'test-acc', 'gap', 'max-weight'];
const PANEL_IDS = ['architecture', 'boundary', 'second'];
const VIEW_IDS = ['second', 'scale'];
const CUSTOM_IDS = ['layers'];
const CONTROL_KEYS = CONTROL_GROUPS.flatMap((g) => g.controls.map((c) => c.key as string));

/* ---------------- the model, built the way Page.tsx builds it ---------------- */

function featuresOf(params: NeuralNetParams): InputFeature[] {
  const chosen: InputFeature[] = [];
  if (params.featX1) chosen.push('x1');
  if (params.featX2) chosen.push('x2');
  if (params.featX1sq) chosen.push('x1sq');
  if (params.featX2sq) chosen.push('x2sq');
  if (params.featX1x2) chosen.push('x1x2');
  if (params.featSinX1) chosen.push('sinx1');
  if (params.featSinX2) chosen.push('sinx2');
  return chosen;
}

function build(params: NeuralNetParams): { split: TrainingData; features: InputFeature[]; config: NetConfig } {
  const dataset = getClassificationDataset(params.dataset).generate({
    count: params.sampleCount,
    noise: params.noise,
    seed: params.seed,
  });
  const split = splitDataset(dataset.points, params.testFraction, params.seed);
  const features = featuresOf(params);
  const config: NetConfig = {
    hiddenLayers: parseLayers(params.hiddenLayers),
    activation: params.activation as Activation,
    learningRate: params.learningRate,
    batchSize: params.batchSize,
    inputFeatures: features,
    initialiser: params.initialiser as Initialiser,
    regularisation: params.regularisation as Regularisation,
    regRate: params.regRate,
    seed: params.seed,
    testFraction: params.testFraction,
  };
  return { split, features, config };
}

/** A stand-in for the page: params, the simulation and the bits of ui the lessons read. */
class Driver {
  params: NeuralNetParams = { ...DEFAULT_PARAMS };
  split!: TrainingData;
  features!: InputFeature[];
  config!: NetConfig;
  state!: NetState;
  ui: Ui = { secondView: 'loss', open: null, layerEdits: 0 };
  metAt: number | null = null;
  enteredAt = 0;

  constructor() {
    this.rebuild();
  }

  rebuild(): void {
    const built = build(this.params);
    this.split = built.split;
    this.features = built.features;
    this.config = built.config;
    this.state = createState(this.config);
  }

  merge(patch: Partial<NeuralNetParams>): void {
    this.params = { ...this.params, ...patch };
    this.rebuild();
  }

  preset(id: string): void {
    const preset = PRESETS.find((p) => p.id === id);
    assert.ok(preset, 'preset ' + id + ' exists');
    this.params = { ...DEFAULT_PARAMS, ...preset.params };
    this.rebuild();
  }

  apply(actions: readonly LessonAction<NeuralNetParams>[] | undefined): void {
    for (const action of actions ?? []) {
      if (action.type === 'preset') this.preset(action.id);
      else if (action.type === 'params') this.merge(action.patch);
      else if (action.type === 'reset') this.state = createState(this.config);
      else if (action.type === 'view' && action.id === 'second') this.ui = { ...this.ui, secondView: action.value as Ui['secondView'] };
      else if (action.type === 'open') this.ui = { ...this.ui, open: action.target };
      else if (action.type === 'step') this.run(action.count);
      else if (action.type === 'runTo') this.run(action.steps);
    }
  }

  ctx(): NetLessonContext {
    const grids = computeNeuronGrids(this.state.layers, this.features, this.config.activation, 28);
    return {
      ...makeNetContext({
        params: this.params,
        state: this.state,
        split: this.split,
        features: this.features,
        grids,
        ui: this.ui,
      }),
      sim: { iteration: this.state.epoch, isRunning: false, isComplete: this.state.diverged },
      lesson: { metAt: this.metAt, enteredAt: this.enteredAt },
    };
  }

  run(epochs: number): void {
    for (let i = 0; i < epochs && !this.state.diverged; i++) this.state = step(this.state, this.split, this.config);
  }

  /** Step until the condition holds, as the runtime polls it after every epoch. */
  runUntil(condition: (ctx: NetLessonContext) => boolean, cap: number): NetLessonContext {
    let ctx = this.ctx();
    let epochs = 0;
    while (!condition(ctx)) {
      assert.ok(epochs < cap, 'condition holds within ' + cap + ' epochs');
      this.run(1);
      epochs += 1;
      ctx = this.ctx();
    }
    this.metAt = this.state.epoch;
    return ctx;
  }

  enter(): void {
    this.metAt = null;
    this.enteredAt = this.state.epoch;
    this.ui = { ...this.ui, open: null, layerEdits: 0 };
  }
}

/* ---------------- copy rules ---------------- */

const DASH = new RegExp('[' + String.fromCharCode(0x2013, 0x2014) + ']');

function clean(text: string, where: string): void {
  assert.equal(DASH.test(text), false, where + ': no dashes in ' + JSON.stringify(text));
  assert.equal(text.includes('!'), false, where + ': no exclamation marks');
  assert.equal(/\b(wrong|right)\b/i.test(text), false, where + ': no verdict words');
  for (const bad of ['NaN', 'undefined', 'n/a']) {
    assert.equal(text.includes(bad), false, where + ': ' + JSON.stringify(text) + ' contains ' + bad);
  }
}

function within(text: string, limit: number, where: string): void {
  assert.ok(text.length <= limit, where + ': ' + text.length + ' > ' + limit + ' chars: ' + JSON.stringify(text));
}

function checkFocus(target: FocusTarget | undefined, where: string): void {
  if (!target) return;
  assert.ok(target.label, where + ': focus has a label');
  switch (target.kind) {
    case 'metric':
      assert.ok(METRIC_KEYS.includes(target.key), where + ': metric ' + target.key);
      break;
    case 'control':
      assert.ok(CONTROL_KEYS.includes(target.key), where + ': control ' + target.key);
      break;
    case 'panel':
      assert.ok(PANEL_IDS.includes(target.id), where + ': panel ' + target.id);
      break;
    case 'view':
      assert.ok(VIEW_IDS.includes(target.id), where + ': view ' + target.id);
      break;
    case 'custom':
      assert.ok(CUSTOM_IDS.includes(target.id), where + ': custom ' + target.id);
      break;
    case 'node':
    case 'edge':
      assert.match(target.id, /^(point|output|input:\d+|unit:\d+:\d+|weight:\d+:\d+:\d+)$/, where + ': target ' + target.id);
      break;
    default:
      break;
  }
  assert.ok(focusKey(target).length > 0);
}

/* ---------------- tests ---------------- */

describe('neural network lessons: structure', () => {
  it('is one screen per lesson: a single sandbox step that opens trained, with two to six runs', () => {
    for (const lesson of LESSONS) {
      assert.equal(lesson.steps.length, 1, lesson.id + ' has ' + lesson.steps.length + ' steps');
      const s = lesson.steps[0];
      const drives = (s.enter ?? []).some((a) => a.type === 'runTo' || a.type === 'step' || a.type === 'play');
      assert.ok(drives || lesson.section === 'sandbox', lesson.id + ' must open on its picture');
      const most = lesson.section === 'sandbox' ? 12 : 6;
      assert.ok(s.experiments.length >= 2 && s.experiments.length <= most, lesson.id + ' has ' + s.experiments.length + ' runs');
      for (const key of lesson.knobs ?? []) assert.ok(key in DEFAULT_PARAMS, lesson.id + ': unknown knob ' + key);
    }
  });

  it('ids are unique, presets exist, lessons load their preset, runs restate their knobs', () => {
    const ids = new Set<string>();
    for (const lesson of LESSONS) {
      assert.equal(ids.has(lesson.id), false, 'lesson id ' + lesson.id + ' is unique');
      ids.add(lesson.id);
      assert.ok(PRESETS.some((p) => p.id === lesson.presetId), lesson.id + ': preset ' + lesson.presetId + ' exists');
      const s = lesson.steps[0];
      const first = s.enter?.[0];
      assert.deepEqual(first, { type: 'preset', id: lesson.presetId }, lesson.id + ': the lesson loads its preset');
      for (const action of s.enter ?? []) {
        if (action.type === 'preset') assert.ok(PRESETS.some((p) => p.id === action.id), lesson.id + ': preset ' + action.id);
        if (action.type === 'params') for (const key of Object.keys(action.patch)) assert.ok(key in DEFAULT_PARAMS, key);
      }
      checkFocus(s.focus, lesson.id);
      for (const e of s.experiments) {
        within(e.label, 24, lesson.id + ' run label');
        within(e.say, 220, lesson.id + ' run say');
        clean(e.say, lesson.id + ' run');
        for (const key of Object.keys(e.patch ?? {})) assert.ok(key in DEFAULT_PARAMS, 'patch key ' + key);
        for (const action of e.enter ?? []) {
          if (action.type === 'preset') assert.ok(PRESETS.some((p) => p.id === action.id), lesson.id + ': preset ' + action.id);
          if (action.type === 'params') for (const key of Object.keys(action.patch)) assert.ok(key in DEFAULT_PARAMS, key);
        }
        if (e.until) assert.ok(e.then, lesson.id + ': run ' + e.label + ' has a condition but no outcome');
        checkFocus(e.focus, lesson.id + ' run ' + e.label);
      }
      clean(lesson.title, lesson.id + ' title');
      clean(lesson.hook, lesson.id + ' hook');
    }
    const sections = LESSONS.map((l) => l.section);
    assert.equal(sections[0], 'hook');
    assert.equal(sections[sections.length - 1], 'sandbox');
  });
});

/** The idea to keep reads cleanly on the opening picture. */
function takeaway(s: Step, ctx: NetLessonContext, where: string): string {
  if (s.takeaway === undefined) return '';
  const text = resolveSay(s.takeaway, ctx);
  clean(text, where + ' takeaway');
  within(text, 250, where + ' takeaway');
  return text;
}

/** Open a lesson the way the runtime does: setup, then the drive that opens it on its picture. */
function open(lesson: (typeof LESSONS)[number]): Driver {
  const d = new Driver();
  d.enter();
  d.apply(lesson.steps[0].enter);
  return d;
}

describe('neural network lessons: every lesson and every run against the model', () => {
  for (const lesson of LESSONS) {
    const s = lesson.steps[0];
    it(lesson.id + ' opens on a picture that reads', () => {
      const d = open(lesson);
      const ctx = d.ctx();
      const text = resolveSay(s.say, ctx);
      clean(text, lesson.id);
      within(text, 260, lesson.id + ' say');
      const idea = takeaway(s, ctx, lesson.id);
      const numbers = s.numbers?.(ctx) ?? [];
      for (const n of numbers) clean(n.label + ' ' + n.value, lesson.id + ' number');
      console.log('  ' + lesson.id + ' opens [epoch ' + ctx.state.epoch + '] ' + numbers.map((n) => n.label + ' ' + n.value).join(', ') + (idea ? ' / ' + idea : ''));
    });
    for (const e of s.experiments) {
      it(lesson.id + ' / ' + e.label, () => {
        const d = open(lesson);
        const where = lesson.id + ' / ' + e.label;
        const list: LessonAction<NeuralNetParams>[] = e.enter ?? [
          ...(e.patch ? [{ type: 'params' as const, patch: e.patch }] : []),
          { type: 'reset' as const },
          { type: 'play' as const },
        ];
        d.enter();
        d.apply(list.filter((a) => a.type !== 'play'));
        if (!e.until) return;
        const ctx = list.some((a) => a.type === 'play') ? d.runUntil(e.until, 1000) : d.ctx();
        assert.ok(e.until(ctx), where + ': condition never held');
        const then = resolveSay(e.then, ctx);
        clean(then, where);
        within(then, 220, where + ' then');
        console.log('  ' + where + ' [epoch ' + ctx.state.epoch + '] ' + then);
      });
    }
  }
});

describe('neural network lessons: the numbers the copy quotes', () => {
  const reads = (where: string, text: string, value: string) => assert.ok(text.includes(value), where + ' should read ' + value + ': ' + text);
  const opened = (id: string) => {
    const lesson = LESSONS.find((l) => l.id === id)!;
    return { text: String(lesson.steps[0].say), c: open(lesson).ctx() };
  };
  const pc = (v: number) => Math.round(v * 100) + '%';

  it('hidden units: 17 parameters, both scores', () => {
    const { text, c } = opened('hidden-units');
    reads('hidden-units', text, c.derived.paramCount + ' parameters');
    reads('hidden-units', text, 'train accuracy ' + pc(c.state.trainAccuracy) + ', held-out ' + pc(c.state.testAccuracy));
  });

  it('no hidden layer: 3 parameters, stuck', () => {
    const { text, c } = opened('no-hidden');
    reads('no-hidden', text, c.derived.paramCount + ' parameters');
    reads('no-hidden', text, 'after ' + c.state.epoch + ' epochs');
    reads('no-hidden', text, 'stuck at ' + pc(c.state.trainAccuracy));
  });

  it('feature learning, capacity, initialisation: the stalled numbers', () => {
    const xor = opened('feature-learning');
    reads('feature-learning', xor.text, 'after ' + xor.c.state.epoch + ' epochs');
    reads('feature-learning', xor.text, 'accuracy is ' + pc(xor.c.state.trainAccuracy));
    const cap = opened('capacity');
    reads('capacity', cap.text, 'accuracy ' + pc(cap.c.state.trainAccuracy));
    reads('capacity', cap.text, 'flat at ' + cap.c.state.trainLoss.toFixed(2));
    const zeros = opened('initialisation');
    reads('initialisation', zeros.text, 'accuracy is ' + pc(zeros.c.state.trainAccuracy));
    assert.equal(zeros.c.derived.flatUnits, zeros.c.derived.hiddenUnits);
  });

  it('learning rate, dead relu, overfitting: the opening runs', () => {
    const eta = opened('learning-rate');
    reads('learning-rate', eta.text, 'accuracy ' + pc(eta.c.state.trainAccuracy) + ' by epoch ' + eta.c.state.epoch);
    const relu = opened('dead-relu');
    reads('dead-relu', relu.text, relu.c.derived.flatUnits + ' of ' + relu.c.derived.hiddenUnits + ' units');
    reads('dead-relu', relu.text, 'Epoch ' + relu.c.state.epoch + ': train ' + pc(relu.c.state.trainAccuracy) + ', held-out ' + pc(relu.c.state.testAccuracy));
    const over = opened('overfitting');
    reads('overfitting', over.text, 'Train loss ' + over.c.state.trainLoss.toFixed(3));
    reads('overfitting', over.text, 'bottomed at ' + over.c.derived.bestTest!.loss.toFixed(3) + ' near epoch ' + over.c.derived.bestTest!.epoch);
    reads('overfitting', over.text, 'is ' + over.c.state.testLoss.toFixed(3) + ' at epoch ' + over.c.state.epoch);
    reads('overfitting', over.text, 'Train ' + pc(over.c.state.trainAccuracy) + ', held-out ' + pc(over.c.state.testAccuracy));
  });
});

describe('neural network lessons: the sandbox claims', () => {
  const sandbox = LESSONS[LESSONS.length - 1];
  const experiments = sandbox.steps[0].kind === 'sandbox' ? sandbox.steps[0].experiments : [];
  const start = (patch: Partial<NeuralNetParams> | undefined) => {
    const d = new Driver();
    d.preset(sandbox.presetId);
    d.merge(patch ?? {});
    return d;
  };
  const flat = (d: Driver) => countFlatUnits(computeNeuronGrids(d.state.layers, d.features, d.config.activation, 28));

  it('sigmoid stays near 54% after 1,500 epochs', () => {
    const d = start(experiments[0].patch);
    d.run(1500);
    assert.ok(Math.abs(d.state.trainAccuracy - 0.54) < 0.03, 'train accuracy ' + d.state.trainAccuracy);
  });

  it('no activation stays near 54%', () => {
    const d = start(experiments[1].patch);
    d.run(300);
    assert.ok(Math.abs(d.state.trainAccuracy - 0.54) < 0.03, 'train accuracy ' + d.state.trainAccuracy);
  });

  it('seed 3 reaches 90% at epoch 116', () => {
    const d = start(experiments[2].patch);
    d.runUntil((c) => c.state.trainAccuracy >= 0.9, 400);
    assert.equal(d.state.epoch, 116);
  });

  it('tanh at 0.5 keeps every unit alive, 90% by 118 and 100% by 125', () => {
    const d = start(experiments[3].patch);
    d.runUntil((c) => c.state.trainAccuracy >= 0.9, 400);
    assert.equal(d.state.epoch, 118);
    d.runUntil((c) => c.state.trainAccuracy >= 0.999, 400);
    assert.equal(d.state.epoch, 125);
    assert.equal(flat(d), 0);
  });

  it('L1 at 0.3 flattens 21 of 24 units and lands near 78%', () => {
    const d = start(experiments[4].patch);
    d.run(500);
    assert.equal(flat(d), 21);
    assert.ok(Math.abs(d.state.trainAccuracy - 0.78) < 0.01, 'train accuracy ' + d.state.trainAccuracy);
  });

  it('four units on the memorising data: gap about 0.06, 96% held-out', () => {
    const d = start(experiments[5].patch);
    d.run(500);
    assert.ok(Math.abs(d.state.testLoss - d.state.trainLoss - 0.06) < 0.005, 'gap ' + (d.state.testLoss - d.state.trainLoss));
    assert.ok(Math.abs(d.state.testAccuracy - 0.96) < 0.001, 'held-out ' + d.state.testAccuracy);
  });
});
