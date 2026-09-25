/** The activations and losses lessons, driven against the real model. */

import { strict as assert } from 'node:assert';
import { describe, it } from 'node:test';

import { CONTROL_GROUPS, DEFAULT_PARAMS, PRESETS } from '../src/explainers/activation-functions/config.ts';
import type { ActParams } from '../src/explainers/activation-functions/config.ts';
import { LESSONS, deriveNumbers } from '../src/explainers/activation-functions/lessons.ts';
import type { ActLessonContext } from '../src/explainers/activation-functions/lessons.ts';
import { MODEL_KEYS, buildModel } from '../src/explainers/activation-functions/model.ts';
import type { ActModel } from '../src/explainers/activation-functions/model.ts';
import type { FocusTarget } from '../src/explainer/lessons.ts';
import { createTrainer, stepTrainer } from '../src/lib/ml/mlpTrainer.ts';
import type { TrainerState } from '../src/lib/ml/mlpTrainer.ts';
import { LessonRig, checkStructure, exerciseRun, exerciseScreen, openLesson, reads } from './lessonRig.ts';

type Ui = ActLessonContext['ui'];
type Rig = LessonRig<ActParams, TrainerState, ActModel, Ui>;

const METRIC_KEYS = ['loss', 'accuracy', 'r2', 'grad-ratio', 'flat', 'dead', 'test-loss'];
const PANEL_IDS = ['architecture', 'fit', 'second'];
const VIEW_IDS = ['graph', 'surface', 'second'];
const CONTROL_KEYS = CONTROL_GROUPS.flatMap((g) => g.controls.map((c) => c.key as string));

const make = (): Rig =>
  new LessonRig({
    defaults: DEFAULT_PARAMS,
    presets: PRESETS,
    build: buildModel,
    create: (m) => createTrainer(m.config, m.data),
    step: (s, m) => stepTrainer(s, m.data, m.config),
    complete: (s, m) => s.diverged || (s.epoch > 10 && s.trainLoss < m.fitted),
    ui: () => ({ view: 'network', surfaceOf: 'output', secondView: 'loss', open: null }),
    resets: (key) => (MODEL_KEYS as readonly string[]).includes(key),
  });

/** The ui the page would show, read from the params the way Page.tsx does. */
const context = (rig: Rig): ActLessonContext => {
  const m = rig.model();
  const view = rig.params.view === 'space' ? 'space' : rig.params.view === 'surface' && m.classify ? 'surface' : rig.params.view === 'loss' ? 'loss' : rig.params.view === 'activation' ? 'activation' : 'network';
  const surfaceOf = rig.params.surfaceOf === 'units' && rig.params.depth >= 1 ? 'units' : 'output';
  const ui: Ui = { ...rig.ui, view, surfaceOf };
  return { params: rig.params, state: rig.state, derived: deriveNumbers(rig.state, m.data, m.act, m.lossName, m.fitted), ui, ...rig.base() };
};

function checkFocus(target: FocusTarget | undefined, where: string): void {
  if (!target) return;
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
      assert.match(target.id, /^(network|output|function:(activation|loss)|layer:\d+|unit:(\d+\.)?\d+|point:\d+)$/, where + ': custom ' + target.id);
      break;
    default:
      assert.fail(where + ': unexpected focus kind ' + target.kind);
  }
}

describe('activation lessons: structure', () => {
  it('follows the shared rules, one screen per lesson', () => checkStructure(LESSONS, PRESETS, DEFAULT_PARAMS, { oneScreen: true }));

  it('every focus names something the page renders', () => {
    for (const lesson of LESSONS) {
      for (const s of lesson.steps) {
        checkFocus(s.focus, lesson.id + '/' + s.id);
        if (s.kind === 'sandbox') for (const e of s.experiments) checkFocus(e.focus, lesson.id + ' ' + e.label);
      }
    }
  });
});

const reveals: string[] = [];

describe('activation lessons: every lesson and every run against the model', () => {
  for (const lesson of LESSONS) {
    it(lesson.id + ' opens on a picture that reads', () => exerciseScreen(openLesson(make, lesson, context), lesson, context, reveals));
    const s = lesson.steps[0];
    if (s.kind !== 'sandbox') continue;
    for (const e of s.experiments) {
      it(lesson.id + ' / ' + e.label, () => exerciseRun(openLesson(make, lesson, context), e, lesson.id + ' / ' + e.label, context, reveals));
    }
  }

  it('prints the live numbers', () => {
    for (const line of reveals) console.log('  ' + line);
  });
});

describe('activation lessons: the numbers the copy quotes', () => {
  const opened = (id: string) => {
    const lesson = LESSONS.find((l) => l.id === id)!;
    return { text: String(lesson.steps[0].say), c: context(openLesson(make, lesson, context)) };
  };

  it('bend opens fitted, ridges at epoch 80, vanishing at epoch 20 with its ratio', () => {
    const bend = opened('bend');
    assert.ok(bend.c.derived.fitted, 'bend opens fitted');
    const ridges = opened('ridges');
    reads('ridges', ridges.text, 'after ' + ridges.c.derived.epoch + ' epochs');
    const vanishing = opened('vanishing');
    assert.equal(vanishing.c.derived.epoch, 20);
    reads('vanishing', vanishing.text, 'after ' + vanishing.c.derived.epoch + ' epochs');
    reads('vanishing', vanishing.text, 'h₁ gets ' + String(vanishing.c.derived.gradRatio === null ? '' : vanishing.c.derived.gradRatio.toExponential(1).replace('-', '−')));
  });
});

describe('activation lessons: the claims hold', () => {
  const run = (over: Partial<ActParams>, epochs: number) => {
    const rig = make();
    rig.merge(over);
    rig.stepOnce(epochs);
    const m = rig.model();
    return { rig, n: deriveNumbers(rig.state, m.data, m.act, m.lossName, m.fitted) };
  };
  const preset = (id: string) => PRESETS.find((p) => p.id === id)!.params;
  const sandbox = LESSONS[LESSONS.length - 1].steps[0];
  const experiment = (label: string): Partial<ActParams> => {
    assert.equal(sandbox.kind, 'sandbox');
    const found = sandbox.kind === 'sandbox' ? sandbox.experiments.find((e) => e.label === label) : undefined;
    assert.ok(found, 'experiment ' + label);
    return { ...preset('relu-deep'), ...found!.patch };
  };
  /** Mean relative change of the train loss over the last hundred epochs. */
  const jitter = (rig: Rig) => {
    const h = rig.state.lossHistory.slice(-100);
    let sum = 0;
    for (let i = 1; i < h.length; i++) sum += Math.abs(h[i].train - h[i - 1].train) / h[i - 1].train;
    return sum / (h.length - 1);
  };

  it('bend: two tanh layers of two fit the moons below 0.03; linear layers stall above 0.4', () => {
    const bent = run(preset('bend'), 300);
    assert.ok(bent.rig.complete && bent.n.trainLoss < 0.03, 'tanh loss ' + bent.n.trainLoss + ' at ' + bent.n.epoch);
    const flat = run({ ...preset('bend'), activation: 'linear' }, 300).n;
    assert.ok(flat.trainLoss > 0.4, 'linear loss ' + flat.trainLoss);
  });

  it('ridges: four ReLU units on the moons are past 90% by epoch 80', () => {
    const n = run(preset('ridges'), 80).n;
    assert.ok((n.accuracy ?? 0) >= 0.9, 'accuracy ' + n.accuracy);
  });

  it('vanishing: six sigmoid layers lose more than a thousandfold at epoch 20; tanh has fitted', () => {
    const sig = run(preset('sigmoid-deep'), 20).n;
    assert.ok(sig.gradRatio !== null && sig.gradRatio < 1e-3, 'sigmoid ratio ' + sig.gradRatio);
    assert.ok((sig.accuracy ?? 0) < 0.5, 'sigmoid accuracy ' + sig.accuracy);
    const tanh = run({ ...preset('sigmoid-deep'), activation: 'tanh' }, 20).n;
    assert.ok(tanh.gradRatio !== null && tanh.gradRatio > 1, 'tanh ratio ' + tanh.gradRatio);
    assert.ok((tanh.accuracy ?? 0) >= 0.95, 'tanh accuracy ' + tanh.accuracy);
  });

  it('experiments: squared error on classes stalls at 0.200 and 43% where cross-entropy reaches 87% by epoch 60', () => {
    const mse = run(experiment('Squared error on classes'), 60).n;
    assert.ok(Math.abs(mse.trainLoss - 0.2) < 0.002, 'mse loss ' + mse.trainLoss);
    assert.ok(Math.abs((mse.accuracy ?? 0) - 0.43) < 0.01, 'mse accuracy ' + mse.accuracy);
    const bce = run(experiment('Cross-entropy instead'), 60).n;
    assert.ok(Math.abs((bce.accuracy ?? 0) - 0.87) < 0.01, 'bce accuracy ' + bce.accuracy);
  });

  it('experiments: every run names the task and the view it needs, so runs never stack', () => {
    assert.equal(sandbox.kind, 'sandbox');
    if (sandbox.kind !== 'sandbox') return;
    for (const e of sandbox.experiments) {
      for (const key of ['task', 'view', 'activation', 'depth', 'initScale', 'learningRate']) assert.ok(key in (e.patch ?? {}), e.label + ' leaves ' + key + ' to the previous run');
      const view = e.patch?.view;
      assert.notEqual(view, 'activation', e.label + ' opens on the bare curve');
      assert.notEqual(view, 'loss', e.label + ' opens on the bare curve');
    }
  });

  it('experiments: dead ReLUs by epoch 3 at loss 0.69; leaky ReLU fits by epoch 16', () => {
    const dead = run(experiment('Dead ReLUs'), 3).n;
    assert.ok(Math.abs(dead.deadShare - 0.71) < 0.01, 'dead share ' + dead.deadShare);
    const later = run(experiment('Dead ReLUs'), 60).n;
    assert.ok(Math.abs(later.trainLoss - 0.69) < 0.01, 'dead loss ' + later.trainLoss);
    const leaky = run(experiment('Leaky ReLU'), 60);
    assert.equal(leaky.n.deadShare, 0, 'leaky dead ' + leaky.n.deadShare);
    assert.ok(leaky.rig.complete && leaky.n.epoch === 16, 'leaky fitted at ' + leaky.n.epoch);
  });

  it('experiments: the hinge clears the margin for 90% of the batch by epoch 8', () => {
    const hinge = run(experiment('Hinge loss'), 8).n;
    assert.ok(Math.abs(hinge.pastMargin - 0.9) < 0.01, 'past margin ' + hinge.pastMargin);
  });

  it('experiments: one sigmoid unit reaches 97% on the blobs by epoch 45 and stalls at 0.45 on the moons', () => {
    const blobs = run(experiment('Logistic regression'), 45).n;
    assert.ok(Math.abs((blobs.accuracy ?? 0) - 0.97) < 0.01, 'blobs ' + blobs.accuracy);
    const moons = run({ ...experiment('Logistic regression'), pointsDataset: 'moons' }, 300).n;
    assert.ok(Math.abs(moons.trainLoss - 0.45) < 0.01, 'moons loss ' + moons.trainLoss);
  });

  it('experiments: eight linear layers stay near 60% on the ring', () => {
    const linear = run(experiment('Eight linear layers'), 300).n;
    assert.ok((linear.accuracy ?? 0) < 0.75, 'eight linear layers ' + linear.accuracy);
  });

  it('experiments: Huber 0.9 vs squared error 0.7 on the outliers; absolute error jitters twice as much', () => {
    const huber = run(experiment('Huber on outliers'), 400).n;
    assert.ok(Math.abs((huber.r2 ?? 0) - 0.9) < 0.05, 'huber ' + huber.r2);
    const mse = run(experiment('Squared error, outliers'), 400).n;
    assert.ok(Math.abs((mse.r2 ?? 0) - 0.7) < 0.1, 'mse ' + mse.r2);
    const mae = run(experiment('Absolute error'), 400).rig;
    const sq = run({ ...experiment('Absolute error'), regLoss: 'mse' }, 400).rig;
    assert.ok(jitter(mae) > 1.5 * jitter(sq), 'jitter ' + jitter(mae) + ' vs ' + jitter(sq));
  });

  it('experiments: softmax reaches 100% within 20 epochs', () => {
    const soft = run(experiment('Softmax, three classes'), 20).n;
    assert.equal(soft.accuracy, 1, 'softmax ' + soft.accuracy);
  });
});
