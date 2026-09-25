/** The convolutional network lessons, driven against the real model. */

import { strict as assert } from 'node:assert';
import { describe, it } from 'node:test';

import { CONTROL_GROUPS, DEFAULT_PARAMS, PRESETS } from '../src/explainers/cnn/config.ts';
import type { CnnParams } from '../src/explainers/cnn/config.ts';
import { LESSONS, REFERENCE, isFitted, makeCnnContext } from '../src/explainers/cnn/lessons.ts';
import type { CnnLessonContext } from '../src/explainers/cnn/lessons.ts';
import { generateGlyphs } from '../src/lib/datasets/images.ts';
import type { ImageDataset } from '../src/lib/datasets/images.ts';
import { createState, forwardProbe, parameterCount, step } from '../src/lib/ml/conv.ts';
import type { CnnSpec, CnnState } from '../src/lib/ml/conv.ts';
import type { OptimiserName } from '../src/lib/ml/optim.ts';
import { LessonRig, checkStructure, exerciseRun, exerciseScreen, openLesson, reads } from './lessonRig.ts';

interface Model {
  data: ImageDataset;
  spec: CnnSpec;
}

type Ui = CnnLessonContext['ui'];
type Rig = LessonRig<CnnParams, CnnState, Model, Ui>;

const METRIC_KEYS = ['test-acc', 'train-loss', 'train-acc', 'test-loss', 'params', 'epoch'];
const PANEL_IDS = ['architecture', 'gallery', 'second'];
const VIEW_IDS = ['arch', 'second'];
const CONTROL_KEYS = CONTROL_GROUPS.flatMap((g) => g.controls.map((c) => c.key as string));

/** The model built the way Page.tsx builds it. */
function build(p: CnnParams): Model {
  const data = generateGlyphs({ size: 16, classCount: p.classCount, count: p.count, jitter: p.jitter, thickness: p.thickness, noise: p.noise, seed: p.seed, trainFraction: 0.75 });
  const spec: CnnSpec = {
    size: 16,
    model: p.model === 'dense' ? 'dense' : 'cnn',
    kernel: Number(p.kernel) === 5 ? 5 : 3,
    filters: p.filters,
    padding: p.padding === 'same' ? 'same' : 'valid',
    pool: p.pool === 'avg' ? 'avg' : p.pool === 'none' ? 'none' : 'max',
    secondConv: p.secondConv,
    secondFilters: p.filters,
    dense: p.dense,
    classCount: data.classNames.length,
    seed: p.seed,
    optimiser: { name: p.optimiser as OptimiserName },
    learningRate: p.learningRate,
    batchSize: p.batchSize,
    evalEvery: 5,
  };
  return { data, spec };
}

const make = (): Rig =>
  new LessonRig({
    defaults: DEFAULT_PARAMS,
    presets: PRESETS,
    build,
    create: (m) => createState(m.spec, m.data),
    step: (s, m) => step(s, m.data, m.spec),
    complete: (s) => s.diverged || isFitted(s),
    ui: () => ({ view: '2d', secondView: 'curves', open: null }),
    // Every knob but the probe rebuilds the run.
    resets: (key) => key !== 'probe',
  });

const context = (rig: Rig): CnnLessonContext => {
  const { data, spec } = rig.model();
  const probeIndex = Math.min(rig.params.probe, data.testIndex.length - 1);
  const image = data.testIndex[probeIndex];
  const cache = forwardProbe(rig.state.params, data.images[image], spec);
  return { ...makeCnnContext({ params: rig.params, state: rig.state, data, spec, cache, label: data.labels[image], ui: rig.ui }), ...rig.base() };
};

describe('cnn lessons: structure', () => {
  it('follows the shared rules, one screen per lesson', () => checkStructure(LESSONS, PRESETS, DEFAULT_PARAMS, { oneScreen: true }));

  it('every focus points at something the page renders', () => {
    for (const lesson of LESSONS) {
      for (const s of lesson.steps) {
        const targets = [s.focus, ...s.experiments.map((e) => e.focus)];
        for (const target of targets) {
          if (!target) continue;
          const where = lesson.id + '/' + s.id;
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
            case 'node':
            case 'edge':
              assert.match(target.id, /^(image|flat|filter:\d+:\d+|map:\d+:\d+|pool:\d+:\d+|unit:\d+|out:\d+|w:\d+:\d+)$/, where + ': target ' + target.id);
              break;
            default:
              break;
          }
        }
      }
    }
  });
});

const reveals: string[] = [];

describe('cnn lessons: every lesson and every run against the model', () => {
  for (const lesson of LESSONS) {
    it(lesson.id + ' opens on a picture that reads', () => exerciseScreen(openLesson(make, lesson, context), lesson, context, reveals));
    const s = lesson.steps[0];
    for (const e of s.experiments) {
      it(lesson.id + ' / ' + e.label, () => exerciseRun(openLesson(make, lesson, context), e, lesson.id + ' / ' + e.label, context, reveals));
    }
  }

  it('prints the live numbers', () => {
    for (const line of reveals) console.log('  ' + line);
  });
});

describe('cnn lessons: the numbers the copy quotes', () => {
  const opened = (id: string) => {
    const lesson = LESSONS.find((l) => l.id === id)!;
    return { text: String(lesson.steps[0].say), c: context(openLesson(make, lesson, context)) };
  };
  const pc = (v: number) => Math.round(v * 100) + '%';
  const n = (v: number) => v.toLocaleString('en-US');

  it('stacking, few kernels, noise and raw pixels quote their opening numbers', () => {
    const two = opened('stacking');
    assert.equal(two.c.derived.paramCount, REFERENCE.twoConvSixClasses.params);
    const one = opened('few-kernels');
    reads('few-kernels', one.text, pc(one.c.state.testAccuracy) + ' held-out at batch ' + one.c.state.step);
    reads('few-kernels', one.text, n(one.c.derived.paramCount) + ' parameters');
    const speckle = opened('noise');
    reads('noise', speckle.text, 'held-out accuracy ' + pc(speckle.c.state.testAccuracy) + ' at batch ' + speckle.c.state.step);
    const dense = opened('raw-pixels');
    reads('raw-pixels', dense.text, 'After ' + dense.c.state.step + ' batches: training accuracy ' + pc(dense.c.state.trainAccuracy) + ', held-out ' + pc(dense.c.state.testAccuracy));
    reads('raw-pixels', dense.text, 'only ' + dense.c.derived.trainCount + ' training images');
    reads('raw-pixels', dense.text, dense.c.derived.denseUnits + ' dense units');
  });

  it('the receptive field and the head quote the map and the vector', () => {
    const field = opened('receptive-field');
    reads('receptive-field', field.text, n(field.c.derived.positions) + ' pixels share');
    const head = opened('dense-head');
    reads('dense-head', head.text, 'the ' + head.c.derived.kernels + ' maps');
    reads('dense-head', head.text, 'vector of ' + n(head.c.derived.flatSize) + ' numbers');
    reads('dense-head', head.text, head.c.derived.denseUnits + ' dense units');
  });
});

describe('cnn lessons: the claims hold', () => {
  const fresh = (patch: Partial<CnnParams>) => {
    const rig = make();
    rig.merge(patch);
    return rig;
  };
  const runUntil = (rig: Rig, until: (s: CnnState) => boolean, cap: number) => {
    let n = 0;
    while (!until(rig.state) && n < cap) {
      rig.stepOnce();
      n += 1;
    }
    return rig.state;
  };
  const fitAt = (patch: Partial<CnnParams>, cap = 400) => runUntil(fresh(patch), isFitted, cap).step;
  const to90 = (patch: Partial<CnnParams>, cap = 400) => runUntil(fresh(patch), (s) => s.testAccuracy >= 0.9, cap).step;

  it('four kernels fit at the reference batch; one kernel reaches 90% at its reference', () => {
    assert.equal(fitAt({}), REFERENCE.fourKernelsFit);
    assert.equal(to90({ filters: 1 }), REFERENCE.oneKernelTo90);
    assert.ok(to90({}) < REFERENCE.oneKernelTo90, 'four kernels reach 90% sooner');
  });

  it('max pooling reaches 90% at its reference and no pooling takes longer with four times the inputs', () => {
    assert.equal(to90({ jitter: 3 }), REFERENCE.maxPoolTo90);
    const none = fresh({ jitter: 3, pool: 'none' });
    assert.equal(none.state.params.hidden!.inSize, 4 * 14 * 14);
    assert.ok(to90({ jitter: 3, pool: 'none' }) > REFERENCE.maxPoolTo90);
    assert.ok(parameterCount(none.state.params) > 3 * parameterCount(fresh({ jitter: 3 }).state.params));
  });

  it('two convolutions on six classes beat one with far fewer parameters', () => {
    const two = fresh({ secondConv: true, classCount: 6 });
    assert.equal(parameterCount(two.state.params), REFERENCE.twoConvSixClasses.params);
    runUntil(two, (s) => s.step >= 300, 300);
    assert.ok(Math.abs(two.state.testAccuracy - REFERENCE.twoConvSixClasses.test) < 0.011, 'two layers ' + two.state.testAccuracy);
    const one = fresh({ classCount: 6 });
    assert.equal(parameterCount(one.state.params), REFERENCE.oneConvSixClasses.params);
    runUntil(one, (s) => s.step >= 300, 300);
    assert.ok(Math.abs(one.state.testAccuracy - REFERENCE.oneConvSixClasses.test) < 0.011, 'one layer ' + one.state.testAccuracy);
    assert.ok(one.state.testAccuracy < two.state.testAccuracy, 'one layer ' + one.state.testAccuracy);
  });

  it('the dense model memorises shifted glyphs while the convolutional one generalises', () => {
    const dense = fresh({ model: 'dense', jitter: 4, count: 120 });
    runUntil(dense, (s) => s.step >= 200, 200);
    assert.ok(dense.state.trainAccuracy >= 0.99, 'dense train ' + dense.state.trainAccuracy);
    assert.ok(dense.state.testAccuracy < 0.9, 'dense held-out ' + dense.state.testAccuracy);
    const cnn = fresh({ jitter: 4, count: 120 });
    runUntil(cnn, (s) => s.step >= 200, 200);
    assert.ok(cnn.state.testAccuracy > dense.state.testAccuracy + 0.05, 'cnn held-out ' + cnn.state.testAccuracy);
  });

  it('noise 0.5 opens a gap between training and held-out accuracy', () => {
    const noisy = fresh({ noise: 0.5 });
    runUntil(noisy, (s) => s.step >= 200, 200);
    assert.ok(noisy.state.trainAccuracy - noisy.state.testAccuracy > 0.1, 'gap ' + (noisy.state.trainAccuracy - noisy.state.testAccuracy));
  });

  describe('the experiments', () => {
    const sandbox = LESSONS[LESSONS.length - 1].steps[0];
    const experiments = sandbox.experiments;
    const patchOf = (label: string) => {
      const found = experiments.find((e) => e.label === label);
      assert.ok(found, 'experiment ' + label);
      return found.patch ?? {};
    };

    it('average pooling on shifted glyphs: 90% at batch 24, fitted at 48', () => {
      assert.equal(to90(patchOf('Average pooling')), 24);
      assert.equal(fitAt(patchOf('Average pooling')), 48);
    });

    it('eight kernels: 6,436 parameters, fitted at batch 24', () => {
      assert.equal(parameterCount(fresh(patchOf('Eight kernels')).state.params), 6436);
      assert.equal(fitAt(patchOf('Eight kernels')), 24);
    });

    it('kernel 5 × 5: 2,492 parameters, 12 × 12 maps, fitted at batch 45', () => {
      const rig = fresh(patchOf('Kernel 5 × 5'));
      assert.equal(parameterCount(rig.state.params), 2492);
      assert.equal(context(rig).derived.mapSize, 12);
      assert.equal(fitAt(patchOf('Kernel 5 × 5')), 45);
    });

    it('same padding: 16 × 16 maps, 256 values, 4,220 parameters, fitted at batch 45', () => {
      const rig = fresh(patchOf('Same padding'));
      assert.equal(parameterCount(rig.state.params), 4220);
      const ctx = context(rig);
      assert.equal(ctx.derived.mapSize, 16);
      assert.equal(ctx.derived.flatSize, 256);
      assert.equal(fitAt(patchOf('Same padding')), 45);
    });

    it('no dense layer: 828 parameters and 100% held-out by batch 45', () => {
      const rig = fresh(patchOf('No dense layer'));
      assert.equal(parameterCount(rig.state.params), 828);
      runUntil(rig, (s) => s.step >= 45, 45);
      assert.equal(rig.state.testAccuracy, 1);
    });

    it('batch 64 fits after 21 batches', () => {
      assert.equal(fitAt(patchOf('Batch 64')), 21);
    });

    it('plain SGD reaches 90% at batch 50 and fits at 95', () => {
      assert.equal(to90(patchOf('Plain SGD')), 50);
      assert.equal(fitAt(patchOf('Plain SGD')), 95);
    });

    it('η = 0.1 kills every map and unit and sits at chance', () => {
      const rig = fresh(patchOf('η = 0.1'));
      runUntil(rig, (s) => s.step >= 400, 400);
      const ctx = context(rig);
      assert.equal(ctx.derived.deadMaps, 4);
      assert.equal(ctx.derived.deadUnits, 16);
      assert.ok(rig.state.trainAccuracy < 0.35, 'train ' + rig.state.trainAccuracy);
    });

    it('seed 3 fits at batch 21', () => {
      assert.equal(fitAt(patchOf('Seed 3')), 21);
    });
  });
});
