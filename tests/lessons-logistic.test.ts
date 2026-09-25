/** The logistic regression lessons, driven against the real model. */

import { strict as assert } from 'node:assert';
import { describe, it } from 'node:test';

import { DEFAULT_PARAMS, PRESETS } from '../src/explainers/logistic-regression/config.ts';
import type { LogRegParams } from '../src/explainers/logistic-regression/config.ts';
import { LESSONS, makeLogRegContext } from '../src/explainers/logistic-regression/lessons.ts';
import type { LogRegLessonContext } from '../src/explainers/logistic-regression/lessons.ts';
import { getClassificationDataset } from '../src/lib/datasets/points.ts';
import type { PointData } from '../src/lib/datasets/types.ts';
import { createState, makeStandardiser, step } from '../src/lib/ml/logisticRegression.ts';
import type { FeatureMap, LogRegConfig, LogRegState, Standardiser } from '../src/lib/ml/logisticRegression.ts';
import { LessonRig, checkStructure, exerciseRun, exerciseScreen, openLesson, reads } from './lessonRig.ts';

interface Model {
  data: PointData;
  config: LogRegConfig;
  std: Standardiser;
}

type Ui = LogRegLessonContext['ui'];
type Rig = LessonRig<LogRegParams, LogRegState, Model, Ui>;

const make = (): Rig =>
  new LessonRig({
    defaults: DEFAULT_PARAMS,
    presets: PRESETS,
    build: (p) => {
      const data = getClassificationDataset(p.dataset).generate({ count: p.sampleCount, noise: p.noise, seed: p.seed });
      const config: LogRegConfig = {
        learningRate: p.learningRate,
        featureMap: p.featureMap as FeatureMap,
        l2: p.l2,
        batchSize: p.batchSize,
        threshold: p.threshold,
        standardise: p.standardise,
        seed: p.seed,
      };
      return { data, config, std: makeStandardiser(data.points, p.standardise) };
    },
    create: (m) => createState(m.config),
    step: (s, m) => step(s, m.data.points, m.config, m.std),
    complete: (s) => s.converged || s.diverged,
    ui: () => ({ view: 'training', secondView: 'sigmoid', open: null, pointEdits: 0 }),
    resets: (key) => !['threshold', 'brushClass', 'showField', 'showMisclassified', 'showProjection'].includes(key),
  });

const context = (rig: Rig): LogRegLessonContext => {
  const { data, config, std } = rig.model();
  return { ...makeLogRegContext({ params: rig.params, state: rig.state, data, config, std, ui: rig.ui }), ...rig.base() };
};

describe('logistic regression lessons: structure', () => {
  it('follows the shared rules, one screen per lesson', () => checkStructure(LESSONS, PRESETS, DEFAULT_PARAMS, { oneScreen: true }));
});

const reveals: string[] = [];

describe('logistic regression lessons: every lesson and every run against the model', () => {
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

describe('logistic regression lessons: the numbers the copy quotes', () => {
  const opened = (id: string) => {
    const lesson = LESSONS.find((l) => l.id === id)!;
    return { text: String(lesson.steps[0].say), c: context(openLesson(make, lesson, context)) };
  };
  const pc = (v: number) => Math.round(v * 100) + '%';

  it('sigmoid: the trained blobs', () => {
    const { text, c } = opened('sigmoid');
    reads('sigmoid', text, 'accuracy ' + pc(c.derived.accuracy));
    reads('sigmoid', text, 'log loss ' + c.derived.logLoss.toFixed(3));
    assert.equal(c.derived.n, 120);
  });

  it('features: the ring on a line', () => {
    const { text, c } = opened('features');
    assert.ok(c.state.converged);
    reads('features', text, 'step ' + c.state.epoch);
    reads('features', text, 'accuracy ' + pc(c.derived.accuracy));
    reads('features', text, 'score ' + pc(c.derived.baseline));
  });

  it('threshold: the imbalanced matrix', () => {
    const { text, c } = opened('threshold');
    reads('threshold', text, pc(c.derived.positives / c.derived.n) + ' of the points');
    reads('threshold', text, 'accuracy ' + pc(c.derived.accuracy));
    reads('threshold', text, 'scores ' + pc(c.derived.baseline));
    reads('threshold', text, 'recall ' + c.derived.recall.toFixed(2));
    reads('threshold', text, c.derived.fn + ' of ' + c.derived.positives + ' positives missed');
  });

  it('learning rate: the reckless run', () => {
    const { text, c } = opened('learning-rate');
    assert.ok(c.state.converged);
    reads('learning-rate', text, 'step ' + c.state.epoch);
    reads('learning-rate', text, 'log loss ' + c.derived.logLoss.toFixed(3));
    reads('learning-rate', text, 'accuracy ' + pc(c.derived.accuracy));
    assert.equal(c.params.learningRate, 100);
  });
});

describe('logistic regression lessons: the claims hold', () => {
  const run = (over: Partial<LogRegParams>, cap = 3000) => {
    const rig = make();
    rig.merge({ ...PRESETS[0].params, ...over });
    rig.stepOnce(cap);
    return { rig, ctx: context(rig) };
  };

  it('threshold 0.2 on the imbalanced preset trades precision for recall', () => {
    const at = (threshold: number) => run({ ...PRESETS.find((p) => p.id === 'imbalanced')!.params, threshold }).ctx.derived;
    const half = at(0.5);
    const low = at(0.2);
    assert.ok(low.recall > half.recall + 0.1, 'recall ' + half.recall + ' -> ' + low.recall);
    assert.ok(low.fp > half.fp, 'false alarms ' + half.fp + ' -> ' + low.fp);
  });

  it('overlapping blobs flatten at loss 0.38 and accuracy 80%', () => {
    const { ctx } = run({ noise: 0.8 });
    assert.ok(Math.abs(ctx.derived.logLoss - 0.38) < 0.01, 'loss ' + ctx.derived.logLoss);
    assert.ok(Math.abs(ctx.derived.accuracy - 0.8) < 0.01, 'accuracy ' + ctx.derived.accuracy);
  });

  it('a single point per step no longer claims convergence from one quiet point', () => {
    const { rig } = run({ batchSize: 1, learningRate: 0.3 });
    assert.ok(!rig.state.converged, 'converged at ' + rig.state.epoch);
    assert.ok(rig.state.epoch === 3000);
  });

  it('moons with the quadratic map reach about 88%; the spiral stalls under 60%', () => {
    const moons = run({ dataset: 'moons', featureMap: 'quadratic' }).ctx.derived.accuracy;
    assert.ok(Math.abs(moons - 0.88) < 0.02, 'moons ' + moons);
    const spiral = run({ dataset: 'spiral', featureMap: 'quadratic' }).ctx.derived.accuracy;
    assert.ok(spiral > 0.5 && spiral < 0.6, 'spiral ' + spiral);
  });

  it('threshold 0.9 on the overlapping blobs gives precision 1, recall 0.45 and no false alarms', () => {
    const strict = run({ noise: 0.8, threshold: 0.9 }).ctx.derived;
    assert.equal(strict.precision, 1);
    assert.equal(strict.fp, 0);
    assert.ok(Math.abs(strict.recall - 0.45) < 0.01, 'recall ' + strict.recall);
  });

  it('a high rate with an L2 penalty of 1 overflows within five steps', () => {
    const { rig } = run({ learningRate: 100, l2: 1 });
    assert.ok(rig.state.diverged, 'diverged');
    assert.ok(rig.state.epoch <= 5, 'step ' + rig.state.epoch);
  });

  it('the raw quadratic moons start higher and end less accurate than the standardised run', () => {
    const raw = run({ dataset: 'moons', featureMap: 'quadratic', standardise: false });
    const scaled = run({ dataset: 'moons', featureMap: 'quadratic', standardise: true });
    assert.ok(raw.ctx.derived.initialLoss > 1.3 * scaled.ctx.derived.initialLoss, raw.ctx.derived.initialLoss + ' vs ' + scaled.ctx.derived.initialLoss);
    assert.ok(raw.ctx.derived.accuracy < scaled.ctx.derived.accuracy, raw.ctx.derived.accuracy + ' vs ' + scaled.ctx.derived.accuracy);
  });
});
