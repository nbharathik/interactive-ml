/** The linear regression lessons, driven against the real model. */

import { strict as assert } from 'node:assert';
import { describe, it } from 'node:test';

import { resolveSay } from '../src/explainer/lessons.ts';
import { DEFAULT_PARAMS, PRESETS } from '../src/explainers/linear-regression/config.ts';
import type { LinRegParams } from '../src/explainers/linear-regression/config.ts';
import { LESSONS, makeLinRegContext } from '../src/explainers/linear-regression/lessons.ts';
import type { LinRegLessonContext } from '../src/explainers/linear-regression/lessons.ts';
import { getRegressionDataset } from '../src/lib/datasets/regression.ts';
import type { RegressionData } from '../src/lib/datasets/types.ts';
import {
  closedFormSolution,
  computeLoss,
  createState,
  makeNormaliser,
  step,
  toOriginalUnits,
} from '../src/lib/ml/linearRegression.ts';
import type { BatchMode, LinRegConfig, LinRegState, Normaliser, OptimiserName } from '../src/lib/ml/linearRegression.ts';
import { LessonRig, checkStructure, exerciseRun, exerciseScreen, openLesson, reads } from './lessonRig.ts';

interface Model {
  data: RegressionData;
  config: LinRegConfig;
  norm: Normaliser;
  closedFormLoss: number | null;
  initialLoss: number;
}

type Ui = LinRegLessonContext['ui'];
type Rig = LessonRig<LinRegParams, LinRegState, Model, Ui>;

function build(p: LinRegParams): Model {
  const data = getRegressionDataset(p.dataset).generate({ count: p.sampleCount, noise: p.noise, seed: p.seed });
  const config: LinRegConfig = {
    learningRate: p.learningRate,
    degree: p.degree,
    optimiser: p.optimiser as OptimiserName,
    momentum: p.momentum,
    batchMode: p.batchMode as BatchMode,
    batchSize: p.batchSize,
    l2: p.l2,
    standardise: p.standardise,
    seed: p.seed,
  };
  const norm = makeNormaliser(data.points, p.standardise);
  const closedForm = closedFormSolution(data.points, config.degree, norm, config.l2);
  const closedFormLoss = closedForm ? computeLoss(closedForm, data.points, config, norm) : null;
  const initialLoss = computeLoss(new Array<number>(config.degree + 1).fill(0), data.points, config, norm);
  return { data, config, norm, closedFormLoss, initialLoss };
}

const make = (): Rig =>
  new LessonRig({
    defaults: DEFAULT_PARAMS,
    presets: PRESETS,
    build,
    create: (m) => createState(m.config),
    step: (s, m) => step(s, m.data.points, m.config, m.norm),
    complete: (s) => s.converged || s.diverged,
    ui: () => ({ view: 'training', lossView: 'curve', logScale: false, open: null, pointEdits: 0 }),
  });

const context = (rig: Rig): LinRegLessonContext => {
  const { data, config, norm, closedFormLoss, initialLoss } = rig.model();
  return {
    ...makeLinRegContext({ params: rig.params, state: rig.state, data, config, norm, closedFormLoss, initialLoss, ui: rig.ui }),
    ...rig.base(),
  };
};

describe('linear regression lessons: structure', () => {
  it('follows the shared rules, one screen per lesson', () => checkStructure(LESSONS, PRESETS, DEFAULT_PARAMS, { oneScreen: true }));

  it('every run with a condition drives the model', () => {
    for (const lesson of LESSONS) {
      for (const e of lesson.steps[0].experiments) {
        if (!e.until) continue;
        const drives = (e.enter ?? [{ type: 'play' }]).some((a) => a.type === 'play' || a.type === 'step' || a.type === 'runTo');
        assert.ok(drives, lesson.id + ': run ' + e.label + ' has a condition but never drives the model');
      }
    }
  });
});

const reveals: string[] = [];

describe('linear regression lessons: every lesson and every run against the model', () => {
  for (const lesson of LESSONS) {
    it(lesson.id + ' opens on a picture that reads', () => exerciseScreen(openLesson(make, lesson, context), lesson, context, reveals));
    for (const e of lesson.steps[0].experiments) {
      it(lesson.id + ' / ' + e.label, () => exerciseRun(openLesson(make, lesson, context), e, lesson.id + ' / ' + e.label, context, reveals));
    }
  }

  it('prints the live numbers', () => {
    for (const line of reveals) console.log('  ' + line);
  });
});

/* ---------------- the specific claims the copy makes ---------------- */

const lessonOf = (id: string) => LESSONS.find((l) => l.id === id)!;
const open = (id: string) => openLesson(make, lessonOf(id), context);
const sayOf = (id: string) => resolveSay(lessonOf(id).steps[0].say, context(open(id)));

describe('linear regression lessons: the claims hold', () => {
  const converge = (over: Partial<LinRegParams>, cap = 100_000) => {
    const rig = make();
    rig.merge({ ...PRESETS[0].params, ...over });
    rig.stepOnce(cap);
    return rig;
  };
  const fmt2 = (v: number) => v.toFixed(2);

  it('fitting a line: 46.56 from 4,983, converged at step 267', () => {
    const rig = open('fitting-a-line');
    assert.ok(rig.state.converged);
    reads('fitting-a-line', sayOf('fitting-a-line'), fmt2(rig.state.loss));
    reads('fitting-a-line', sayOf('fitting-a-line'), Math.round(rig.model().initialLoss).toLocaleString('en-US'));
    reads('learning-rate', sayOf('learning-rate'), String(rig.state.epoch));
    assert.ok(Math.abs(rig.state.loss - rig.model().closedFormLoss!) < 0.01);
  });

  it('too-small takes over ten thousand steps and 2.2 diverges', () => {
    const slow = converge({ learningRate: 0.0005 });
    assert.ok(slow.state.converged && slow.state.epoch > 10_000);
    assert.ok(Math.abs(slow.state.loss - 46.56) < 0.01);
    const wild = converge({ learningRate: 2.2 });
    assert.ok(wild.state.diverged);
  });

  it('outliers: three bad readings carry 96% of the loss, slope 1.49 against 1.60', () => {
    const c = context(open('outliers'));
    assert.equal(c.derived.outlierCount, 3);
    assert.equal(c.derived.n, 60);
    const share = c.derived.lossShareOfTop(3);
    reads('outliers', sayOf('outliers'), Math.round(share * 100) + '%');
    reads('outliers', sayOf('outliers'), fmt2(c.derived.line.slope));
    reads('outliers', sayOf('outliers'), fmt2(c.derived.truthSlope));
  });

  it('feature scaling: 96.68 to 3.5e+13 on step 1, overflow at step 3', () => {
    const rig = open('feature-scaling');
    assert.ok(rig.state.diverged);
    reads('feature-scaling', sayOf('feature-scaling'), fmt2(rig.model().initialLoss));
    reads('feature-scaling', sayOf('feature-scaling'), rig.state.lossHistory[0]!.toExponential(1));
    reads('feature-scaling', sayOf('feature-scaling'), 'step ' + rig.state.epoch);
    assert.equal(10 ** rig.params.degree, 100_000);
  });

  it('overfitting: loss 3.54 beats the truth at 5.40, 1.38 from the truth; degree 1 and 200 points sit closer', () => {
    const c = context(open('overfitting'));
    reads('overfitting', sayOf('overfitting'), fmt2(c.state.loss));
    reads('overfitting', sayOf('overfitting'), fmt2(c.derived.truthLoss));
    reads('overfitting', sayOf('overfitting'), fmt2(c.derived.truthRms));
    assert.ok(c.state.loss < c.derived.truthLoss);
    const overfit = PRESETS.find((p) => p.id === 'overfit')!.params;
    const one = context(converge({ ...overfit, degree: 1 }, 2000));
    assert.ok(one.derived.truthRms < c.derived.truthRms && one.state.loss > c.state.loss);
    const many = context(converge({ ...overfit, sampleCount: 200 }, 2000));
    assert.ok(many.derived.truthRms < one.derived.truthRms);
    assert.ok(Math.abs(c.derived.largestWeight - 1.84) < 0.01, 'largest weight ' + c.derived.largestWeight);
  });

  it('α = 0.5 lands on the exact answer in one step', () => {
    const rig = converge({ learningRate: 0.5 }, 1);
    assert.ok(Math.abs(rig.state.loss - rig.model().closedFormLoss!) < 1e-6);
  });

  it('α = 1 bounces at the starting loss without diverging', () => {
    const rig = converge({ learningRate: 1 }, 200);
    assert.ok(!rig.state.diverged && !rig.state.converged);
    assert.ok(Math.abs(rig.state.loss - rig.model().initialLoss) < 1e-6);
  });

  it('momentum on the slow rate arrives about eleven times sooner', () => {
    const slow = converge({ learningRate: 0.0005 });
    const fast = converge({ learningRate: 0.0005, optimiser: 'momentum' });
    assert.ok(fast.state.converged);
    const ratio = slow.state.epoch / fast.state.epoch;
    assert.ok(ratio > 10 && ratio < 12, 'ratio ' + ratio);
  });

  it('Adam tames the wild rate near step 300', () => {
    const rig = converge({ learningRate: 2.2, optimiser: 'adam' });
    assert.ok(rig.state.converged && rig.state.epoch > 250 && rig.state.epoch < 350, 'step ' + rig.state.epoch);
  });

  it('degree 1, 2 and 3 on the curve stop at 17.3, 1.85 and 1.85', () => {
    const at = (degree: number) => converge({ dataset: 'curved', degree }).state.loss;
    assert.ok(Math.abs(at(1) - 17.3) < 0.1);
    assert.ok(Math.abs(at(2) - 1.85) < 0.01);
    assert.ok(Math.abs(at(3) - 1.85) < 0.01);
  });

  it('ten points swing the slope between about 6 and 12; sixty stay near 9 to 10', () => {
    const slope = (over: Partial<LinRegParams>) => {
      const rig = converge(over);
      return toOriginalUnits(rig.state.weights, rig.model().norm).slope;
    };
    const tens = Array.from({ length: 30 }, (_, i) => slope({ sampleCount: 10, seed: i + 1 }));
    const sixties = Array.from({ length: 30 }, (_, i) => slope({ seed: i + 1 }));
    assert.ok(Math.min(...tens) < 7 && Math.max(...tens) > 11);
    assert.ok(Math.min(...sixties) > 8.5 && Math.max(...sixties) < 10.5);
  });
});
