/** The k-means lessons, driven against the real model. */

import { strict as assert } from 'node:assert';
import { describe, it } from 'node:test';

import { DEFAULT_PARAMS, PRESETS } from '../src/explainers/k-means/config.ts';
import type { KMeansParams } from '../src/explainers/k-means/config.ts';
import { LESSONS, makeKMeansContext, settleRun } from '../src/explainers/k-means/lessons.ts';
import type { KMeansLessonContext } from '../src/explainers/k-means/lessons.ts';
import { getClusteringDataset } from '../src/lib/datasets/points.ts';
import type { PointData } from '../src/lib/datasets/types.ts';
import { createState, isComplete, step } from '../src/lib/ml/kmeans.ts';
import type { InitMethod, KMeansConfig, KMeansState } from '../src/lib/ml/kmeans.ts';
import { LessonRig, checkStructure, exerciseRun, exerciseScreen, openLesson, reads } from './lessonRig.ts';

interface Model {
  data: PointData;
  config: KMeansConfig;
  bounds: { xRange: [number, number]; yRange: [number, number] };
}

type Ui = KMeansLessonContext['ui'];
type Rig = LessonRig<KMeansParams, KMeansState, Model, Ui>;

const complete = (s: KMeansState, m: Model) =>
  isComplete(s, m.config.tolerance) || (s.phase === 'update' && s.iteration > 0 && s.shift < m.config.tolerance);

const make = (): Rig =>
  new LessonRig({
    defaults: DEFAULT_PARAMS,
    presets: PRESETS,
    build: (p) => {
      const data = getClusteringDataset(p.dataset).generate({
        count: p.sampleCount,
        noise: p.noise,
        seed: p.seed,
        classCount: p.trueClusters,
      });
      const config: KMeansConfig = { k: p.k, init: p.init as InitMethod, seed: p.seed, tolerance: p.tolerance };
      return { data, config, bounds: { xRange: data.xRange, yRange: data.yRange } };
    },
    create: (m) => createState(m.data.points, m.config, m.bounds),
    step: (s, m) => step(s, m.data.points),
    complete,
    ui: () => ({ secondView: 'sizes', open: null }),
    resets: (key) => ['dataset', 'sampleCount', 'noise', 'trueClusters', 'seed', 'k', 'init'].includes(key),
  });

const context = (rig: Rig): KMeansLessonContext => {
  const { data, config, bounds } = rig.model();
  return { ...makeKMeansContext({ params: rig.params, state: rig.state, data, config, bounds, ui: rig.ui }), ...rig.base() };
};

describe('k-means lessons: structure', () => {
  it('follows the shared rules, one screen per lesson', () => checkStructure(LESSONS, PRESETS, DEFAULT_PARAMS, { oneScreen: true }));
});

const reveals: string[] = [];

describe('k-means lessons: every lesson and every run against the model', () => {
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

describe('k-means lessons: the numbers the copy quotes', () => {
  const opened = (id: string) => {
    const lesson = LESSONS.find((l) => l.id === id)!;
    return { text: String(lesson.steps[0].say), c: context(openLesson(make, lesson, context)) };
  };
  const round = (v: number) => String(Math.round(v));

  it('assignment and update: the settled textbook run', () => {
    const { text, c } = opened('assignment-update');
    assert.ok(c.sim.isComplete);
    reads('assignment-update', text, c.derived.rounds + ' rounds');
    reads('assignment-update', text, 'inertia ' + round(c.derived.inertia));
    reads('assignment-update', text, 'sizes ' + c.derived.sizes.join(' / '));
  });

  it('initialisation: the corner start', () => {
    const { text, c } = opened('initialisation');
    reads('initialisation', text, c.derived.rounds + ' rounds');
    reads('initialisation', text, 'sizes ' + c.derived.sizes.join(' / '));
    reads('initialisation', text, 'inertia ' + round(c.derived.inertia));
  });

  it('choosing k: six against three', () => {
    const { text, c } = opened('choosing-k');
    const three = c.derived.settle({ k: 3 });
    reads('choosing-k', text, 'inertia ' + round(c.derived.inertia));
    reads('choosing-k', text, 'the ' + round(three.inertia) + ' that k = 3');
    reads('choosing-k', text, 'Silhouette ' + c.derived.silhouette.toFixed(2) + ' against ' + three.silhouette.toFixed(2));
  });

  it('shapes: the rings split in half', () => {
    const { text, c } = opened('shapes');
    reads('shapes', text, Math.round(c.derived.purity * 100) + '% of the points');
  });
});

describe('k-means lessons: the claims hold', () => {
  const settled = (over: Partial<KMeansParams>) => {
    const rig = make();
    rig.merge({ ...PRESETS[0].params, ...over });
    const { data, config, bounds } = rig.model();
    return settleRun(data, config, bounds);
  };

  it('k = 1 finishes in one round at the overall mean', () => {
    const one = settled({ k: 1 });
    assert.equal(one.rounds, 1);
    const rig = make();
    rig.merge({ ...PRESETS[0].params, k: 1 });
    rig.stepOnce(400);
    const ctx = context(rig);
    assert.ok(Math.abs(ctx.derived.inertia - ctx.derived.k1Inertia) < 1e-6);
  });

  it('random coordinates still find the three blobs', () => {
    const r = settled({ init: 'random-coords' });
    assert.ok(r.purity > 0.97, 'purity ' + r.purity);
  });

  it('spread 0.8 leaves about a tenth of the points with another group', () => {
    const r = settled({ noise: 0.8 });
    assert.ok(r.purity > 0.85 && r.purity < 0.95, 'purity ' + r.purity);
  });

  it('tolerance 1 stops a round early, before an assignment confirms the answer', () => {
    const rig = make();
    rig.merge({ ...PRESETS[0].params, tolerance: 1 });
    rig.stepOnce(400);
    const full = settled({});
    assert.ok(rig.state.iteration < full.rounds, rig.state.iteration + ' vs ' + full.rounds);
    assert.ok(!rig.state.converged);
  });

  it('400 points settle in a handful of rounds', () => {
    const r = settled({ sampleCount: 400 });
    assert.ok(r.rounds <= 8, 'rounds ' + r.rounds);
    assert.ok(r.purity > 0.97, 'purity ' + r.purity);
  });

  it('the corner seeding lands on a worse inertia than k-means++ on the same points', () => {
    const good = settled({ seed: 21, init: 'kmeans++' });
    const bad = settled({ seed: 21, init: 'forgy-far' });
    assert.ok(bad.inertia > 3 * good.inertia, bad.inertia + ' vs ' + good.inertia);
  });
});
