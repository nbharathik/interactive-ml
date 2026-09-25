/**
 * The k-nearest neighbours lessons, driven against the real model: every step
 * runs the way the page would, then its copy must read cleanly and the claims
 * the sandbox makes must hold.
 */

import { strict as assert } from 'node:assert';
import { describe, it } from 'node:test';

import { DEFAULT_PARAMS, PRESETS } from '../src/explainers/knn/config.ts';
import type { KnnParams } from '../src/explainers/knn/config.ts';
import { LESSONS, makeKnnContext } from '../src/explainers/knn/lessons.ts';
import type { KnnLessonContext } from '../src/explainers/knn/lessons.ts';
import { getClassificationDataset } from '../src/lib/datasets/points.ts';
import type { PointData } from '../src/lib/datasets/types.ts';
import { accuracyAcrossK, makeScaling } from '../src/lib/ml/knn.ts';
import type { DistanceMetric, KnnConfig, Scaling, WeightingScheme } from '../src/lib/ml/knn.ts';
import { LessonRig, checkStructure, exerciseRun, exerciseScreen, openLesson, reads } from './lessonRig.ts';

interface Model {
  data: PointData;
  config: KnnConfig;
  scaling: Scaling;
  query: { x: number; y: number };
  countedK: number;
  curve: Array<{ k: number; accuracy: number }>;
}

type State = { revealed: number };
type Ui = KnnLessonContext['ui'];
type Rig = LessonRig<KnnParams, State, Model, Ui>;

const MAX_K = 60;

const make = (): Rig =>
  new LessonRig({
    defaults: DEFAULT_PARAMS,
    presets: PRESETS,
    build: (p) => {
      const raw = getClassificationDataset(p.dataset).generate({
        count: p.sampleCount,
        noise: p.noise,
        seed: p.seed,
        classCount: p.classCount,
      });
      const stretch = p.xStretch;
      const data: PointData =
        stretch === 1
          ? raw
          : { ...raw, points: raw.points.map((q) => ({ ...q, x: q.x * stretch })), xRange: [raw.xRange[0] * stretch, raw.xRange[1] * stretch] };
      const config: KnnConfig = {
        k: Math.max(1, p.k),
        metric: p.metric as DistanceMetric,
        weighting: p.weighting as WeightingScheme,
        classCount: data.classCount,
        standardise: p.standardise,
      };
      const scaling = makeScaling(data.points, p.standardise);
      const query = { x: p.queryX * stretch, y: p.queryY };
      const { k: _k, ...rest } = config;
      return {
        data,
        config,
        scaling,
        query,
        countedK: Math.min(config.k, data.points.length),
        curve: accuracyAcrossK(data.points, MAX_K, rest, scaling),
      };
    },
    create: () => ({ revealed: 0 }),
    step: (s) => ({ revealed: s.revealed + 1 }),
    complete: (s, m) => s.revealed >= m.countedK,
    ui: () => ({ secondView: 'table', queryOpen: false, queryMoves: 0, pointEdits: 0 }),
    resets: (key) => !['paintClass', 'clickAdds', 'showRegions', 'showLinks', 'showBall'].includes(key),
  });

const context = (rig: Rig): KnnLessonContext => {
  const { data, config, scaling, query, curve } = rig.model();
  return {
    ...makeKnnContext({ params: rig.params, state: rig.state, data, config, scaling, query, curve, ui: rig.ui }),
    ...rig.base(),
  };
};

describe('k-NN lessons: structure', () => {
  it('follows the shared rules, one screen per lesson', () => checkStructure(LESSONS, PRESETS, DEFAULT_PARAMS, { oneScreen: true, instant: true }));
});

const reveals: string[] = [];

describe('k-NN lessons: every lesson and every run against the model', () => {
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

describe('k-NN lessons: the numbers the copy quotes', () => {
  const opened = (id: string) => {
    const lesson = LESSONS.find((l) => l.id === id)!;
    return { text: String(lesson.steps[0].say), c: context(openLesson(make, lesson, context)) };
  };
  const pc = (v: number) => Math.round(v * 100) + '%';

  it('majority vote: the finished count', () => {
    const { text, c } = opened('majority-vote');
    assert.ok(c.sim.isComplete);
    reads('majority-vote', text, c.derived.n + ' stored points in ' + c.derived.classNames.length + ' classes');
    reads('majority-vote', text, 'The ' + c.derived.countedK + ' nearest');
    reads('majority-vote', text, c.derived.classNames[c.derived.vote.winner] + ' ' + c.derived.vote.scores[c.derived.vote.winner] + ' of ' + c.derived.countedK);
  });

  it('choosing k: k = 1', () => {
    const { text, c } = opened('choosing-k');
    assert.equal(c.derived.k, 1);
    reads('choosing-k', text, 'reads ' + pc(c.derived.trainAccuracy));
    reads('choosing-k', text, 'from itself, ' + pc(c.derived.looAccuracy));
  });

  it('distance: the manhattan neighbours shared with euclidean', () => {
    const { text, c } = opened('distance');
    reads('distance', text, c.derived.sharedWith('euclidean') + ' of these ' + c.derived.countedK + ' neighbours');
  });

  it('imbalance: the rare class at k = 35', () => {
    const { text, c } = opened('imbalance-ties');
    reads('imbalance-ties', text, pc(1 - c.derived.baseline) + ' of the points');
    reads('imbalance-ties', text, 'k = ' + c.derived.k + ' leave-one-out recall for them is ' + pc(c.derived.looRecall(1)));
  });
});

describe('k-NN lessons: the claims hold', () => {
  const at = (over: Partial<KnnParams>) => {
    const rig = make();
    rig.merge({ ...PRESETS[0].params, ...over });
    return context(rig).derived;
  };

  it('k = 1 reads 100% training accuracy on the moons and leave-one-out falls once k spans both moons', () => {
    const d = at(PRESETS.find((p) => p.id === 'k-one')!.params);
    assert.equal(d.trainAccuracy, 1);
    assert.ok(d.looAccuracy < 1, 'loo ' + d.looAccuracy);
    assert.ok(d.looAt(59) < d.looAccuracy - 0.05, 'k = 59 ' + d.looAt(59));
  });

  it('the metric presets sit where the neighbour list changes with the metric', () => {
    for (const id of ['manhattan', 'chebyshev']) {
      const d = at(PRESETS.find((p) => p.id === id)!.params);
      assert.ok(d.sharedWith('euclidean') < d.countedK, id + ' shares every neighbour with euclidean');
    }
  });

  it('distance weighting at k = 45 beats uniform voting on the moons', () => {
    const flat = at(PRESETS.find((p) => p.id === 'k-large')!.params);
    const weighted = at(PRESETS.find((p) => p.id === 'weighted-rescue')!.params);
    assert.ok(weighted.looAccuracy > flat.looAccuracy, flat.looAccuracy + ' -> ' + weighted.looAccuracy);
  });

  it('stretching x hurts leave-one-out and standardising undoes it', () => {
    const plain = at({ dataset: 'gaussians', sampleCount: 100, k: 9 });
    const stretched = at({ dataset: 'gaussians', sampleCount: 100, k: 9, xStretch: 20 });
    const fixed = at({ dataset: 'gaussians', sampleCount: 100, k: 9, xStretch: 20, standardise: true });
    assert.ok(stretched.looAccuracy < plain.looAccuracy - 0.05, plain.looAccuracy + ' -> ' + stretched.looAccuracy);
    assert.ok(fixed.looAccuracy > stretched.looAccuracy + 0.05, stretched.looAccuracy + ' -> ' + fixed.looAccuracy);
  });

  it('the rare class is never predicted at k = 35 and comes back at k = 5', () => {
    const rare = PRESETS.find((p) => p.id === 'rare-class')!.params;
    assert.equal(at(rare).looRecall(1), 0);
    assert.ok(at({ ...rare, k: 5 }).looRecall(1) > 0.3);
  });

  it('the tie preset puts the query where k = 4 deadlocks two to two', () => {
    const d = at(PRESETS.find((p) => p.id === 'tie')!.params);
    assert.ok(d.finalVote.tied, 'vote ' + d.finalVote.scores.join('/'));
    for (const [dx, dy] of [[0.1, 0], [-0.1, 0], [0, 0.1], [0, -0.1]]) {
      const near = at({ ...PRESETS.find((p) => p.id === 'tie')!.params, queryX: 0 + dx, queryY: 0.4 + dy });
      assert.ok(near.finalVote.tied, 'still tied at ' + dx + ', ' + dy);
    }
  });

  it('heavy noise moves the best k up', () => {
    const quiet = at({});
    const noisy = at({ noise: 1 });
    assert.ok(noisy.bestK > quiet.bestK, quiet.bestK + ' -> ' + noisy.bestK);
  });

  it('the spiral at k = 3 scores high leave-one-out; k = 45 collapses to about 40%', () => {
    const tight = at({ dataset: 'spiral', sampleCount: 200, noise: 0.12, k: 3 });
    const loose = at({ dataset: 'spiral', sampleCount: 200, noise: 0.12, k: 45 });
    assert.ok(tight.looAccuracy > 0.95, 'k = 3 ' + tight.looAccuracy);
    assert.ok(Math.abs(loose.looAccuracy - 0.4) < 0.05, 'k = 45 ' + loose.looAccuracy);
  });
});
