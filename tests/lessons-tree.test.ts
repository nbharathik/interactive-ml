/**
 * The decision tree lessons, driven against the real model: every step runs
 * the way the page would, then its copy must read cleanly and the claims the
 * sandbox makes must hold.
 */

import { strict as assert } from 'node:assert';
import { describe, it } from 'node:test';

import { DEFAULT_PARAMS, PRESETS } from '../src/explainers/decision-tree/config.ts';
import type { TreeParams } from '../src/explainers/decision-tree/config.ts';
import { LESSONS, growTree, makeTreeContext } from '../src/explainers/decision-tree/lessons.ts';
import type { GrowthPoint, TreeLessonContext } from '../src/explainers/decision-tree/lessons.ts';
import { getClassificationDataset, splitData } from '../src/lib/datasets/points.ts';
import type { Point2D } from '../src/lib/datasets/types.ts';
import { accuracy, countLeaves, createState, growOne, prune } from '../src/lib/ml/decisionTree.ts';
import type { Criterion, TreeConfig, TreeState } from '../src/lib/ml/decisionTree.ts';
import { LessonRig, checkStructure, exerciseRun, exerciseScreen, openLesson, reads } from './lessonRig.ts';

interface Model {
  train: Point2D[];
  test: Point2D[];
  config: TreeConfig;
  bounds: { x0: number; x1: number; y0: number; y1: number };
  featureNames: [string, string];
  growth: GrowthPoint[];
}

type Ui = TreeLessonContext['ui'];
type Rig = LessonRig<TreeParams, TreeState, Model, Ui>;

/** The page's growth curve: the whole run from scratch, pruned as the page prunes. */
function growthCurve(train: Point2D[], test: Point2D[], config: TreeConfig, bounds: Model['bounds'], alpha: number): GrowthPoint[] {
  let current = createState(train, config, bounds);
  const out: GrowthPoint[] = [];
  const record = () => {
    const tree = alpha > 0 ? prune(current.root, alpha, Math.max(1, train.length)) : current.root;
    out.push({ train: accuracy(tree, train), test: test.length > 0 ? accuracy(tree, test) : 0, leaves: countLeaves(tree) });
  };
  record();
  for (let i = 0; i < 220 && !current.finished; i++) {
    const next = growOne(current, train, config);
    if (next.nodeCount === current.nodeCount) break;
    current = next;
    record();
  }
  return out;
}

const make = (): Rig =>
  new LessonRig({
    defaults: DEFAULT_PARAMS,
    presets: PRESETS,
    build: (p) => {
      const data = getClassificationDataset(p.dataset).generate({
        count: p.sampleCount,
        noise: p.noise,
        seed: p.seed,
        classCount: p.classCount,
      });
      const { train, test } = splitData(data.points, p.trainFraction, p.seed + 977);
      const config: TreeConfig = {
        criterion: p.criterion as Criterion,
        maxDepth: p.maxDepth,
        minSamplesSplit: p.minSamplesSplit,
        minSamplesLeaf: p.minSamplesLeaf,
        minImpurityDecrease: p.minImpurityDecrease,
        ccpAlpha: p.ccpAlpha,
        classCount: data.classCount,
      };
      const bounds = { x0: data.xRange[0], x1: data.xRange[1], y0: data.yRange[0], y1: data.yRange[1] };
      const featureNames: [string, string] = [data.xLabel, data.yLabel];
      return { train, test, config, bounds, featureNames, growth: growthCurve(train, test, config, bounds, p.ccpAlpha) };
    },
    create: (m) => createState(m.train, m.config, m.bounds),
    step: (s, m) => growOne(s, m.train, m.config),
    complete: (s) => s.finished,
    ui: () => ({ secondView: 'search', open: null }),
    // Pruning happens after growth, so α re-prunes the tree already grown.
    resets: (key) => !['ccpAlpha', 'showRegions', 'showSplitLines', 'showTestPoints', 'markMistakes'].includes(key),
  });

const context = (rig: Rig): TreeLessonContext => {
  const { train, test, config, bounds, featureNames, growth } = rig.model();
  return {
    ...makeTreeContext({ params: rig.params, state: rig.state, train, test, config, bounds, featureNames, growth, ui: rig.ui }),
    ...rig.base(),
  };
};

describe('decision tree lessons: structure', () => {
  it('follows the shared rules, one screen per lesson', () => checkStructure(LESSONS, PRESETS, DEFAULT_PARAMS, { oneScreen: true }));
});

const reveals: string[] = [];

describe('decision tree lessons: every lesson and every run against the model', () => {
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

describe('decision tree lessons: the numbers the copy quotes', () => {
  const opened = (id: string) => {
    const lesson = LESSONS.find((l) => l.id === id)!;
    return { text: String(lesson.steps[0].say), c: context(openLesson(make, lesson, context)) };
  };
  const pc = (v: number) => Math.round(v * 100) + '%';

  it('splitting: the grown tree', () => {
    const { text, c } = opened('splitting');
    assert.ok(c.sim.isComplete);
    reads('splitting', text, c.derived.trainCount + ' training points');
    reads('splitting', text, c.derived.splits + ' splits, ' + c.derived.leaves + ' leaves, depth ' + c.derived.depth);
    reads('splitting', text, 'train ' + pc(c.derived.trainAcc) + ', held-out ' + pc(c.derived.testAcc));
  });

  it('gini and entropy: the root question and the leaves', () => {
    const { text, c } = opened('gini-entropy');
    reads('gini-entropy', text, 'asks ' + c.derived.grown({}).rootQuestion);
    reads('gini-entropy', text, c.derived.leaves + ' leaves, held-out ' + pc(c.derived.testAcc));
  });

  it('max depth: the stump', () => {
    const { text, c } = opened('max-depth');
    assert.equal(c.derived.leaves, 2);
    reads('max-depth', text, 'train ' + pc(c.derived.trainAcc) + ', held-out ' + pc(c.derived.testAcc));
  });

  it('pruning: the memorising tree', () => {
    const { text, c } = opened('pruning');
    reads('pruning', text, c.derived.leaves + ' leaves, train ' + pc(c.derived.trainAcc) + ', held-out ' + pc(c.derived.testAcc));
    reads('pruning', text, 'peaked at ' + pc(c.derived.bestTest) + ' after ' + c.derived.bestSplit + ' splits');
  });

  it('axis-aligned: the diagonal stripes', () => {
    const { text, c } = opened('axis-aligned');
    reads('axis-aligned', text, c.derived.splits + ' splits');
    reads('axis-aligned', text, 'held-out ' + pc(c.derived.testAcc));
  });
});

describe('decision tree lessons: the claims hold', () => {
  const grown = (over: Partial<TreeParams>) => {
    const rig = make();
    rig.merge({ ...PRESETS[0].params, ...over });
    const { train, test, config, bounds, featureNames } = rig.model();
    return growTree(train, test, config, bounds, featureNames);
  };

  it('pruning at α = 0.008 collapses the memorising tree and lifts held-out accuracy', () => {
    const deep = grown(PRESETS.find((p) => p.id === 'memorise')!.params);
    const pruned = grown({ ...PRESETS.find((p) => p.id === 'memorise')!.params, ccpAlpha: 0.008 });
    assert.ok(pruned.leaves < deep.leaves / 2, deep.leaves + ' -> ' + pruned.leaves);
    assert.ok(pruned.testAcc > deep.testAcc, deep.testAcc + ' -> ' + pruned.testAcc);
  });

  it('the stump beats nothing but is within reach of the depth 4 tree on two blobs', () => {
    const stump = grown({ maxDepth: 1 });
    const four = grown({});
    assert.equal(stump.leaves, 2);
    assert.ok(Math.abs(four.testAcc - stump.testAcc) < 0.1, stump.testAcc + ' vs ' + four.testAcc);
  });

  it('min gain 0.05 stops the tree after a handful of questions', () => {
    const r = grown({ minImpurityDecrease: 0.05 });
    assert.ok(r.splits >= 1 && r.splits <= 6, 'splits ' + r.splits);
  });

  it('training on 30% leaves 60 training and 140 held-out points', () => {
    const rig = make();
    rig.merge({ ...PRESETS[0].params, trainFraction: 0.3 });
    assert.equal(rig.model().train.length, 60);
    assert.equal(rig.model().test.length, 140);
  });

  it('depth 2 allows at most four leaves', () => {
    assert.ok(grown({ maxDepth: 2 }).leaves <= 4);
  });

  it('the spiral at depth 8 uses over a dozen rectangles and still scores well at low noise; noise opens the gap', () => {
    const quiet = grown({ dataset: 'spiral', maxDepth: 8, noise: 0.1, sampleCount: 260 });
    assert.ok(quiet.leaves >= 12, 'leaves ' + quiet.leaves);
    assert.ok(quiet.testAcc > 0.9, 'held-out ' + quiet.testAcc);
    const noisy = grown({ dataset: 'spiral', maxDepth: 8, noise: 0.5, sampleCount: 260 });
    assert.ok(noisy.trainAcc - noisy.testAcc > quiet.trainAcc - quiet.testAcc, 'gap did not open');
  });

  it('on XOR the first cut buys little and the full tree is near perfect', () => {
    const rig = make();
    rig.applyPreset('xor');
    const growth = rig.model().growth;
    assert.ok(growth[1].test < 0.7, 'after one split ' + growth[1].test);
    assert.ok(growth[growth.length - 1].test > 0.9, 'final ' + growth[growth.length - 1].test);
  });
});
