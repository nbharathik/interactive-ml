/** Decision tree lessons, one screen each. Pure data, driven against the real model in tests/lessons-tree.test.ts. */

import type { Lesson, LessonAction, LessonContextBase, LiveNumber } from '../../explainer/lessons';
import type { Point2D } from '../../lib/datasets/types';
import { fmt, fmtPercent } from '../../lib/math/stats';
import { accuracy, countLeaves, createState, growAll, prune, treeDepth, walk } from '../../lib/ml/decisionTree';
import type { TreeConfig, TreeNode, TreeState } from '../../lib/ml/decisionTree';
import type { TreeParams } from './config';

/* ---------------- context ---------------- */

export type SecondView = 'search' | 'impurity' | 'growth';

export interface GrowthPoint {
  train: number;
  test: number;
  leaves: number;
}

export interface SplitSummary {
  /** "x₁ ≤ -0.08". */
  question: string;
  gain: number;
  left: number;
  right: number;
}

/** A whole tree grown to completion, for comparisons. */
export interface Grown {
  leaves: number;
  depth: number;
  splits: number;
  trainAcc: number;
  testAcc: number;
  rootQuestion: string | null;
}

export interface TreeLessonContext extends LessonContextBase {
  params: TreeParams;
  state: TreeState;
  derived: {
    trainCount: number;
    testCount: number;
    hasTest: boolean;
    /** Of the tree on show, after pruning. */
    leaves: number;
    grownLeaves: number;
    depth: number;
    splits: number;
    trainAcc: number;
    testAcc: number;
    gap: number;
    rootImpurity: number;
    /** Impurity left in the leaves, weighted by the data each holds. */
    weightedImpurity: number;
    lastSplit: SplitSummary | null;
    /** Splits in the full growth run. */
    totalSplits: number;
    /** Where on the growth curve the held-out accuracy peaks. */
    bestSplit: number;
    bestTest: number;
    /** Train and held-out accuracy after `splits` splits of the full run. */
    growthAt: (splits: number) => GrowthPoint | null;
    /** The same data grown to completion under different settings. */
    grown: (over: Partial<TreeConfig>) => Grown;
  };
  ui: {
    secondView: SecondView;
    /** The node whose card is open, or null. */
    open: number | null;
  };
}

export interface TreeLessonInput {
  params: TreeParams;
  state: TreeState;
  train: readonly Point2D[];
  test: readonly Point2D[];
  config: TreeConfig;
  bounds: { x0: number; x1: number; y0: number; y1: number };
  featureNames: readonly [string, string];
  /** The page's growth curve: one entry per split of the full run, pruned as the page prunes. */
  growth: readonly GrowthPoint[];
  ui: TreeLessonContext['ui'];
}

export function questionText(feature: 0 | 1, threshold: number, featureNames: readonly [string, string]): string {
  return featureNames[feature] + ' ≤ ' + fmt(threshold, 2);
}

function weightedImpurityOf(root: TreeNode, total: number): number {
  let sum = 0;
  walk(root, (node) => {
    if (node.isLeaf) sum += (node.samples.length / Math.max(1, total)) * node.impurity;
  });
  return sum;
}

/** Grow the whole tree under `config`, then prune it the way the page does. */
export function growTree(
  train: readonly Point2D[],
  test: readonly Point2D[],
  config: TreeConfig,
  bounds: TreeLessonInput['bounds'],
  featureNames: readonly [string, string],
): Grown {
  const state = growAll(createState(train, config, bounds), train, config, 400);
  const root = prune(state.root, config.ccpAlpha, Math.max(1, train.length));
  const leaves = countLeaves(root);
  return {
    leaves,
    depth: treeDepth(root),
    splits: leaves - 1,
    trainAcc: accuracy(root, train),
    testAcc: test.length > 0 ? accuracy(root, test) : Number.NaN,
    rootQuestion: root.split && !root.isLeaf ? questionText(root.split.feature, root.split.threshold, featureNames) : null,
  };
}

/** The page's share of the context; the lesson hook adds `sim` and `lesson`. */
export function makeTreeContext(input: TreeLessonInput): Omit<TreeLessonContext, keyof LessonContextBase> {
  const { params, state, train, test, config, bounds, featureNames, growth, ui } = input;
  const root = prune(state.root, params.ccpAlpha, Math.max(1, train.length));
  const leaves = countLeaves(root);
  const hasTest = test.length > 0;
  const trainAcc = accuracy(root, train);
  const testAcc = hasTest ? accuracy(root, test) : Number.NaN;
  let bestSplit = 0;
  for (let i = 1; i < growth.length; i++) if (growth[i].test > growth[bestSplit].test) bestSplit = i;
  const last = state.lastSplit;
  return {
    params,
    state,
    derived: {
      trainCount: train.length,
      testCount: test.length,
      hasTest,
      leaves,
      grownLeaves: countLeaves(state.root),
      depth: treeDepth(root),
      splits: leaves - 1,
      trainAcc,
      testAcc,
      gap: hasTest ? trainAcc - testAcc : 0,
      rootImpurity: state.root.impurity,
      weightedImpurity: weightedImpurityOf(root, train.length),
      lastSplit:
        last && last.node.split
          ? {
              question: questionText(last.node.split.feature, last.node.split.threshold, featureNames),
              gain: last.candidate.gain,
              left: last.candidate.leftCount,
              right: last.candidate.rightCount,
            }
          : null,
      totalSplits: Math.max(0, growth.length - 1),
      bestSplit,
      bestTest: growth[bestSplit]?.test ?? Number.NaN,
      growthAt: (splits) => growth[Math.min(Math.max(0, splits), growth.length - 1)] ?? null,
      grown: (over) => growTree(train, test, { ...config, ...over }, bounds, featureNames),
    },
    ui,
  };
}

/* ---------------- copy helpers ---------------- */

type Ctx = TreeLessonContext;
type L = Lesson<Ctx, TreeParams>;

const pc = (v: number) => fmtPercent(v, 0);
const done = (c: Ctx) => c.sim.isComplete;
const plural = (n: number, word: string) => n + ' ' + word + (n === 1 ? '' : 's');
const leaves = (n: number) => n + (n === 1 ? ' leaf' : ' leaves');
const scores = (c: Ctx) => 'train ' + pc(c.derived.trainAcc) + ', held-out ' + pc(c.derived.testAcc);

/* ---------------- the lessons ---------------- */

type Act = LessonAction<TreeParams>;

const SEARCH: Act = { type: 'view', id: 'second', value: 'search' };
const IMPURITY: Act = { type: 'view', id: 'second', value: 'impurity' };
const GROWTH: Act = { type: 'view', id: 'second', value: 'growth' };
const PLAY: Act = { type: 'play' };
/** Opens the lesson on the grown tree. */
const GROW: Act = { type: 'runTo', steps: 400 };

/** A run restates every knob it depends on: the preset, then its own changes, then a fresh start. */
const on = (preset: string, patch: Partial<TreeParams> | null, ...rest: Act[]): Act[] => [
  { type: 'preset', id: preset },
  ...(patch ? [{ type: 'params' as const, patch }] : []),
  { type: 'reset' },
  ...rest,
];

const numbers = (c: Ctx): LiveNumber[] => [
  { label: 'splits', value: String(c.derived.splits) },
  { label: 'leaves', value: String(c.derived.leaves) },
  { label: 'train', value: pc(c.derived.trainAcc) },
  { label: 'held out', value: pc(c.derived.testAcc) },
];

const impurityAndSplitting: L = {
  id: 'splitting',
  title: 'Impurity and splitting',
  hook: 'Each split removes the most impurity.',
  section: 'hook',
  presetId: 'watch-it-grow',
  steps: [
    {
      id: 'grown',
      kind: 'sandbox',
      say: '140 training points; the 60 faded ones are held out. Fully grown: 11 splits, 12 leaves, depth 4, train 96%, held-out 90%. Each split is one question about one feature, the one that removes the most impurity; each leaf is one rectangle. Try it: First split.',
      takeaway: 'Impurity is how mixed a node is: 0 for one class, 0.5 for an even two-class mix. The whole model is the rectangles the questions carve out.',
      focus: { kind: 'panel', id: 'field', label: 'The feature space' },
      enter: [{ type: 'preset', id: 'watch-it-grow' }, { type: 'reset' }, SEARCH, GROW],
      numbers,
      experiments: [
        {
          label: 'First split',
          say: 'The split search scores every threshold on both features; the peak is the root’s question.',
          enter: on('watch-it-grow', null, SEARCH, { type: 'step', count: 1 }),
          until: (c) => c.sim.iteration >= c.lesson.enteredAt + 1,
          then: (c) =>
            c.derived.lastSplit
              ? 'From one leaf at Gini impurity ' + fmt(c.derived.rootImpurity, 3) + ' the root asks ' + c.derived.lastSplit.question + ': gain ' + fmt(c.derived.lastSplit.gain, 3) + ', ' +
                c.derived.lastSplit.left + ' points go left and ' + c.derived.lastSplit.right + ' right.'
              : 'The tree asked its first question.',
          focus: { kind: 'panel', id: 'second', label: 'The split search' },
        },
        {
          label: 'Grow to the end',
          say: 'Each step takes the largest impurity reduction available anywhere in the tree.',
          enter: on('watch-it-grow', null, SEARCH, PLAY),
          speed: 'normal',
          until: done,
          then: (c) => 'Fully grown after ' + plural(c.derived.splits, 'split') + ': ' + leaves(c.derived.leaves) + ', depth ' + c.derived.depth + ', ' + scores(c) + '.',
          focus: { kind: 'panel', id: 'field', label: 'The feature space' },
        },
      ],
    },
  ],
};

const giniAndEntropy: L = {
  id: 'gini-entropy',
  title: 'Gini and entropy',
  hook: 'Two impurity measures, one gain rule.',
  section: 'core',
  presetId: 'watch-it-grow',
  knobs: ['criterion'],
  steps: [
    {
      id: 'criterion',
      kind: 'sandbox',
      say: 'Gini, 1 − Σp², and entropy, −Σp log₂p, over a node’s class balance: both are 0 for a pure node and largest at an even mix. Grown with Gini: the root asks x₁ ≤ −0.08, 12 leaves, held-out 90%. Try it: Entropy.',
      takeaway: 'Gain is the parent impurity minus the weighted impurity of the children, whichever measure is used. The two almost always pick the same split; Gini skips the logarithm.',
      focus: { kind: 'panel', id: 'second', label: 'Gini against entropy' },
      enter: [{ type: 'preset', id: 'watch-it-grow' }, { type: 'reset' }, IMPURITY, GROW],
      numbers,
      experiments: [
        {
          label: 'Entropy',
          say: 'Same data, the other measure.',
          enter: on('watch-it-grow', { criterion: 'entropy' }, IMPURITY, PLAY),
          speed: 'normal',
          until: done,
          then: (c) => 'Grown with entropy: the root asks ' + (c.derived.grown({}).rootQuestion ?? 'nothing') + ', ' + leaves(c.derived.leaves) + ', depth ' + c.derived.depth + ', ' + scores(c) + '.',
          focus: { kind: 'control', key: 'criterion', label: 'Impurity' },
        },
        {
          label: 'Gini again',
          say: 'Back to 1 − Σp², live.',
          enter: on('watch-it-grow', null, IMPURITY, PLAY),
          speed: 'normal',
          until: done,
          then: (c) => 'Grown with Gini: the root asks ' + (c.derived.grown({}).rootQuestion ?? 'nothing') + ', ' + leaves(c.derived.leaves) + ', ' + scores(c) + '.',
          focus: { kind: 'control', key: 'criterion', label: 'Impurity' },
        },
      ],
    },
  ],
};

const maxDepth: L = {
  id: 'max-depth',
  title: 'Max depth',
  hook: 'Depth 1, 4 and 8, judged held-out.',
  section: 'core',
  presetId: 'stump',
  knobs: ['maxDepth'],
  steps: [
    {
      id: 'depth',
      kind: 'sandbox',
      say: 'Depth 1: the tree may ask one question. Two leaves, train 89%, held-out 92%; on two blobs a single cut already gets most of the way. Training accuracy can only rise with depth; held-out accuracy decides whether the depth paid. Try it: Depth 4.',
      takeaway: 'Every cut is parallel to an axis, so a curve costs a staircase of them. Judge depth on points the tree has not seen.',
      focus: { kind: 'control', key: 'maxDepth', label: 'Max depth' },
      enter: [{ type: 'preset', id: 'stump' }, { type: 'reset' }, GROWTH, GROW],
      numbers: (c) => [{ label: 'depth', value: String(c.derived.depth) }, ...numbers(c)],
      experiments: [
        {
          label: 'Depth 4',
          say: 'The same blobs with up to four questions in a row.',
          enter: on('watch-it-grow', null, GROWTH, PLAY),
          speed: 'normal',
          until: done,
          then: (c) => {
            const stump = c.derived.grown({ maxDepth: 1 });
            const bought = c.derived.testAcc - stump.testAcc;
            return (
              'Depth 4: ' + leaves(c.derived.leaves) + ', ' + scores(c) + '. The stump scored ' + pc(stump.testAcc) + ' held-out, so ' + plural(c.derived.splits - 1, 'extra question') +
              (bought > 0.005 ? ' bought ' + fmtPercent(bought, 1) + '.' : ' bought nothing on new data; they fit the training set instead.')
            );
          },
          focus: { kind: 'metric', key: 'test', label: 'Held out' },
        },
        {
          label: 'Two moons, depth 8',
          say: 'A curved boundary and eight questions in a row.',
          enter: on('staircase', null, GROWTH, PLAY),
          speed: 'fast',
          until: done,
          then: (c) => 'Two moons at depth 8: ' + plural(c.derived.leaves, 'rectangle') + ' tracing one curve, held-out ' + pc(c.derived.testAcc) + '.',
          focus: { kind: 'panel', id: 'field', label: 'The feature space' },
        },
      ],
    },
  ],
};

const overfittingAndPruning: L = {
  id: 'pruning',
  title: 'Overfitting and pruning',
  hook: 'Leaves of one point, then pruned back.',
  section: 'failure',
  presetId: 'memorise',
  knobs: ['ccpAlpha', 'minSamplesLeaf', 'maxDepth'],
  steps: [
    {
      id: 'memorise',
      kind: 'sandbox',
      say: 'Depth 10 with leaves as small as one point: 28 leaves, train 100%, held-out 85%. On the growth curve held-out peaked at 91% after 3 splits, then fell while train kept climbing: the tree fenced off single points. Try it: Prune with α = 0.008.',
      takeaway: 'Pruning charges every leaf a rent of α and folds any subtree that does not earn it. A minimum leaf size is pruning before the fact.',
      focus: { kind: 'panel', id: 'second', label: 'Accuracy as the tree grows' },
      enter: [{ type: 'preset', id: 'memorise' }, { type: 'reset' }, GROWTH, GROW],
      numbers: (c) => [{ label: 'leaves', value: String(c.derived.leaves) }, { label: 'train', value: pc(c.derived.trainAcc) }, { label: 'held out', value: pc(c.derived.testAcc) }, { label: 'gap', value: fmtPercent(c.derived.gap, 1) }],
      experiments: [
        {
          label: 'Prune with α = 0.008',
          say: 'Cost-complexity pruning on the grown tree.',
          enter: on('memorise', { ccpAlpha: 0.008 }, GROWTH, GROW),
          until: done,
          then: (c) =>
            'α = ' + fmt(c.params.ccpAlpha, 3) + ' folds the ' + c.derived.grownLeaves + ' leaves back to ' + c.derived.leaves + ': ' + scores(c) +
            '. The large rectangles stay; the fences around single points go.',
          focus: { kind: 'control', key: 'ccpAlpha', label: 'Pruning strength (α)' },
        },
        {
          label: 'At least 20 points per leaf',
          say: 'The other approach: refuse questions that would fence off a handful of points.',
          enter: on('starved', null, SEARCH, GROW),
          until: done,
          then: (c) =>
            'With at least 20 points per leaf growth stopped at depth ' + c.derived.depth + ' with ' + leaves(c.derived.leaves) + ' and impurity ' + fmt(c.derived.weightedImpurity, 3) +
            ' still left: ' + scores(c) + '.',
          focus: { kind: 'control', key: 'minSamplesLeaf', label: 'Min samples per leaf' },
        },
        {
          label: 'Depth 10 again',
          say: 'Watch the two scores drift apart.',
          enter: on('memorise', null, GROWTH, PLAY),
          speed: 'fast',
          until: done,
          then: (c) =>
            'Depth 10: ' + leaves(c.derived.leaves) + ', ' + scores(c) + ', a gap of ' + fmtPercent(c.derived.gap, 1) + '. Held-out peaked at ' + pc(c.derived.bestTest) + ' after ' +
            plural(c.derived.bestSplit, 'split') + '.',
          focus: { kind: 'panel', id: 'second', label: 'Accuracy as the tree grows' },
        },
      ],
    },
  ],
};

const axisAlignedSplits: L = {
  id: 'axis-aligned',
  title: 'Axis-aligned splits',
  hook: 'Diagonals and XOR cost a tree dearly.',
  section: 'failure',
  presetId: 'diagonal',
  knobs: ['dataset'],
  steps: [
    {
      id: 'diagonal',
      kind: 'sandbox',
      say: 'Diagonal stripes, the worst case: 27 splits to approximate two straight lines, held-out 74%. A tree cannot ask about x₁ + x₂; every cut is parallel to an axis, so anything diagonal costs a staircase of them. Try it: XOR.',
      takeaway: 'Greedy growth judges one question at a time. On XOR the first cut buys nothing alone; the ones under it make it pay.',
      focus: { kind: 'panel', id: 'field', label: 'The feature space' },
      enter: [{ type: 'preset', id: 'diagonal' }, { type: 'reset' }, GROWTH, GROW],
      numbers,
      experiments: [
        {
          label: 'XOR',
          say: 'Opposite quadrants share a label.',
          enter: on('xor', null, GROWTH, PLAY),
          speed: 'normal',
          until: done,
          then: (c) => {
            const first = c.derived.growthAt(1);
            return 'XOR: after one split held-out was ' + pc(first?.test ?? Number.NaN) + '. After ' + plural(c.derived.splits, 'split') + ' it is ' + pc(c.derived.testAcc) + '.';
          },
          focus: { kind: 'panel', id: 'second', label: 'Accuracy as the tree grows' },
        },
        {
          label: 'Diagonal stripes again',
          say: 'The staircase, cut by cut.',
          enter: on('diagonal', null, GROWTH, PLAY),
          speed: 'fast',
          until: done,
          then: (c) => 'Diagonal stripes: ' + plural(c.derived.splits, 'split') + ' to approximate two straight lines, ' + scores(c) + '.',
          focus: { kind: 'panel', id: 'field', label: 'The feature space' },
        },
      ],
    },
  ],
};

const experiments: L = {
  id: 'experiments',
  title: 'Experiments',
  hook: 'Six experiments with known outcomes.',
  section: 'sandbox',
  presetId: 'watch-it-grow',
  steps: [
    {
      id: 'sandbox',
      kind: 'sandbox',
      say: 'Pick an experiment: the settings change, the run replays, and the outcome opens under it.',
      enter: [{ type: 'preset', id: 'watch-it-grow' }, { type: 'reset' }, SEARCH],
      experiments: [
        {
          label: 'Three classes',
          say: 'Three classes need no extra machinery: each leaf votes for the class it holds most of.',
          patch: { dataset: 'three-class', classCount: 3, maxDepth: 5, sampleCount: 240, noise: 0.3 },
          focus: { kind: 'control', key: 'dataset', label: 'Dataset' },
        },
        {
          label: 'Four classes',
          say: 'Four classes, same algorithm. The impurity is computed over four shares instead of two.',
          patch: { dataset: 'three-class', classCount: 4, maxDepth: 5, sampleCount: 240, noise: 0.3 },
          focus: { kind: 'control', key: 'classCount', label: 'Classes' },
        },
        {
          label: 'Min gain 0.05',
          say: 'Refusing any split that gains less than 0.05 stops the tree after a handful of questions.',
          patch: { minImpurityDecrease: 0.05 },
          focus: { kind: 'control', key: 'minImpurityDecrease', label: 'Min gain' },
        },
        {
          label: 'Train on 30%',
          say: 'With 60 training points the tree has less to learn from, and the held-out score is measured on 140.',
          patch: { trainFraction: 0.3 },
          focus: { kind: 'control', key: 'trainFraction', label: 'Train split' },
        },
        {
          label: 'Depth 2',
          say: 'Depth 2 allows at most four leaves. Compare its held-out score with the depth 4 tree.',
          patch: { maxDepth: 2 },
          focus: { kind: 'control', key: 'maxDepth', label: 'Max depth' },
        },
        {
          label: 'Spiral',
          say: 'The spiral at depth 8: over a dozen rectangles chasing two arms. With little noise they still score well; raise the noise and watch the gap open.',
          patch: { dataset: 'spiral', maxDepth: 8, noise: 0.1, sampleCount: 260 },
          focus: { kind: 'control', key: 'dataset', label: 'Dataset' },
        },
      ],
    },
  ],
};

export const LESSONS: L[] = [impurityAndSplitting, giniAndEntropy, maxDepth, overfittingAndPruning, axisAlignedSplits, experiments];
