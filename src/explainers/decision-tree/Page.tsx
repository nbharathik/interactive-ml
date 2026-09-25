/** Decision tree explainer. One transport step is one split, chosen best-first. */

import { useCallback, useMemo, useState } from 'react';

import { ControlPanel, ControlRow, railGroupsOf, topGroupsOf } from '../../explainer/components/ControlPanel';
import { ColumnHead, Studio } from '../../explainer/components/Studio';
import {
  Legend,
  MetricStrip,
  Panel,
  ParamsReset,
  ViewSwitch,
} from '../../explainer/components/Panels';
import {
  Annotation,
  DetailHead,
  Scalar,
  Scalars,
} from '../../explainer/components/Detail';
import { Button } from '../../explainer/components/Controls';
import { LessonPanel } from '../../explainer/components/LessonPanel';
import { LessonFocusContext, LessonOpenContext } from '../../explainer/lessonFocus';
import { LESSON_QUERY_KEYS } from '../../explainer/lessons';
import { useLesson } from '../../explainer/useLesson';
import { Transport } from '../../explainer/components/Transport';
import { useExplainerParams } from '../../explainer/useExplainerParams';
import { useMediaQuery } from '../../explainer/useMediaQuery';
import { useSimulation } from '../../explainer/useSimulation';
import type { GuideStatus, Metric } from '../../explainer/types';
import { getClassificationDataset, splitData } from '../../lib/datasets/points';
import type { Point2D } from '../../lib/datasets/types';
import { fmt, fmtPercent } from '../../lib/math/stats';
import {
  accuracy as treeAccuracy,
  countLeaves,
  createState,
  decisionPath,
  evaluateSplits,
  growOne,
  predict,
  prune,
  treeDepth,
  walk,
} from '../../lib/ml/decisionTree';
import type { Criterion, SplitCandidate, TreeConfig, TreeNode } from '../../lib/ml/decisionTree';
import { classColour } from '../../lib/viz/plots';
import { usePalette } from '../../components/ThemeProvider';
import { getExplainer } from '../registry';
import { CONTROL_GROUPS, DEFAULT_PARAMS, PRESETS } from './config';
import type { TreeParams } from './config';
import { LESSONS, makeTreeContext } from './lessons';
import type { SecondView } from './lessons';
import {
  FeatureSpace,
  GrowthCurve,
  ImpurityCurves,
  SplitSearch,
  TreeDiagram,
  conditionText,
  useIntro,
} from './panels';
import type { GrowthPoint } from './panels';

const META = getExplainer('decision-tree')!;
const TOP_GROUPS = topGroupsOf(CONTROL_GROUPS);
const RAIL_GROUPS = railGroupsOf(CONTROL_GROUPS);

const SECOND_VIEW_TITLES: Record<SecondView, string> = {
  search: 'The split search',
  impurity: 'Gini against entropy',
  growth: 'Accuracy as the tree grows',
};

const SECOND_VIEW_BLURBS: Record<SecondView, string> = {
  search: 'Every threshold scored for the leaf the last step expanded.',
  impurity: 'Two ways to measure the same thing, over the class balance p.',
  growth: 'The full growth run, with a marker at the step you are on.',
};

/** Marker shapes, in the order `classShape` assigns them. */
const LEGEND_SHAPES = ['dot', 'cross', 'triangle', 'square'] as const;

function legendShape(label: number): (typeof LEGEND_SHAPES)[number] {
  return LEGEND_SHAPES[((label % 4) + 4) % 4];
}

/** Depth-first search for a node by id, used to resolve the clicked node. */
function findNode(root: TreeNode, id: number): TreeNode | null {
  const stack: TreeNode[] = [root];
  while (stack.length > 0) {
    const node = stack.pop();
    if (!node) break;
    if (node.id === id) return node;
    if (node.left) stack.push(node.left);
    if (node.right) stack.push(node.right);
  }
  return null;
}

export default function DecisionTreePage() {
  const paramsApi = useExplainerParams<TreeParams>(DEFAULT_PARAMS, PRESETS, LESSON_QUERY_KEYS, CONTROL_GROUPS);
  const { params, set, reset, lastPreset, isDirty, shareUrl } = paramsApi;

  /* ---------------- data ---------------- */

  const data = useMemo(
    () =>
      getClassificationDataset(params.dataset).generate({
        count: params.sampleCount,
        noise: params.noise,
        seed: params.seed,
        classCount: params.classCount,
      }),
    [params.dataset, params.sampleCount, params.noise, params.seed, params.classCount],
  );

  const { train, test } = useMemo(
    () => splitData(data.points, params.trainFraction, params.seed + 977),
    [data.points, params.trainFraction, params.seed],
  );

  const points = useMemo(() => train.concat(test), [train, test]);
  const testStart = train.length;
  const hasTest = test.length > 0;

  const bounds = useMemo(
    () => ({ x0: data.xRange[0], x1: data.xRange[1], y0: data.yRange[0], y1: data.yRange[1] }),
    [data.xRange, data.yRange],
  );

  const featureNames = useMemo<[string, string]>(
    () => [data.xLabel, data.yLabel],
    [data.xLabel, data.yLabel],
  );

  const config = useMemo<TreeConfig>(
    () => ({
      criterion: params.criterion as Criterion,
      maxDepth: params.maxDepth,
      minSamplesSplit: params.minSamplesSplit,
      minSamplesLeaf: params.minSamplesLeaf,
      minImpurityDecrease: params.minImpurityDecrease,
      ccpAlpha: params.ccpAlpha,
      classCount: data.classCount,
    }),
    [
      params.criterion,
      params.maxDepth,
      params.minSamplesSplit,
      params.minSamplesLeaf,
      params.minImpurityDecrease,
      params.ccpAlpha,
      data.classCount,
    ],
  );

  /* ---------------- simulation ---------------- */

  // Alpha is not in `deps`: pruning re-applies to the grown tree.
  const sim = useSimulation({
    create: () => createState(train, config, bounds),
    step: (state) => growOne(state, train, config),
    isComplete: (state) => state.finished,
    deps: [
      train,
      bounds,
      params.criterion,
      params.maxDepth,
      params.minSamplesSplit,
      params.minSamplesLeaf,
      params.minImpurityDecrease,
      data.classCount,
    ],
    baseStepsPerSecond: 2.4,
    maxSteps: 400,
  });

  const state = sim.state;

  /** What the reader sees: the grown tree with cost-complexity pruning applied. */
  const root = useMemo(
    () => prune(state.root, params.ccpAlpha, Math.max(1, train.length)),
    [state.root, params.ccpAlpha, train.length],
  );

  /* ---------------- derived numbers ---------------- */

  const leaves = useMemo(() => countLeaves(root), [root]);
  const grownLeaves = useMemo(() => countLeaves(state.root), [state.root]);
  const depth = useMemo(() => treeDepth(root), [root]);
  const splits = leaves - 1;

  const trainAcc = useMemo(() => treeAccuracy(root, train), [root, train]);
  const testAcc = useMemo(() => (hasTest ? treeAccuracy(root, test) : NaN), [root, test, hasTest]);
  const gap = hasTest ? trainAcc - testAcc : 0;

  const mistakes = useMemo(() => {
    const out = new Set<number>();
    points.forEach((p, i) => {
      if (predict(root, p.x, p.y) !== p.label) out.add(i);
    });
    return out;
  }, [points, root]);

  /** Impurity still left in the tree, weighted by how much data each leaf holds. */
  const weightedImpurity = useMemo(() => {
    const total = Math.max(1, train.length);
    let sum = 0;
    walk(root, (node) => {
      if (node.isLeaf) sum += (node.samples.length / total) * node.impurity;
    });
    return sum;
  }, [root, train.length]);

  /* ---------------- the split search ---------------- */

  const focus = state.lastSplit ? state.lastSplit.node : state.root;

  // For one render the tree can still index a training set that has since shrunk.
  const inSync = state.root.samples.length === train.length;

  /** `lastEvaluated` is capped at 120 candidates; re-run the search when the curve needs the whole scan. */
  const candidates = useMemo<SplitCandidate[]>(() => {
    if (!inSync) return [];
    if (state.lastEvaluated.length > 0 && state.lastEvaluated.length < 120) {
      return state.lastEvaluated.slice();
    }
    return evaluateSplits(train, focus, config);
  }, [inSync, state.lastEvaluated, train, focus, config]);

  // Both sources are sorted best-first, so the greedy winner is always index 0.
  const chosen = candidates.length > 0 ? candidates[0] : null;

  /** The leaves still eligible to grow, ranked the way `growOne` ranks them. */
  const frontier = useMemo(() => {
    const rows: Array<{ node: TreeNode; candidate: SplitCandidate; score: number }> = [];
    if (!inSync) return rows;
    const total = Math.max(1, train.length);
    walk(state.root, (node) => {
      if (!node.isLeaf) return;
      if (node.depth >= config.maxDepth) return;
      if (node.samples.length < config.minSamplesSplit) return;
      if (node.impurity <= 1e-12) return;
      const found = evaluateSplits(train, node, config);
      if (found.length === 0) return;
      rows.push({
        node,
        candidate: found[0],
        score: (node.samples.length / total) * found[0].gain,
      });
    });
    rows.sort((a, b) => b.score - a.score);
    return rows;
  }, [inSync, state.root, train, config]);

  /** Of those, the ones whose gain clears the minimum-gain floor. */
  const growable = useMemo(
    () => frontier.filter((row) => row.candidate.gain >= params.minImpurityDecrease),
    [frontier, params.minImpurityDecrease],
  );
  const blockedByMinGain = frontier.length > 0 && growable.length === 0;

  /* ---------------- growth curve ---------------- */

  /** Train and held-out accuracy after every split, grown from scratch so the curve is always complete. */
  const growth = useMemo<GrowthPoint[]>(() => {
    let current = createState(train, config, bounds);
    const out: GrowthPoint[] = [];
    const record = () => {
      const tree =
        params.ccpAlpha > 0
          ? prune(current.root, params.ccpAlpha, Math.max(1, train.length))
          : current.root;
      out.push({
        train: treeAccuracy(tree, train),
        test: test.length > 0 ? treeAccuracy(tree, test) : 0,
        leaves: countLeaves(tree),
      });
    };
    record();
    for (let i = 0; i < 220 && !current.finished; i++) {
      const next = growOne(current, train, config);
      if (next.nodeCount === current.nodeCount) break;
      current = next;
      record();
    }
    return out;
  }, [train, test, config, bounds, params.ccpAlpha]);

  const totalSplits = Math.max(1, growth.length - 1);

  const bestIndex = useMemo(() => {
    if (!hasTest) return -1;
    let best = 0;
    for (let i = 1; i < growth.length; i++) {
      if (growth[i].test > growth[best].test) best = i;
    }
    return best;
  }, [growth, hasTest]);

  /* ---------------- interaction ---------------- */

  const [hoverIndex, setHoverIndex] = useState<number | null>(null);
  const [selectedId, setSelectedId] = useState<number | null>(null);
  const [searchMode, setSearchMode] = useState<'gain' | 'impurity'>('gain');
  const [scaleGini, setScaleGini] = useState(true);
  const [secondView, setSecondView] = useState<SecondView>('search');

  const handleHover = useCallback((index: number | null) => setHoverIndex(index), []);
  const handleSelect = useCallback((id: number | null) => setSelectedId(id), []);

  // The lessons drive the presets, the transport, the second chart and the open card.
  const lessonActions = useMemo(
    () => ({
      view: (id: string, value: string) => {
        if (id === 'second') setSecondView(value as SecondView);
      },
      open: (target: string | null) => {
        if (target === null) setSelectedId(null);
        else if (target.startsWith('node:')) setSelectedId(Number(target.slice(5)));
      },
    }),
    [],
  );
  const lessonContext = useMemo(
    () =>
      makeTreeContext({
        params,
        state,
        train,
        test,
        config,
        bounds,
        featureNames,
        growth,
        ui: { secondView, open: selectedId },
      }),
    [params, state, train, test, config, bounds, featureNames, growth, secondView, selectedId],
  );
  const lessonApi = useLesson({ id: META.slug, lessons: LESSONS, params: paramsApi, sim, actions: lessonActions, context: lessonContext });

  /* ---------------- the chapter reset ---------------- */

  // Free play with no lesson marks, default knobs and views, a fresh run.
  const { reset: resetLessons } = lessonApi;
  const { reset: resetRun } = sim;
  const resetAll = useCallback(() => {
    resetLessons();
    reset();
    resetRun();
    setSecondView('search');
    setSearchMode('gain');
    setScaleGini(true);
    setSelectedId(null);
  }, [resetLessons, reset, resetRun]);
  const canReset = isDirty || lessonApi.dirty || sim.iteration > 0 || secondView !== 'search' || searchMode !== 'gain' || !scaleGini || selectedId !== null;

  /* ---------------- the guide ---------------- */

  /** Why the leaves that are left cannot split: the depth budget, purity, or size. */
  const stopReason = useMemo(() => {
    let pure = 0;
    let deep = 0;
    let small = 0;
    let cramped = 0;
    walk(state.root, (node) => {
      if (!node.isLeaf) return;
      if (node.impurity <= 1e-12) pure += 1;
      else if (node.depth >= config.maxDepth) deep += 1;
      else if (node.samples.length < config.minSamplesSplit) small += 1;
      else cramped += 1;
    });
    const parts: string[] = [];
    if (deep > 0) parts.push(deep + (deep === 1 ? ' leaf hit' : ' leaves hit') + ' the depth budget (' + config.maxDepth + ')');
    if (pure > 0) parts.push(pure + (pure === 1 ? ' is pure' : ' are pure'));
    if (small > 0) parts.push(small + (small === 1 ? ' is' : ' are') + ' too small to split');
    // What is left had no split that kept both sides at the minimum leaf size.
    if (cramped > 0 && config.minSamplesLeaf > 1 && !blockedByMinGain) {
      parts.push(cramped + (cramped === 1 ? ' has' : ' have') + ' no split leaving ' + config.minSamplesLeaf + ' points on each side');
    }
    if (blockedByMinGain) parts.push('no remaining split clears the minimum gain ' + fmt(params.minImpurityDecrease, 3));
    return parts.join(', ');
  }, [state.root, config.maxDepth, config.minSamplesSplit, config.minSamplesLeaf, blockedByMinGain, params.minImpurityDecrease]);

  const guide = useMemo<GuideStatus>(() => {
    const scores =
      ' · train ' + fmtPercent(trainAcc, 0) + (hasTest ? ', held-out ' + fmtPercent(testAcc, 0) : '');
    const memorising = hasTest && gap > 0.15;
    const pruned =
      params.ccpAlpha > 0 && grownLeaves > leaves
        ? ' · α = ' + fmt(params.ccpAlpha, 3) + ' prunes ' + grownLeaves + ' leaves to ' + leaves
        : '';
    if (sim.iteration === 0 && !state.finished) {
      return {
        text:
          'One leaf holding all ' + train.length + ' training points, impurity ' + fmt(state.root.impurity, 3) +
          '. Press play: every step asks the one question that removes the most impurity. Click any node for its numbers.',
      };
    }
    if (state.finished) {
      if (sim.iteration === 0) {
        return {
          text: 'Nothing to grow: ' + (stopReason || 'the root cannot be split') + ' · 1 leaf' + scores,
          tone: 'warn',
        };
      }
      return {
        text:
          'Fully grown after ' + sim.iteration + ' split' + (sim.iteration === 1 ? '' : 's') + ': ' + stopReason +
          ' · ' + leaves + ' leaves, depth ' + depth + pruned + scores +
          (memorising ? ' · held-out is falling behind: the tree is memorising' : ''),
        tone: memorising ? 'warn' : 'good',
      };
    }
    const last = state.lastSplit;
    let text = 'Split ' + sim.iteration + ' of ' + totalSplits;
    if (last && last.node.split) {
      text +=
        ' · asked ' + conditionText(last.node.split.feature, last.node.split.threshold, featureNames) +
        ' at depth ' + last.node.depth + ', gain ' + fmt(last.candidate.gain, 3) + ', ' +
        last.candidate.leftCount + ' points left / ' + last.candidate.rightCount + ' right';
    }
    text += pruned + scores;
    if (memorising) text += ' · held-out is falling behind: the tree is memorising';
    return { text, tone: memorising ? 'warn' : 'neutral' };
  }, [state, sim.iteration, train.length, trainAcc, testAcc, hasTest, gap, params.ccpAlpha, grownLeaves, leaves, depth, stopReason, totalSplits, featureNames]);

  const selected = useMemo(
    () => (selectedId === null ? null : findNode(root, selectedId)),
    [root, selectedId],
  );

  const hovered: Point2D | null = hoverIndex === null ? null : (points[hoverIndex] ?? null);

  const pathNodes = useMemo(
    () => (hovered ? decisionPath(root, hovered.x, hovered.y) : []),
    [hovered, root],
  );

  const pathIds = useMemo(() => new Set(pathNodes.map((node) => node.id)), [pathNodes]);

  /** The hovered point's rule chain. */
  const pathText = useMemo(() => {
    if (!hovered || pathNodes.length === 0) return null;
    const parts: string[] = [];
    for (let i = 0; i < pathNodes.length - 1; i++) {
      const node = pathNodes[i];
      if (!node.split) break;
      const value = node.split.feature === 0 ? hovered.x : hovered.y;
      const goesLeft = value <= node.split.threshold;
      parts.push(
        featureNames[node.split.feature] +
          (goesLeft ? ' ≤ ' : ' > ') +
          fmt(node.split.threshold, 2),
      );
    }
    const leaf = pathNodes[pathNodes.length - 1];
    const predicted = data.classNames[leaf.prediction] ?? 'class ' + leaf.prediction;
    const correct = leaf.prediction === hovered.label;
    return (
      (parts.length > 0 ? parts.join('  →  ') + '  →  ' : '') +
      'predict ' +
      predicted +
      (correct ? ', correct' : ', wrong, it is ' + data.classNames[hovered.label])
    );
  }, [hovered, pathNodes, featureNames, data.classNames]);

  const intro = useIntro(sim.iteration);
  const touch = useMediaQuery('(hover: none)');

  /* ---------------- metrics ---------------- */

  const palette = usePalette();

  const headline: Metric[] = useMemo(() => {
    const previous = growth[Math.max(0, Math.min(growth.length - 1, splits) - 1)];
    return [
      {
        key: 'train',
        label: 'train',
        value: fmtPercent(trainAcc, 0),
        tone: 'neutral',
        trend: previous && trainAcc > previous.train ? 'up' : 'flat',
        colour: palette.violet,
        help:
          'The share of the ' + train.length + ' training points whose leaf votes for their true class. It can only go up as the tree grows, which is exactly why it cannot tell you when to stop.',
      },
      {
        key: 'test',
        label: 'held out',
        value: hasTest ? fmtPercent(testAcc, 0) : 'n/a',
        tone: !hasTest ? 'neutral' : testAcc > 0.85 ? 'good' : testAcc > 0.7 ? 'neutral' : 'warn',
        colour: palette.orange,
        help:
          'The same measurement on the ' + test.length + ' points the tree was not grown from. This is the number that starts falling once the tree begins memorising.',
      },
    ];
  }, [growth, splits, trainAcc, testAcc, hasTest, train.length, test.length, palette.violet, palette.orange]);

  const more: Metric[] = useMemo(() => {
    return [
      {
        key: 'gap',
        label: 'Generalisation gap',
        value: hasTest ? fmtPercent(gap, 1) : 'n/a',
        caption: 'train − held-out',
        tone: !hasTest ? 'neutral' : gap > 0.15 ? 'bad' : gap > 0.07 ? 'warn' : 'good',
        help: 'How much of the training score is memorisation rather than pattern. A few points is normal; twenty is a tree that has fenced off individual observations.',
      },
      {
        key: 'leaves',
        label: 'Leaves',
        value: String(leaves),
        caption:
          params.ccpAlpha > 0 && grownLeaves > leaves
            ? 'pruned from ' + grownLeaves
            : growable.length + (growable.length === 1 ? ' leaf' : ' leaves') + ' can still split',
        tone: params.ccpAlpha > 0 && grownLeaves > leaves ? 'accent' : 'neutral',
        help: 'One leaf is one rectangle of feature space and one constant prediction. The whole model is this many rectangles.',
      },
      {
        key: 'depth',
        label: 'Depth',
        value: String(depth),
        caption: 'budget ' + params.maxDepth,
        tone: depth < params.maxDepth ? 'good' : 'neutral',
        help: 'The longest root-to-leaf path actually built. When it sits below the budget, something other than depth stopped the growth.',
      },
      {
        key: 'impurity',
        label: 'Impurity left',
        value: fmt(weightedImpurity, 3),
        caption:
          growable.length > 0
            ? 'next split gains ' + fmt(growable[0].candidate.gain, 3)
            : blockedByMinGain
              ? 'below the minimum gain'
              : 'nothing left to ask',
        tone: 'accent',
        trend: splits > 0 ? 'down' : 'flat',
        help: 'Each leaf’s impurity, weighted by how much of the data it holds. Every split is chosen to make this number fall as fast as possible.',
      },
    ];
  }, [
    splits,
    hasTest,
    gap,
    leaves,
    grownLeaves,
    depth,
    weightedImpurity,
    growable,
    blockedByMinGain,
    params.ccpAlpha,
    params.maxDepth,
  ]);

  /* ---------------- render ---------------- */

  const classLegend = data.classNames.map((name, index) => ({
    label: name,
    colour: classColour(palette, index),
    shape: legendShape(index),
  }));

  const renderNodeCard = useCallback(
    (node: TreeNode) => {
      const total = Math.max(1, root.samples.length);
      const majority = data.classNames[node.prediction] ?? String(node.prediction);
      return (
        <>
          <DetailHead
            eyebrow={(node.isLeaf ? 'Leaf' : 'Node') + ' · depth ' + node.depth}
            title={
              node.isLeaf
                ? 'Predicts ' + majority
                : node.split
                  ? conditionText(node.split.feature, node.split.threshold, featureNames)
                  : 'Node'
            }
            shape={node.samples.length + ' of ' + total + ' points · ' + fmtPercent(node.samples.length / total, 0) + ' of the data'}
          />
          <Annotation>
            {node.isLeaf
              ? 'Nothing is asked here: every point that reaches this leaf gets the majority class of the training points in it.'
              : 'One question about one feature. Points answering yes go left, the rest right, and each side gets its own question.'}
          </Annotation>
          <Scalars>
            {node.counts.map((count, index) => (
              <Scalar
                key={index}
                label={data.classNames[index] ?? 'class ' + index}
                value={count + ' · ' + fmtPercent(count / Math.max(1, node.samples.length), 0)}
                tone={index === node.prediction ? 'accent' : undefined}
              />
            ))}
            <Scalar label={config.criterion === 'entropy' ? 'entropy' : 'gini'} value={node.impurity} decimals={3} />
            {node.split ? (
              <>
                <Scalar label="left / right" value={node.split.leftCount + ' / ' + node.split.rightCount} />
                <Scalar label="gain" value={node.split.gain} decimals={4} tone="good" />
              </>
            ) : null}
          </Scalars>
        </>
      );
    },
    [root.samples.length, data.classNames, featureNames, config.criterion],
  );

  return (
    <LessonFocusContext.Provider value={lessonApi.focus}>
    <LessonOpenContext.Provider value={lessonApi.view.active}>
    <Studio
      meta={META}
      shareUrl={shareUrl}
      lesson={<LessonPanel view={lessonApi.view} status={guide.text} tone={guide.tone} exitLabel="Leave lesson" />}
      transport={
        <Transport sim={sim} unit="split" total={totalSplits} completeLabel="Fully grown" />
      }
      topControls={<ControlRow groups={TOP_GROUPS} params={params} onChange={set} lesson={lastPreset} named={lessonApi.view.knobs} />}
      tools={<ParamsReset onReset={resetAll} isDirty={canReset} title="Reset the chapter: defaults, free play, lesson marks cleared" />}
      controls={
        <ControlPanel
          groups={RAIL_GROUPS}
          params={params}
          onChange={set}
        />
      }
      wide
      architecture={
        <Panel
          id="architecture"
          title="The tree"
          actions={
            <>
              {selectedId !== null ? (
                <Button
                  size="sm"
                  variant="ghost"
                  onClick={() => {
                    setSelectedId(null);
                    document.querySelector<HTMLElement>('.mlx-tree[tabindex]')?.focus();
                  }}
                >
                  Clear selection
                </Button>
              ) : null}
              <Legend
                dense
                items={[{ label: 'class mix', colour: palette.textMuted, shape: 'square' }]}
              />
            </>
          }
        >
          <TreeDiagram
            root={root}
            classNames={data.classNames}
            featureNames={featureNames}
            selectedId={selectedId}
            pathIds={pathIds}
            lastSplitId={state.lastSplit ? state.lastSplit.node.id : null}
            intro={intro}
            onSelect={handleSelect}
            renderDetail={renderNodeCard}
          />
          <p className="mlx-note">
            {pathText
              ? pathText
              : touch
                ? 'Tap a point in the feature space to print its rule chain here. Tap any node to outline the rectangle it owns.'
                : 'Hover a point in the feature space to print its rule chain here. Click any node to outline the rectangle it owns.'}
          </p>
        </Panel>
      }
      output={
        <>
          <ColumnHead title="Output" blurb="Every leaf is a rectangle in the feature space." />

          <Panel
            id="field"
            fill
            title="The feature space"
            subtitle="Every cut is one question, drawn only across the region that asked it."
            actions={
              <Legend
                dense
                items={[
                  ...classLegend,
                  { label: 'newest cut', colour: palette.text, shape: 'line' as const },
                ]}
              />
            }
          >
            <FeatureSpace
              data={data}
              points={points}
              testStart={testStart}
              root={root}
              showRegions={params.showRegions}
              showSplitLines={params.showSplitLines}
              showTestPoints={params.showTestPoints}
              markMistakes={params.markMistakes}
              mistakes={mistakes}
              hoverIndex={hoverIndex}
              pathIds={pathIds}
              selected={selected}
              lastSplitId={state.lastSplit ? state.lastSplit.node.id : null}
              intro={intro}
              onHover={handleHover}
            />
          </Panel>

          <Panel
            id="second"
            fill
            title={SECOND_VIEW_TITLES[secondView]}
            subtitle={SECOND_VIEW_BLURBS[secondView]}
            actions={
              <>
                <Legend
                  dense
                  items={
                    secondView === 'search'
                      ? [
                          { label: featureNames[0], colour: palette.violet, shape: 'line' },
                          { label: featureNames[1], colour: palette.cyan, shape: 'line' },
                        ]
                      : secondView === 'impurity'
                        ? [
                            { label: 'Gini', colour: palette.violet, shape: 'line' },
                            { label: 'entropy', colour: palette.cyan, shape: 'line' },
                          ]
                        : [
                            { label: 'train', colour: palette.violet, shape: 'dashed' },
                            { label: 'held out', colour: palette.orange, shape: 'line' },
                          ]
                  }
                />
                <ViewSwitch
                  id="second"
                  label="Second chart"
                  value={secondView}
                  options={[
                    { value: 'search', label: 'Split search' },
                    { value: 'impurity', label: 'Impurity' },
                    { value: 'growth', label: 'Growth' },
                  ]}
                  onChange={(next) => setSecondView(next as SecondView)}
                />
                {secondView === 'search' ? (
                  <ViewSwitch
                    label="Score"
                    value={searchMode}
                    options={[
                      { value: 'gain', label: 'Gain' },
                      { value: 'impurity', label: 'Child impurity' },
                    ]}
                    onChange={(next) => setSearchMode(next as 'gain' | 'impurity')}
                  />
                ) : null}
                {secondView === 'impurity' ? (
                  <ViewSwitch
                    label="Gini scale"
                    value={scaleGini ? 'x2' : 'raw'}
                    options={[
                      { value: 'raw', label: 'Gini' },
                      { value: 'x2', label: 'Gini ×2' },
                    ]}
                    onChange={(next) => setScaleGini(next === 'x2')}
                  />
                ) : null}
              </>
            }
          >
            {secondView === 'search' ? (
              <SplitSearch
                candidates={candidates}
                chosen={chosen}
                parentImpurity={focus.impurity}
                featureNames={featureNames}
                xDomain={data.xRange}
                minGain={params.minImpurityDecrease}
                mode={searchMode}
                criterion={config.criterion}
              />
            ) : secondView === 'impurity' ? (
              <ImpurityCurves
                counts={focus.counts}
                criterion={config.criterion}
                scaleGini={scaleGini}
                nodeLabel={state.lastSplit ? 'last split' : 'root'}
              />
            ) : (
              <GrowthCurve
                history={growth}
                current={sim.iteration}
                best={bestIndex}
                hasTest={hasTest}
              />
            )}
          </Panel>

          <MetricStrip headline={headline} more={more} />
        </>
      }
    />
    </LessonOpenContext.Provider>
    </LessonFocusContext.Provider>
  );
}
