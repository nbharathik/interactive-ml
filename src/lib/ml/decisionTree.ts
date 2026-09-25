/** CART decision tree on two features. `growOne` expands one leaf per call, best-first. */

import type { Point2D } from '../datasets/types';

export type Criterion = 'gini' | 'entropy';

export interface TreeConfig {
  criterion: Criterion;
  maxDepth: number;
  minSamplesSplit: number;
  minSamplesLeaf: number;
  /** Refuse a split that improves impurity by less than this. */
  minImpurityDecrease: number;
  /** Cost-complexity pruning strength, applied after growth. */
  ccpAlpha: number;
  classCount: number;
}

export interface SplitCandidate {
  feature: 0 | 1;
  threshold: number;
  /** Weighted impurity of the children. */
  impurity: number;
  /** Parent impurity minus weighted child impurity; the node's share of the data is applied when leaves compete. */
  gain: number;
  leftCount: number;
  rightCount: number;
}

export interface TreeNode {
  id: number;
  depth: number;
  /** Indices into the training array. */
  samples: number[];
  counts: number[];
  impurity: number;
  /** Majority class. */
  prediction: number;
  /** Fraction of this node's samples that are the majority class. */
  confidence: number;
  split: SplitCandidate | null;
  left: TreeNode | null;
  right: TreeNode | null;
  /** The rectangle of feature space this node owns, used to draw the cuts. */
  bounds: { x0: number; x1: number; y0: number; y1: number };
  /** True while the node is still a candidate for expansion. */
  isLeaf: boolean;
  /** Marked by cost-complexity pruning. */
  pruned: boolean;
}

export interface TreeState {
  root: TreeNode;
  nodeCount: number;
  leafCount: number;
  depth: number;
  /** The split chosen on the most recent growth step, for the narration. */
  lastSplit: { node: TreeNode; candidate: SplitCandidate } | null;
  /** Leaves that could still be split. */
  frontier: number[];
  finished: boolean;
  /** Every split evaluated for the most recent expansion, for the split chart. */
  lastEvaluated: SplitCandidate[];
}

/* ---------------- impurity ---------------- */

export function classCounts(
  points: readonly Point2D[],
  samples: readonly number[],
  classCount: number,
): number[] {
  const counts = new Array<number>(classCount).fill(0);
  for (const i of samples) {
    // A tree from a previous dataset can briefly hold out-of-range indices.
    const point = points[i];
    if (!point) continue;
    const label = point.label;
    if (label >= 0 && label < classCount) counts[label] += 1;
  }
  return counts;
}

export function impurityOf(counts: readonly number[], criterion: Criterion): number {
  const total = counts.reduce((a, b) => a + b, 0);
  if (total === 0) return 0;
  if (criterion === 'entropy') {
    let sum = 0;
    for (const c of counts) {
      if (c === 0) continue;
      const p = c / total;
      sum -= p * Math.log2(p);
    }
    return sum;
  }
  let sum = 1;
  for (const c of counts) {
    const p = c / total;
    sum -= p * p;
  }
  return sum;
}

function majority(counts: readonly number[]): { prediction: number; confidence: number } {
  let best = 0;
  let bestCount = -1;
  let total = 0;
  for (let i = 0; i < counts.length; i++) {
    total += counts[i];
    if (counts[i] > bestCount) {
      bestCount = counts[i];
      best = i;
    }
  }
  return { prediction: best, confidence: total === 0 ? 0 : bestCount / total };
}

/* ---------------- split search ---------------- */

/** Every candidate split for a node (midpoints between distinct values), sorted best-first. */
export function evaluateSplits(
  points: readonly Point2D[],
  node: TreeNode,
  config: TreeConfig,
): SplitCandidate[] {
  const out: SplitCandidate[] = [];
  const n = node.samples.length;
  if (n < config.minSamplesSplit) return out;

  for (const feature of [0, 1] as const) {
    const sorted = node.samples
      .slice()
      .sort((a, b) => value(points[a], feature) - value(points[b], feature));

    const leftCounts = new Array<number>(config.classCount).fill(0);
    const rightCounts = node.counts.slice();

    for (let i = 0; i < sorted.length - 1; i++) {
      const label = points[sorted[i]].label;
      if (label >= 0 && label < config.classCount) {
        leftCounts[label] += 1;
        rightCounts[label] -= 1;
      }
      const current = value(points[sorted[i]], feature);
      const next = value(points[sorted[i + 1]], feature);
      if (next - current < 1e-9) continue;

      const leftCount = i + 1;
      const rightCount = n - leftCount;
      if (leftCount < config.minSamplesLeaf || rightCount < config.minSamplesLeaf) continue;

      const weighted =
        (leftCount / n) * impurityOf(leftCounts, config.criterion) +
        (rightCount / n) * impurityOf(rightCounts, config.criterion);

      out.push({
        feature,
        threshold: (current + next) / 2,
        impurity: weighted,
        gain: node.impurity - weighted,
        leftCount,
        rightCount,
      });
    }
  }

  out.sort((a, b) => b.gain - a.gain);
  return out;
}

function value(point: Point2D, feature: 0 | 1): number {
  return feature === 0 ? point.x : point.y;
}

/* ---------------- growth ---------------- */

function makeNode(
  points: readonly Point2D[],
  samples: number[],
  depth: number,
  id: number,
  config: TreeConfig,
  bounds: TreeNode['bounds'],
): TreeNode {
  const counts = classCounts(points, samples, config.classCount);
  const { prediction, confidence } = majority(counts);
  return {
    id,
    depth,
    samples,
    counts,
    impurity: impurityOf(counts, config.criterion),
    prediction,
    confidence,
    split: null,
    left: null,
    right: null,
    bounds,
    isLeaf: true,
    pruned: false,
  };
}

export function createState(
  points: readonly Point2D[],
  config: TreeConfig,
  bounds: TreeNode['bounds'],
): TreeState {
  const samples = points.map((_, i) => i);
  const root = makeNode(points, samples, 0, 0, config, bounds);

  // Must apply the same test as `growOne`, including minImpurityDecrease.
  const growable =
    canSplit(root, config) &&
    (() => {
      const candidates = evaluateSplits(points, root, config);
      return candidates.length > 0 && candidates[0].gain >= config.minImpurityDecrease;
    })();

  return {
    root,
    nodeCount: 1,
    leafCount: 1,
    depth: 0,
    lastSplit: null,
    frontier: growable ? [0] : [],
    finished: !growable,
    lastEvaluated: [],
  };
}

function canSplit(node: TreeNode, config: TreeConfig): boolean {
  if (node.depth >= config.maxDepth) return false;
  if (node.samples.length < config.minSamplesSplit) return false;
  if (node.impurity <= 1e-12) return false;
  return true;
}

/** Depth-first walk over every node in the tree. */
export function walk(node: TreeNode | null, visit: (node: TreeNode) => void): void {
  if (!node) return;
  visit(node);
  walk(node.left, visit);
  walk(node.right, visit);
}

function collectLeaves(root: TreeNode): TreeNode[] {
  const out: TreeNode[] = [];
  walk(root, (node) => {
    if (node.isLeaf) out.push(node);
  });
  return out;
}

/** Expand the one leaf whose best split buys the largest weighted impurity reduction. */
export function growOne(
  state: TreeState,
  points: readonly Point2D[],
  config: TreeConfig,
): TreeState {
  if (state.finished) return state;

  const leaves = collectLeaves(state.root).filter((leaf) => canSplit(leaf, config));
  if (leaves.length === 0) {
    return { ...state, finished: true, lastSplit: null, lastEvaluated: [] };
  }

  let bestLeaf: TreeNode | null = null;
  let bestCandidate: SplitCandidate | null = null;
  let bestScore = -Infinity;
  let bestEvaluated: SplitCandidate[] = [];

  const total = state.root.samples.length || 1;
  for (const leaf of leaves) {
    const candidates = evaluateSplits(points, leaf, config);
    if (candidates.length === 0) continue;
    const top = candidates[0];
    if (top.gain < config.minImpurityDecrease) continue;
    const score = (leaf.samples.length / total) * top.gain;
    if (score > bestScore) {
      bestScore = score;
      bestLeaf = leaf;
      bestCandidate = top;
      bestEvaluated = candidates.slice(0, 120);
    }
  }

  if (!bestLeaf || !bestCandidate) {
    return { ...state, finished: true, lastSplit: null, lastEvaluated: [] };
  }

  // Rebuild the path to the split node so the state stays immutable.
  const nextId = state.nodeCount;
  const root = cloneWithSplit(state.root, bestLeaf.id, points, bestCandidate, config, nextId);

  let nodeCount = 0;
  let leafCount = 0;
  let depth = 0;
  let splitNode: TreeNode | null = null;
  walk(root, (node) => {
    nodeCount += 1;
    if (node.isLeaf) leafCount += 1;
    if (node.depth > depth) depth = node.depth;
    if (node.id === bestLeaf.id) splitNode = node;
  });

  const remaining = collectLeaves(root).filter((leaf) => {
    if (!canSplit(leaf, config)) return false;
    const candidates = evaluateSplits(points, leaf, config);
    return candidates.length > 0 && candidates[0].gain >= config.minImpurityDecrease;
  });

  return {
    root,
    nodeCount,
    leafCount,
    depth,
    lastSplit: splitNode ? { node: splitNode, candidate: bestCandidate } : null,
    frontier: remaining.map((leaf) => leaf.id),
    finished: remaining.length === 0,
    lastEvaluated: bestEvaluated,
  };
}

function cloneWithSplit(
  node: TreeNode,
  targetId: number,
  points: readonly Point2D[],
  candidate: SplitCandidate,
  config: TreeConfig,
  nextId: number,
): TreeNode {
  if (node.id === targetId) {
    const leftSamples: number[] = [];
    const rightSamples: number[] = [];
    for (const i of node.samples) {
      if (value(points[i], candidate.feature) <= candidate.threshold) leftSamples.push(i);
      else rightSamples.push(i);
    }

    const leftBounds = { ...node.bounds };
    const rightBounds = { ...node.bounds };
    if (candidate.feature === 0) {
      leftBounds.x1 = candidate.threshold;
      rightBounds.x0 = candidate.threshold;
    } else {
      leftBounds.y1 = candidate.threshold;
      rightBounds.y0 = candidate.threshold;
    }

    return {
      ...node,
      isLeaf: false,
      split: candidate,
      left: makeNode(points, leftSamples, node.depth + 1, nextId, config, leftBounds),
      right: makeNode(points, rightSamples, node.depth + 1, nextId + 1, config, rightBounds),
    };
  }

  if (!node.left && !node.right) return node;
  return {
    ...node,
    left: node.left ? cloneWithSplit(node.left, targetId, points, candidate, config, nextId) : null,
    right: node.right
      ? cloneWithSplit(node.right, targetId, points, candidate, config, nextId)
      : null,
  };
}

/** Grow to completion in one go. */
export function growAll(
  state: TreeState,
  points: readonly Point2D[],
  config: TreeConfig,
  limit = 200,
): TreeState {
  let current = state;
  for (let i = 0; i < limit && !current.finished; i++) {
    const next = growOne(current, points, config);
    if (next === current) break;
    current = next;
  }
  return current;
}

/* ---------------- prediction ---------------- */

/** Walk the tree for one point, returning the leaf it lands in. */
export function decide(root: TreeNode, x: number, y: number): TreeNode {
  let node = root;
  while (!node.isLeaf && !node.pruned && node.split && node.left && node.right) {
    const v = node.split.feature === 0 ? x : y;
    node = v <= node.split.threshold ? node.left : node.right;
  }
  return node;
}

/** The sequence of nodes a point passes through, the "explain this prediction" path. */
export function decisionPath(root: TreeNode, x: number, y: number): TreeNode[] {
  const path: TreeNode[] = [];
  let node: TreeNode | null = root;
  while (node) {
    path.push(node);
    const current: TreeNode = node;
    if (current.isLeaf || current.pruned || !current.split || !current.left || !current.right) break;
    const v: number = current.split.feature === 0 ? x : y;
    node = v <= current.split.threshold ? current.left : current.right;
  }
  return path;
}

export function predict(root: TreeNode, x: number, y: number): number {
  return decide(root, x, y).prediction;
}

export function accuracy(root: TreeNode, points: readonly Point2D[]): number {
  if (points.length === 0) return 0;
  let hits = 0;
  for (const p of points) if (predict(root, p.x, p.y) === p.label) hits += 1;
  return hits / points.length;
}

/* ---------------- pruning ---------------- */

/** Cost-complexity pruning. Returns a new tree. */
export function prune(root: TreeNode, alpha: number, totalSamples: number): TreeNode {
  if (alpha <= 0) return root;

  const pruneNode = (node: TreeNode): { node: TreeNode; cost: number; leaves: number } => {
    const weight = node.samples.length / Math.max(1, totalSamples);
    const leafCost = weight * node.impurity;
    if (node.isLeaf || !node.left || !node.right) {
      return { node, cost: leafCost, leaves: 1 };
    }
    const left = pruneNode(node.left);
    const right = pruneNode(node.right);
    const subtreeCost = left.cost + right.cost;
    const subtreeLeaves = left.leaves + right.leaves;

    // Collapse when keeping the subtree costs more than the flat leaf does.
    if (leafCost + alpha <= subtreeCost + alpha * subtreeLeaves) {
      return {
        node: { ...node, isLeaf: true, pruned: true, left: null, right: null, split: null },
        cost: leafCost,
        leaves: 1,
      };
    }
    return {
      node: { ...node, left: left.node, right: right.node },
      cost: subtreeCost,
      leaves: subtreeLeaves,
    };
  };

  return pruneNode(root).node;
}

export function countLeaves(root: TreeNode): number {
  let count = 0;
  walk(root, (node) => {
    if (node.isLeaf) count += 1;
  });
  return count;
}

export function treeDepth(root: TreeNode): number {
  let depth = 0;
  walk(root, (node) => {
    if (node.isLeaf && node.depth > depth) depth = node.depth;
  });
  return depth;
}

/** Every leaf rectangle, for painting the decision regions without sampling. */
export function leafRegions(root: TreeNode): Array<{ node: TreeNode; bounds: TreeNode['bounds'] }> {
  const out: Array<{ node: TreeNode; bounds: TreeNode['bounds'] }> = [];
  walk(root, (node) => {
    if (node.isLeaf) out.push({ node, bounds: node.bounds });
  });
  return out;
}

/** Every internal split, ordered by depth, the cut lines drawn over the data. */
export function splitLines(root: TreeNode): Array<{
  node: TreeNode;
  feature: 0 | 1;
  threshold: number;
  bounds: TreeNode['bounds'];
}> {
  const out: Array<{ node: TreeNode; feature: 0 | 1; threshold: number; bounds: TreeNode['bounds'] }> =
    [];
  walk(root, (node) => {
    if (!node.isLeaf && node.split) {
      out.push({
        node,
        feature: node.split.feature,
        threshold: node.split.threshold,
        bounds: node.bounds,
      });
    }
  });
  return out.sort((a, b) => a.node.depth - b.node.depth);
}
