/** k-nearest neighbours. `findNeighbours` returns the full ranked list so the vote can be replayed. */

import type { Point2D } from '../datasets/types';

export type DistanceMetric = 'euclidean' | 'manhattan' | 'chebyshev' | 'minkowski3';
export type WeightingScheme = 'uniform' | 'distance';

export interface KnnConfig {
  k: number;
  metric: DistanceMetric;
  weighting: WeightingScheme;
  classCount: number;
  /** Rescale both axes to unit variance before measuring distance. */
  standardise: boolean;
}

export interface Neighbour {
  index: number;
  point: Point2D;
  distance: number;
  /** Vote weight this neighbour contributes. */
  weight: number;
}

export interface Scaling {
  sx: number;
  sy: number;
}

export function makeScaling(points: readonly Point2D[], enabled: boolean): Scaling {
  if (!enabled || points.length === 0) return { sx: 1, sy: 1 };
  let mx = 0;
  let my = 0;
  for (const p of points) {
    mx += p.x;
    my += p.y;
  }
  mx /= points.length;
  my /= points.length;
  let vx = 0;
  let vy = 0;
  for (const p of points) {
    vx += (p.x - mx) ** 2;
    vy += (p.y - my) ** 2;
  }
  const sx = Math.sqrt(vx / points.length);
  const sy = Math.sqrt(vy / points.length);
  return { sx: sx < 1e-9 ? 1 : sx, sy: sy < 1e-9 ? 1 : sy };
}

export const METRIC_LABELS: Record<DistanceMetric, string> = {
  euclidean: 'Euclidean (L2)',
  manhattan: 'Manhattan (L1)',
  chebyshev: 'Chebyshev (L∞)',
  minkowski3: 'Minkowski (p = 3)',
};

export function distance(
  ax: number,
  ay: number,
  bx: number,
  by: number,
  metric: DistanceMetric,
  scaling: Scaling = { sx: 1, sy: 1 },
): number {
  const dx = Math.abs(ax - bx) / scaling.sx;
  const dy = Math.abs(ay - by) / scaling.sy;
  switch (metric) {
    case 'manhattan':
      return dx + dy;
    case 'chebyshev':
      return Math.max(dx, dy);
    case 'minkowski3':
      return Math.cbrt(dx ** 3 + dy ** 3);
    default:
      return Math.sqrt(dx * dx + dy * dy);
  }
}

/** All training points, ranked by distance from the query. */
export function rankNeighbours(
  points: readonly Point2D[],
  qx: number,
  qy: number,
  config: KnnConfig,
  scaling: Scaling,
): Neighbour[] {
  const scored = points.map((point, index) => {
    const d = distance(point.x, point.y, qx, qy, config.metric, scaling);
    return {
      index,
      point,
      distance: d,
      weight: config.weighting === 'distance' ? 1 / (d * d + 1e-9) : 1,
    };
  });
  scored.sort((a, b) => a.distance - b.distance);
  return scored;
}

/** The k closest, in the same order a full ranking would give. Keeps a sorted buffer of k instead of sorting every point. */
export function findNeighbours(
  points: readonly Point2D[],
  qx: number,
  qy: number,
  config: KnnConfig,
  scaling: Scaling,
): Neighbour[] {
  const k = Math.max(1, config.k);
  if (k * 4 >= points.length) return rankNeighbours(points, qx, qy, config, scaling).slice(0, k);
  const best: Neighbour[] = [];
  for (let index = 0; index < points.length; index++) {
    const point = points[index];
    const d = distance(point.x, point.y, qx, qy, config.metric, scaling);
    const full = best.length === k;
    if (full && d >= best[k - 1].distance) continue;
    const neighbour: Neighbour = {
      index,
      point,
      distance: d,
      weight: config.weighting === 'distance' ? 1 / (d * d + 1e-9) : 1,
    };
    let i = best.length;
    if (full) i = k - 1;
    else best.push(neighbour);
    while (i > 0 && best[i - 1].distance > d) {
      best[i] = best[i - 1];
      i -= 1;
    }
    best[i] = neighbour;
  }
  return best;
}

export interface Vote {
  /** Weighted vote per class. */
  scores: number[];
  winner: number;
  /** Winner's share of the total weight, in [0, 1]. */
  confidence: number;
  /** True when two or more classes are exactly level. */
  tied: boolean;
}

export function tally(neighbours: readonly Neighbour[], classCount: number): Vote {
  const scores = new Array<number>(classCount).fill(0);
  let total = 0;
  for (const n of neighbours) {
    const label = n.point.label;
    if (label >= 0 && label < classCount) {
      scores[label] += n.weight;
      total += n.weight;
    }
  }

  // No votes is not a tie.
  if (total === 0) {
    return { scores, winner: 0, confidence: 0, tied: false };
  }

  let winner = 0;
  let best = -Infinity;
  let tied = false;
  for (let c = 0; c < classCount; c++) {
    if (scores[c] > best + 1e-12) {
      best = scores[c];
      winner = c;
      tied = false;
    } else if (Math.abs(scores[c] - best) <= 1e-12) {
      tied = true;
    }
  }
  return { scores, winner, confidence: total === 0 ? 0 : best / total, tied };
}

export function classify(
  points: readonly Point2D[],
  qx: number,
  qy: number,
  config: KnnConfig,
  scaling: Scaling,
): Vote {
  return tally(findNeighbours(points, qx, qy, config, scaling), config.classCount);
}

/** Leave-one-out accuracy: classify each point using every other point. */
export function leaveOneOutAccuracy(
  points: readonly Point2D[],
  config: KnnConfig,
  scaling: Scaling,
): number {
  if (points.length < 2) return 0;
  let hits = 0;
  for (let i = 0; i < points.length; i++) {
    const ranked = rankNeighbours(points, points[i].x, points[i].y, config, scaling)
      .filter((n) => n.index !== i)
      .slice(0, Math.max(1, config.k));
    if (tally(ranked, config.classCount).winner === points[i].label) hits += 1;
  }
  return hits / points.length;
}

/** Leave-one-out accuracy across a range of k, for the "which k?" chart. */
export function accuracyAcrossK(
  points: readonly Point2D[],
  maxK: number,
  config: Omit<KnnConfig, 'k'>,
  scaling: Scaling,
): Array<{ k: number; accuracy: number }> {
  if (points.length < 2) return [];
  const limit = Math.min(maxK, points.length - 1);

  // Rank once per point, then read off prefixes.
  const rankings = points.map((p, i) =>
    rankNeighbours(points, p.x, p.y, { ...config, k: 1 }, scaling).filter((n) => n.index !== i),
  );

  const out: Array<{ k: number; accuracy: number }> = [];
  for (let k = 1; k <= limit; k++) {
    let hits = 0;
    for (let i = 0; i < points.length; i++) {
      const vote = tally(rankings[i].slice(0, k), config.classCount);
      if (vote.winner === points[i].label) hits += 1;
    }
    out.push({ k, accuracy: hits / points.length });
  }
  return out;
}

/** Training-set accuracy, always 100% at k = 1, which is the point. */
export function trainingAccuracy(
  points: readonly Point2D[],
  config: KnnConfig,
  scaling: Scaling,
): number {
  if (points.length === 0) return 0;
  let hits = 0;
  for (const p of points) {
    if (classify(points, p.x, p.y, config, scaling).winner === p.label) hits += 1;
  }
  return hits / points.length;
}
