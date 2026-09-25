/** K-means (Lloyd's algorithm). A step is half an iteration: assign, then update. */

import type { Point2D } from '../datasets/types';
import { makeRng } from '../math/rng';

export type InitMethod = 'random-points' | 'kmeans++' | 'random-coords' | 'forgy-far';
export type Phase = 'seeded' | 'assign' | 'update';

export interface Centroid {
  x: number;
  y: number;
}

export interface KMeansConfig {
  k: number;
  init: InitMethod;
  seed: number;
  /** Stop once no centroid moves further than this. */
  tolerance: number;
}

export interface KMeansState {
  centroids: Centroid[];
  /** Where the centroids were before the most recent update, drawn as ghosts. */
  previousCentroids: Centroid[];
  /** Cluster index per point, or -1 before the first assignment. */
  assignments: number[];
  /** Which half-step produced this state. */
  phase: Phase;
  /** Full assign+update rounds completed. */
  iteration: number;
  /** Within-cluster sum of squares after the most recent move. */
  inertia: number;
  inertiaHistory: number[];
  /** Largest distance any centroid travelled in the last update. */
  shift: number;
  converged: boolean;
  /** Points whose cluster changed in the most recent assignment. */
  changedCount: number;
  /** Trail of every position each centroid has occupied. */
  trails: Centroid[][];
  /** Clusters that ended up with no points, the classic failure mode. */
  emptyClusters: number[];
}

function dist2(ax: number, ay: number, bx: number, by: number): number {
  const dx = ax - bx;
  const dy = ay - by;
  return dx * dx + dy * dy;
}

/* ---------------- initialisation ---------------- */

export function initCentroids(
  points: readonly Point2D[],
  config: KMeansConfig,
  bounds: { xRange: [number, number]; yRange: [number, number] },
): Centroid[] {
  const rng = makeRng(config.seed);
  const k = Math.max(1, config.k);
  if (points.length === 0) {
    return Array.from({ length: k }, () => ({ x: 0, y: 0 }));
  }

  switch (config.init) {
    case 'random-coords': {
      // Uniform in the bounding box.
      return Array.from({ length: k }, () => ({
        x: bounds.xRange[0] + rng() * (bounds.xRange[1] - bounds.xRange[0]),
        y: bounds.yRange[0] + rng() * (bounds.yRange[1] - bounds.yRange[0]),
      }));
    }

    case 'forgy-far': {
      // Deliberately bad: all seeds crowded into one corner.
      let cx = 0;
      let cy = 0;
      for (const p of points) {
        cx += p.x;
        cy += p.y;
      }
      cx /= points.length;
      cy /= points.length;
      let farthest = points[0];
      let best = -1;
      for (const p of points) {
        const d = dist2(p.x, p.y, cx, cy);
        if (d > best) {
          best = d;
          farthest = p;
        }
      }
      return Array.from({ length: k }, (_, i) => ({
        x: farthest.x + (rng() - 0.5) * 0.35 * (i + 1),
        y: farthest.y + (rng() - 0.5) * 0.35 * (i + 1),
      }));
    }

    case 'kmeans++': {
      const chosen: Centroid[] = [];
      const first = points[Math.floor(rng() * points.length)];
      chosen.push({ x: first.x, y: first.y });
      while (chosen.length < k) {
        // Distance to the nearest chosen centre, squared.
        const weights = points.map((p) => {
          let best = Infinity;
          for (const c of chosen) {
            const d = dist2(p.x, p.y, c.x, c.y);
            if (d < best) best = d;
          }
          return best;
        });
        const total = weights.reduce((a, b) => a + b, 0);
        if (total <= 0) {
          const p = points[Math.floor(rng() * points.length)];
          chosen.push({ x: p.x, y: p.y });
          continue;
        }
        // Sample proportionally to D².
        let target = rng() * total;
        let index = 0;
        for (let i = 0; i < weights.length; i++) {
          target -= weights[i];
          if (target <= 0) {
            index = i;
            break;
          }
          index = i;
        }
        chosen.push({ x: points[index].x, y: points[index].y });
      }
      return chosen;
    }

    default: {
      // Pick k distinct data points at random.
      const pool = points.map((_, i) => i);
      for (let i = 0; i < Math.min(k, pool.length); i++) {
        const j = i + Math.floor(rng() * (pool.length - i));
        [pool[i], pool[j]] = [pool[j], pool[i]];
      }
      return Array.from({ length: k }, (_, i) => {
        const p = points[pool[i % pool.length]];
        return { x: p.x, y: p.y };
      });
    }
  }
}

export function createState(
  points: readonly Point2D[],
  config: KMeansConfig,
  bounds: { xRange: [number, number]; yRange: [number, number] },
): KMeansState {
  const centroids = initCentroids(points, config, bounds);
  return {
    centroids,
    previousCentroids: centroids.map((c) => ({ ...c })),
    assignments: new Array<number>(points.length).fill(-1),
    phase: 'seeded',
    iteration: 0,
    inertia: 0,
    inertiaHistory: [],
    shift: Infinity,
    converged: false,
    changedCount: 0,
    trails: centroids.map((c) => [{ ...c }]),
    emptyClusters: [],
  };
}

/* ---------------- the two half-steps ---------------- */

export function assign(state: KMeansState, points: readonly Point2D[]): KMeansState {
  const assignments = new Array<number>(points.length);
  let changed = 0;
  for (let i = 0; i < points.length; i++) {
    let best = 0;
    let bestDist = Infinity;
    for (let c = 0; c < state.centroids.length; c++) {
      const d = dist2(points[i].x, points[i].y, state.centroids[c].x, state.centroids[c].y);
      if (d < bestDist) {
        bestDist = d;
        best = c;
      }
    }
    assignments[i] = best;
    if (state.assignments[i] !== best) changed += 1;
  }

  const inertia = computeInertia(points, assignments, state.centroids);
  return {
    ...state,
    assignments,
    phase: 'assign',
    inertia,
    changedCount: changed,
    // An assignment that moves nobody is convergence.
    converged: state.phase !== 'seeded' && changed === 0,
  };
}

export function update(state: KMeansState, points: readonly Point2D[]): KMeansState {
  const k = state.centroids.length;
  const sumX = new Float64Array(k);
  const sumY = new Float64Array(k);
  const counts = new Int32Array(k);

  for (let i = 0; i < points.length; i++) {
    const c = state.assignments[i];
    // A stale assignments array can outlive a reduction in k.
    if (!Number.isInteger(c) || c < 0 || c >= k) continue;
    sumX[c] += points[i].x;
    sumY[c] += points[i].y;
    counts[c] += 1;
  }

  const emptyClusters: number[] = [];
  const centroids: Centroid[] = [];
  for (let c = 0; c < k; c++) {
    if (counts[c] > 0) {
      centroids.push({ x: sumX[c] / counts[c], y: sumY[c] / counts[c] });
    } else {
      // An empty cluster keeps its position and is flagged.
      emptyClusters.push(c);
      centroids.push({ ...state.centroids[c] });
    }
  }

  let shift = 0;
  for (let c = 0; c < k; c++) {
    const d = Math.sqrt(dist2(centroids[c].x, centroids[c].y, state.centroids[c].x, state.centroids[c].y));
    if (d > shift) shift = d;
  }

  const inertia = computeInertia(points, state.assignments, centroids);
  const trails = state.trails.map((trail, c) => {
    const next = trail.concat({ ...centroids[c] });
    return next.length > 60 ? next.slice(-60) : next;
  });

  return {
    ...state,
    centroids,
    previousCentroids: state.centroids.map((c) => ({ ...c })),
    phase: 'update',
    iteration: state.iteration + 1,
    inertia,
    inertiaHistory: state.inertiaHistory.concat(inertia),
    shift,
    converged: shift < 1e-9,
    trails,
    emptyClusters,
  };
}

/** One half-step: whichever move comes next. */
export function step(state: KMeansState, points: readonly Point2D[]): KMeansState {
  if (state.converged) return state;
  return state.phase === 'assign' ? update(state, points) : assign(state, points);
}

export function isComplete(state: KMeansState, tolerance: number): boolean {
  if (state.phase === 'seeded') return false;
  // Stop when an assignment moved nobody, or when the last update fell under the tolerance.
  if (state.converged) return true;
  return state.phase === 'update' && state.iteration > 0 && state.shift < tolerance;
}

/* ---------------- metrics ---------------- */

export function computeInertia(
  points: readonly Point2D[],
  assignments: readonly number[],
  centroids: readonly Centroid[],
): number {
  let sum = 0;
  const n = Math.min(points.length, assignments.length);
  for (let i = 0; i < n; i++) {
    const c = assignments[i];
    // Also rejects undefined from a stale assignments array.
    if (!Number.isInteger(c) || c < 0 || c >= centroids.length) continue;
    sum += dist2(points[i].x, points[i].y, centroids[c].x, centroids[c].y);
  }
  return sum;
}

export function clusterSizes(assignments: readonly number[], k: number): number[] {
  const sizes = new Array<number>(k).fill(0);
  for (const a of assignments) if (a >= 0 && a < k) sizes[a] += 1;
  return sizes;
}

/** Mean silhouette score in [-1, 1]. Capped at 600 points because it is O(n²). */
export function silhouetteScore(
  points: readonly Point2D[],
  assignments: readonly number[],
  k: number,
): number {
  // Points and assignments can briefly disagree in length while a dataset changes.
  const n = Math.min(points.length, assignments.length);
  if (n < 2 || k < 2 || n > 600) return Number.NaN;

  const members: number[][] = Array.from({ length: k }, () => []);
  for (let i = 0; i < n; i++) {
    const label = assignments[i];
    if (Number.isInteger(label) && label >= 0 && label < k) members[label].push(i);
  }

  let total = 0;
  let counted = 0;
  for (let i = 0; i < n; i++) {
    const own = assignments[i];
    if (!Number.isInteger(own) || own < 0 || own >= k) continue;
    // A point alone in its cluster scores 0 (Rousseeuw's convention), rather than dropping out of the mean.
    if (members[own].length <= 1) {
      counted += 1;
      continue;
    }

    let a = 0;
    for (const j of members[own]) {
      if (j === i) continue;
      a += Math.sqrt(dist2(points[i].x, points[i].y, points[j].x, points[j].y));
    }
    a /= members[own].length - 1;

    let b = Infinity;
    for (let c = 0; c < k; c++) {
      if (c === own || members[c].length === 0) continue;
      let sum = 0;
      for (const j of members[c]) {
        sum += Math.sqrt(dist2(points[i].x, points[i].y, points[j].x, points[j].y));
      }
      const avg = sum / members[c].length;
      if (avg < b) b = avg;
    }
    if (!Number.isFinite(b)) continue;

    total += (b - a) / Math.max(a, b);
    counted += 1;
  }
  return counted === 0 ? Number.NaN : total / counted;
}

/** Final inertia for each k, best of `restarts` runs. Not guaranteed monotonic. */
export function elbowCurve(
  points: readonly Point2D[],
  maxK: number,
  config: Omit<KMeansConfig, 'k'>,
  bounds: { xRange: [number, number]; yRange: [number, number] },
  restarts = 3,
): Array<{ k: number; inertia: number; silhouette: number }> {
  const out: Array<{ k: number; inertia: number; silhouette: number }> = [];
  for (let k = 1; k <= maxK; k++) {
    let best = Infinity;
    let bestAssignments: number[] = [];
    for (let r = 0; r < restarts; r++) {
      const cfg: KMeansConfig = { ...config, k, seed: config.seed + r * 977 };
      let state = createState(points, cfg, bounds);
      for (let i = 0; i < 60; i++) {
        const next = step(state, points);
        if (next === state) break;
        state = next;
        if (state.phase === 'update' && state.shift < cfg.tolerance) {
          state = assign(state, points);
          break;
        }
      }
      if (state.inertia < best) {
        best = state.inertia;
        bestAssignments = state.assignments;
      }
    }
    out.push({
      k,
      inertia: best,
      silhouette: k >= 2 ? silhouetteScore(points, bestAssignments, k) : Number.NaN,
    });
  }
  return out;
}

/** Nearest centroid for a coordinate, used to paint the Voronoi regions. */
export function nearestCentroid(centroids: readonly Centroid[], x: number, y: number): number {
  let best = 0;
  let bestDist = Infinity;
  for (let c = 0; c < centroids.length; c++) {
    const d = dist2(x, y, centroids[c].x, centroids[c].y);
    if (d < bestDist) {
      bestDist = d;
      best = c;
    }
  }
  return best;
}
