/** The grids and samplers the page, the lessons and their test share. */

import { CURVE_X_RANGE, refitSeed, renoise } from '../../lib/datasets/curves';
import type { RegressionPoint } from '../../lib/datasets/types';
import { biasVariance, gridOver } from '../../lib/ml/polynomialRegression';
import type { BiasVariance, RefitState } from '../../lib/ml/polynomialRegression';

/** Refit curves are drawn on the whole range. */
export const CURVE_GRID = gridOver(CURVE_X_RANGE, 81);

/** Bias and variance are read off the inner 90%, clear of the edges where every polynomial swings. */
const INSET = (CURVE_X_RANGE[1] - CURVE_X_RANGE[0]) * 0.05;
const INNER = CURVE_GRID.map((_, i) => i).filter(
  (i) => CURVE_GRID[i] >= CURVE_X_RANGE[0] + INSET - 1e-9 && CURVE_GRID[i] <= CURVE_X_RANGE[1] - INSET + 1e-9,
);
export const BV_GRID = INNER.map((i) => CURVE_GRID[i]);

/** The refits so far, scored on the inner grid. */
export function innerBiasVariance(state: RefitState, truthOnInner: readonly number[]): BiasVariance {
  return biasVariance(
    { ...state, mean: INNER.map((i) => state.mean[i]), m2: INNER.map((i) => state.m2[i]) },
    truthOnInner,
  );
}

/** Fresh noise on the same x's, one sample per refit. */
export function samplerFor(
  points: readonly RegressionPoint[],
  truth: (x: number) => number,
  noise: number,
  seed: number,
): (k: number) => RegressionPoint[] {
  return (k) => renoise(points, truth, noise, refitSeed(seed, k));
}
