/** Several optimisers walking one landscape from one start. Every step is pure. */

import { gauss, makeRng } from '../math/rng';
import type { Landscape } from './landscapes';
import { applyDelta, createOptState, lookahead, lrAt, optStep } from './optim';
import type { OptState, OptimiserName, Schedule } from './optim';

export interface DescentConfig {
  landscape: Landscape;
  start: [number, number];
  optimisers: readonly OptimiserName[];
  learningRate: number;
  beta1: number;
  beta2: number;
  schedule: Schedule;
  /** Standard deviation of the noise added to every gradient component. */
  noise: number;
  seed: number;
  /** Gradient norm below which a run counts as settled. */
  tolerance: number;
}

export interface RunState {
  name: OptimiserName;
  opt: OptState;
  position: [number, number];
  /** Where the last gradient was taken; Nesterov peeks ahead of the position. */
  probe: [number, number];
  lastGrad: [number, number];
  lastDelta: [number, number];
  lastLr: number;
  loss: number;
  path: Array<[number, number]>;
  losses: number[];
  gradNorms: number[];
  stepSizes: number[];
  steps: number;
  converged: boolean;
  diverged: boolean;
}

export interface DescentState {
  step: number;
  runs: RunState[];
}

const MAX_PATH = 600;
const MAX_HISTORY = 3000;

function trim<T>(list: T[], limit: number): T[] {
  return list.length > limit ? list.slice(-limit) : list;
}

function createRun(name: OptimiserName, config: DescentConfig): RunState {
  const [x, y] = config.start;
  const loss = config.landscape.f(x, y);
  const grad = config.landscape.grad(x, y);
  return {
    name,
    opt: createOptState({ name, beta1: config.beta1, beta2: config.beta2 }, 2),
    position: [x, y],
    probe: [x, y],
    lastGrad: grad,
    lastDelta: [0, 0],
    lastLr: lrAt(config.schedule, 0, config.learningRate),
    loss,
    path: [[x, y]],
    losses: [loss],
    gradNorms: [Math.hypot(grad[0], grad[1])],
    stepSizes: [],
    steps: 0,
    converged: false,
    diverged: false,
  };
}

export function createDescent(config: DescentConfig): DescentState {
  return { step: 0, runs: config.optimisers.map((name) => createRun(name, config)) };
}

/** A position far outside the landscape counts as diverged. */
function escaped(landscape: Landscape, x: number, y: number): boolean {
  const spanX = landscape.domain.x[1] - landscape.domain.x[0];
  const spanY = landscape.domain.y[1] - landscape.domain.y[0];
  return !Number.isFinite(x) || !Number.isFinite(y) || Math.abs(x) > 50 * spanX || Math.abs(y) > 50 * spanY;
}

function stepRun(run: RunState, index: number, state: DescentState, config: DescentConfig): RunState {
  if (run.converged || run.diverged) return run;
  const { landscape } = config;
  const optConfig = { name: run.name, beta1: config.beta1, beta2: config.beta2 };
  const lr = lrAt(config.schedule, run.steps, config.learningRate);
  const probe = applyDelta(run.position, lookahead(run.opt, optConfig)) as [number, number];
  const grad = landscape.grad(probe[0], probe[1]);
  if (config.noise > 0) {
    const rng = makeRng(config.seed + state.step * 7919 + index * 104729);
    grad[0] += gauss(rng, 0, config.noise);
    grad[1] += gauss(rng, 0, config.noise);
  }
  const { state: opt, delta } = optStep(run.opt, grad, lr, optConfig);
  const position = applyDelta(run.position, delta) as [number, number];
  const diverged = escaped(landscape, position[0], position[1]);
  const loss = diverged ? Number.POSITIVE_INFINITY : landscape.f(position[0], position[1]);
  const trueGrad = diverged ? [Infinity, Infinity] : landscape.grad(position[0], position[1]);
  const gradNorm = Math.hypot(trueGrad[0], trueGrad[1]);
  const stepSize = Math.hypot(delta[0], delta[1]);
  return {
    name: run.name,
    opt,
    position,
    probe,
    lastGrad: [grad[0], grad[1]],
    lastDelta: [delta[0], delta[1]],
    lastLr: lr,
    loss,
    path: diverged ? run.path : trim(run.path.concat([position]), MAX_PATH),
    losses: trim(run.losses.concat(diverged ? Number.NaN : loss), MAX_HISTORY),
    gradNorms: trim(run.gradNorms.concat(diverged ? Number.NaN : gradNorm), MAX_HISTORY),
    stepSizes: trim(run.stepSizes.concat(diverged ? Number.NaN : stepSize), MAX_HISTORY),
    steps: run.steps + 1,
    converged: !diverged && gradNorm < config.tolerance,
    diverged,
  };
}

/** Advance every live run by one update. */
export function stepDescent(state: DescentState, config: DescentConfig): DescentState {
  return {
    step: state.step + 1,
    runs: state.runs.map((run, index) => stepRun(run, index, state, config)),
  };
}

export function isComplete(state: DescentState): boolean {
  return state.runs.length > 0 && state.runs.every((run) => run.converged || run.diverged);
}

/** The run with the lowest loss right now, or null when none is live. */
export function bestRun(state: DescentState): RunState | null {
  let best: RunState | null = null;
  for (const run of state.runs) {
    if (run.diverged) continue;
    if (!best || run.loss < best.loss) best = run;
  }
  return best;
}

/** Share of the last `window` moves where a coordinate that mattered reversed sign, the zig-zag signal. */
export function flipRate(run: RunState, window = 20): number {
  const path = run.path;
  const n = Math.min(window, path.length - 2);
  if (n < 4) return 0;
  let flips = 0;
  for (let i = path.length - n; i < path.length - 1; i++) {
    const a = [path[i][0] - path[i - 1][0], path[i][1] - path[i - 1][1]];
    const b = [path[i + 1][0] - path[i][0], path[i + 1][1] - path[i][1]];
    const size = Math.hypot(b[0], b[1]);
    const flipped = (k: number) => a[k] * b[k] < 0 && Math.abs(b[k]) > 0.05 * size;
    if (flipped(0) || flipped(1)) flips++;
  }
  return flips / n;
}
