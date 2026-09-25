/** First-order optimisers and learning-rate schedules, shared by every trainer. */

export type OptimiserName = 'gd' | 'momentum' | 'nesterov' | 'adagrad' | 'rmsprop' | 'adam';

export interface OptimiserConfig {
  name: OptimiserName;
  /** Momentum coefficient, and beta1 for Adam. */
  beta1?: number;
  /** Second-moment decay: Adam 0.999, RMSProp 0.9. */
  beta2?: number;
  eps?: number;
}

/** Accumulators for one parameter vector. `m` is momentum or the first moment, `v` the second. */
export interface OptState {
  name: OptimiserName;
  t: number;
  m: number[];
  v: number[];
}

export const OPTIMISER_NAMES: readonly OptimiserName[] = ['gd', 'momentum', 'nesterov', 'adagrad', 'rmsprop', 'adam'];

export const OPTIMISER_LABELS: Record<OptimiserName, string> = {
  gd: 'Gradient descent',
  momentum: 'Momentum',
  nesterov: 'Nesterov',
  adagrad: 'Adagrad',
  rmsprop: 'RMSProp',
  adam: 'Adam',
};

const DEFAULT_EPS = 1e-8;

function beta1Of(config: OptimiserConfig): number {
  return config.beta1 ?? 0.9;
}

function beta2Of(config: OptimiserConfig): number {
  return config.beta2 ?? (config.name === 'rmsprop' ? 0.9 : 0.999);
}

export function createOptState(config: OptimiserConfig, size: number): OptState {
  return { name: config.name, t: 0, m: new Array<number>(size).fill(0), v: new Array<number>(size).fill(0) };
}

/** Where the gradient must be taken: Nesterov peeks ahead along the momentum, the rest stay put. */
export function lookahead(state: OptState, config: OptimiserConfig): number[] {
  const out = new Array<number>(state.m.length).fill(0);
  if (config.name !== 'nesterov') return out;
  const b1 = beta1Of(config);
  for (let i = 0; i < out.length; i++) out[i] = b1 * state.m[i];
  return out;
}

/** One update. Returns the new accumulators and the delta to add to the parameters. */
export function optStep(
  state: OptState,
  grad: ArrayLike<number>,
  lr: number,
  config: OptimiserConfig,
): { state: OptState; delta: number[] } {
  const n = state.m.length;
  const m = state.m.slice();
  const v = state.v.slice();
  const delta = new Array<number>(n).fill(0);
  const eps = config.eps ?? DEFAULT_EPS;
  const t = state.t + 1;

  switch (config.name) {
    case 'momentum':
    case 'nesterov': {
      const b1 = beta1Of(config);
      for (let i = 0; i < n; i++) {
        m[i] = b1 * m[i] - lr * grad[i];
        delta[i] = m[i];
      }
      break;
    }
    case 'adagrad':
      for (let i = 0; i < n; i++) {
        v[i] += grad[i] * grad[i];
        delta[i] = (-lr * grad[i]) / (Math.sqrt(v[i]) + eps);
      }
      break;
    case 'rmsprop': {
      const b2 = beta2Of(config);
      for (let i = 0; i < n; i++) {
        v[i] = b2 * v[i] + (1 - b2) * grad[i] * grad[i];
        delta[i] = (-lr * grad[i]) / (Math.sqrt(v[i]) + eps);
      }
      break;
    }
    case 'adam': {
      const b1 = beta1Of(config);
      const b2 = beta2Of(config);
      const c1 = 1 - Math.pow(b1, t);
      const c2 = 1 - Math.pow(b2, t);
      for (let i = 0; i < n; i++) {
        m[i] = b1 * m[i] + (1 - b1) * grad[i];
        v[i] = b2 * v[i] + (1 - b2) * grad[i] * grad[i];
        delta[i] = (-lr * (m[i] / c1)) / (Math.sqrt(v[i] / c2) + eps);
      }
      break;
    }
    default:
      for (let i = 0; i < n; i++) delta[i] = -lr * grad[i];
      break;
  }
  return { state: { name: state.name, t, m, v }, delta };
}

export function applyDelta(params: ArrayLike<number>, delta: ArrayLike<number>): number[] {
  const out = new Array<number>(params.length);
  for (let i = 0; i < params.length; i++) out[i] = params[i] + delta[i];
  return out;
}

/* ---------------- schedules ---------------- */

export type ScheduleName = 'constant' | 'step' | 'exponential' | 'cosine' | 'warmup';

export interface Schedule {
  name: ScheduleName;
  /** Decay factor for step and exponential. */
  gamma?: number;
  /** Steps between step-decay drops. */
  every?: number;
  /** Steps over which cosine falls to the floor. */
  horizon?: number;
  warmupSteps?: number;
  /** Fraction of the base rate cosine ends at. */
  floor?: number;
}

export const SCHEDULE_LABELS: Record<ScheduleName, string> = {
  constant: 'Constant',
  step: 'Step decay',
  exponential: 'Exponential',
  cosine: 'Cosine',
  warmup: 'Warmup',
};

function cosineAt(step: number, base: number, horizon: number, floor: number): number {
  const t = Math.min(1, Math.max(0, step / Math.max(1, horizon)));
  return base * (floor + (1 - floor) * 0.5 * (1 + Math.cos(Math.PI * t)));
}

/** The learning rate at `step`, counting from zero. */
export function lrAt(schedule: Schedule, step: number, base: number): number {
  switch (schedule.name) {
    case 'step':
      return base * Math.pow(schedule.gamma ?? 0.5, Math.floor(step / Math.max(1, schedule.every ?? 100)));
    case 'exponential':
      return base * Math.pow(schedule.gamma ?? 0.99, step);
    case 'cosine':
      return cosineAt(step, base, schedule.horizon ?? 500, schedule.floor ?? 0.02);
    case 'warmup': {
      const warm = Math.max(1, schedule.warmupSteps ?? 50);
      if (step < warm) return (base * (step + 1)) / warm;
      return cosineAt(step - warm, base, schedule.horizon ?? 500, schedule.floor ?? 0.02);
    }
    default:
      return base;
  }
}
