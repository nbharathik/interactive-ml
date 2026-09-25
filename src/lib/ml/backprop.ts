/** Backpropagation one phase at a time: a two-layer network, one sample, seven ticks per pass. */

import type { ActivationName } from './activations';
import type { LossName } from './losses';
import {
  addGradients,
  applyGradients,
  createMlp,
  gradientCheck,
  layerDelta,
  layerForward,
  layerGradients,
  lossFnOf,
  outputDelta,
  scaleGradients,
  zeroGradients,
} from './mlp';
import type { Gradients, Mlp, MlpSpec } from './mlp';
import type { Initialiser } from './neuralNetwork';

export type BackpropPhase =
  | 'forward-hidden'
  | 'forward-output'
  | 'loss'
  | 'delta-output'
  | 'delta-hidden'
  | 'gradients'
  | 'update';

export const PHASES: readonly BackpropPhase[] = [
  'forward-hidden',
  'forward-output',
  'loss',
  'delta-output',
  'delta-hidden',
  'gradients',
  'update',
];

export const PHASE_LABELS: Record<BackpropPhase, string> = {
  'forward-hidden': 'Forward: hidden layer',
  'forward-output': 'Forward: output',
  loss: 'Loss',
  'delta-output': 'Backward: output δ',
  'delta-hidden': 'Backward: hidden δ',
  gradients: 'Gradients',
  update: 'Update',
};

export type UpdateMode = 'sample' | 'batch';

export interface BackpropConfig {
  hiddenUnits: number;
  hiddenActivation: ActivationName;
  outputActivation: ActivationName;
  loss: LossName;
  learningRate: number;
  /** Update after every sample, or accumulate and update once per epoch. */
  mode: UpdateMode;
  init: Initialiser | 'fixed';
  seed: number;
  inputs: number[][];
  targets: number[][];
}

/** Everything a phase has produced so far; null until its phase has run. */
export interface PhaseCache {
  z: number[] | null;
  a: number[] | null;
  delta: number[] | null;
  dW: number[] | null;
  db: number[] | null;
}

export interface BackpropState {
  net: Mlp;
  spec: MlpSpec;
  phase: BackpropPhase;
  /** Which sample the current pass is about. */
  sampleIndex: number;
  epoch: number;
  tick: number;
  updates: number;
  input: number[];
  target: number[];
  cache: [PhaseCache, PhaseCache];
  loss: number | null;
  /** The weights the update phase will commit, so before and after can both be shown. */
  pending: Mlp | null;
  /** Gradients summed over the epoch in batch mode. */
  accumulated: Gradients | null;
  epochLossSum: number;
  /** Loss of every completed pass. */
  lossHistory: number[];
  /** Mean loss per completed epoch. */
  epochLosses: number[];
  /** Largest gap between the analytic and the numerical gradient in the last pass. */
  checkError: number | null;
  diverged: boolean;
}

const MAX_HISTORY = 3000;

/** The textbook two-input, two-hidden, one-output weights, so a walkthrough matches a worked example. */
const FIXED_HIDDEN = { W: [0.15, 0.2, 0.25, 0.3], b: [0.35, 0.35] };
const FIXED_OUTPUT = { W: [0.4, 0.45], b: [0.6] };

function emptyCache(): PhaseCache {
  return { z: null, a: null, delta: null, dW: null, db: null };
}

export function specOf(config: BackpropConfig): MlpSpec {
  return {
    sizes: [2, Math.max(1, config.hiddenUnits), 1],
    activations: [config.hiddenActivation, config.outputActivation],
    loss: config.loss,
    init: config.init === 'fixed' ? 'small' : config.init,
    seed: config.seed,
  };
}

function buildNet(config: BackpropConfig, spec: MlpSpec): Mlp {
  const net = createMlp(spec);
  if (config.init !== 'fixed') return net;
  const hidden = net.layers[0];
  const output = net.layers[1];
  if (hidden.outSize === 2) {
    return {
      layers: [
        { ...hidden, W: FIXED_HIDDEN.W.slice(), b: FIXED_HIDDEN.b.slice() },
        { ...output, W: FIXED_OUTPUT.W.slice(), b: FIXED_OUTPUT.b.slice() },
      ],
    };
  }
  // Other widths get a deterministic spread of the same flavour.
  return {
    layers: [
      { ...hidden, W: hidden.W.map((_, i) => 0.15 + 0.05 * i), b: hidden.b.map(() => 0.35) },
      { ...output, W: output.W.map((_, i) => 0.4 + 0.05 * i), b: [0.6] },
    ],
  };
}

export function createState(config: BackpropConfig): BackpropState {
  const spec = specOf(config);
  const net = buildNet(config, spec);
  return {
    net,
    spec,
    phase: 'forward-hidden',
    sampleIndex: 0,
    epoch: 0,
    tick: 0,
    updates: 0,
    input: config.inputs[0] ?? [0, 0],
    target: config.targets[0] ?? [0],
    cache: [emptyCache(), emptyCache()],
    loss: null,
    pending: null,
    accumulated: config.mode === 'batch' ? zeroGradients(net) : null,
    epochLossSum: 0,
    lossHistory: [],
    epochLosses: [],
    checkError: null,
    diverged: false,
  };
}

export function phaseAfter(phase: BackpropPhase): BackpropPhase {
  return PHASES[(PHASES.indexOf(phase) + 1) % PHASES.length];
}

/** One micro-phase. Pure. */
export function step(state: BackpropState, config: BackpropConfig): BackpropState {
  if (state.diverged || config.inputs.length === 0) return state;
  const { net, spec } = state;
  const [hidden, output] = net.layers;
  const [h, o] = state.cache;
  const next: BackpropState = { ...state, tick: state.tick + 1, cache: [{ ...h }, { ...o }] };

  switch (state.phase) {
    case 'forward-hidden': {
      const { z, a } = layerForward(hidden, state.input, spec.activationParams);
      next.cache[0] = { ...emptyCache(), z, a };
      next.cache[1] = emptyCache();
      next.loss = null;
      next.pending = null;
      break;
    }
    case 'forward-output': {
      const { z, a } = layerForward(output, h.a ?? [], spec.activationParams);
      next.cache[1] = { ...emptyCache(), z, a };
      break;
    }
    case 'loss':
      next.loss = lossFnOf(spec).f(o.a ?? [], state.target);
      break;
    case 'delta-output':
      next.cache[1] = { ...o, delta: outputDelta(output, o.z ?? [], o.a ?? [], state.target, lossFnOf(spec), spec.activationParams) };
      break;
    case 'delta-hidden':
      next.cache[0] = { ...h, delta: layerDelta(hidden, h.z ?? [], output, o.delta ?? [], spec.activationParams) };
      break;
    case 'gradients': {
      const gh = layerGradients(h.delta ?? [], state.input);
      const go = layerGradients(o.delta ?? [], h.a ?? []);
      next.cache[0] = { ...h, dW: gh.dW, db: gh.db };
      next.cache[1] = { ...o, dW: go.dW, db: go.db };
      const grads: Gradients = { dW: [gh.dW, go.dW], db: [gh.db, go.db] };
      const check = gradientCheck(net, state.input, state.target, spec);
      next.checkError = check.maxAbs;
      if (config.mode === 'batch') {
        const layerGrads = [
          { delta: h.delta ?? [], dW: gh.dW, db: gh.db },
          { delta: o.delta ?? [], dW: go.dW, db: go.db },
        ];
        next.accumulated = addGradients(state.accumulated ?? zeroGradients(net), layerGrads);
        const last = state.sampleIndex === config.inputs.length - 1;
        next.pending = last ? applyGradients(net, scaleGradients(next.accumulated, 1 / config.inputs.length), config.learningRate) : null;
      } else {
        next.pending = applyGradients(net, grads, config.learningRate);
      }
      break;
    }
    case 'update': {
      const last = state.sampleIndex === config.inputs.length - 1;
      const passLoss = state.loss ?? 0;
      const history = state.lossHistory.concat(passLoss);
      next.lossHistory = history.length > MAX_HISTORY ? history.slice(-MAX_HISTORY) : history;
      next.epochLossSum = state.epochLossSum + passLoss;
      if (state.pending) {
        next.net = state.pending;
        next.updates = state.updates + 1;
        if (config.mode === 'batch') next.accumulated = zeroGradients(net);
      }
      next.diverged = next.net.layers.some((l) => l.W.some((w) => !Number.isFinite(w) || Math.abs(w) > 1e6));
      if (last) {
        next.epoch = state.epoch + 1;
        const losses = state.epochLosses.concat(next.epochLossSum / config.inputs.length);
        next.epochLosses = losses.length > MAX_HISTORY ? losses.slice(-MAX_HISTORY) : losses;
        next.epochLossSum = 0;
      }
      next.sampleIndex = (state.sampleIndex + 1) % config.inputs.length;
      next.input = config.inputs[next.sampleIndex];
      next.target = config.targets[next.sampleIndex];
      break;
    }
    default:
      break;
  }
  next.phase = phaseAfter(state.phase);
  return next;
}

/** Run every phase of every sample once. */
export function runEpoch(state: BackpropState, config: BackpropConfig): BackpropState {
  let current = state;
  const ticks = PHASES.length * config.inputs.length;
  for (let i = 0; i < ticks; i++) current = step(current, config);
  return current;
}

/** How far through the seven phases the current pass is, 1-based. */
export function phaseIndex(phase: BackpropPhase): number {
  return PHASES.indexOf(phase) + 1;
}
