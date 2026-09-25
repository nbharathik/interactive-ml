/** The shared training library: optimisers, schedules, activations, losses, contours. */

import { strict as assert } from 'node:assert';
import { describe, it } from 'node:test';

import { applyDelta, createOptState, lookahead, lrAt, optStep } from '../src/lib/ml/optim.ts';
import type { OptimiserConfig } from '../src/lib/ml/optim.ts';
import { ACTIVATION_NAMES, activation, sigmoid, softplus } from '../src/lib/ml/activations.ts';
import { LOSS_NAMES, loss, softmax } from '../src/lib/ml/losses.ts';
import { LANDSCAPE_NAMES, makeLandscape, nearestMinimum, numericGrad } from '../src/lib/ml/landscapes.ts';
import { bestRun, createDescent, flipRate, isComplete, stepDescent } from '../src/lib/ml/descent.ts';
import type { DescentConfig } from '../src/lib/ml/descent.ts';
import { isoLevels, isoSegments } from '../src/lib/viz/contours.ts';
import type { SampledGrid } from '../src/lib/viz/contours.ts';

function close(actual: number, expected: number, tolerance = 1e-6, message?: string) {
  assert.ok(
    Math.abs(actual - expected) <= tolerance,
    (message ?? 'value') + ': expected ' + expected + ', got ' + actual,
  );
}

/** Central difference of a scalar function. */
function numeric(f: (x: number) => number, x: number, h = 1e-5): number {
  return (f(x + h) - f(x - h)) / (2 * h);
}

/* ================================================================== */

describe('optimisers', () => {
  const ravine = (w: readonly number[]) => [w[0], 20 * w[1]];

  function run(config: OptimiserConfig, lr: number, steps: number): { w: number[]; states: number } {
    let w = [3, 1];
    let state = createOptState(config, 2);
    let count = 0;
    for (let i = 0; i < steps; i++) {
      const probe = applyDelta(w, lookahead(state, config));
      const out = optStep(state, ravine(probe), lr, config);
      state = out.state;
      w = applyDelta(w, out.delta);
      count++;
      if (Math.hypot(w[0], w[1]) < 1e-3) break;
    }
    return { w, states: count };
  }

  it('optStep is pure and returns fresh arrays', () => {
    const config: OptimiserConfig = { name: 'adam' };
    const state = createOptState(config, 2);
    const before = JSON.stringify(state);
    const out = optStep(state, [1, -2], 0.1, config);
    assert.equal(JSON.stringify(state), before);
    assert.notEqual(out.state.m, state.m);
    assert.notEqual(out.delta, state.m);
  });

  it('Adam takes a step of size lr per coordinate first, whatever the gradient scale', () => {
    const config: OptimiserConfig = { name: 'adam' };
    for (const scale of [1e-3, 1, 1e3]) {
      const out = optStep(createOptState(config, 2), [scale, -scale], 0.05, config);
      close(Math.abs(out.delta[0]), 0.05, 1e-6, 'scale ' + scale);
      close(Math.abs(out.delta[1]), 0.05, 1e-6, 'scale ' + scale);
      assert.ok(out.delta[0] < 0 && out.delta[1] > 0);
    }
  });

  it('nesterov with no momentum is plain gradient descent, and its lookahead is beta1 times m', () => {
    const still = run({ name: 'nesterov', beta1: 0 }, 0.05, 30).w;
    const plain = run({ name: 'gd' }, 0.05, 30).w;
    close(still[0], plain[0], 1e-12);
    close(still[1], plain[1], 1e-12);
    const config: OptimiserConfig = { name: 'nesterov', beta1: 0.9 };
    const state = optStep(createOptState(config, 2), [1, 2], 0.1, config).state;
    const peek = lookahead(state, config);
    close(peek[0], 0.9 * state.m[0], 1e-12);
    close(peek[1], 0.9 * state.m[1], 1e-12);
    assert.deepEqual(lookahead(state, { name: 'momentum' }), [0, 0]);
  });

  it('adagrad never lengthens its step for a constant gradient', () => {
    const config: OptimiserConfig = { name: 'adagrad' };
    let state = createOptState(config, 1);
    let last = Infinity;
    for (let i = 0; i < 20; i++) {
      const out = optStep(state, [2], 0.1, config);
      state = out.state;
      assert.ok(Math.abs(out.delta[0]) <= last + 1e-12);
      last = Math.abs(out.delta[0]);
    }
  });

  it('momentum settles the ravine in fewer steps than plain descent', () => {
    const gd = run({ name: 'gd' }, 0.05, 2000);
    const mom = run({ name: 'momentum', beta1: 0.8 }, 0.05, 2000);
    assert.ok(Math.hypot(...gd.w) < 1e-3 && Math.hypot(...mom.w) < 1e-3, 'both settle');
    assert.ok(mom.states < gd.states, mom.states + ' vs ' + gd.states);
  });

  it('rmsprop and adam stay finite where plain descent at the same rate diverges', () => {
    const gd = run({ name: 'gd' }, 0.2, 200).w;
    assert.ok(!Number.isFinite(gd[1]) || Math.abs(gd[1]) > 1e6);
    for (const name of ['rmsprop', 'adam'] as const) {
      const w = run({ name }, 0.2, 200).w;
      assert.ok(Number.isFinite(w[0]) && Number.isFinite(w[1]), name);
      assert.ok(Math.hypot(w[0], w[1]) < 0.5, name + ' near the minimum');
    }
  });

  it('schedules have the expected shape', () => {
    close(lrAt({ name: 'constant' }, 500, 0.1), 0.1);
    close(lrAt({ name: 'step', gamma: 0.5, every: 100 }, 99, 0.1), 0.1);
    close(lrAt({ name: 'step', gamma: 0.5, every: 100 }, 100, 0.1), 0.05);
    close(lrAt({ name: 'exponential', gamma: 0.9 }, 3, 1), 0.729);
    close(lrAt({ name: 'cosine', horizon: 100, floor: 0.1 }, 0, 1), 1);
    close(lrAt({ name: 'cosine', horizon: 100, floor: 0.1 }, 100, 1), 0.1);
    close(lrAt({ name: 'cosine', horizon: 100, floor: 0.1 }, 1000, 1), 0.1);
    const warm = { name: 'warmup' as const, warmupSteps: 10, horizon: 100, floor: 0 };
    assert.ok(lrAt(warm, 0, 1) < lrAt(warm, 5, 1) && lrAt(warm, 5, 1) < lrAt(warm, 9, 1));
    close(lrAt(warm, 9, 1), 1);
    assert.ok(lrAt(warm, 60, 1) < lrAt(warm, 20, 1));
  });
});

/* ================================================================== */

describe('activations', () => {
  const zs = [-3, -1, -0.3, 0.3, 1, 3];

  it('every derivative matches a central difference', () => {
    for (const name of ACTIVATION_NAMES) {
      const fn = activation(name, { leakSlope: 0.2, eluAlpha: 1.5 });
      for (const z of zs) close(fn.df(z), numeric(fn.f, z), 1e-5, name + ' at ' + z);
    }
  });

  it('outputs stay inside the declared range', () => {
    for (const name of ACTIVATION_NAMES) {
      const fn = activation(name);
      if (!fn.range) continue;
      for (let z = -8; z <= 8; z += 0.25) {
        const v = fn.f(z);
        assert.ok(v >= fn.range[0] - 1e-12 && v <= fn.range[1] + 1e-12, name + ' at ' + z);
      }
    }
  });

  it('softplus approaches relu and everything stays finite at extreme z', () => {
    close(softplus(30), 30, 1e-9);
    close(softplus(-30), 0, 1e-9);
    for (const name of ACTIVATION_NAMES) {
      const fn = activation(name);
      for (const z of [-1000, 1000]) {
        assert.ok(Number.isFinite(fn.f(z)) && Number.isFinite(fn.df(z)), name + ' at ' + z);
      }
    }
    close(sigmoid(0), 0.5, 1e-12);
  });

  it('relu, elu and sigmoid saturate on the left; leaky relu does not', () => {
    assert.equal(activation('relu').df(-2), 0);
    assert.ok(activation('elu').df(-8) < 1e-3);
    assert.ok(activation('sigmoid').df(-12) < 1e-4);
    close(activation('leakyRelu', { leakSlope: 0.1 }).df(-5), 0.1, 1e-12);
  });
});

/* ================================================================== */

describe('losses', () => {
  it('every gradient matches a central difference', () => {
    for (const name of LOSS_NAMES) {
      const fn = loss(name, { huberDelta: 1, classCount: 4 });
      const width = fn.outputWidth(4);
      const target = fn.kind === 'regression' ? fn.encodeValue(0.4) : fn.encodeClass(1, 4);
      for (const seed of [0.3, -0.8, 1.7, 2.6]) {
        const output = Array.from({ length: width }, (_, i) => seed * (i + 1) - 0.5 * i);
        const grad = fn.grad(output, target);
        output.forEach((_, i) => {
          const f = (x: number) => fn.f(output.map((o, j) => (j === i ? x : o)), target);
          close(grad[i], numeric(f, output[i]), 1e-5, name + ' coord ' + i + ' seed ' + seed);
        });
      }
    }
  });

  it('huber matches squared error inside delta and absolute error outside', () => {
    const fn = loss('huber', { huberDelta: 1 });
    close(fn.f([0.5], [0]), 0.125, 1e-12);
    close(fn.f([3], [0]), 1 * (3 - 0.5), 1e-12);
    close(fn.grad([3], [0])[0], 1, 1e-12);
  });

  it('softmax sums to one and survives huge logits; its cross-entropy gradient sums to zero', () => {
    const p = softmax([1000, 999, -1000]);
    close(p.reduce((a, b) => a + b, 0), 1, 1e-9);
    assert.ok(p.every(Number.isFinite));
    const fn = loss('softmaxCE', { classCount: 3 });
    const g = fn.grad([2, -1, 0.5], fn.encodeClass(2, 3));
    close(g.reduce((a, b) => a + b, 0), 0, 1e-9);
    assert.ok(g[2] < 0 && g[0] > 0);
    close(fn.f([0, 0, 0], fn.encodeClass(1, 3)), Math.log(3), 1e-9);
  });

  it('binary cross-entropy on a logit is softplus of the wrong-signed logit', () => {
    const fn = loss('bce');
    close(fn.f([2], [1]), softplus(-2), 1e-12);
    close(fn.f([2], [0]), softplus(2), 1e-12);
    close(fn.grad([0], [1])[0], -0.5, 1e-12);
  });

  it('hinge is flat once the margin is met', () => {
    const fn = loss('hinge');
    close(fn.f([1.5], fn.encodeClass(1, 2)), 0, 1e-12);
    close(fn.grad([1.5], fn.encodeClass(1, 2))[0], 0, 1e-12);
    close(fn.f([-0.5], fn.encodeClass(1, 2)), 1.5, 1e-12);
  });

  it('targets round-trip through encode and decode', () => {
    for (const name of LOSS_NAMES) {
      const fn = loss(name, { classCount: 3 });
      for (const label of [0, 1]) {
        const encoded = fn.encodeClass(label, 3);
        // The decoder reads a confident model output, not the target itself, except for softmax.
        const output = name === 'softmaxCE' ? encoded.map((v) => v * 5) : name === 'hinge' ? encoded : [encoded[0] * 4 - 2];
        assert.equal(fn.decodeClass(output), label, name + ' label ' + label);
      }
      if (fn.kind !== 'multiclass') close(fn.decodeValue(fn.encodeValue(0.7)), name === 'bce' ? sigmoid(0.7) : 0.7, 1e-12, name);
    }
  });

  it('the plotted curve agrees with the sample loss', () => {
    const bce = loss('bce');
    close(bce.curve.f(1.2, 1), bce.f([1.2], [1]), 1e-12);
    close(bce.curve.df(1.2, 0), bce.grad([1.2], [0])[0], 1e-12);
    const mse = loss('mse');
    close(mse.curve.f(2, 0), mse.f([2], [0]), 1e-12);
    const ce = loss('softmaxCE', { classCount: 4 });
    close(ce.curve.f(1, 0), ce.f([1, 0, 0, 0], ce.encodeClass(0, 4)), 1e-12);
  });
});

/* ================================================================== */

describe('contours', () => {
  function gridOf(f: (c: number, r: number) => number, cols: number, rows: number): SampledGrid {
    const values = new Float32Array(cols * rows);
    let min = Infinity;
    let max = -Infinity;
    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        const v = f(c, r);
        values[r * cols + c] = v;
        min = Math.min(min, v);
        max = Math.max(max, v);
      }
    }
    return { values, cols, rows, min, max };
  }

  it('crosses a linear ramp at the exact fractional column', () => {
    const grid = gridOf((c) => c, 5, 3);
    const segments = isoSegments(grid, 1.5);
    assert.ok(segments.length > 0);
    for (let i = 0; i < segments.length; i += 4) {
      close(segments[i], 1.5, 1e-6);
      close(segments[i + 2], 1.5, 1e-6);
    }
  });

  it('gives no segments for a level outside the range', () => {
    const grid = gridOf((c, r) => c + r, 4, 4);
    assert.equal(isoSegments(grid, 100).length, 0);
    assert.equal(isoSegments(grid, -1).length, 0);
  });

  it('isoLevels are increasing, strictly inside the range, and log mode bunches near the floor', () => {
    const lin = isoLevels(0, 10, 4);
    assert.deepEqual(lin, [2, 4, 6, 8]);
    const log = isoLevels(0, 10, 9, 'log');
    assert.equal(log.length, 9);
    for (let i = 1; i < log.length; i++) assert.ok(log[i] > log[i - 1]);
    assert.ok(log[0] > 0 && log[log.length - 1] < 10);
    assert.ok(log[0] < lin[0]);
    assert.deepEqual(isoLevels(3, 3, 4), []);
  });
});

/* ================================================================== */

describe('landscapes', () => {
  it('every gradient matches a central difference', () => {
    for (const name of LANDSCAPE_NAMES) {
      const land = makeLandscape(name, { condition: 12 });
      const probes: Array<[number, number]> = [
        [0.3, -0.7],
        [-1.5, 1.2],
        [2, 2],
        [-0.4, 0.9],
        [1.7, -2.1],
        land.start,
      ];
      for (const [x, y] of probes) {
        const g = land.grad(x, y);
        const n = numericGrad(land.f, x, y);
        close(g[0], n[0], 1e-4, name + ' dx at ' + x + ',' + y);
        close(g[1], n[1], 1e-4, name + ' dy at ' + x + ',' + y);
      }
    }
  });

  it('every listed minimum is stationary and the start sits inside the domain', () => {
    for (const name of LANDSCAPE_NAMES) {
      const land = makeLandscape(name);
      for (const [x, y] of land.minima) {
        const g = land.grad(x, y);
        assert.ok(Math.hypot(g[0], g[1]) < 1e-3, name + ' minimum ' + x + ',' + y);
      }
      const [sx, sy] = land.start;
      assert.ok(sx >= land.domain.x[0] && sx <= land.domain.x[1] && sy >= land.domain.y[0] && sy <= land.domain.y[1], name);
    }
  });

  it('the ravine condition number sets the curvature ratio', () => {
    const land = makeLandscape('ravine', { condition: 30 });
    const gx = land.grad(1, 0)[0];
    const gy = land.grad(0, 1)[1];
    close(gy / gx, 30, 1e-12);
  });

  it('nearestMinimum picks the closest basin', () => {
    const land = makeLandscape('himmelblau');
    const near = nearestMinimum(land, 2.8, 2.2);
    assert.equal(near.index, 0);
    assert.ok(near.distance < 0.4);
  });
});

/* ================================================================== */

describe('descent runner', () => {
  function config(overrides: Partial<DescentConfig> = {}): DescentConfig {
    const landscape = makeLandscape('ravine', { condition: 20 });
    return {
      landscape,
      start: landscape.start,
      optimisers: ['gd', 'momentum', 'adam'],
      learningRate: 0.09,
      beta1: 0.5,
      beta2: 0.999,
      schedule: { name: 'constant' },
      noise: 0,
      seed: 3,
      tolerance: 1e-3,
      ...overrides,
    };
  }

  function runFor(cfg: DescentConfig, steps: number) {
    let state = createDescent(cfg);
    for (let i = 0; i < steps && !isComplete(state); i++) state = stepDescent(state, cfg);
    return state;
  }

  it('stepDescent is pure', () => {
    const cfg = config();
    const state = createDescent(cfg);
    const before = JSON.stringify(state);
    const next = stepDescent(state, cfg);
    assert.equal(JSON.stringify(state), before);
    assert.notEqual(next.runs[0].path, state.runs[0].path);
    assert.equal(next.step, 1);
  });

  it('plain descent zig-zags across the ravine and momentum settles first', () => {
    const state = runFor(config(), 2000);
    const gd = state.runs.find((r) => r.name === 'gd')!;
    const mom = state.runs.find((r) => r.name === 'momentum')!;
    assert.ok(gd.converged && mom.converged, 'both settle');
    assert.ok(mom.steps < gd.steps, mom.steps + ' vs ' + gd.steps);
    // The first twenty steps flip direction almost every time.
    const early = runFor(config({ optimisers: ['gd'] }), 20).runs[0];
    assert.ok(flipRate(early, 18) > 0.6, 'flip rate ' + flipRate(early, 18));
  });

  it('a rate above 2 over the curvature diverges and is flagged', () => {
    const state = runFor(config({ optimisers: ['gd', 'adam'], learningRate: 0.2 }), 300);
    const gd = state.runs[0];
    const adam = state.runs[1];
    assert.ok(gd.diverged);
    assert.ok(!adam.diverged && Number.isFinite(adam.loss));
    assert.ok(isComplete(state) || state.step === 300);
    assert.equal(bestRun(state)?.name, 'adam');
  });

  it('gradient noise replays for the same seed and differs for another', () => {
    const a = runFor(config({ noise: 0.5 }), 40);
    const b = runFor(config({ noise: 0.5 }), 40);
    const c = runFor(config({ noise: 0.5, seed: 4 }), 40);
    assert.deepEqual(a.runs[0].position, b.runs[0].position);
    assert.notDeepEqual(a.runs[0].position, c.runs[0].position);
  });

  it('nesterov probes ahead of its position once it has momentum', () => {
    const state = runFor(config({ optimisers: ['nesterov'] }), 3);
    const run = state.runs[0];
    assert.ok(Math.hypot(run.probe[0] - run.path[run.path.length - 2][0], run.probe[1] - run.path[run.path.length - 2][1]) > 1e-6);
  });

  it('Adam leaves the saddle long before plain descent does', () => {
    const landscape = makeLandscape('saddle');
    const cfg = config({ landscape, start: landscape.start, optimisers: ['gd', 'adam'], learningRate: 0.1 });
    const state = runFor(cfg, 400);
    const gd = state.runs[0];
    const adam = state.runs[1];
    assert.ok(adam.converged, 'adam settles');
    assert.ok(nearestMinimum(landscape, adam.position[0], adam.position[1]).distance < 0.05);
    const escape = (path: Array<[number, number]>) => path.findIndex(([, y]) => Math.abs(y) > 0.5);
    assert.ok(escape(adam.path) > 0 && escape(adam.path) * 3 < escape(gd.path), escape(adam.path) + ' vs ' + escape(gd.path));
  });

  it('a cosine schedule lets a noisy run settle closer than a constant rate', () => {
    const cfg = (schedule: DescentConfig['schedule']) =>
      config({ landscape: makeLandscape('bowl'), start: [3, 2.5], optimisers: ['gd'], learningRate: 0.1, noise: 0.5, schedule });
    const flat = runFor(cfg({ name: 'constant' }), 400).runs[0];
    const cool = runFor(cfg({ name: 'cosine', horizon: 300, floor: 0.02 }), 400).runs[0];
    assert.ok(Math.hypot(...cool.position) < Math.hypot(...flat.position), cool.position + ' vs ' + flat.position);
  });
});
