/** The dense network, the backpropagation phase machine, and the trainers built on them. */

import { strict as assert } from 'node:assert';
import { describe, it } from 'node:test';

import { ACTIVATION_NAMES } from '../src/lib/ml/activations.ts';
import type { ActivationName } from '../src/lib/ml/activations.ts';
import { LOSS_NAMES, loss } from '../src/lib/ml/losses.ts';
import type { LossName } from '../src/lib/ml/losses.ts';
import * as mlp from '../src/lib/ml/mlp.ts';
import type { MlpSpec } from '../src/lib/ml/mlp.ts';
import * as nn from '../src/lib/ml/neuralNetwork.ts';
import { PHASES, createState, phaseIndex, runEpoch, step } from '../src/lib/ml/backprop.ts';
import type { BackpropConfig } from '../src/lib/ml/backprop.ts';
import { createTrainer, dataFromPoints, dataFromRegression, lossAxisValue, signedScore, signedScoreFromHidden, stepTrainer, valueScale } from '../src/lib/ml/mlpTrainer.ts';
import { extent2, gridLines, outputFromHidden, principalBasis, projectOnto, traceLayers } from '../src/lib/ml/warp.ts';
import { makeProjector } from '../src/lib/viz/surface.ts';
import { decodeTarget, encodeTarget, sameTarget } from '../src/explainers/activation-functions/targets.ts';
import { TILE_N, creaseSegments, rowRange, sampleUnit, sampleUnits, tileNorm, weightedSum } from '../src/explainers/activation-functions/tiles.ts';
import { activation } from '../src/lib/ml/activations.ts';
import type { TrainerConfig } from '../src/lib/ml/mlpTrainer.ts';
import { getClassificationDataset, splitData } from '../src/lib/datasets/points.ts';
import { REGRESSION_DATASETS } from '../src/lib/datasets/regression.ts';

function close(actual: number, expected: number, tolerance = 1e-6, message?: string) {
  assert.ok(
    Math.abs(actual - expected) <= tolerance,
    (message ?? 'value') + ': expected ' + expected + ', got ' + actual,
  );
}

/* ================================================================== */

describe('mlp', () => {
  const spec: MlpSpec = { sizes: [2, 3, 2], activations: ['tanh', 'linear'], loss: 'softmaxCE', lossParams: { classCount: 2 }, init: 'xavier', seed: 4 };

  it('has the right shapes and parameter count', () => {
    const net = mlp.createMlp(spec);
    assert.equal(net.layers.length, 2);
    assert.equal(net.layers[0].W.length, 6);
    assert.equal(net.layers[1].W.length, 6);
    assert.equal(mlp.parameterCount(net), 6 + 3 + 6 + 2);
  });

  it('forward matches a hand computation on a one-unit chain', () => {
    const tiny: MlpSpec = { sizes: [1, 1, 1], activations: ['sigmoid', 'linear'], loss: 'mse', init: 'zeros', seed: 1 };
    const net = mlp.createMlp(tiny);
    net.layers[0].W = [2];
    net.layers[0].b = [-1];
    net.layers[1].W = [3];
    net.layers[1].b = [0.5];
    const out = mlp.predict(net, [1], tiny)[0];
    const hidden = 1 / (1 + Math.exp(-(2 * 1 - 1)));
    close(out, 3 * hidden + 0.5, 1e-12);
  });

  it('the analytic gradient matches finite differences for every activation and loss', () => {
    for (const act of ACTIVATION_NAMES) {
      for (const name of LOSS_NAMES) {
        const lossFn = loss(name, { classCount: 3 });
        const outputs = lossFn.outputWidth(3);
        // Squared error goes through a sigmoid output, the fused losses through a linear one.
        const outputAct: ActivationName = name === 'mse' ? 'sigmoid' : 'linear';
        const s: MlpSpec = {
          sizes: [2, 3, outputs],
          activations: [act, outputAct],
          activationParams: { leakSlope: 0.2 },
          loss: name as LossName,
          lossParams: { classCount: 3, huberDelta: 0.5 },
          init: 'xavier',
          seed: 7,
        };
        const net = mlp.createMlp(s);
        const target = lossFn.kind === 'regression' ? lossFn.encodeValue(0.3) : lossFn.encodeClass(1, 3);
        const check = mlp.gradientCheck(net, [0.4, -0.7], target, s);
        assert.ok(check.maxRel < 1e-5, act + ' + ' + name + ': rel ' + check.maxRel);
      }
    }
  });

  it('backward leaves the forward caches untouched and a small step lowers the loss', () => {
    const net = mlp.createMlp(spec);
    const target = [0, 1];
    const caches = mlp.forward(net, [0.5, 0.2], spec);
    const before = JSON.stringify(caches);
    const grads = mlp.backward(net, caches, target, spec);
    assert.equal(JSON.stringify(caches), before);
    const gradients = { dW: grads.map((g) => g.dW), db: grads.map((g) => g.db) };
    const after = mlp.applyGradients(net, gradients, 0.05);
    assert.ok(mlp.sampleLoss(after, [0.5, 0.2], target, spec) < mlp.sampleLoss(net, [0.5, 0.2], target, spec));
    assert.notEqual(after.layers[0].W, net.layers[0].W);
  });

  it('batch gradients are the mean of the per-sample gradients and report the pre-activations', () => {
    const net = mlp.createMlp(spec);
    const inputs = [
      [0.1, 0.2],
      [-0.5, 0.9],
    ];
    const targets = [
      [1, 0],
      [0, 1],
    ];
    const batch = mlp.batchGradients(net, inputs, targets, spec);
    const one = mlp.backward(net, mlp.forward(net, inputs[0], spec), targets[0], spec);
    const two = mlp.backward(net, mlp.forward(net, inputs[1], spec), targets[1], spec);
    close(batch.grads.dW[0][0], (one[0].dW[0] + two[0].dW[0]) / 2, 1e-12);
    assert.equal(batch.zs[0].length, 6);
    assert.equal(batch.outputs.length, 2);
  });

  it('agrees with the neuron-array network on the same weights', () => {
    const config: nn.NetConfig = {
      hiddenLayers: [3],
      activation: 'tanh',
      learningRate: 0.1,
      batchSize: 1,
      inputFeatures: ['x1', 'x2'],
      initialiser: 'xavier',
      regularisation: 'none',
      regRate: 0,
      seed: 3,
      testFraction: 0,
    };
    const old = nn.createState(config);
    const s: MlpSpec = { sizes: [2, 3, 1], activations: ['tanh', 'linear'], loss: 'mse', init: 'zeros', seed: 1 };
    const net = mlp.createMlp(s);
    old.layers.forEach((layer, l) => {
      layer.forEach((neuron, o) => {
        neuron.weights.forEach((w, i) => {
          net.layers[l].W[o * net.layers[l].inSize + i] = w;
        });
        net.layers[l].b[o] = neuron.bias;
      });
    });
    const input = [0.3, -0.6];
    close(mlp.predict(net, input, s)[0], nn.evaluate(old.layers, input, 'tanh'), 1e-12);
  });

  it('an optimiser update moves every tensor', () => {
    const net = mlp.createMlp(spec);
    const opt = mlp.createOptStates(net, { name: 'adam' });
    assert.equal(opt.length, 4);
    const batch = mlp.batchGradients(net, [[0.1, 0.2]], [[1, 0]], spec);
    const next = mlp.updateWithOptimiser(net, batch.grads, opt, 0.01, { name: 'adam' });
    assert.ok(next.net.layers[0].W.some((w, i) => w !== net.layers[0].W[i]));
    assert.equal(next.opt[0].t, 1);
  });
});

/* ================================================================== */

describe('backprop phases', () => {
  const config: BackpropConfig = {
    hiddenUnits: 2,
    hiddenActivation: 'sigmoid',
    outputActivation: 'linear',
    loss: 'bce',
    learningRate: 0.5,
    mode: 'sample',
    init: 'fixed',
    seed: 1,
    inputs: [
      [0.05, 0.1],
      [-0.4, 0.3],
      [0.7, -0.2],
    ],
    targets: [[1], [0], [1]],
  };

  it('takes exactly seven ticks per sample and fills only its own fields', () => {
    let state = createState(config);
    assert.equal(state.phase, 'forward-hidden');
    assert.equal(state.cache[0].z, null);
    state = step(state, config);
    assert.ok(state.cache[0].z && state.cache[0].a && state.cache[0].delta === null);
    state = step(state, config);
    assert.ok(state.cache[1].a && state.loss === null);
    state = step(state, config);
    assert.ok(state.loss !== null && state.cache[1].delta === null);
    state = step(state, config);
    assert.ok(state.cache[1].delta && state.cache[0].delta === null);
    state = step(state, config);
    assert.ok(state.cache[0].delta && state.cache[0].dW === null);
    state = step(state, config);
    assert.ok(state.cache[0].dW && state.cache[1].dW && state.pending);
    assert.equal(state.sampleIndex, 0);
    state = step(state, config);
    assert.equal(state.phase, 'forward-hidden');
    assert.equal(state.sampleIndex, 1);
    assert.equal(state.updates, 1);
    assert.equal(state.tick, PHASES.length);
    assert.equal(state.lossHistory.length, 1);
    assert.equal(phaseIndex('update'), 7);
  });

  it('the pass gradients equal a whole-network backward and a numerical check', () => {
    let state = createState(config);
    for (let i = 0; i < 6; i++) state = step(state, config);
    const grads = mlp.backward(state.net, mlp.forward(state.net, state.input, state.spec), state.target, state.spec);
    state.cache[0].dW!.forEach((v, i) => close(v, grads[0].dW[i], 1e-12, 'hidden dW ' + i));
    state.cache[1].dW!.forEach((v, i) => close(v, grads[1].dW[i], 1e-12, 'output dW ' + i));
    assert.ok(state.checkError !== null && state.checkError < 1e-6, 'check ' + state.checkError);
  });

  it('is pure and the fixed weights match the textbook', () => {
    const state = createState(config);
    assert.deepEqual(state.net.layers[0].W, [0.15, 0.2, 0.25, 0.3]);
    assert.deepEqual(state.net.layers[1].W, [0.4, 0.45]);
    const before = JSON.stringify(state);
    step(state, config);
    assert.equal(JSON.stringify(state), before);
  });

  it('epochs advance and the mean loss falls in both update modes', () => {
    for (const mode of ['sample', 'batch'] as const) {
      let state = createState({ ...config, mode, init: 'xavier' });
      for (let e = 0; e < 40; e++) state = runEpoch(state, { ...config, mode, init: 'xavier' });
      assert.equal(state.epoch, 40, mode + ' epochs');
      assert.equal(state.epochLosses.length, 40);
      assert.ok(state.epochLosses[39] < state.epochLosses[0], mode + ' loss falls');
      assert.equal(state.updates, mode === 'sample' ? 120 : 40, mode + ' updates');
    }
  });
});

/* ================================================================== */

describe('mlp trainer', () => {
  const blobs = getClassificationDataset('gaussians').generate({ count: 120, noise: 0.3, seed: 4 });
  const split = splitData(blobs.points, 0.75, 4);
  const base: TrainerConfig = {
    task: 'classify',
    depth: 1,
    width: 6,
    activation: 'relu',
    loss: 'bce',
    classCount: 2,
    init: 'he',
    initScale: 1,
    seed: 2,
    optimiser: { name: 'adam' },
    learningRate: 0.02,
    batchSize: 16,
  };

  function train(config: TrainerConfig, epochs: number) {
    const lossFn = loss(config.loss, { classCount: config.classCount });
    const data = dataFromPoints(split.train, split.test, lossFn, config.classCount);
    let state = createTrainer(config, data);
    for (let i = 0; i < epochs && !state.diverged; i++) state = stepTrainer(state, data, config);
    return state;
  }

  it('is pure and separates two blobs', () => {
    const lossFn = loss('bce');
    const data = dataFromPoints(split.train, split.test, lossFn, 2);
    const state = createTrainer(base, data);
    const before = JSON.stringify(state);
    stepTrainer(state, data, base);
    assert.equal(JSON.stringify(state), before);
    const trained = train(base, 40);
    assert.ok((trained.trainAccuracy ?? 0) > 0.9, 'accuracy ' + trained.trainAccuracy);
    assert.equal(trained.lossHistory.length, 41);
    assert.ok(trained.layers[0].zSamples.length > 0 && trained.layers[0].zSamples.length <= 256);
  });

  it('a deep sigmoid stack has a far smaller first-layer gradient than its last layer', () => {
    const deep = train({ ...base, depth: 8, width: 6, activation: 'sigmoid', init: 'xavier', learningRate: 0.01 }, 3);
    const first = deep.layers[0].gradNorm;
    const last = deep.layers[deep.layers.length - 1].gradNorm;
    assert.ok(first < last * 0.05, first + ' vs ' + last);
  });

  it('reports dead units for an oversized ReLU initialisation', () => {
    const dead = train({ ...base, depth: 3, initScale: 4, learningRate: 0.5, optimiser: { name: 'gd' } }, 10);
    const share = dead.layers.slice(0, -1).reduce((s, l) => s + l.smallGradFraction, 0) / (dead.layers.length - 1);
    assert.ok(share > 0.3 || dead.diverged, 'dead share ' + share);
  });

  it('fits a curve in regression', () => {
    const curve = REGRESSION_DATASETS.find((d) => d.id === 'curved')!.generate({ count: 100, noise: 0.1, seed: 5 });
    const scale = valueScale(curve.points);
    const data = dataFromRegression(curve.points.slice(0, 75), curve.points.slice(75), curve.xRange, scale);
    const config: TrainerConfig = { ...base, task: 'regress', loss: 'mse', depth: 2, width: 8, activation: 'tanh', init: 'xavier', learningRate: 0.02 };
    let state = createTrainer(config, data);
    for (let i = 0; i < 120; i++) state = stepTrainer(state, data, config);
    assert.ok(state.trainLoss < 0.05, 'train loss ' + state.trainLoss);
    assert.equal(state.trainAccuracy, null);
  });

  it('an absurd rate is flagged as diverged', () => {
    const wild = train({ ...base, optimiser: { name: 'gd' }, learningRate: 1e6, initScale: 3 }, 20);
    assert.ok(wild.diverged);
  });

  it('signed scores and loss-axis values agree with the loss', () => {
    const state = train(base, 5);
    const s = signedScore(state, [0.2, -0.3]);
    assert.ok(s >= -1 && s <= 1);
    assert.equal(lossAxisValue('bce', [1.5], [1]), 1.5);
    assert.equal(lossAxisValue('mse', [1.5], [1]), 0.5);
    assert.equal(lossAxisValue('softmaxCE', [0.2, 1.7, -1], [0, 1, 0]), 1.7);
  });

  it('depth zero is logistic regression: one sigmoid unit on the inputs', () => {
    const state = train({ ...base, depth: 0 }, 30);
    assert.equal(state.net.layers.length, 1);
    assert.deepEqual(state.spec.sizes, [2, 1]);
    const [w1, w2] = state.net.layers[0].W;
    const b = state.net.layers[0].b[0];
    const x = [0.3, -0.4];
    const p = 1 / (1 + Math.exp(-(w1 * x[0] + w2 * x[1] + b)));
    close((signedScore(state, x) + 1) / 2, p, 1e-12, 'sigmoid of the linear score');
    assert.ok((state.trainAccuracy ?? 0) > 0.9, 'accuracy ' + state.trainAccuracy);
  });

  it('a linear stack of any depth is one affine map: it keeps every point on the line between two others', () => {
    const state = train({ ...base, depth: 3, width: 2, activation: 'linear', init: 'xavier' }, 5);
    const a = [-0.6, 0.2];
    const b = [0.7, -0.8];
    const mid = a.map((v, i) => 0.3 * v + 0.7 * b[i]);
    const [ha, hb, hm] = traceLayers(state.net, state.spec, [a, b, mid])[2].a;
    for (let i = 0; i < 2; i++) close(hm[i], 0.3 * ha[i] + 0.7 * hb[i], 1e-9, 'layer 3, unit ' + i);
  });

  it('the output layer alone reproduces the full network from the last hidden layer', () => {
    const state = train(base, 3);
    const x = [0.25, -0.5];
    const traces = traceLayers(state.net, state.spec, [x]);
    const hidden = traces[0].a[0];
    assert.deepEqual(outputFromHidden(state.net, state.spec, hidden), traces[1].a[0]);
    close(signedScoreFromHidden(state, hidden), signedScore(state, x), 1e-12);
  });
});

describe('warp helpers', () => {
  it('lays a grid of eleven lines each way, or one segment for a one-dimensional input', () => {
    const lines = gridLines(2);
    assert.equal(lines.length, 22);
    assert.equal(lines.filter((l) => l.axis).length, 2);
    assert.equal(lines[0].points.length, 41);
    assert.deepEqual(lines[0].points[0], [-1, -1]);
    assert.deepEqual(lines[0].points[40], [1, -1]);
    const segment = gridLines(1);
    assert.equal(segment.length, 1);
    assert.deepEqual(segment[0].points[0], [-1]);
  });

  it('finds the two directions of largest spread and keeps two dimensions as they are', () => {
    const cloud: number[][] = [];
    for (let i = 0; i < 200; i++) {
      const t = (i / 199) * 2 - 1;
      const s = Math.sin(i * 12.9898) * 0.05;
      // Spread along (1, 1, 0, 0), a little along (0, 0, 1, -1), nothing elsewhere.
      cloud.push([t + s, t - s, 0.2 * Math.cos(i), -0.2 * Math.cos(i)]);
    }
    const basis = principalBasis(cloud);
    const [first, second] = basis.axes;
    close(Math.abs(first[0] * first[1]), 0.5, 0.02, 'first axis along (1, 1, 0, 0)');
    close(Math.abs(second[2] * second[3]), 0.5, 0.05, 'second axis along (0, 0, 1, -1)');
    close(first.reduce((s, v, i) => s + v * second[i], 0), 0, 1e-6, 'orthogonal');
    const flat = principalBasis([
      [1, 2],
      [3, 4],
    ]);
    assert.deepEqual(projectOnto(flat, [3, 4]), [3, 4]);
  });

  it('pads an extent by a fraction of its larger side', () => {
    const e = extent2([
      [0, 0],
      [2, 1],
    ]);
    close(e.x[0], -0.16);
    close(e.x[1], 2.16);
    close(e.y[1], 1.16);
    assert.deepEqual(extent2([]), { x: [-1, 1], y: [-1, 1] });
  });
});

describe('unit tiles', () => {
  const blobs = getClassificationDataset('gaussians').generate({ count: 60, noise: 0.3, seed: 4 });
  const split = splitData(blobs.points, 0.75, 4);
  const config: TrainerConfig = { task: 'classify', depth: 2, width: 4, activation: 'relu', loss: 'bce', classCount: 2, init: 'he', initScale: 1, seed: 2, optimiser: { name: 'adam' }, learningRate: 0.02, batchSize: 16 };
  const data = dataFromPoints(split.train, split.test, loss('bce'), 2);
  const state = createTrainer(config, data);

  it('samples every hidden unit over the input plane, a = relu(z) at every vertex', () => {
    const layers = sampleUnits(state);
    assert.equal(layers.length, 2);
    assert.equal(layers[0].units.length, 4);
    const unit = layers[0].units[1];
    assert.equal(unit.z.length, TILE_N * TILE_N);
    for (let i = 0; i < unit.z.length; i++) close(unit.a[i], Math.max(0, unit.z[i]), 1e-6);
    assert.ok(layers[0].zMax > 0 && layers[0].aMin >= 0);
    // The corner vertex is the input (-1, -1) through the first layer.
    const [w1, w2] = [state.net.layers[0].W[2], state.net.layers[0].W[3]];
    close(unit.z[0], -w1 - w2 + state.net.layers[0].b[1], 1e-6);
    const one = sampleUnit(state, 2, 2);
    assert.ok(one && one.field.z.length === TILE_N * TILE_N && one.layer.units.length === 4);
    assert.equal(sampleUnit(state, 0, 3), null);
  });

  it('colours z against the layer, a against the range of f or the layer when f is unbounded', () => {
    const layer = { units: [], zMax: 4, zLo: -4, zHi: 2, aMin: 0, aMax: 2 };
    const before = tileNorm(activation('relu'), layer, false);
    close(before(2), 0.5);
    close(before(-8), -1);
    const relu = tileNorm(activation('relu'), layer, true);
    close(relu(1), 0.5);
    close(relu(0), 0);
    const sigmoid = tileNorm(activation('sigmoid'), layer, true);
    close(sigmoid(0.5), 0);
    close(sigmoid(1), 1);
    close(tileNorm(activation('tanh'), layer, true)(-0.5), -0.5);
  });

  it('sums a layer’s sheets into the next stage and boxes them on the activation’s range once folded', () => {
    const layers = sampleUnits(state);
    const last = layers[1];
    const out = state.net.layers[2];
    const sum = weightedSum(last, Array.from(out.W.slice(0, out.inSize)), out.b[0]);
    const i = 17 * TILE_N + 5;
    const u = (c: number) => -1 + (2 * c) / (TILE_N - 1);
    const z = mlp.forward(state.net, [u(5), u(17)], state.spec).slice(-1)[0].z[0];
    close(sum[i], z, 1e-5, 'the sum of the sheets is the output’s z');
    const relu = activation('relu');
    const [lo0, hi0] = rowRange(relu, last, 0);
    const [lo1, hi1] = rowRange(relu, last, 1);
    assert.ok(lo0 <= last.zLo && lo0 <= 0 && hi0 >= last.zHi, 'the replay starts with everything z reached');
    close(lo1, Math.min(last.aMin, 0));
    close(hi1, Math.max(last.aMax, 1e-6));
    assert.deepEqual(rowRange(activation('tanh'), last, 0.3), [-1, 1]);
    const segments = creaseSegments(layers[0].units[0].z, '#000', 1, []);
    for (const seg of segments) for (const p of [seg.from, seg.to]) assert.ok(p[0] >= -1 && p[0] <= 1 && p[1] >= -1 && p[1] <= 1, 'creases stay on the floor');
  });

  it('names the units fold of a layer as a target', () => {
    const fold = decodeTarget('units:2');
    assert.deepEqual(fold, { kind: 'units', layer: 2 });
    assert.equal(encodeTarget(fold!), 'units:2');
    assert.ok(sameTarget(fold, { kind: 'units', layer: 2 }));
    assert.ok(!sameTarget(fold, { kind: 'units', layer: 1 }));
    assert.equal(decodeTarget('units:0'), null);
    assert.deepEqual(decodeTarget('unit:2.1'), { kind: 'unit', index: 1, layer: 2 });
  });
});

describe('surface projector', () => {
  const rect = { x: 0, y: 0, w: 400, h: 300 };

  it('lifts a higher point up the screen and pushes a farther point back', () => {
    const project = makeProjector(rect, { azimuth: 0, elevation: 0.6 });
    const low = project(0, 0, 0);
    const high = project(0, 0, 1);
    assert.ok(high.y < low.y, 'z rises up the screen');
    close(high.x, low.x, 1e-9, 'straight up');
    const near = project(0, -1, 0);
    const far = project(0, 1, 0);
    assert.ok(far.depth > near.depth && far.y < near.y, 'far is behind and above');
  });

  it('keeps the box inside the rect at every turn', () => {
    for (let k = 0; k < 12; k++) {
      const project = makeProjector(rect, { azimuth: (k * Math.PI) / 6, elevation: 0.4 + (k % 3) * 0.3 });
      for (const [u, v, z] of [
        [-1, -1, 0],
        [1, -1, 1],
        [1, 1, 0],
        [-1, 1, 1],
      ]) {
        const p = project(u, v, z);
        assert.ok(p.x >= 0 && p.x <= rect.w && p.y >= 0 && p.y <= rect.h, 'corner inside at turn ' + k + ': ' + p.x + ',' + p.y);
      }
    }
  });
});
