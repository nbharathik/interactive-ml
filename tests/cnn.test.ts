/** The tiny convolutional network and the glyph images it trains on. */

import { strict as assert } from 'node:assert';
import { describe, it } from 'node:test';

import { DEFAULT_GLYPHS, GLYPHS, generateGlyphs, paintGlyph } from '../src/lib/datasets/images.ts';
import * as cnn from '../src/lib/ml/conv.ts';
import type { CnnSpec } from '../src/lib/ml/conv.ts';

function close(actual: number, expected: number, tolerance = 1e-6, message?: string) {
  assert.ok(
    Math.abs(actual - expected) <= tolerance,
    (message ?? 'value') + ': expected ' + expected + ', got ' + actual,
  );
}

const SPEC: CnnSpec = {
  size: 8,
  model: 'cnn',
  kernel: 3,
  filters: 3,
  padding: 'same',
  pool: 'max',
  secondConv: true,
  secondFilters: 2,
  dense: 4,
  classCount: 3,
  seed: 3,
  optimiser: { name: 'adam' },
  learningRate: 0.01,
  batchSize: 8,
  evalEvery: 5,
};

describe('glyph images', () => {
  it('is deterministic, partitions its rows and paints distinct classes', () => {
    const a = generateGlyphs({ ...DEFAULT_GLYPHS, count: 60, seed: 2 });
    const b = generateGlyphs({ ...DEFAULT_GLYPHS, count: 60, seed: 2 });
    assert.deepEqual(Array.from(a.images[3]), Array.from(b.images[3]));
    assert.equal(a.images.length, 60);
    assert.equal(a.images[0].length, 16 * 16);
    assert.equal(a.trainIndex.length + a.testIndex.length, 60);
    assert.equal(new Set([...a.trainIndex, ...a.testIndex]).size, 60);
    assert.equal(a.classNames.length, 4);
    assert.equal(a.classSymbols.length, 4);
    // Every class differs from every other at its centre and pixels stay in [0, 1].
    const centre = GLYPHS.map((_, label) => Array.from(paintGlyph(16, label, 8, 8, 2, 3.5)));
    for (let i = 0; i < centre.length; i++) {
      for (let j = i + 1; j < centre.length; j++) {
        const diff = centre[i].reduce((s, v, p) => s + Math.abs(v - centre[j][p]), 0);
        assert.ok(diff > 10, GLYPHS[i].name + ' vs ' + GLYPHS[j].name);
      }
    }
    for (const image of a.images) for (const v of image) assert.ok(v >= 0 && v <= 1);
  });

  it('a horizontal bar lights a row and a vertical bar a column', () => {
    const bar = paintGlyph(16, 0, 8, 8, 2, 3.5);
    const row = Array.from({ length: 16 }, (_, x) => bar[8 * 16 + x]).filter((v) => v > 0.5).length;
    const col = Array.from({ length: 16 }, (_, y) => bar[y * 16 + 8]).filter((v) => v > 0.5).length;
    assert.ok(row > 8 && col <= 3, row + ' lit in the row, ' + col + ' in the column');
  });
});

describe('cnn', () => {
  it('reports the stage shapes and the parameter count', () => {
    const shapes = cnn.stageShapes(SPEC);
    assert.deepEqual(shapes.stages[0].conv, [3, 8, 8]);
    assert.deepEqual(shapes.stages[0].pooled, [3, 4, 4]);
    assert.deepEqual(shapes.stages[1].conv, [2, 4, 4]);
    assert.deepEqual(shapes.stages[1].pooled, [2, 2, 2]);
    assert.equal(shapes.flat, 8);
    const valid = cnn.stageShapes({ ...SPEC, padding: 'valid', kernel: 5, secondConv: false, size: 16 });
    assert.deepEqual(valid.stages[0].conv, [3, 12, 12]);
    assert.deepEqual(valid.stages[0].pooled, [3, 6, 6]);
    const simple = cnn.createCnn({ ...SPEC, secondConv: false, dense: 0, filters: 4, classCount: 2 });
    // 4 kernels of 9 plus 4 biases, then a dense head from 4 x 4 x 4 to 2.
    assert.equal(cnn.parameterCount(simple), 40 + 64 * 2 + 2);
    assert.equal(cnn.stageShapes({ ...SPEC, model: 'dense' }).flat, 64);
  });

  it('a centre-only kernel with same padding reproduces the input', () => {
    const layer: cnn.ConvLayer = { kernel: 3, inC: 1, outC: 1, pad: 1, W: new Float64Array(9), b: new Float64Array(1) };
    layer.W[4] = 1;
    const x = cnn.tensor(1, 4, 4, Float64Array.from({ length: 16 }, (_, i) => i / 16));
    const out = cnn.convForward(x, layer);
    assert.deepEqual(Array.from(out.data), Array.from(x.data));
  });

  it('max pooling routes the gradient to the winning cell and average pooling spreads it', () => {
    const a = cnn.tensor(1, 2, 2, Float64Array.from([1, 5, 2, 3]));
    const max = cnn.poolForward(a, 'max');
    assert.equal(max.out.data[0], 5);
    const back = cnn.poolBackward(cnn.tensor(1, 1, 1, Float64Array.from([2])), a, 'max', max.argmax);
    assert.deepEqual(Array.from(back.data), [0, 2, 0, 0]);
    const avg = cnn.poolBackward(cnn.tensor(1, 1, 1, Float64Array.from([2])), a, 'avg', null);
    assert.deepEqual(Array.from(avg.data), [0.5, 0.5, 0.5, 0.5]);
  });

  it('every analytic gradient matches finite differences', () => {
    const data = generateGlyphs({ ...DEFAULT_GLYPHS, size: 8, classCount: 3, count: 12, seed: 4 });
    for (const pool of ['max', 'avg', 'none'] as const) {
      const spec = { ...SPEC, pool };
      const params = cnn.createCnn(spec);
      const image = data.images[1];
      const label = data.labels[1];
      const cache = cnn.forwardProbe(params, image, spec);
      const grads = cnn.backwardProbe(params, cache, label, spec);
      const lossAt = (p: cnn.CnnParams) => cnn.sampleLoss(cnn.forwardProbe(p, image, spec), label);
      const check = (get: (p: cnn.CnnParams) => Float64Array, analytic: Float64Array, name: string) => {
        const h = 1e-5;
        for (let i = 0; i < analytic.length; i += Math.max(1, Math.floor(analytic.length / 12))) {
          const clone = (): cnn.CnnParams => {
            const copy = {
              convs: params.convs.map((c) => ({ ...c, W: Float64Array.from(c.W), b: Float64Array.from(c.b) })),
              hidden: params.hidden ? { ...params.hidden, W: Float64Array.from(params.hidden.W), b: Float64Array.from(params.hidden.b) } : null,
              out: { ...params.out, W: Float64Array.from(params.out.W), b: Float64Array.from(params.out.b) },
            };
            return copy;
          };
          const up = clone();
          get(up)[i] += h;
          const down = clone();
          get(down)[i] -= h;
          close(analytic[i], (lossAt(up) - lossAt(down)) / (2 * h), 1e-5, pool + ' ' + name + '[' + i + ']');
        }
      };
      check((p) => p.convs[0].W, grads.convs[0].dW, 'conv1.W');
      check((p) => p.convs[0].b, grads.convs[0].db, 'conv1.b');
      check((p) => p.convs[1].W, grads.convs[1].dW, 'conv2.W');
      check((p) => p.hidden!.W, grads.hidden!.dW, 'hidden.W');
      check((p) => p.out.W, grads.out.dW, 'out.W');
      check((p) => p.out.b, grads.out.db, 'out.b');
    }
  });

  it('step is pure, learns two glyph classes within a second, and evaluates on schedule', () => {
    const data = generateGlyphs({ ...DEFAULT_GLYPHS, size: 8, classCount: 2, count: 40, jitter: 1, noise: 0.05, seed: 6 });
    const spec: CnnSpec = { ...SPEC, classCount: 2, secondConv: false, dense: 0, filters: 4, batchSize: 10, learningRate: 0.02, evalEvery: 4 };
    let state = cnn.createState(spec, data);
    const before = JSON.stringify(state.params.out.W);
    const next = cnn.step(state, data, spec);
    assert.equal(JSON.stringify(state.params.out.W), before);
    assert.notEqual(next.params.out.W, state.params.out.W);
    const started = Date.now();
    for (let i = 0; i < 60; i++) state = cnn.step(state, data, spec);
    assert.ok(Date.now() - started < 3000, 'took ' + (Date.now() - started) + ' ms');
    assert.equal(state.step, 60);
    assert.ok(state.epoch >= 15, 'epochs ' + state.epoch);
    assert.ok(state.trainAccuracy > 0.8, 'accuracy ' + state.trainAccuracy);
    assert.ok(state.history.length > 10);
    assert.equal(state.testPredictions.length, data.testIndex.length);
    const dense: CnnSpec = { ...spec, model: 'dense', dense: 8 };
    let plain = cnn.createState(dense, data);
    assert.equal(plain.params.convs.length, 0);
    for (let i = 0; i < 40; i++) plain = cnn.step(plain, data, dense);
    assert.ok(plain.trainAccuracy > 0.7, 'dense accuracy ' + plain.trainAccuracy);
  });

  it('the probe cache holds every stage and a filter picture', () => {
    const data = generateGlyphs({ ...DEFAULT_GLYPHS, size: 8, classCount: 3, count: 12, seed: 4 });
    const params = cnn.createCnn(SPEC);
    const cache = cnn.forwardProbe(params, data.images[0], SPEC);
    assert.equal(cache.stages.length, 2);
    assert.equal(cache.stages[0].a.data.length, 3 * 8 * 8);
    assert.equal(cache.flat.length, 8);
    close(Array.from(cache.probs).reduce((a, b) => a + b, 0), 1, 1e-9);
    const picture = cnn.filterTensor(params.convs[1], 0);
    assert.equal(picture.data.length, 9);
    assert.equal(cnn.channel(cache.stages[0].a, 2).length, 64);
  });
});
