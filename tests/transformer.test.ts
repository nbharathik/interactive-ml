/** The transformer and its sequence tasks: the data, the backward pass against finite differences, purity, and what each task learns. */

import { strict as assert } from 'node:assert';
import { describe, it } from 'node:test';

import { generateSequences, labelTokens, parseTokens, spellTokens } from '../src/lib/datasets/sequences.ts';
import type { SequenceTask } from '../src/lib/datasets/sequences.ts';
import { backward, createState, createTransformer, forward, generate, parameterCount, sequenceLoss, step, tensorsOf, zerosLike } from '../src/lib/ml/transformer.ts';
import type { PositionMode, TransformerSpec } from '../src/lib/ml/transformer.ts';

function setup(task: SequenceTask, patch: Partial<TransformerSpec> = {}, digits = task === 'count' ? 4 : 8) {
  const data = generateSequences({ task, size: 6, digits, count: 600, seed: 7, trainFraction: 0.75 });
  const spec: TransformerSpec = {
    vocab: data.vocab.length,
    outVocab: data.outVocab.length,
    length: data.length,
    width: 16,
    heads: 2,
    layers: 1,
    ffn: 64,
    positions: 'sinusoidal',
    causal: data.nextToken,
    norm: true,
    seed: 7,
    optimiser: { name: 'adam' },
    learningRate: 0.01,
    batchSize: 16,
    evalEvery: 10,
    ...patch,
  };
  return { data, spec };
}

describe('sequence tasks', () => {
  it('reverse answers each digit with the one opposite it', () => {
    const { data } = setup('reverse');
    assert.equal(data.inputs.length, 600);
    for (let i = 0; i < 20; i++) {
      const digits = data.inputs[i].slice(1).map((t) => t - 1);
      assert.equal(data.inputs[i][0], 0, 'every sequence starts with the start token');
      assert.deepEqual(data.targets[i], [-1, ...digits.slice().reverse()]);
      assert.deepEqual(data.scored[i], [1, 2, 3, 4, 5, 6]);
    }
    assert.equal(new Set(data.inputs.map((s) => s.join(','))).size, 600, 'no sequence twice');
    assert.equal(data.trainIndex.length + data.testIndex.length, 600);
  });

  it('count answers how often each digit appears, as the class count - 1', () => {
    const { data } = setup('count');
    for (let i = 0; i < 20; i++) {
      const digits = data.inputs[i].slice(1);
      digits.forEach((d, k) => assert.equal(data.targets[i][k + 1] + 1, digits.filter((e) => e === d).length));
    }
    assert.deepEqual(data.outVocab, ['1', '2', '3', '4', '5', '6']);
  });

  it('the animal sentences predict the next word and score the sound after says', () => {
    const { data } = setup('animals');
    assert.equal(data.inputs.length, 72);
    assert.ok(data.nextToken);
    for (let i = 0; i < data.inputs.length; i++) {
      const words = spellTokens(data, data.inputs[i]).split(' ');
      const says = data.scored[i][0];
      assert.equal(data.vocab[data.inputs[i][says]], 'says');
      assert.equal(data.targets[i][says], data.inputs[i][says + 1], 'the sound follows says');
      assert.equal(words[words.length - 1], '.');
    }
  });

  it('typed text parses by the task rules, and wrong text says why', () => {
    const { data } = setup('reverse');
    const parsed = parseTokens(data, '1 2 3 4 5 6');
    assert.ok('tokens' in parsed);
    assert.deepEqual(labelTokens('reverse', data.vocab, parsed.tokens).targets, [-1, 6, 5, 4, 3, 2, 1]);
    assert.ok('error' in parseTokens(data, '12345'));
    assert.ok('error' in parseTokens(data, '12345x'));
    const words = setup('animals').data;
    const dog = parseTokens(words, 'the old dog says');
    assert.ok('tokens' in dog);
    const label = labelTokens('animals', words.vocab, dog.tokens);
    assert.deepEqual(label.scored, [4]);
    assert.equal(words.vocab[label.targets[4]], 'woof');
    assert.ok('error' in parseTokens(words, 'the old unicorn says'));
  });
});

describe('the backward pass', () => {
  const cases: Array<[string, SequenceTask, Partial<TransformerSpec>]> = [
    ['layer norm, learned positions, two layers', 'reverse', { width: 8, heads: 2, layers: 2, ffn: 8, positions: 'learned' as PositionMode }],
    ['no norm, causal, no feed-forward', 'animals', { width: 8, heads: 4, ffn: 0, norm: false }],
    ['one head, no positions', 'count', { width: 8, heads: 1, positions: 'none' as PositionMode, ffn: 12 }],
  ];
  for (const [name, task, patch] of cases) {
    it('matches finite differences: ' + name, () => {
      const { data, spec } = setup(task, patch);
      const weights = createTransformer(spec);
      const tokens = data.inputs[3];
      const targets = data.targets[3];
      const lossOf = () => sequenceLoss(forward(weights, tokens, spec), targets, spec.outVocab).loss;
      const grads = zerosLike(weights);
      backward(weights, forward(weights, tokens, spec), targets, spec, grads, 1);
      const ws = tensorsOf(weights);
      const gs = tensorsOf(grads);
      let checked = 0;
      ws.forEach((w, k) => {
        for (let i = 0; i < w.length; i += Math.max(1, Math.floor(w.length / 5))) {
          const old = w[i];
          w[i] = old + 1e-5;
          const up = lossOf();
          w[i] = old - 1e-5;
          const down = lossOf();
          w[i] = old;
          const numeric = (up - down) / 2e-5;
          const scale = Math.max(1e-4, Math.abs(numeric) + Math.abs(gs[k][i]));
          assert.ok(Math.abs(numeric - gs[k][i]) / scale < 1e-4, 'tensor ' + k + ' index ' + i + ': ' + numeric + ' vs ' + gs[k][i]);
          checked += 1;
        }
      });
      assert.ok(checked > 40);
    });
  }
});

describe('the model', () => {
  it('attention rows are probabilities, and a causal mask zeroes the future', () => {
    const { data, spec } = setup('animals');
    const cache = forward(createTransformer(spec), data.inputs[0], spec);
    const T = cache.length;
    for (let h = 0; h < spec.heads; h++) {
      for (let i = 0; i < T; i++) {
        let sum = 0;
        for (let j = 0; j < T; j++) {
          const a = cache.layers[0].attn[(h * T + i) * T + j];
          if (j > i) assert.equal(a, 0);
          sum += a;
        }
        assert.ok(Math.abs(sum - 1) < 1e-9);
      }
    }
  });

  it('a step does not touch the state it is given', () => {
    const { data, spec } = setup('reverse');
    const s0 = createState(spec, data);
    const before = tensorsOf(s0.weights).map((t) => Array.from(t));
    const s1 = step(s0, data, spec);
    tensorsOf(s0.weights).forEach((t, k) => assert.deepEqual(Array.from(t), before[k]));
    assert.equal(s1.step, 1);
    assert.notDeepEqual(Array.from(s1.weights.E), before[0]);
  });

  it('counts its parameters', () => {
    const { spec } = setup('reverse');
    // E 9x16, U 16x8 + 8, a block: 2 norms, 4 attention matrices + bias, 16x64 + 64 + 64x16 + 16.
    assert.equal(parameterCount(createTransformer(spec)), 9 * 16 + 16 * 8 + 8 + 4 * 16 + 4 * 256 + 16 + 16 * 64 + 64 + 64 * 16 + 16 + 2 * 16);
  });

  it('learns to reverse, and the mirror is what it attends to', () => {
    const { data, spec } = setup('reverse');
    let s = createState(spec, data);
    for (let i = 0; i < 200 && s.testAccuracy < 0.99; i++) s = step(s, data, spec);
    assert.ok(s.testAccuracy >= 0.99, 'held out ' + s.testAccuracy);
    const cache = forward(s.weights, data.inputs[data.testIndex[0]], spec);
    const T = cache.length;
    let best = 0;
    for (let h = 0; h < spec.heads; h++) {
      let share = 0;
      for (let i = 1; i < T; i++) share += cache.layers[0].attn[(h * T + i) * T + (T - i)];
      best = Math.max(best, share / (T - 1));
    }
    assert.ok(best > 0.6, 'mirror share ' + best);
  });

  it('writes the right sound after training on the animal sentences', () => {
    const { data, spec } = setup('animals');
    let s = createState(spec, data);
    for (let i = 0; i < 150; i++) s = step(s, data, spec);
    const id = (w: string) => data.vocab.indexOf(w);
    const written = generate(s.weights, [0, id('the'), id('old'), id('owl')], spec, id('.'));
    const text = written.map((t) => data.vocab[t]).join(' ');
    assert.ok(text.includes('says hoot'), text);
  });
});
