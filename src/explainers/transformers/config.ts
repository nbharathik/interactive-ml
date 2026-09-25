/** Attention and transformers: params, controls and presets. */

import type { ControlGroup, Preset } from '../../explainer/types';
import { fmtKnob } from '../../lib/math/stats';

export interface TfParams extends Record<string, number | string | boolean> {
  task: string;
  size: number;
  digits: number;
  count: number;
  seed: number;
  /** Start from random weights, or from the shipped trained letter model. */
  start: string;

  positions: string;
  mask: string;
  heads: number;
  layers: number;
  width: number;
  ffn: boolean;
  norm: boolean;

  learningRate: number;
  optimiser: string;
  batchSize: number;

  probe: number;
  /** A sequence the reader typed; empty follows the held-out probe. */
  input: string;
}

export const DEFAULT_PARAMS: TfParams = {
  task: 'letters',
  size: 6,
  digits: 8,
  count: 600,
  seed: 7,
  start: 'random',

  positions: 'sinusoidal',
  mask: 'causal',
  heads: 2,
  layers: 2,
  width: 16,
  ffn: true,
  norm: true,

  learningRate: 0.01,
  optimiser: 'adam',
  batchSize: 16,

  probe: 0,
  input: '',
};

/** The mask a task is written for: the language tasks predict what comes next, the digit tasks read the whole string. */
export function maskFor(task: string): string {
  return task === 'animals' || task === 'letters' ? 'causal' : 'none';
}

/** The layers a task is set up with: letters need a second block to know which word a space ends. */
export function layersFor(task: string): number {
  return task === 'letters' ? 2 : 1;
}

/** The alphabet a task starts with: counting needs repeats, so it uses fewer digits. */
export function digitsFor(task: string): number {
  return task === 'count' ? 4 : 8;
}

const digitTask = (p: Record<string, unknown>) => p.task === 'reverse' || p.task === 'count';

export const CONTROL_GROUPS: ControlGroup<keyof TfParams & string>[] = [
  {
    title: 'Task',
    zone: 'top',
    controls: [
      {
        kind: 'select',
        key: 'task',
        label: 'Task',
        help: 'What the model learns. Changing it also sets the mask the task is written for.',
        options: [
          { value: 'letters', label: 'Next letter', hint: 'Predict the next letter of "the big dog often says woof", then write whole sentences one letter at a time, as a language model does.' },
          { value: 'reverse', label: 'Reverse', hint: 'Write the digits backwards. Every answer sits at a fixed other position.' },
          { value: 'count', label: 'Count', hint: 'Say how many times each digit appears. The answer depends on content, not position.' },
          { value: 'animals', label: 'Animal sounds', hint: 'Predict the next word of "the big dog often says woof". The sound depends on a word several places back.' },
        ],
      },
    ],
  },
  {
    title: 'Attention',
    zone: 'top',
    controls: [
      {
        kind: 'select',
        key: 'positions',
        label: 'Positions',
        help: 'What is added to each token so attention can tell first from last.',
        options: [
          { value: 'sinusoidal', label: 'Sinusoidal', hint: 'Fixed waves of different speeds: every position gets its own pattern.' },
          { value: 'learned', label: 'Learned', hint: 'One trained vector per position.' },
          { value: 'none', label: 'None', hint: 'Nothing: attention sees a bag of tokens and cannot tell their order.' },
        ],
      },
      {
        kind: 'select',
        key: 'mask',
        label: 'Mask',
        help: 'Which positions each position may look at.',
        options: [
          { value: 'none', label: 'None', hint: 'Every position sees every other: an encoder.' },
          { value: 'causal', label: 'Causal', hint: 'A position sees only itself and what came before: a decoder that writes left to right.' },
        ],
      },
      {
        kind: 'select',
        key: 'heads',
        label: 'Heads',
        help: 'Attention patterns side by side. The width d is split between them.',
        options: [
          { value: '1', label: '1' },
          { value: '2', label: '2' },
          { value: '4', label: '4' },
        ],
      },
    ],
  },
  {
    title: 'Model',
    zone: 'top',
    controls: [
      { kind: 'stepper', key: 'layers', label: 'Layers', min: 1, max: 2, step: 1, help: 'Blocks of attention and feed-forward, one on top of the other.' },
      {
        kind: 'select',
        key: 'width',
        advanced: true,
        label: 'Width d',
        help: 'Numbers per token vector.',
        options: [
          { value: '8', label: '8' },
          { value: '16', label: '16' },
          { value: '32', label: '32' },
        ],
      },
      { kind: 'toggle', key: 'ffn', advanced: true, label: 'Feed-forward', help: 'The per-token network after attention: 4d ReLU units, then back to d.' },
      { kind: 'toggle', key: 'norm', advanced: true, label: 'Layer norm', help: 'Rescale every token vector to mean 0 and spread 1 before each block reads it.' },
    ],
  },
  {
    title: 'Training',
    zone: 'top',
    controls: [
      { kind: 'stepper', key: 'learningRate', label: 'Learning rate (η)', min: 0.0001, max: 1, step: 0.0001, scale: 'log', format: fmtKnob, help: 'The step size of every update.' },
    ],
  },
  {
    title: 'Data',
    blurb: 'Sequences drawn in code; a quarter are held out.',
    controls: [
      { kind: 'slider', key: 'size', label: 'Length', min: 4, max: 8, step: 1, visibleWhen: digitTask, help: 'Digits per sequence.' },
      { kind: 'slider', key: 'digits', label: 'Digits', min: 2, max: 10, step: 1, visibleWhen: digitTask, help: 'How many different digits the sequences use.' },
      { kind: 'slider', key: 'count', label: 'Sequences', min: 100, max: 1000, step: 100, visibleWhen: digitTask, help: 'How many sequences are drawn; a quarter are held out. The language tasks always use all their sentences.' },
      { kind: 'seed', key: 'seed', label: 'Seed', help: 'Fixes the sequences, the split and the starting weights.' },
      {
        kind: 'segmented',
        key: 'start',
        label: 'Start from',
        visibleWhen: (p) => p.task === 'letters',
        help: 'Random weights, or a copy of this letter model already trained, so it writes from the first moment.',
        options: [
          { value: 'random', label: 'Random' },
          { value: 'trained', label: 'Trained' },
        ],
      },
    ],
  },
  {
    title: 'Updates',
    controls: [
      { kind: 'slider', key: 'batchSize', label: 'Batch', min: 4, max: 64, step: 4, help: 'Sequences per update.' },
      {
        kind: 'select',
        key: 'optimiser',
        label: 'Optimiser',
        help: 'How the gradient becomes a step.',
        options: [
          { value: 'adam', label: 'Adam' },
          { value: 'momentum', label: 'Momentum' },
          { value: 'gd', label: 'Plain SGD' },
        ],
      },
    ],
  },
  {
    title: 'Probe',
    blurb: 'The held-out sequence the diagram follows. Type your own above the diagram instead.',
    controls: [
      { kind: 'slider', key: 'probe', label: 'Held-out sequence', min: 0, max: 199, step: 1, format: (v) => String(v + 1), help: 'Which held-out sequence the diagram follows, counted from 1; past the last one it stays on the last.' },
    ],
  },
];

/** The digit tasks read the whole string with one block; the word task has one block and the mask. */
const DIGITS = { task: 'reverse', mask: 'none', layers: 1 } as const;
const WORDS = { task: 'animals', mask: 'causal', layers: 1 } as const;

export const PRESETS: Preset<TfParams>[] = [
  {
    id: 'letters',
    name: 'Next letter',
    blurb: 'A tiny language, letter by letter: "the big dog often says woof". The model guesses every next letter from the ones before it, then writes whole sentences one letter at a time, the way a large language model writes words.',
    params: {},
    autoRun: true,
  },
  {
    id: 'letters-trained',
    name: 'Next letter, trained',
    blurb: 'The same letter model, already trained: it writes from the first moment, so every part can be read on a model that works.',
    params: { start: 'trained' },
  },
  {
    id: 'reverse',
    name: 'Reverse the digits',
    blurb: 'Six digits in, the same six backwards out. Watch the attention settle into a mirror: each position looks at the one opposite it and copies what it finds.',
    params: { ...DIGITS },
    autoRun: true,
  },
  {
    id: 'reverse-no-positions',
    name: 'Reverse, no positions',
    blurb: 'The same task with nothing added to say where a token sits. Attention sees a bag of digits, so it cannot find the one opposite: accuracy stalls far below the mirror.',
    params: { ...DIGITS, positions: 'none' },
    autoRun: true,
  },
  {
    id: 'reverse-causal',
    name: 'Reverse, causal mask',
    blurb: 'The same task, but each position may only look back. The last three can see their mirror; the first three would need to look ahead, so about half the answers stay guesses.',
    params: { ...DIGITS, mask: 'causal' },
    autoRun: true,
  },
  {
    id: 'count',
    name: 'Count each digit',
    blurb: 'Six digits from 0 to 3, and at each one: how many times does it appear? Order does not matter here, only which digits match, and the feed-forward block turns what attention gathers into a number.',
    params: { ...DIGITS, task: 'count', digits: 4, learningRate: 0.003 },
    autoRun: true,
  },
  {
    id: 'count-no-ffn',
    name: 'Count, no feed-forward',
    blurb: 'Counting with attention alone. Attention can gather which digits match, but turning that summary into a count takes the feed-forward block: without it accuracy stalls well short.',
    params: { ...DIGITS, task: 'count', digits: 4, ffn: false, learningRate: 0.003 },
    autoRun: true,
  },
  {
    id: 'animals',
    name: 'Animal sounds',
    blurb: 'A tiny language: "the big dog often says woof". The model predicts every next word, and the word after says depends on the animal a few words back. A causal mask keeps it honest.',
    params: { ...WORDS },
    autoRun: true,
  },
  {
    id: 'animals-no-mask',
    name: 'Animal sounds, no mask',
    blurb: 'The language task with the mask off: every word can see the next one, so predicting it is copying. The training loss falls to zero, and the model still cannot write a sound it has not seen.',
    params: { ...WORDS, mask: 'none' },
    autoRun: true,
  },
  {
    id: 'reverse-hot',
    name: 'Learning rate too high',
    blurb: 'Reverse with η = 0.1. The first updates are so large that the attention collapses and the model settles on guessing; it never finds the mirror.',
    params: { ...DIGITS, learningRate: 0.1 },
    autoRun: true,
  },
];
