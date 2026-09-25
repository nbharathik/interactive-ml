/** Attention and transformer lessons, one screen each. Pure data, driven against the real model in tests/lessons-transformer.test.ts. */

import type { Lesson, LessonAction, LessonContextBase, LiveNumber } from '../../explainer/lessons';
import { letterAnimal, spellTokens } from '../../lib/datasets/sequences';
import type { SequenceDataset } from '../../lib/datasets/sequences';
import { fmt, fmtPercent } from '../../lib/math/stats';
import { parameterCount } from '../../lib/ml/transformer';
import type { TransformerCache, TransformerSpec, TransformerState } from '../../lib/ml/transformer';
import type { HeadPick } from './attention';
import type { TfParams } from './config';
import { aheadShare, animalShare, mirrorShare, sharpness, tokenName, weightAt } from './model';

/* ---------------- context ---------------- */

export type ArchView = 'model' | 'flow' | 'attention' | 'head';
export type SecondView = 'loss' | 'accuracy' | 'embeddings' | 'positions';

export interface TfLessonContext extends LessonContextBase {
  params: TfParams;
  state: TransformerState;
  derived: {
    paramCount: number;
    length: number;
    fitted: boolean;
    /** Reverse: the mean weight on the mirror in the head that does it most. */
    mirror: number;
    mirrorHead: number;
    /** Language: the weight from says to the animal, in the head that does it most, and in the others. */
    animal: number;
    animalHead: number;
    animalOther: number;
    /** The mean largest weight per row: 1 / T for a blur, 1 for a pointer. */
    sharp: number;
    /** The share of attention that lands on later positions, which a causal mask forbids. */
    ahead: number;
    uniform: number;
    /** The head view's row: its query and the key it weights most. */
    pick: { query: string; position: number; head: number; key: string; keyPosition: number; weight: number };
    /** The sentence written from the probe's prompt. */
    writes: string;
    prompt: string;
    /** Letters: the probe's text, and the three likeliest next letters with their shares. */
    text: string;
    next: Array<{ letter: string; p: number }>;
    /** Letters: from the last row, the head that puts the most weight on the animal's letters. */
    focus: { layer: number; head: number; share: number; animal: string };
    /** Letters: the feed-forward units a row keeps, on average, in each block. */
    kept: number[];
  };
  ui: { view: ArchView; secondView: SecondView; open: string | null; pick: HeadPick };
}

export interface TfContextInput {
  params: TfParams;
  state: TransformerState;
  data: SequenceDataset;
  spec: TransformerSpec;
  cache: TransformerCache;
  written: { prompt: number[]; tokens: number[] } | null;
  ui: TfLessonContext['ui'];
}

/**
 * Every held-out answer right; a next-word model trains on a while so its guesses for the other words settle.
 * The letter model is done when it writes every held-out sentence and gets nine in ten of the sound's letters.
 */
export function isFitted(state: TransformerState, task = 'reverse'): boolean {
  if (task === 'letters') return state.step >= 150 && state.testExact >= 0.99 && state.testAccuracy >= 0.9;
  return state.step >= (task === 'animals' ? 150 : 20) && state.testAccuracy >= 0.99 && state.trainAccuracy >= 0.99;
}

/** The head a pick reads: its own, or the one that weights its top key most. */
export function resolveHead(cache: TransformerCache, spec: TransformerSpec, pick: HeadPick): number {
  if (pick.head >= 0 && pick.head < spec.heads) return pick.head;
  let best = 0;
  let top = -1;
  for (let h = 0; h < spec.heads; h++) {
    for (let j = 0; j < cache.length; j++) {
      const w = weightAt(cache, pick.layer, h, pick.query, j);
      if (w > top) {
        top = w;
        best = h;
      }
    }
  }
  return best;
}

/** The page's share of the lesson context; the hook adds `sim` and `lesson`. */
export function makeTfContext({ params, state, data, spec, cache, written, ui }: TfContextInput): Omit<TfLessonContext, keyof LessonContextBase> {
  const mirror = data.task === 'reverse' ? mirrorShare(cache, spec) : { head: 0, share: 0, shares: [] };
  const animal = data.task === 'animals' ? animalShare(cache, spec, data) : null;
  const query = Math.min(ui.pick.query, cache.length - 1);
  const head = resolveHead(cache, spec, { ...ui.pick, query });
  let keyPosition = 0;
  let weight = -1;
  for (let j = 0; j < cache.length; j++) {
    const w = weightAt(cache, Math.min(ui.pick.layer, spec.layers - 1), head, query, j);
    if (w > weight) {
      weight = w;
      keyPosition = j;
    }
  }
  const others = animal ? animal.shares.filter((_, h) => h !== animal.head) : [];
  const letters = data.task === 'letters';
  const text = letters ? spellTokens(data, cache.tokens) : '';
  return {
    params,
    state,
    derived: {
      text,
      next: letters ? nextLetters(data, spec, cache) : [],
      focus: letters ? animalFocus(text, cache, spec) : { layer: 0, head: 0, share: 0, animal: '' },
      kept: letters ? keptUnits(cache, spec) : [],
      paramCount: parameterCount(state.weights),
      length: cache.length,
      fitted: isFitted(state, data.task),
      mirror: mirror.share,
      mirrorHead: mirror.head,
      animal: animal?.share ?? 0,
      animalHead: animal?.head ?? 0,
      animalOther: others.length ? Math.max(...others) : 0,
      sharp: sharpness(cache, spec),
      ahead: aheadShare(cache, spec),
      uniform: 1 / cache.length,
      pick: {
        query: tokenName(data, cache.tokens[query]),
        position: query,
        head,
        key: tokenName(data, cache.tokens[keyPosition]),
        keyPosition,
        weight: Math.max(0, weight),
      },
      // Letters read as the text they spell; words as words after the prompt.
      writes: written
        ? letters
          ? spellTokens(data, written.tokens)
          : written.tokens
              .slice(written.prompt.length)
              .map((t) => tokenName(data, t))
              .filter((word) => word !== '.')
              .join(' ')
        : '',
      prompt: written ? (letters ? spellTokens(data, written.prompt) : written.prompt.slice(1).map((t) => tokenName(data, t)).join(' ')) : '',
    },
    ui,
  };
}

/** The three likeliest letters after the last one, the space shown as a space. */
function nextLetters(data: SequenceDataset, spec: TransformerSpec, cache: TransformerCache): Array<{ letter: string; p: number }> {
  const V = spec.outVocab;
  const t = cache.length - 1;
  return Array.from({ length: V }, (_, c) => c)
    .filter((c) => c !== 0)
    .sort((a, b) => cache.probs[t * V + b] - cache.probs[t * V + a])
    .slice(0, 3)
    .map((c) => ({ letter: tokenName(data, c), p: cache.probs[t * V + c] }));
}

/** From the last row, each head's weight on the animal's letters; the largest wins. */
function animalFocus(text: string, cache: TransformerCache, spec: TransformerSpec): { layer: number; head: number; share: number; animal: string } {
  const animal = letterAnimal(text);
  const best = { layer: 0, head: 0, share: 0, animal: animal?.name ?? '' };
  if (!animal) return best;
  const at = (' ' + text + ' ').indexOf(' ' + animal.name + ' ') + 1;
  const T = cache.length;
  for (let l = 0; l < spec.layers; l++) {
    for (let h = 0; h < spec.heads; h++) {
      let share = 0;
      for (let k = at; k < at + animal.name.length; k++) share += weightAt(cache, l, h, T - 1, k);
      if (share > best.share) Object.assign(best, { layer: l, head: h, share });
    }
  }
  return best;
}

/** The mean number of feed-forward units a row keeps after ReLU, per block. */
function keptUnits(cache: TransformerCache, spec: TransformerSpec): number[] {
  const T = cache.length;
  return cache.layers.map((layer) => {
    if (!layer.act) return 0;
    let kept = 0;
    for (let i = 0; i < T * spec.ffn; i++) if (layer.act[i] > 0) kept += 1;
    return kept / T;
  });
}

/* ---------------- copy helpers ---------------- */

type Ctx = TfLessonContext;
type L = Lesson<Ctx, TfParams>;
type Act = LessonAction<TfParams>;

const pc = (v: number) => fmtPercent(v, 0);
const n = (v: number) => v.toLocaleString('en-US');
const at = (c: Ctx) => 'Step ' + n(c.state.step);
const w2 = (v: number) => fmt(v, 2);

/** Steps the reference runs take, checked by the tests. */
export const REFERENCE = {
  /** Reverse at the defaults: every held-out digit right at this step. */
  reverseFit: 80,
  /** Reverse with no positions after 400 steps. */
  noPositions: 0.4,
  /** Count at the defaults after 400 steps, and without the feed-forward block after 800. */
  count400: 0.98,
  countNoFfn: 0.53,
  /** The language task with the mask off after 300 steps: held-out sentences it writes right. */
  noMask: 0.17,
  /** Reverse at η = 0.1 after 400 steps. */
  hot: 0.19,
};

/**
 * Two places under the fixed position vectors: how far the two fastest hands turn between them, in degrees,
 * and the cosine similarity of their rows, which is the mean cosine of every hand's turn.
 */
export function placeGap(width: number, a: number, b: number): { fast: number; next: number; similarity: number } {
  const pairs = Math.floor(width / 2);
  const turn = (k: number) => (b - a) / Math.pow(10000, (2 * k) / width);
  let sum = 0;
  for (let k = 0; k < pairs; k++) sum += Math.cos(turn(k));
  const deg = (k: number) => Math.round(((turn(k) * 180) / Math.PI) % 360);
  return { fast: deg(0), next: deg(1), similarity: sum / pairs };
}

/** The architecture with one of its parts open; 'none' shows the whole of it. */
const MODEL = (block: string): Act[] => [
  { type: 'view', id: 'arch', value: 'model' },
  { type: 'view', id: 'block', value: block },
];
const GRID: Act = { type: 'view', id: 'arch', value: 'attention' };
/** Rows lit on an open part, as '0,6'; empty clears them. */
const PLACES = (list: string): Act => ({ type: 'view', id: 'place', value: list });
const PLAY: Act = { type: 'play' };
const NOW = () => true;
/** The head the diagram shows, blocks and heads counted from 0, read from the last row. */
const pickHead = (layer: number, head: number): Act => ({ type: 'view', id: 'pick', value: layer + ':' + head + ':99' });
/** The model writes on from a text, one letter a beat. */
const WRITE = (text: string): Act => ({ type: 'view', id: 'write', value: text });
/** A highlight walks down the rows once. */
const SWEEP: Act = { type: 'view', id: 'sweep', value: 'rows' };

/** A run restates every knob it depends on: the preset, then its own changes, then a fresh start. */
const on = (preset: string, patch: Partial<TfParams> | null, ...rest: Act[]): Act[] => [
  { type: 'preset', id: preset },
  ...(patch ? [{ type: 'params' as const, patch }] : []),
  { type: 'reset' },
  ...rest,
];

/** A letter lesson: its preset, a typed text, nothing lit. */
const letters = (preset: string, patch: Partial<TfParams> | null, text: string, ...rest: Act[]): Act[] => on(preset, { ...(patch ?? {}), input: text }, PLACES(''), ...rest);

const numbers = (c: Ctx): LiveNumber[] => [
  { label: 'step', value: n(c.state.step) },
  { label: 'held out', value: pc(c.state.testAccuracy) },
  { label: 'parameters', value: n(c.derived.paramCount) },
];

const letterNumbers = (c: Ctx): LiveNumber[] => [
  { label: 'step', value: n(c.state.step) },
  { label: 'sound letters', value: pc(c.state.testAccuracy) },
  { label: 'sentences', value: pc(c.state.testExact) },
];

/** The three likeliest next letters, as the page shows them. */
const bet = (c: Ctx) => c.derived.next.map((x) => x.letter + ' ' + pc(x.p)).join(', ');
const headName = (layer: number, head: number) => 'head ' + (head + 1) + ' of block ' + (layer + 1);
const CAT = 'the cat often says ';

const nextLetter: L = {
  id: 'next-letter',
  title: 'Next-letter prediction',
  hook: 'Guess the next letter, add it, repeat.',
  section: 'hook',
  presetId: 'letters-trained',
  steps: [
    {
      id: 'write',
      kind: 'sandbox',
      say: 'A language model does one thing: guess the next letter from the letters before it. Add the guess, guess again, and it writes. Watch it write from “the ”: each new letter becomes a new row in the model. Try it: From “the h”.',
      takeaway: 'Guess, append, repeat: that loop is all a language model does. The rest of the chapter is how one good guess is made.',
      focus: { kind: 'panel', id: 'answer', label: 'The next letter' },
      enter: [...letters('letters-trained', null, 'the '), ...MODEL('none'), WRITE('the ')],
      numbers: letterNumbers,
      experiments: [
        {
          label: 'From “the h”',
          say: 'A start two words share: happy and hen.',
          enter: [...letters('letters-trained', null, 'the h'), ...MODEL('none'), WRITE('the h')],
          until: NOW,
          then: (c) => 'It writes “' + c.derived.writes + '”. Each letter is its likeliest guess from the letters before, and the sound at the end follows the animal many letters back.',
          focus: { kind: 'panel', id: 'answer', label: 'Writing' },
        },
        {
          label: 'From “the dog”',
          say: 'An animal already in the text.',
          enter: [...letters('letters-trained', null, 'the dog'), ...MODEL('none'), WRITE('the dog')],
          until: NOW,
          then: (c) => 'It writes “' + c.derived.writes + '”: with the animal already written, the letters after says follow from it.',
          focus: { kind: 'panel', id: 'answer', label: 'Writing' },
        },
        {
          label: 'Before training',
          say: 'The same model with random weights.',
          enter: [...letters('letters', null, 'the '), ...MODEL('none'), WRITE('the ')],
          until: NOW,
          then: (c) => 'Untrained, it writes “' + c.derived.writes + '”. Every letter is still its likeliest guess, but from random weights the guesses mean nothing.',
          focus: { kind: 'panel', id: 'answer', label: 'Writing' },
        },
        {
          label: 'Watch it learn',
          say: 'Random weights, trained live on 120 sentences.',
          enter: [...letters('letters', null, 'the '), ...MODEL('none'), PLAY],
          speed: 'turbo',
          until: (c) => c.derived.fitted || c.state.step >= 1500,
          then: (c) => at(c) + ': it writes every held-out sentence and gets ' + pc(c.state.testAccuracy) + ' of the sound letters. From “the ” it now writes “' + c.derived.writes + '”.',
          focus: { kind: 'metric', key: 'test-acc', label: 'Sound letters' },
        },
      ],
    },
  ],
};

const embeddings: L = {
  id: 'embeddings',
  title: 'Embeddings and positions',
  hook: 'Which letter, and where.',
  section: 'core',
  presetId: 'letters-trained',
  knobs: ['positions'],
  steps: [
    {
      id: 'clock',
      kind: 'sandbox',
      say: 'Each letter looks up its own row of numbers, e, so the t’s of “the cat often” all get the same row (outlined). A place vector p is added: each pair of its numbers turns like a clock hand, fast on the left. Watch the hands turn as the place moves down.',
      takeaway: 'e says which letter, p says where. Attention compares x = e + p, so equal letters at different places no longer look the same.',
      focus: { kind: 'node', id: 'position', label: 'Positional encoding' },
      enter: [...letters('letters-trained', null, CAT), ...MODEL('embed'), SWEEP],
      numbers: letterNumbers,
      experiments: [
        {
          label: 'Places 0 and 6',
          say: 'Two places, lit on the clock hands and in every column.',
          enter: [...letters('letters-trained', null, CAT), ...MODEL('embed'), PLACES('0,6')],
          until: NOW,
          then: (c) => {
            const far = placeGap(c.params.width, 0, 6);
            const near = placeGap(c.params.width, 0, 1);
            return (
              'The fastest hand has turned ' + far.fast + '°, nearly back where place 0 points, but the next is at ' + far.next + '°, so the rows still differ: similarity ' + w2(far.similarity) + ', against ' + w2(near.similarity) + ' for places 0 and 1.'
            );
          },
          focus: { kind: 'node', id: 'position', label: 'Positional encoding' },
        },
        {
          label: 'No positions',
          say: 'Nothing added for the place.',
          enter: [...letters('letters-trained', { positions: 'none' }, CAT), ...MODEL('embed')],
          until: NOW,
          then: () => 'Nothing is added, so x = e: the t’s of “the cat often” get identical rows, and attention could not tell the first t from the last.',
          focus: { kind: 'control', key: 'positions', label: 'Positions' },
        },
        {
          label: 'Sweep the places',
          say: 'The highlight walks down the rows again.',
          enter: [...letters('letters-trained', null, CAT), ...MODEL('embed'), SWEEP],
          focus: { kind: 'node', id: 'position', label: 'Positional encoding' },
        },
      ],
    },
  ],
};

const selfAttention: L = {
  id: 'self-attention',
  title: 'Self-attention',
  hook: 'Every letter reads the letters before it.',
  section: 'core',
  presetId: 'letters-trained',
  steps: [
    {
      id: 'reads',
      kind: 'sandbox',
      say: (c) =>
        'Each letter takes a weighted blend of the letters before it; the dots show the weights. Watch the rows light one by one. From the space after says, ' + headName(c.derived.focus.layer, c.derived.focus.head) + ' puts ' + pc(c.derived.focus.share) + ' on ' + c.derived.focus.animal + ': the sound depends on the animal.',
      takeaway: 'Attention lets a letter read any letter before it, as much as it needs. Training decides the weights, so it decides who reads whom.',
      focus: { kind: 'node', id: 'attention', label: 'Attention' },
      enter: [...letters('letters-trained', null, CAT), ...MODEL('none'), pickHead(1, 1), SWEEP],
      numbers: (c) => [{ label: 'on ' + c.derived.focus.animal, value: pc(c.derived.focus.share) }, { label: 'in', value: headName(c.derived.focus.layer, c.derived.focus.head) }],
      experiments: [
        {
          label: 'Another animal',
          say: 'Two words between the animal and says.',
          enter: [...letters('letters-trained', null, 'the small owl loudly says '), ...MODEL('none'), pickHead(1, 0)],
          until: NOW,
          then: (c) => 'In “' + c.derived.text + '”, ' + headName(c.derived.focus.layer, c.derived.focus.head) + ' puts ' + pc(c.derived.focus.share) + ' on ' + c.derived.focus.animal + '. Attention finds the animal by what it is, not by how far back it sits.',
          focus: { kind: 'node', id: 'attention', label: 'Attention' },
        },
        {
          label: 'Digits: reverse',
          say: 'Six digits in, the same six backwards out.',
          enter: on('reverse', null, ...MODEL('none'), PLACES(''), PLAY),
          speed: 'fast',
          until: (c) => c.derived.fitted || c.state.step >= 400,
          then: (c) => at(c) + ': held-out accuracy ' + pc(c.state.testAccuracy) + '. Each digit reads the one opposite it, a mirror: ' + w2(c.derived.mirror) + ' of each row lands there, against 1/' + c.derived.length + ' if it spread evenly.',
          focus: { kind: 'node', id: 'attention', label: 'The mirror' },
        },
        {
          label: 'Before training',
          say: 'The same letters, random weights.',
          enter: [...letters('letters', null, CAT), ...MODEL('none'), pickHead(1, 1)],
          until: NOW,
          then: (c) => 'Random weights: the largest weight in a row averages ' + w2(c.derived.sharp) + ', a blur over the letters before, and the best head puts only ' + pc(c.derived.focus.share) + ' on ' + c.derived.focus.animal + '.',
          focus: { kind: 'node', id: 'attention', label: 'A blur' },
        },
      ],
    },
  ],
};

const queriesKeys: L = {
  id: 'queries-keys-values',
  title: 'Queries, keys and values',
  hook: 'How a letter decides where to look.',
  section: 'core',
  presetId: 'letters-trained',
  knobs: ['heads'],
  steps: [
    {
      id: 'inside',
      kind: 'sandbox',
      say: 'Inside one head. Every letter makes a query Q (what it looks for), a key K (what it offers) and a value V (what it passes on). The lit row’s query meets every key: scores, then softmax, weights that add to 1, then a blend of values. Try it: Head 1.',
      takeaway: 'The query asks, the keys answer with a score, the values are what gets passed on. Each head has its own Q, K and V, so each can look for something else.',
      focus: { kind: 'node', id: 'attention', label: 'Q, K and V' },
      enter: [...letters('letters-trained', null, CAT), ...MODEL('attention'), pickHead(1, 1), PLACES(String(CAT.length))],
      numbers: (c) => [{ label: 'top key', value: c.derived.pick.key + ' at ' + c.derived.pick.keyPosition }, { label: 'weight', value: w2(c.derived.pick.weight) }],
      experiments: [
        {
          label: 'Head 1',
          say: 'The other head of the same block.',
          enter: [pickHead(1, 0)],
          until: NOW,
          then: (c) => 'Head 1 of block 2: from the lit row its top key is ' + c.derived.pick.key + ' at ' + c.derived.pick.keyPosition + ', weight ' + w2(c.derived.pick.weight) + '. Same letters, its own Q and K, so its own pattern.',
          focus: { kind: 'node', id: 'attention', label: 'Head 1' },
        },
        {
          label: 'Block 1',
          say: 'A head of the first block, nearer the letters.',
          enter: [pickHead(0, 0)],
          until: NOW,
          then: (c) => 'Head 1 of block 1: its top key from the lit row is ' + c.derived.pick.key + ' at ' + c.derived.pick.keyPosition + ', weight ' + w2(c.derived.pick.weight) + '. The first block mostly reads the letters close by.',
          focus: { kind: 'node', id: 'attention', label: 'Block 1' },
        },
        {
          label: 'Before training',
          say: 'The same head with random weights.',
          enter: [...letters('letters', null, CAT), ...MODEL('attention'), pickHead(1, 1), PLACES(String(CAT.length))],
          until: NOW,
          then: (c) => 'Random weights: the scores are small and close together, so softmax spreads the row; the top key gets only ' + w2(c.derived.pick.weight) + '.',
          focus: { kind: 'node', id: 'attention', label: 'Flat scores' },
        },
      ],
    },
  ],
};

const feedForward: L = {
  id: 'feed-forward',
  title: 'Feed-forward block',
  hook: 'Attention gathers, the block computes.',
  section: 'core',
  presetId: 'letters-trained',
  knobs: ['ffn'],
  steps: [
    {
      id: 'units',
      kind: 'sandbox',
      say: (c) =>
        'After attention, each letter’s row goes through the same small network on its own: widen to ' + 4 * c.params.width + ' units, ReLU sets the negative ones to 0 (faded), narrow back to ' + c.params.width + '. In block 1 a row keeps about ' + Math.round(c.derived.kept[0] ?? 0) + ' units. Watch the rows go by.',
      takeaway: 'Attention moves information between letters; the feed-forward block computes on it, one letter at a time. A transformer alternates the two.',
      focus: { kind: 'node', id: 'ffn', label: 'Feed Forward' },
      enter: [...letters('letters-trained', null, CAT), ...MODEL('ffn'), pickHead(0, 0), SWEEP],
      numbers: (c) => c.derived.kept.map((k, l) => ({ label: 'block ' + (l + 1) + ' keeps', value: Math.round(k) + ' of ' + 4 * c.params.width })),
      experiments: [
        {
          label: 'Block 2',
          say: 'The block that writes the next letter.',
          enter: [pickHead(1, 0), SWEEP],
          until: NOW,
          then: (c) => 'In block 2 a row keeps about ' + Math.round(c.derived.kept[1] ?? 0) + ' of its ' + 4 * c.params.width + ' units. Which ones stay on depends on the letter and on what attention brought to it.',
          focus: { kind: 'node', id: 'ffn', label: 'Block 2' },
        },
        {
          label: 'Digits: no block',
          say: 'Counting digits with attention alone.',
          enter: on('count-no-ffn', null, ...MODEL('none'), PLACES(''), PLAY),
          speed: 'turbo',
          until: (c) => c.state.step >= 800 || c.state.diverged,
          then: (c) => at(c) + ': counting reaches only ' + pc(c.state.testAccuracy) + ' without the block, against ' + pc(REFERENCE.count400) + ' with it. Attention gathers which digits match; the block turns that into a count.',
          focus: { kind: 'control', key: 'ffn', label: 'Feed-forward' },
        },
      ],
    },
  ],
};

const causalMask: L = {
  id: 'causal-mask',
  title: 'Causal mask',
  hook: 'Write without peeking.',
  section: 'core',
  presetId: 'letters-trained',
  knobs: ['mask'],
  steps: [
    {
      id: 'hidden',
      kind: 'sandbox',
      say: 'A letter being written cannot see the letters after it; they do not exist yet. The mask hides them: every score above the diagonal is hatched, so softmax gives it no weight. Then one sentence trains every position at once, honestly. Try it: No mask.',
      takeaway: 'A decoder predicts each token from the ones before it. The mask lets one sentence train every position at once without any position seeing its answer.',
      focus: { kind: 'node', id: 'attention', label: 'The mask' },
      enter: [...letters('letters-trained', null, CAT), ...MODEL('attention'), pickHead(1, 1)],
      numbers: letterNumbers,
      experiments: [
        {
          label: 'No mask',
          say: 'Every letter may read every other.',
          enter: [...letters('letters-trained', { mask: 'none' }, CAT), ...MODEL('attention'), pickHead(1, 1)],
          until: NOW,
          then: (c) => 'With the mask off every letter can read the next one, and random weights already put ' + pc(c.derived.ahead) + ' of the attention there. Trained, guessing the next letter would be copying it.',
          focus: { kind: 'control', key: 'mask', label: 'Mask' },
        },
        {
          label: 'Digits: look back only',
          say: 'Reversing digits with the mask on.',
          enter: on('reverse-causal', null, ...MODEL('none'), PLACES(''), PLAY),
          speed: 'fast',
          until: (c) => c.state.step >= 400 || c.state.diverged,
          then: (c) => at(c) + ': reversing reaches ' + pc(c.state.testAccuracy) + '. The last three digits can see their mirror; the first three would have to look ahead.',
          focus: { kind: 'control', key: 'mask', label: 'Mask' },
        },
      ],
    },
  ],
};

const softmax: L = {
  id: 'softmax',
  title: 'Softmax and the next letter',
  hook: 'Scores become a bet.',
  section: 'core',
  presetId: 'letters-trained',
  steps: [
    {
      id: 'bet',
      kind: 'sandbox',
      say: (c) => 'The last row scores every letter, and softmax turns the scores into shares that add to 1. After “the ” the model spreads its bet: ' + bet(c) + '. Writing takes the tallest. Try it: After “the c”.',
      takeaway: 'A language model does not know the next letter; it bets. Softmax turns scores into a bet, and writing picks from it.',
      focus: { kind: 'node', id: 'softmax', label: 'Softmax' },
      enter: [...letters('letters-trained', null, 'the '), ...MODEL('output')],
      numbers: (c) => c.derived.next.map((x) => ({ label: x.letter, value: pc(x.p) })),
      experiments: [
        {
          label: 'After “the c”',
          say: 'Two animals start with c.',
          enter: [...letters('letters-trained', null, 'the c'), ...MODEL('output')],
          until: NOW,
          then: (c) => 'After “the c”: ' + bet(c) + '. Cat or cow, and the model cannot know yet; the letter it writes next decides.',
          focus: { kind: 'node', id: 'softmax', label: 'Softmax' },
        },
        {
          label: 'After “the cat says ”',
          say: 'The animal is in the text.',
          enter: [...letters('letters-trained', null, 'the cat says '), ...MODEL('output')],
          until: NOW,
          then: (c) => 'After “the cat says ”: ' + bet(c) + '. With the animal written, the bet is nearly sure.',
          focus: { kind: 'node', id: 'softmax', label: 'Softmax' },
        },
        {
          label: 'Before training',
          say: 'Random weights, the same start.',
          enter: [...letters('letters', null, 'the '), ...MODEL('output')],
          until: NOW,
          then: (c) => 'Random weights: ' + bet(c) + '. Softmax always makes a bet from the scores; only training makes the scores worth betting on.',
          focus: { kind: 'node', id: 'softmax', label: 'Softmax' },
        },
      ],
    },
  ],
};

const noMask: L = {
  id: 'no-mask',
  title: 'Attention without a mask',
  hook: 'A perfect loss and a model that cannot write.',
  section: 'failure',
  presetId: 'animals-no-mask',
  knobs: ['mask'],
  steps: [
    {
      id: 'cheat',
      kind: 'sandbox',
      say: 'The same language with the mask off. Every word can see the next one, so predicting it is copying: the training loss falls to 0.00, below any honest model. Asked to finish held-out sentences, it writes only 17% of them right. Try it: Put the mask back.',
      takeaway: 'Next-word training is honest only if no position can see the word it predicts. Without the mask the loss measures copying, not prediction.',
      focus: { kind: 'panel', id: 'architecture', label: 'Looking ahead' },
      enter: [{ type: 'preset', id: 'animals-no-mask' }, { type: 'reset' }, GRID, { type: 'runTo', steps: 300 }],
      numbers: (c) => [{ label: 'train loss', value: fmt(c.state.trainLoss, 2) }, { label: 'sentences', value: pc(c.state.testExact) }],
      experiments: [
        {
          label: 'Put the mask back',
          say: 'Each word sees only the words before it.',
          enter: on('animals', null, GRID, PLAY),
          speed: 'normal',
          until: (c) => c.derived.fitted || c.state.step >= 300,
          then: (c) => at(c) + ': training loss ' + fmt(c.state.trainLoss, 2) + ', higher because it can no longer see the answer, and ' + pc(c.state.testExact) + ' of held-out sentences written right.',
          focus: { kind: 'control', key: 'mask', label: 'Mask' },
        },
        {
          label: 'Watch it cheat',
          say: 'No mask, from random weights.',
          enter: on('animals-no-mask', null, GRID, PLAY),
          speed: 'fast',
          until: (c) => c.state.step >= 300,
          then: (c) => at(c) + ': training loss ' + fmt(c.state.trainLoss, 2) + ', sentences written right ' + pc(c.state.testExact) + '. ' + pc(c.derived.ahead) + ' of the attention lands on later words, the half the mask hatches out.',
          focus: { kind: 'metric', key: 'train-loss', label: 'Train loss' },
        },
      ],
    },
  ],
};

const hot: L = {
  id: 'learning-rate',
  title: 'Learning rate too high',
  hook: 'Big steps never find the pattern.',
  section: 'failure',
  presetId: 'reverse-hot',
  knobs: ['learningRate'],
  steps: [
    {
      id: 'stuck',
      kind: 'sandbox',
      say: 'Reverse with η = 0.1, ten times the default. The first updates are so large that the weights jump past any useful pattern, and the model settles on guessing: held-out accuracy 19% after 400 steps. Try it: η = 0.01.',
      takeaway: 'Too large a step throws the weights past the pattern; too small a step crawls towards it. Adam at 0.01 suits this model.',
      focus: { kind: 'metric', key: 'test-acc', label: 'Held-out accuracy' },
      enter: [{ type: 'preset', id: 'reverse-hot' }, { type: 'reset' }, GRID, { type: 'runTo', steps: 400 }],
      numbers,
      experiments: [
        {
          label: 'η = 0.01',
          say: 'The default step.',
          enter: on('reverse', null, GRID, PLAY),
          speed: 'normal',
          until: (c) => c.derived.fitted || c.state.step >= 400,
          then: (c) => at(c) + ': held-out accuracy ' + pc(c.state.testAccuracy) + ', and the mirror is back.',
          focus: { kind: 'control', key: 'learningRate', label: 'Learning rate (η)' },
        },
        {
          label: 'η = 0.001',
          say: 'A tenth of the default.',
          enter: on('reverse', { learningRate: 0.001 }, GRID, PLAY),
          speed: 'turbo',
          until: (c) => c.state.testAccuracy >= 0.9 || c.state.step >= 1500,
          then: (c) => at(c) + ': held-out accuracy ' + pc(c.state.testAccuracy) + '. It gets there, several times slower.',
          focus: { kind: 'control', key: 'learningRate', label: 'Learning rate (η)' },
        },
      ],
    },
  ],
};

const experiments: L = {
  id: 'experiments',
  title: 'Experiments',
  hook: 'Nine experiments with known outcomes.',
  section: 'sandbox',
  presetId: 'reverse',
  steps: [
    {
      id: 'sandbox',
      kind: 'sandbox',
      enter: [{ type: 'preset', id: 'reverse' }, { type: 'reset' }, GRID, { type: 'speed', speed: 'normal' }],
      say: 'Pick an experiment: the settings change, the run replays, and the outcome opens under it.',
      experiments: [
        { label: 'Causal mask', patch: { mask: 'causal' }, say: 'Reverse with a causal mask: the last three digits can see their mirror, the first three would have to look ahead. Held-out accuracy stalls at 57%.', focus: { kind: 'control', key: 'mask', label: 'Mask' } },
        { label: 'One head', patch: { heads: 1 }, say: 'One head of 16 numbers: every held-out digit right at step 80, the same as two. Reversing needs only one pattern.', focus: { kind: 'control', key: 'heads', label: 'Heads' } },
        { label: 'Two layers', patch: { layers: 2 }, say: 'A second block on top: 6,776 parameters instead of 3,544, and fitted at step 70. One layer already finds the mirror.', focus: { kind: 'control', key: 'layers', label: 'Layers' } },
        { label: 'Width 8', patch: { width: 8 }, say: 'Token vectors of 8 numbers: 1,008 parameters, fitted at step 100, a little later than width 16.', focus: { kind: 'control', key: 'width', label: 'Width d' } },
        { label: 'Width 32', patch: { width: 32 }, say: 'Token vectors of 32 numbers: 13,224 parameters, fitted at step 70.', focus: { kind: 'control', key: 'width', label: 'Width d' } },
        { label: 'No layer norm', patch: { norm: false }, say: 'Without layer norm the blocks read vectors of any size: fitted at step 120 instead of 80.', focus: { kind: 'control', key: 'norm', label: 'Layer norm' } },
        { label: 'Plain SGD', patch: { optimiser: 'gd' }, say: 'Plain gradient descent at the same η = 0.01 crawls: 44% after 800 steps, where Adam fits at 80.', focus: { kind: 'control', key: 'optimiser', label: 'Optimiser' } },
        { label: 'Eight digits long', patch: { size: 8 }, say: 'Sequences of 8 digits: the mirror is found just as well, fitted at step 90.', focus: { kind: 'control', key: 'size', label: 'Length' } },
        { label: '200 sequences', patch: { count: 200 }, say: 'Only 150 training sequences: still fitted at step 80. The mirror is one rule, so a few examples teach it.', focus: { kind: 'control', key: 'count', label: 'Sequences' } },
      ],
    },
  ],
};

export const LESSONS: L[] = [nextLetter, embeddings, selfAttention, queriesKeys, feedForward, causalMask, softmax, noMask, hot, experiments];
