/** Short token sequences for the transformer: reverse a string of digits, count each digit, and a tiny language of animal sounds, word by word or letter by letter. */

import { makeRng, shuffled } from '../math/rng';

export type SequenceTask = 'reverse' | 'count' | 'animals' | 'letters';

export interface SequenceDataset {
  kind: 'sequences';
  task: SequenceTask;
  /** The longest sequence, start token included. */
  length: number;
  /** Digits per sequence for the digit tasks. */
  size: number;
  /** Input token names; index 0 is the start token. */
  vocab: string[];
  /** What the output head chooses between. */
  outVocab: string[];
  inputs: number[][];
  /** The class each position should predict, or -1 where there is nothing to predict. */
  targets: number[][];
  /** The positions accuracy is scored on. */
  scored: number[][];
  /** The targets are the next word: the answer must come from the prefix alone. */
  nextToken: boolean;
  /** How many tokens of each sequence a next-word model is given before it writes the rest. */
  prompt: number[];
  trainIndex: number[];
  testIndex: number[];
}

export interface SequenceConfig {
  task: SequenceTask;
  /** Digits per sequence after the start token. */
  size: number;
  /** How many different digits the sequences use. */
  digits: number;
  count: number;
  seed: number;
  trainFraction: number;
}

export const START = '▸';

const DIGITS = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'];

/** The animals and what they say. */
export const ANIMALS: ReadonlyArray<{ name: string; sound: string }> = [
  { name: 'cat', sound: 'meow' },
  { name: 'dog', sound: 'woof' },
  { name: 'cow', sound: 'moo' },
  { name: 'duck', sound: 'quack' },
  { name: 'sheep', sound: 'baa' },
  { name: 'owl', sound: 'hoot' },
];
export const ADJECTIVES = ['big', 'small', 'old'];
export const ADVERBS = ['often', 'loudly'];
const ANIMAL_VOCAB = [START, 'the', ...ADJECTIVES, ...ANIMALS.map((a) => a.name), ...ADVERBS, 'says', ...ANIMALS.map((a) => a.sound), '.'];

/** The letter task's language is larger, so its 160 sentences leave every pattern in training. */
export const LETTER_ANIMALS: ReadonlyArray<{ name: string; sound: string }> = [...ANIMALS, { name: 'pig', sound: 'oink' }, { name: 'hen', sound: 'cluck' }];
export const LETTER_ADJECTIVES = [...ADJECTIVES, 'happy'];
export const LETTER_ADVERBS = [...ADVERBS, 'softly'];

/** Every sentence of the letter task: the adjective and the adverb may be left out. */
export function letterSentences(): string[] {
  const out: string[] = [];
  for (const animal of LETTER_ANIMALS) {
    for (const adjective of ['', ...LETTER_ADJECTIVES]) {
      for (const adverb of ['', ...LETTER_ADVERBS]) out.push(['the', adjective, animal.name, adverb, 'says', animal.sound].filter(Boolean).join(' ') + '.');
    }
  }
  return out;
}

export const TASK_NAMES: Record<SequenceTask, string> = {
  reverse: 'Reverse',
  count: 'Count',
  animals: 'Animal sounds',
  letters: 'Next letter',
};

/** The text of a letter sequence, without its start token. */
function textOf(vocab: readonly string[], tokens: readonly number[]): string {
  return tokens
    .slice(1)
    .map((t) => vocab[t] ?? '')
    .join('');
}

/** The animal a letter sentence names before its says, if any. */
export function letterAnimal(text: string): { name: string; sound: string } | undefined {
  const says = text.indexOf('says ');
  const words = (says >= 0 ? text.slice(0, says) : text).split(' ');
  return LETTER_ANIMALS.find((a) => words.includes(a.name));
}

function split(rng: () => number, count: number, trainFraction: number) {
  const order = shuffled(rng, Array.from({ length: count }, (_, i) => i));
  const trainCount = Math.max(1, Math.min(count - 1, Math.round(count * trainFraction)));
  return {
    trainIndex: order.slice(0, trainCount).sort((a, b) => a - b),
    testIndex: order.slice(trainCount).sort((a, b) => a - b),
  };
}

/** Distinct random digit strings, so no held-out sequence was seen in training. */
function digitStrings(rng: () => number, count: number, size: number, digits: number): number[][] {
  const seen = new Set<string>();
  const out: number[][] = [];
  const target = Math.min(count, Math.pow(digits, size));
  for (let tries = 0; out.length < target && tries < target * 50; tries++) {
    const s = Array.from({ length: size }, () => Math.floor(rng() * digits));
    const key = s.join('');
    if (seen.has(key)) continue;
    seen.add(key);
    out.push(s);
  }
  return out;
}

/** The answers for any sequence of the task, by its rule: targets per position and the positions that are scored. */
export function labelTokens(task: SequenceTask, vocab: readonly string[], tokens: readonly number[]): { targets: number[]; scored: number[] } {
  if (task === 'letters') {
    // Position t predicts letter t of the text; after says come the animal's sound, letter by letter.
    const targets = tokens.map((_, t) => (t + 1 < tokens.length ? tokens[t + 1] : -1));
    const text = textOf(vocab, tokens);
    const says = text.indexOf('says ');
    const animal = letterAnimal(text);
    const scored: number[] = [];
    if (says >= 0 && animal) {
      for (let k = 0; k < animal.sound.length; k++) {
        const position = says + 5 + k;
        if (position >= tokens.length) break;
        targets[position] = vocab.indexOf(animal.sound[k]);
        scored.push(position);
      }
    }
    return { targets, scored };
  }
  if (task === 'animals') {
    const targets = tokens.map((_, t) => (t + 1 < tokens.length ? tokens[t + 1] : -1));
    const scored: number[] = [];
    const says = tokens.indexOf(vocab.indexOf('says'));
    const animal = ANIMALS.find((a) => tokens.slice(0, Math.max(0, says)).includes(vocab.indexOf(a.name)));
    if (says >= 0 && animal) {
      // After says comes the animal's sound, whatever was typed next.
      targets[says] = vocab.indexOf(animal.sound);
      scored.push(says);
    }
    return { targets, scored };
  }
  const digits = tokens.slice(1).map((t) => t - 1);
  const answers = task === 'count' ? digits.map((d) => digits.filter((e) => e === d).length - 1) : digits.slice().reverse();
  return { targets: [-1, ...answers], scored: digits.map((_, i) => i + 1) };
}

export function generateSequences(config: SequenceConfig): SequenceDataset {
  const rng = makeRng(config.seed);
  const size = Math.max(2, Math.round(config.size));
  const digits = Math.max(2, Math.min(DIGITS.length, Math.round(config.digits)));
  const inputs: number[][] = [];
  const targets: number[][] = [];
  const scored: number[][] = [];
  let vocab: string[];
  let outVocab: string[];
  if (config.task === 'letters') {
    const sentences = letterSentences();
    vocab = [START, ...[...new Set(sentences.join(''))].sort()];
    outVocab = vocab;
    for (const s of sentences) inputs.push([0, ...s.split('').map((c) => vocab.indexOf(c))]);
  } else if (config.task === 'animals') {
    vocab = ANIMAL_VOCAB.slice();
    outVocab = vocab;
    const id = (word: string) => vocab.indexOf(word);
    for (const animal of ANIMALS) {
      for (const adjective of ['', ...ADJECTIVES]) {
        for (const adverb of ['', ...ADVERBS]) {
          const words = ['the', adjective, animal.name, adverb, 'says', animal.sound, '.'].filter(Boolean);
          inputs.push([0, ...words.map(id)]);
        }
      }
    }
  } else {
    vocab = [START, ...DIGITS.slice(0, digits)];
    outVocab = config.task === 'count' ? Array.from({ length: size }, (_, i) => String(i + 1)) : DIGITS.slice(0, digits);
    for (const s of digitStrings(rng, Math.max(8, Math.round(config.count)), size, digits)) inputs.push([0, ...s.map((d) => d + 1)]);
  }
  for (const tokens of inputs) {
    const label = labelTokens(config.task, vocab, tokens);
    targets.push(label.targets);
    scored.push(label.scored);
  }
  const language = config.task === 'animals' || config.task === 'letters';
  const length = language ? Math.max(...inputs.map((s) => s.length)) : size + 1;
  const prompt = inputs.map((tokens) => promptLength(config.task, vocab, tokens));
  return { kind: 'sequences', task: config.task, length, size, vocab, outVocab, inputs, targets, scored, nextToken: language, prompt, ...split(rng, inputs.length, config.trainFraction) };
}

/** A sentence is prompted up to its animal; the rest is the model's to write. Digit strings are read whole. */
export function promptLength(task: SequenceTask, vocab: readonly string[], tokens: readonly number[]): number {
  if (task === 'letters') {
    const text = textOf(vocab, tokens);
    const animal = letterAnimal(text);
    if (!animal) return tokens.length;
    const at = (' ' + text + ' ').indexOf(' ' + animal.name + ' ');
    return Math.min(tokens.length, 1 + at + animal.name.length);
  }
  if (task !== 'animals') return tokens.length;
  const animal = tokens.findIndex((t) => ANIMALS.some((a) => a.name === vocab[t]));
  return animal >= 0 ? animal + 1 : tokens.length;
}

/** Typed text as tokens: digits one by one, or words split on spaces. */
export function parseTokens(data: SequenceDataset, text: string): { tokens: number[] } | { error: string } {
  if (data.task === 'letters') {
    // A trailing space matters here: after "says " comes the sound.
    const letters = text.toLowerCase().replace(/^\s+/, '').replace(/\s+/g, ' ');
    if (letters.length === 0) return { error: 'Type the start of a sentence.' };
    if (letters.length > data.length - 1) return { error: 'At most ' + (data.length - 1) + ' letters.' };
    const unknown = letters.split('').find((c) => c === START || !data.vocab.includes(c));
    if (unknown !== undefined) return { error: 'The model does not know "' + unknown + '".' };
    return { tokens: [0, ...letters.split('').map((c) => data.vocab.indexOf(c))] };
  }
  const clean = text.trim();
  if (data.task === 'animals') {
    const words = clean.toLowerCase().split(/\s+/).filter(Boolean);
    if (words.length === 0) return { error: 'Type a few words.' };
    if (words.length > data.length - 1) return { error: 'At most ' + (data.length - 1) + ' words.' };
    const unknown = words.find((w) => w === START || !data.vocab.includes(w));
    if (unknown) return { error: 'The model does not know "' + unknown + '".' };
    return { tokens: [0, ...words.map((w) => data.vocab.indexOf(w))] };
  }
  const chars = clean.replace(/[\s,]+/g, '').split('');
  if (chars.length !== data.size) return { error: 'Type exactly ' + data.size + ' digits.' };
  const bad = chars.find((c) => !data.vocab.includes(c) || c === START);
  if (bad !== undefined) return { error: 'Use the digits 0 to ' + (data.vocab.length - 2) + '.' };
  return { tokens: [0, ...chars.map((c) => data.vocab.indexOf(c))] };
}

/** A sequence as text, without the start token: digits and letters run together, words keep their spaces. */
export function spellTokens(data: SequenceDataset, tokens: readonly number[]): string {
  const words = tokens.slice(1).map((t) => data.vocab[t] ?? '?');
  return data.task === 'animals' ? words.join(' ') : words.join('');
}
