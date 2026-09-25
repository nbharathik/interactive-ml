/** Seeded randomness so every dataset and initialisation is reproducible. */

export type Rng = () => number;

/** Mulberry32. */
export function makeRng(seed: number): Rng {
  let s = seed >>> 0;
  return () => {
    s = (s + 0x6d2b79f5) | 0;
    let t = Math.imul(s ^ (s >>> 15), 1 | s);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Box-Muller normal sample. */
export function gauss(rng: Rng, mu = 0, sigma = 1): number {
  let u = 0;
  let v = 0;
  while (u === 0) u = rng();
  while (v === 0) v = rng();
  return mu + sigma * Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

/** Uniform sample in [lo, hi). */
export function uniform(rng: Rng, lo: number, hi: number): number {
  return lo + rng() * (hi - lo);
}

/** Fisher-Yates, returning a new array. */
export function shuffled<T>(rng: Rng, items: readonly T[]): T[] {
  const out = items.slice();
  for (let i = out.length - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1));
    [out[i], out[j]] = [out[j], out[i]];
  }
  return out;
}
