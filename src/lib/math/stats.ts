/** Small statistics helpers shared by metrics panels and algorithms. */

export function mean(values: readonly number[]): number {
  if (values.length === 0) return 0;
  let sum = 0;
  for (const v of values) sum += v;
  return sum / values.length;
}

/* ---------------- regression metrics ---------------- */

export function meanSquaredError(actual: readonly number[], predicted: readonly number[]): number {
  if (actual.length === 0) return 0;
  let sum = 0;
  for (let i = 0; i < actual.length; i++) {
    const d = actual[i] - predicted[i];
    sum += d * d;
  }
  return sum / actual.length;
}

export function meanAbsoluteError(actual: readonly number[], predicted: readonly number[]): number {
  if (actual.length === 0) return 0;
  let sum = 0;
  for (let i = 0; i < actual.length; i++) sum += Math.abs(actual[i] - predicted[i]);
  return sum / actual.length;
}

/** Coefficient of determination. Returns 0 when the target has no variance. */
export function rSquared(actual: readonly number[], predicted: readonly number[]): number {
  if (actual.length === 0) return 0;
  const mu = mean(actual);
  let ssRes = 0;
  let ssTot = 0;
  for (let i = 0; i < actual.length; i++) {
    const r = actual[i] - predicted[i];
    const t = actual[i] - mu;
    ssRes += r * r;
    ssTot += t * t;
  }
  return ssTot === 0 ? 0 : 1 - ssRes / ssTot;
}

export function pearson(xs: readonly number[], ys: readonly number[]): number {
  const n = Math.min(xs.length, ys.length);
  if (n < 2) return 0;
  const mx = mean(xs.slice(0, n));
  const my = mean(ys.slice(0, n));
  let num = 0;
  let dx = 0;
  let dy = 0;
  for (let i = 0; i < n; i++) {
    const a = xs[i] - mx;
    const b = ys[i] - my;
    num += a * b;
    dx += a * a;
    dy += b * b;
  }
  const den = Math.sqrt(dx * dy);
  return den === 0 ? 0 : num / den;
}

/* ---------------- classification metrics ---------------- */

export interface ConfusionMatrix {
  /** counts[actual][predicted] */
  counts: number[][];
  labels: number[];
  total: number;
}

export function confusionMatrix(
  actual: readonly number[],
  predicted: readonly number[],
  classCount: number,
): ConfusionMatrix {
  const counts = Array.from({ length: classCount }, () => new Array<number>(classCount).fill(0));
  const n = Math.min(actual.length, predicted.length);
  for (let i = 0; i < n; i++) {
    const a = actual[i];
    const p = predicted[i];
    if (a >= 0 && a < classCount && p >= 0 && p < classCount) counts[a][p] += 1;
  }
  return { counts, labels: Array.from({ length: classCount }, (_, i) => i), total: n };
}

export function accuracy(actual: readonly number[], predicted: readonly number[]): number {
  if (actual.length === 0) return 0;
  let hits = 0;
  for (let i = 0; i < actual.length; i++) if (actual[i] === predicted[i]) hits += 1;
  return hits / actual.length;
}

/** Precision, recall and F1 for one class in a one-vs-rest sense. */
export function precisionRecallF1(
  actual: readonly number[],
  predicted: readonly number[],
  positiveClass = 1,
): { precision: number; recall: number; f1: number; tp: number; fp: number; fn: number; tn: number } {
  let tp = 0;
  let fp = 0;
  let fn = 0;
  let tn = 0;
  for (let i = 0; i < actual.length; i++) {
    const a = actual[i] === positiveClass;
    const p = predicted[i] === positiveClass;
    if (a && p) tp += 1;
    else if (!a && p) fp += 1;
    else if (a && !p) fn += 1;
    else tn += 1;
  }
  const precision = tp + fp === 0 ? 0 : tp / (tp + fp);
  const recall = tp + fn === 0 ? 0 : tp / (tp + fn);
  const f1 = precision + recall === 0 ? 0 : (2 * precision * recall) / (precision + recall);
  return { precision, recall, f1, tp, fp, fn, tn };
}

/** Binary cross-entropy (log loss), averaged, with probabilities clipped for safety. */
export function logLoss(actual: readonly number[], probabilities: readonly number[]): number {
  if (actual.length === 0) return 0;
  const eps = 1e-12;
  let sum = 0;
  for (let i = 0; i < actual.length; i++) {
    const p = Math.min(1 - eps, Math.max(eps, probabilities[i]));
    sum += actual[i] === 1 ? -Math.log(p) : -Math.log(1 - p);
  }
  return sum / actual.length;
}

/** Area under the ROC curve, computed from ranks (handles ties). */
export function rocAuc(actual: readonly number[], scores: readonly number[]): number {
  const n = actual.length;
  if (n === 0) return 0.5;
  const order = Array.from({ length: n }, (_, i) => i).sort((a, b) => scores[a] - scores[b]);
  const ranks = new Array<number>(n);
  let i = 0;
  while (i < n) {
    let j = i;
    while (j + 1 < n && scores[order[j + 1]] === scores[order[i]]) j += 1;
    const avgRank = (i + j) / 2 + 1;
    for (let k = i; k <= j; k++) ranks[order[k]] = avgRank;
    i = j + 1;
  }
  let positives = 0;
  let rankSum = 0;
  for (let k = 0; k < n; k++) {
    if (actual[k] === 1) {
      positives += 1;
      rankSum += ranks[k];
    }
  }
  const negatives = n - positives;
  if (positives === 0 || negatives === 0) return 0.5;
  return (rankSum - (positives * (positives + 1)) / 2) / (positives * negatives);
}

/** ROC curve points, ordered by increasing threshold aggressiveness. */
export function rocCurve(
  actual: readonly number[],
  scores: readonly number[],
): Array<{ fpr: number; tpr: number; threshold: number }> {
  const n = actual.length;
  const order = Array.from({ length: n }, (_, i) => i).sort((a, b) => scores[b] - scores[a]);
  const positives = actual.reduce((acc, v) => acc + (v === 1 ? 1 : 0), 0);
  const negatives = n - positives;
  const out: Array<{ fpr: number; tpr: number; threshold: number }> = [
    { fpr: 0, tpr: 0, threshold: Infinity },
  ];
  let tp = 0;
  let fp = 0;
  // Tied scores share one threshold, so they move the curve in one diagonal step, whatever their order.
  // At least one point per pass, so a NaN score (never equal to itself) cannot stall the loop.
  for (let i = 0; i < order.length; ) {
    const threshold = scores[order[i]];
    do {
      if (actual[order[i]] === 1) tp += 1;
      else fp += 1;
      i += 1;
    } while (i < order.length && scores[order[i]] === threshold);
    out.push({
      fpr: negatives === 0 ? 0 : fp / negatives,
      tpr: positives === 0 ? 0 : tp / positives,
      threshold,
    });
  }
  return out;
}

/* ---------------- scaling ---------------- */

export interface Standardizer {
  means: number[];
  stds: number[];
  transform(row: readonly number[]): number[];
  inverse(row: readonly number[]): number[];
}

/** Column-wise z-score standardiser, fitted on a matrix of rows. */
export function fitStandardizer(rows: readonly number[][]): Standardizer {
  const dim = rows[0]?.length ?? 0;
  const means = new Array<number>(dim).fill(0);
  const stds = new Array<number>(dim).fill(1);
  if (rows.length === 0) {
    return {
      means,
      stds,
      transform: (r) => r.slice(),
      inverse: (r) => r.slice(),
    };
  }
  for (let d = 0; d < dim; d++) {
    let sum = 0;
    for (const row of rows) sum += row[d];
    means[d] = sum / rows.length;
  }
  for (let d = 0; d < dim; d++) {
    let sum = 0;
    for (const row of rows) {
      const diff = row[d] - means[d];
      sum += diff * diff;
    }
    const sd = Math.sqrt(sum / rows.length);
    stds[d] = sd < 1e-9 ? 1 : sd;
  }
  return {
    means,
    stds,
    transform: (row) => row.map((v, d) => (v - means[d]) / stds[d]),
    inverse: (row) => row.map((v, d) => v * stds[d] + means[d]),
  };
}

const SUBSCRIPTS = ['₀', '₁', '₂', '₃', '₄', '₅', '₆', '₇', '₈', '₉'];
const SUPERSCRIPTS = ['⁰', '¹', '²', '³', '⁴', '⁵', '⁶', '⁷', '⁸', '⁹'];

/** Unicode subscript digits: 12 → ₁₂. */
export function sub(index: number): string {
  return String(index).split('').map((d) => SUBSCRIPTS[Number(d)] ?? d).join('');
}

/** Unicode superscript digits: 12 → ¹². */
export function sup(index: number): string {
  return String(index).split('').map((d) => SUPERSCRIPTS[Number(d)] ?? d).join('');
}

const MINUS = '−';
const signed = (s: string) => s.replace('-', MINUS);

/** Round for display without producing "-0" or 17 decimal places. */
export function fmt(value: number, decimals = 3): string {
  if (!Number.isFinite(value)) return 'n/a';
  const rounded = Number(value.toFixed(decimals));
  return signed(Object.is(rounded, -0) ? (0).toFixed(decimals) : rounded.toFixed(decimals));
}

/** `fmt`, except that a runaway value (and, with `tiny`, a vanishing one) goes to 1.2e+13 form. */
export function fmtCell(value: number, decimals = 3, tiny = false): string {
  const abs = Math.abs(value);
  if (Number.isFinite(value) && (abs >= 1e5 || (tiny && abs > 0 && abs < 0.5 * 10 ** -decimals))) {
    return signed(value.toExponential(1));
  }
  return fmt(value, decimals);
}

/** Compact display for big/small numbers in metric tiles. */
export function fmtCompact(value: number): string {
  if (!Number.isFinite(value)) return 'n/a';
  const abs = Math.abs(value);
  if (abs === 0) return '0';
  if (abs >= 1e6) return signed((value / 1e6).toFixed(1) + 'M');
  if (abs >= 1e4) return signed((value / 1e3).toFixed(1) + 'k');
  if (abs >= 100) return signed(value.toFixed(0));
  if (abs >= 1) return signed(value.toFixed(2));
  if (abs >= 0.001) return signed(value.toFixed(4));
  return signed(value.toExponential(1));
}

export function fmtPercent(value: number, decimals = 1): string {
  if (!Number.isFinite(value)) return 'n/a';
  return signed((value * 100).toFixed(decimals) + '%');
}

/** Readout for log-scale knobs: two significant figures, no padding. */
export function fmtKnob(v: number): string {
  if (v === 0) return '0';
  // 0.0005 reads as the lessons write it; only smaller values take an exponent.
  if (Math.abs(v) < 0.0001) return v.toExponential(1);
  return String(Number(v.toPrecision(2)));
}
