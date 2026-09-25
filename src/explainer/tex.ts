/** Small TeX builders for the detail cards. */

/** A number for TeX: fixed decimals, never "-0.00", and 1.2 \times 10^{13} once it runs away. */
export function texNum(value: number, decimals = 2): string {
  if (Number.isNaN(value)) return '\\text{n/a}';
  if (!Number.isFinite(value)) return value < 0 ? '-\\infty' : '\\infty';
  if (Math.abs(value) >= 1e5) {
    const [mantissa, exponent] = value.toExponential(1).split('e');
    return mantissa + ' \\times 10^{' + Number(exponent) + '}';
  }
  const text = value.toFixed(decimals);
  return Number(text) === 0 ? (0).toFixed(decimals) : text;
}

export interface TexTerm {
  coef: number;
  symbol: string;
}

/** `0.99\,x_1 + 0.49\,x_2 + 0.06`, wrapped every `perLine` terms. */
export function texSum(
  terms: readonly TexTerm[],
  bias?: number,
  { decimals = 2, perLine = 3 }: { decimals?: number; perLine?: number } = {},
): string {
  const parts: string[] = [];
  terms.forEach((term, i) => {
    const sign = term.coef < 0 ? '-' : i === 0 ? '' : '+';
    parts.push(sign + ' ' + texNum(Math.abs(term.coef), decimals) + '\\,' + term.symbol);
  });
  if (bias !== undefined) {
    const sign = bias < 0 ? '-' : parts.length === 0 ? '' : '+';
    parts.push(sign + ' ' + texNum(Math.abs(bias), decimals));
  }
  if (parts.length === 0) return '0';
  const lines: string[] = [];
  for (let i = 0; i < parts.length; i += perLine) lines.push(parts.slice(i, i + perLine).join(' '));
  return lines.join(' \\\\ &\\quad ');
}

/** `w_1 x_1 + w_2 x_2 + b` for a few inputs, `\mathbf{w}\cdot\mathbf{x} + b` for many. */
export function texSymbolicSum(symbols: readonly string[], vector = '\\mathbf{x}', max = 3): string {
  if (symbols.length === 0) return 'b';
  if (symbols.length > max) return '\\mathbf{w} \\cdot ' + vector + ' + b';
  return symbols.map((s, i) => 'w_' + (i + 1) + ' ' + s).join(' + ') + ' + b';
}

/** Rows of `lhs = rhs` in one aligned block. */
export function texLines(rows: ReadonlyArray<readonly [string, string]>): string {
  return '\\begin{aligned}' + rows.map(([lhs, rhs]) => lhs + ' &= ' + rhs).join(' \\\\ ') + '\\end{aligned}';
}

const SUBSCRIPTS = '₀₁₂₃₄₅₆₇₈₉';
const SUPERSCRIPTS = '⁰¹²³⁴⁵⁶⁷⁸⁹';

/** Unicode symbols (x₁, x̃₂², x¹², ŷ, sin x₁) as TeX; a run of digits shares one group. */
export function texSymbol(symbol: string): string {
  let out = '';
  let run: '_' | '^' | null = null;
  for (const ch of symbol) {
    const sub = SUBSCRIPTS.indexOf(ch);
    const sup = SUPERSCRIPTS.indexOf(ch);
    if (sub >= 0 || sup >= 0) {
      const mark = sub >= 0 ? '_' : '^';
      const digit = sub >= 0 ? sub : sup;
      out = run === mark ? out.slice(0, -1) + digit + '}' : out + mark + '{' + digit + '}';
      run = mark;
      continue;
    }
    run = null;
    if (ch === 'ŷ') out += '\\hat{y}';
    else if (ch === '̃') out = out.replace(/([a-z])$/, '\\tilde{$1}');
    else if (ch === ' ') out += '\\,';
    else out += ch;
  }
  return out.replace(/sin\\,/g, '\\sin ');
}
