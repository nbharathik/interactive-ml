/** The activation catalogue: each entry knows its value, its derivative in z, and its formula. */

export type ActivationName =
  | 'linear'
  | 'sigmoid'
  | 'tanh'
  | 'relu'
  | 'leakyRelu'
  | 'elu'
  | 'gelu'
  | 'softplus'
  | 'swish';

export interface ActivationParams {
  /** Slope left of zero for leaky ReLU. */
  leakSlope?: number;
  eluAlpha?: number;
}

export interface ActivationFn {
  name: ActivationName;
  label: string;
  tex: string;
  dtex: string;
  f(z: number): number;
  /** Derivative as a function of the pre-activation z. */
  df(z: number): number;
  /** Output range, or null when unbounded. */
  range: [number, number] | null;
  /** Whether the derivative dies for very negative z (dead or saturated units). */
  saturatesLeft: boolean;
  saturatesRight: boolean;
}

export const ACTIVATION_NAMES: readonly ActivationName[] = [
  'sigmoid',
  'tanh',
  'relu',
  'leakyRelu',
  'elu',
  'gelu',
  'softplus',
  'swish',
  'linear',
];

export const ACTIVATION_LABELS: Record<ActivationName, string> = {
  linear: 'Linear',
  sigmoid: 'Sigmoid',
  tanh: 'Tanh',
  relu: 'ReLU',
  leakyRelu: 'Leaky ReLU',
  elu: 'ELU',
  gelu: 'GELU',
  softplus: 'Softplus',
  swish: 'Swish',
};

/** Sign-stable logistic function. */
export function sigmoid(z: number): number {
  if (z >= 0) {
    const e = Math.exp(-z);
    return 1 / (1 + e);
  }
  const e = Math.exp(z);
  return e / (1 + e);
}

/** log(1 + e^z) without overflow. */
export function softplus(z: number): number {
  return Math.max(z, 0) + Math.log1p(Math.exp(-Math.abs(z)));
}

const GELU_K = Math.sqrt(2 / Math.PI);
const GELU_C = 0.044715;

export function activation(name: ActivationName, params: ActivationParams = {}): ActivationFn {
  const leak = params.leakSlope ?? 0.1;
  const alpha = params.eluAlpha ?? 1;
  const label = ACTIVATION_LABELS[name];
  switch (name) {
    case 'sigmoid':
      return {
        name,
        label,
        tex: '\\sigma(z) = \\frac{1}{1 + e^{-z}}',
        dtex: "\\sigma'(z) = \\sigma(z)\\,(1 - \\sigma(z))",
        f: sigmoid,
        df: (z) => {
          const s = sigmoid(z);
          return s * (1 - s);
        },
        range: [0, 1],
        saturatesLeft: true,
        saturatesRight: true,
      };
    case 'tanh':
      return {
        name,
        label,
        tex: 'f(z) = \\tanh(z)',
        dtex: "f'(z) = 1 - \\tanh^2(z)",
        f: Math.tanh,
        df: (z) => {
          const t = Math.tanh(z);
          return 1 - t * t;
        },
        range: [-1, 1],
        saturatesLeft: true,
        saturatesRight: true,
      };
    case 'relu':
      return {
        name,
        label,
        tex: 'f(z) = \\max(0, z)',
        dtex: "f'(z) = [z > 0]",
        f: (z) => (z > 0 ? z : 0),
        df: (z) => (z > 0 ? 1 : 0),
        range: [0, Infinity],
        saturatesLeft: true,
        saturatesRight: false,
      };
    case 'leakyRelu':
      return {
        name,
        label,
        tex: 'f(z) = \\max(\\alpha z, z)',
        dtex: "f'(z) = 1 \\text{ if } z > 0 \\text{ else } \\alpha",
        f: (z) => (z > 0 ? z : leak * z),
        df: (z) => (z > 0 ? 1 : leak),
        range: null,
        saturatesLeft: false,
        saturatesRight: false,
      };
    case 'elu':
      return {
        name,
        label,
        tex: 'f(z) = z \\text{ if } z > 0 \\text{ else } \\alpha(e^{z} - 1)',
        dtex: "f'(z) = 1 \\text{ if } z > 0 \\text{ else } \\alpha e^{z}",
        f: (z) => (z > 0 ? z : alpha * (Math.exp(z) - 1)),
        df: (z) => (z > 0 ? 1 : alpha * Math.exp(z)),
        range: [-alpha, Infinity],
        saturatesLeft: true,
        saturatesRight: false,
      };
    case 'gelu':
      return {
        name,
        label,
        tex: 'f(z) = \\tfrac{1}{2} z \\left(1 + \\tanh\\left(\\sqrt{2/\\pi}\\,(z + 0.044715 z^3)\\right)\\right)',
        dtex: "f'(z) = \\tfrac{1}{2}(1 + \\tanh u) + \\tfrac{1}{2} z\\,(1 - \\tanh^2 u)\\,u'",
        f: (z) => 0.5 * z * (1 + Math.tanh(GELU_K * (z + GELU_C * z * z * z))),
        df: (z) => {
          const u = GELU_K * (z + GELU_C * z * z * z);
          const t = Math.tanh(u);
          const du = GELU_K * (1 + 3 * GELU_C * z * z);
          return 0.5 * (1 + t) + 0.5 * z * (1 - t * t) * du;
        },
        range: null,
        saturatesLeft: true,
        saturatesRight: false,
      };
    case 'softplus':
      return {
        name,
        label,
        tex: 'f(z) = \\ln(1 + e^{z})',
        dtex: "f'(z) = \\sigma(z)",
        f: softplus,
        df: sigmoid,
        range: [0, Infinity],
        saturatesLeft: true,
        saturatesRight: false,
      };
    case 'swish':
      return {
        name,
        label,
        tex: 'f(z) = z\\,\\sigma(z)',
        dtex: "f'(z) = \\sigma(z)\\,(1 + z\\,(1 - \\sigma(z)))",
        f: (z) => z * sigmoid(z),
        df: (z) => {
          const s = sigmoid(z);
          return s * (1 + z * (1 - s));
        },
        range: null,
        saturatesLeft: true,
        saturatesRight: false,
      };
    default:
      return {
        name: 'linear',
        label,
        tex: 'f(z) = z',
        dtex: "f'(z) = 1",
        f: (z) => z,
        df: () => 1,
        range: null,
        saturatesLeft: false,
        saturatesRight: false,
      };
  }
}
