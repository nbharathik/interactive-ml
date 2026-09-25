/** Two-dimensional test functions for watching optimisers. Each knows its gradient and its minima. */

export type LandscapeName = 'bowl' | 'ravine' | 'rosenbrock' | 'saddle' | 'himmelblau' | 'eggcrate';

export interface Landscape {
  name: LandscapeName;
  label: string;
  tex: string;
  f(x: number, y: number): number;
  grad(x: number, y: number): [number, number];
  /** Global minimum first. */
  minima: Array<[number, number]>;
  domain: { x: [number, number]; y: [number, number] };
  start: [number, number];
  suggestedLr: number;
  /** Gradient norm that counts as settled; flat valleys need a tighter one. */
  tolerance: number;
}

export const LANDSCAPE_NAMES: readonly LandscapeName[] = ['bowl', 'ravine', 'rosenbrock', 'saddle', 'himmelblau', 'eggcrate'];

export const LANDSCAPE_LABELS: Record<LandscapeName, string> = {
  bowl: 'Bowl',
  ravine: 'Ravine',
  rosenbrock: 'Valley',
  saddle: 'Saddle',
  himmelblau: 'Four basins',
  eggcrate: 'Egg crate',
};

/** Local minimum of x² + 25 sin²x nearest to π. */
const CRATE_MIN = 3.0196019;

export function makeLandscape(name: LandscapeName, options: { condition?: number } = {}): Landscape {
  const label = LANDSCAPE_LABELS[name];
  switch (name) {
    case 'ravine': {
      const k = Math.max(1, options.condition ?? 20);
      return {
        name,
        label,
        tex: 'f(x, y) = \\tfrac{1}{2}\\left(x^2 + ' + k + '\\,y^2\\right)',
        f: (x, y) => 0.5 * (x * x + k * y * y),
        grad: (x, y) => [x, k * y],
        minima: [[0, 0]],
        domain: { x: [-4, 4], y: [-4, 4] },
        start: [3, 1.5],
        suggestedLr: 0.09,
        tolerance: 1e-3,
      };
    }
    case 'rosenbrock':
      return {
        name,
        label,
        tex: 'f(x, y) = \\tfrac{1}{100}\\left((1 - x)^2 + 100\\,(y - x^2)^2\\right)',
        f: (x, y) => ((1 - x) * (1 - x) + 100 * (y - x * x) * (y - x * x)) / 100,
        grad: (x, y) => [(-2 * (1 - x) - 400 * x * (y - x * x)) / 100, (200 * (y - x * x)) / 100],
        minima: [[1, 1]],
        domain: { x: [-2, 2], y: [-1, 3] },
        start: [-1.2, 1],
        suggestedLr: 0.05,
        tolerance: 1e-5,
      };
    case 'saddle':
      return {
        name,
        label,
        tex: 'f(x, y) = \\tfrac{1}{2}x^2 + \\tfrac{1}{4}y^4 - \\tfrac{1}{2}y^2',
        f: (x, y) => 0.5 * x * x + 0.25 * y * y * y * y - 0.5 * y * y,
        grad: (x, y) => [x, y * y * y - y],
        minima: [
          [0, 1],
          [0, -1],
        ],
        domain: { x: [-3, 3], y: [-2, 2] },
        start: [-2, 0.02],
        suggestedLr: 0.1,
        tolerance: 1e-3,
      };
    case 'himmelblau':
      return {
        name,
        label,
        tex: 'f(x, y) = \\tfrac{1}{20}\\left((x^2 + y - 11)^2 + (x + y^2 - 7)^2\\right)',
        f: (x, y) => ((x * x + y - 11) ** 2 + (x + y * y - 7) ** 2) / 20,
        grad: (x, y) => {
          const a = x * x + y - 11;
          const b = x + y * y - 7;
          return [(4 * a * x + 2 * b) / 20, (2 * a + 4 * b * y) / 20];
        },
        minima: [
          [3, 2],
          [-2.805118, 3.131312],
          [-3.779310, -3.283186],
          [3.584428, -1.848126],
        ],
        domain: { x: [-5, 5], y: [-5, 5] },
        start: [0.5, -0.5],
        suggestedLr: 0.02,
        tolerance: 1e-3,
      };
    case 'eggcrate': {
      const spots = [0, CRATE_MIN, -CRATE_MIN];
      const minima: Array<[number, number]> = [];
      for (const x of spots) for (const y of spots) minima.push([x, y]);
      return {
        name,
        label,
        tex: 'f(x, y) = \\tfrac{1}{10}\\left(x^2 + y^2 + 25\\,(\\sin^2 x + \\sin^2 y)\\right)',
        f: (x, y) => (x * x + y * y + 25 * (Math.sin(x) ** 2 + Math.sin(y) ** 2)) / 10,
        grad: (x, y) => [(2 * x + 25 * Math.sin(2 * x)) / 10, (2 * y + 25 * Math.sin(2 * y)) / 10],
        minima,
        domain: { x: [-5, 5], y: [-5, 5] },
        start: [4.4, 4.4],
        suggestedLr: 0.1,
        tolerance: 1e-3,
      };
    }
    default:
      return {
        name: 'bowl',
        label,
        tex: 'f(x, y) = \\tfrac{1}{2}\\left(x^2 + y^2\\right)',
        f: (x, y) => 0.5 * (x * x + y * y),
        grad: (x, y) => [x, y],
        minima: [[0, 0]],
        domain: { x: [-4, 4], y: [-4, 4] },
        start: [3, 2.5],
        suggestedLr: 0.1,
        tolerance: 1e-3,
      };
  }
}

/** Central-difference gradient, for tests and sanity checks. */
export function numericGrad(f: (x: number, y: number) => number, x: number, y: number, h = 1e-5): [number, number] {
  return [(f(x + h, y) - f(x - h, y)) / (2 * h), (f(x, y + h) - f(x, y - h)) / (2 * h)];
}

/** The closest listed minimum and how far away it is. */
export function nearestMinimum(landscape: Landscape, x: number, y: number): { index: number; distance: number } {
  let index = -1;
  let distance = Infinity;
  landscape.minima.forEach(([mx, my], i) => {
    const d = Math.hypot(x - mx, y - my);
    if (d < distance) {
      distance = d;
      index = i;
    }
  });
  return { index, distance };
}
