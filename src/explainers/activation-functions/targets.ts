/** Everything a pointer can land on across the three views, and how a lesson names it. */

export type DiagramTarget =
  | { kind: 'function'; which: 'activation' | 'loss'; x?: number }
  | { kind: 'layer'; index: number }
  | { kind: 'point'; index: number }
  | { kind: 'input'; x: number; y: number }
  | { kind: 'unit'; index: number; layer?: number }
  | { kind: 'units'; layer: number }
  | { kind: 'output' };

/** The hidden layer a unit target belongs to, from 1. */
export function unitLayer(t: { layer?: number }): number {
  return t.layer ?? 1;
}

/** A lesson's spotlight on the canvas: the target and the label drawn beside it. */
export interface Spot {
  target: DiagramTarget;
  label: string;
}

/** Same thing, ignoring where along a curve the pointer sits. */
export function sameTarget(a: DiagramTarget | null, b: DiagramTarget | null): boolean {
  if (a === null || b === null) return a === b;
  if (a.kind !== b.kind) return false;
  switch (a.kind) {
    case 'function':
      return a.which === (b as typeof a).which;
    case 'output':
      return true;
    case 'input':
      return a.x === (b as typeof a).x && a.y === (b as typeof a).y;
    case 'unit':
      return a.index === (b as typeof a).index && unitLayer(a) === unitLayer(b as typeof a);
    case 'units':
      return a.layer === (b as typeof a).layer;
    default:
      return a.index === (b as typeof a).index;
  }
}

export function targetKey(t: DiagramTarget | null): string {
  if (!t) return '';
  switch (t.kind) {
    case 'function':
      return 'function:' + t.which + (t.x === undefined ? '' : '@' + t.x.toFixed(2));
    case 'input':
      return 'input:' + t.x.toFixed(3) + ',' + t.y.toFixed(3);
    case 'unit':
      return 'unit:' + (unitLayer(t) === 1 ? '' : unitLayer(t) + '.') + t.index;
    case 'units':
      return 'units:' + t.layer;
    case 'output':
      return 'output';
    default:
      return t.kind + ':' + t.index;
  }
}

/** The id a lesson uses: 'function:activation', 'layer:2', 'unit:0', 'unit:2.1' for a deeper layer, 'units:1' for a layer's tiles, 'point:5'. */
export function encodeTarget(t: DiagramTarget): string {
  switch (t.kind) {
    case 'function':
      return 'function:' + t.which;
    case 'input':
      return 'input:' + t.x.toFixed(3) + ',' + t.y.toFixed(3);
    case 'unit':
      return 'unit:' + (unitLayer(t) === 1 ? '' : unitLayer(t) + '.') + t.index;
    case 'units':
      return 'units:' + t.layer;
    case 'output':
      return 'output';
    default:
      return t.kind + ':' + t.index;
  }
}

export function decodeTarget(id: string): DiagramTarget | null {
  const [kind, rest] = id.split(':');
  switch (kind) {
    case 'function':
      return rest === 'activation' || rest === 'loss' ? { kind, which: rest } : null;
    case 'output':
      return { kind };
    case 'units': {
      const layer = Number(rest);
      return Number.isInteger(layer) && layer >= 1 ? { kind, layer } : null;
    }
    case 'layer':
    case 'point': {
      const index = Number(rest);
      return Number.isInteger(index) && index >= 0 ? { kind, index } : null;
    }
    case 'unit': {
      const [a, b] = (rest ?? '').split('.').map(Number);
      if (b === undefined) return Number.isInteger(a) && a >= 0 ? { kind, index: a } : null;
      return Number.isInteger(a) && a >= 1 && Number.isInteger(b) && b >= 0 ? { kind, index: b, layer: a } : null;
    }
    case 'input': {
      const [x, y] = (rest ?? '').split(',').map(Number);
      return Number.isFinite(x) && Number.isFinite(y) ? { kind, x, y } : null;
    }
    default:
      return null;
  }
}
