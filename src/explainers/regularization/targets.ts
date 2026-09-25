/** What the pointer can be on, shared by the map, the charts and the inspector. */

/** The stations of the model map, in reading order. */
export type ModelNode = 'data' | 'features' | 'sum' | 'prediction' | 'error' | 'penalty' | 'objective';

export const MODEL_NODES: ModelNode[] = ['data', 'features', 'sum', 'prediction', 'error', 'penalty', 'objective'];

export function isModelNode(id: string): id is ModelNode {
  return (MODEL_NODES as string[]).includes(id);
}

export type RegTarget =
  | { kind: 'feature'; index: number }
  | { kind: 'point'; index: number }
  | { kind: 'node'; id: ModelNode }
  /** A mark on the weight plane: the answer, the other shape's answer, the least-squares cross. */
  | { kind: 'mark'; which: 'now' | 'other' | 'centre' }
  /** The fitted curve at one x. */
  | { kind: 'curve'; x: number };

export function targetKey(target: RegTarget | null): string {
  if (!target) return '';
  switch (target.kind) {
    case 'feature':
    case 'point':
      return target.kind + ':' + target.index;
    case 'node':
      return 'node:' + target.id;
    case 'mark':
      return 'mark:' + target.which;
    case 'curve':
      return 'curve:' + target.x.toFixed(2);
  }
}

export function sameTarget(a: RegTarget | null, b: RegTarget | null): boolean {
  return targetKey(a) === targetKey(b);
}

/** The feature under the pointer or open, if any. */
export function featureOf(target: RegTarget | null): number | null {
  return target && target.kind === 'feature' ? target.index : null;
}

export function nodeOf(target: RegTarget | null): ModelNode | null {
  return target && target.kind === 'node' ? target.id : null;
}

/** A lesson spotlight on the map or a pane. */
export interface Spot {
  target: RegTarget | { kind: 'pane'; id: 'fit' | 'geometry' };
  label: string;
}

/** Decodes a lesson's custom focus id: `model:<station>`, `feature:<j>`, `pane:<fit|geometry>`. */
export function spotOf(id: string, label: string): Spot | null {
  const [kind, rest] = id.split(':');
  if (kind === 'model' && rest && isModelNode(rest)) return { target: { kind: 'node', id: rest }, label };
  if (kind === 'feature' && rest !== undefined && Number.isInteger(Number(rest))) return { target: { kind: 'feature', index: Number(rest) }, label };
  if (kind === 'pane' && (rest === 'fit' || rest === 'geometry')) return { target: { kind: 'pane', id: rest }, label };
  return null;
}
