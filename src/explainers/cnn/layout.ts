/** Where every part of the scene sits: columns and boxes for the 2D view, sheets and discs in world units for the 3D view. */

import type { EdgePath, Rect } from '../../explainer/architecture';
import type { Vec3 } from '../../lib/viz/volume';
import type { CnnLayer, CnnLayerKind, CnnScene } from './scene';

/* ---------------- 2D ---------------- */

export type Shape = 'box' | 'circle' | 'strip';

export interface Placed {
  id: string;
  shape: Shape;
  rect: Rect;
  cx: number;
  cy: number;
}

export interface Column {
  layer: CnnLayer;
  x: number;
  width: number;
}

export interface Layout2D {
  nodes: Placed[];
  byId: Map<string, Placed>;
  columns: Column[];
  headerY: number;
  /** Boxes too small for a label under them. */
  compact: boolean;
}

const WEIGHT: Record<CnnLayerKind, number> = { image: 1, filters: 0.62, maps: 1, pooled: 0.9, flat: 0.42, dense: 0.62, scores: 0.85 };
const LABEL = 18;

function clamp(v: number, lo: number, hi: number): number {
  return Math.max(lo, Math.min(hi, v));
}

export function layoutScene2D(scene: CnnScene, width: number, height: number): Layout2D {
  const pad = width < 480 ? 10 : 20;
  const padTop = 44;
  const padBottom = 30;
  const usableW = Math.max(1, width - pad * 2);
  const usableH = Math.max(1, height - padTop - padBottom);
  const centreY = padTop + usableH / 2;
  const totalWeight = scene.layers.reduce((a, l) => a + WEIGHT[l.kind], 0);
  const unit = Math.min(usableW / totalWeight, 230);
  const startX = pad + (usableW - unit * totalWeight) / 2;

  const columns: Column[] = [];
  let acc = 0;
  for (const layer of scene.layers) {
    const w = WEIGHT[layer.kind] * unit;
    columns.push({ layer, x: startX + acc + w / 2, width: w });
    acc += w;
  }

  let stackRows = 1;
  for (const layer of scene.layers) if (layer.kind === 'maps' || layer.kind === 'pooled') stackRows = Math.max(stackRows, layer.nodes.length);
  const rowGap = 8 + LABEL;
  const box = clamp(Math.min(112, (usableH - (stackRows - 1) * rowGap) / stackRows, WEIGHT.maps * unit - 24), 14, 112);
  const compact = box < 30;
  const kernel = clamp(box * 0.55, 12, 48);
  const image = Math.min(box * 1.3, WEIGHT.image * unit - 16, 136);
  const pooled = box * 0.8;
  const score = clamp(box * 0.75, 22, 56);

  const nodes: Placed[] = [];
  const byId = new Map<string, Placed>();
  const place = (id: string, shape: Shape, cx: number, cy: number, w: number, h: number) => {
    const placed: Placed = { id, shape, rect: { x: cx - w / 2, y: cy - h / 2, w, h }, cx, cy };
    nodes.push(placed);
    byId.set(id, placed);
  };
  const stack = (column: Column, sizes: number[], gap: number, shape: Shape, widths?: number[]) => {
    const total = sizes.reduce((a, s) => a + s, 0) + gap * (sizes.length - 1);
    let y = centreY - total / 2;
    column.layer.nodes.forEach((id, i) => {
      const s = sizes[i];
      place(id, shape, column.x, y + s / 2, widths ? widths[i] : s, s);
      y += s + gap;
    });
  };

  for (const column of columns) {
    const n = column.layer.nodes.length;
    switch (column.layer.kind) {
      case 'image':
        stack(column, [image], 0, 'box');
        break;
      case 'filters':
        stack(column, new Array(n).fill(kernel), rowGap + (box - kernel), 'box');
        break;
      case 'maps':
        stack(column, new Array(n).fill(box), rowGap, 'box');
        break;
      case 'pooled':
        stack(column, new Array(n).fill(pooled), rowGap + (box - pooled), 'box');
        break;
      case 'flat': {
        const h = Math.min(usableH * 0.72, 260);
        stack(column, [h], 0, 'strip', [clamp(unit * 0.16, 12, 22)]);
        break;
      }
      case 'dense': {
        // Units keep in proportion with the pictures once the stage is narrow.
        const d = clamp(Math.min((usableH - (n - 1) * 4) / n, box * 0.55), 6, 18);
        stack(column, new Array(n).fill(d), d < 10 ? 2 : 4, 'circle');
        break;
      }
      case 'scores':
        stack(column, new Array(n).fill(score), Math.min(14, score * 0.5), 'box');
        break;
      default:
        break;
    }
  }

  let top = Infinity;
  for (const node of nodes) top = Math.min(top, node.rect.y);
  return { nodes, byId, columns, headerY: Math.max(12, Math.min(top - 26, padTop - 18)), compact };
}

/** A wire from the right of one box to the left of the next. */
export function wirePath(from: Placed, to: Placed): EdgePath {
  const start = { x: from.rect.x + from.rect.w + 2, y: from.cy };
  const end = { x: to.rect.x - 2, y: to.cy };
  const midX = (start.x + end.x) / 2;
  return {
    start,
    c1: { x: midX, y: start.y },
    c2: { x: midX, y: end.y },
    end,
    angle: 0,
    backwards: false,
    label: { x: midX, y: (start.y + end.y) / 2 },
  };
}

/* ---------------- 3D ---------------- */

export interface Sheet3D {
  id: string;
  centre: Vec3;
  w: number;
  h: number;
}

export interface Disc3D {
  id: string;
  at: Vec3;
  /** Radius as a share of the image's side. */
  r: number;
  label?: string;
}

export interface LayerBounds {
  layer: CnnLayer;
  x: number;
  /** Where the title sits: above the layer, or under it when the space above is taken. */
  titleAt: number;
  below: boolean;
  depth: number;
}

export interface Layout3D {
  sheets: Sheet3D[];
  discs: Disc3D[];
  byId: Map<string, Sheet3D | Disc3D>;
  layers: LayerBounds[];
  /** Every corner, for fitting the view. */
  points: Vec3[];
  floor: { x0: number; x1: number; z0: number; z1: number; y: number };
}

/** Sheets face the viewer, stacks recede along z, the pipeline runs along x. Kernels float above their maps. */
export function layoutScene3D(scene: CnnScene, size: number): Layout3D {
  const KERNEL = 0.28;
  // Small sheets stack closer, so a pooled stack does not sprawl.
  const spacing = (side: number) => Math.max(0.3, Math.min(0.5, 0.55 * side));
  const sheets: Sheet3D[] = [];
  const discs: Disc3D[] = [];
  const byId = new Map<string, Sheet3D | Disc3D>();
  const layers: LayerBounds[] = [];
  const points: Vec3[] = [];
  let cursor = 0;
  let maxDepth = 0;

  const sheet = (id: string, centre: Vec3, w: number, h: number) => {
    const s = { id, centre, w, h };
    sheets.push(s);
    byId.set(id, s);
    for (const [sx, sy] of [
      [-1, 1],
      [1, 1],
      [1, -1],
      [-1, -1],
    ]) points.push({ x: centre.x + (sx * w) / 2, y: centre.y + (sy * h) / 2, z: centre.z });
  };
  const disc = (id: string, at: Vec3, r: number, label?: string) => {
    const d = { id, at, r, label };
    discs.push(d);
    byId.set(id, d);
    points.push({ x: at.x - r, y: at.y - r, z: at.z }, { x: at.x + r, y: at.y + r, z: at.z });
  };
  const sideOf = (id: string) => {
    const node = scene.byId.get(id);
    return node?.picture ? Math.max(0.42, node.picture.rows / size) : 0.42;
  };

  const gapBefore = (kind: CnnLayerKind): number => {
    switch (kind) {
      case 'flat':
        return 0.42;
      case 'dense':
        return 0.5;
      case 'scores':
        return 0.55;
      default:
        return 0.38;
    }
  };

  scene.layers.forEach((layer, index) => {
    const n = layer.nodes.length;
    const maps = layer.kind === 'filters' ? scene.layers[index + 1] : layer;
    const dz = spacing(maps && maps.nodes.length > 0 ? sideOf(maps.nodes[0]) : 1);
    const z0 = ((n - 1) * dz) / 2;
    if (layer.kind === 'filters') {
      // Over the maps they produce, so they take no room along the pipeline.
      const side = maps ? sideOf(maps.nodes[0]) : 0.5;
      const x = cursor + gapBefore('maps') + side / 2;
      const y = side / 2 + 0.22 + KERNEL / 2;
      layer.nodes.forEach((id, i) => sheet(id, { x, y, z: z0 - i * dz }, KERNEL, KERNEL));
      layers.push({ layer, x, titleAt: y + KERNEL / 2 + 0.16, below: false, depth: (n - 1) * dz });
      return;
    }
    let half = 0.5;
    let titleAt = 0.66;
    let below = false;
    let depth = 0;
    switch (layer.kind) {
      case 'maps':
      case 'pooled':
        half = sideOf(layer.nodes[0]) / 2;
        depth = (n - 1) * dz;
        break;
      case 'flat':
        half = 0.08;
        break;
      case 'dense':
        half = 0.12;
        break;
      case 'scores':
        half = 0.2;
        break;
      default:
        break;
    }
    const x = index === 0 ? half : cursor + gapBefore(layer.kind) + half;
    switch (layer.kind) {
      case 'image':
        sheet(layer.nodes[0], { x, y: 0, z: 0 }, 1, 1);
        break;
      case 'maps':
      case 'pooled': {
        const side = half * 2;
        layer.nodes.forEach((id, i) => sheet(id, { x, y: 0, z: z0 - i * dz }, side, side));
        // The kernels sit above a maps layer, so its title goes underneath.
        below = layer.kind === 'maps';
        titleAt = below ? -side / 2 - 0.12 : side / 2 + 0.16;
        break;
      }
      case 'flat':
        sheet(layer.nodes[0], { x, y: 0, z: 0 }, 0.14, 1.2);
        titleAt = 0.76;
        break;
      case 'dense': {
        const span = Math.min(1.5, 0.11 * n);
        layer.nodes.forEach((id, i) => disc(id, { x, y: n > 1 ? span / 2 - (span * i) / (n - 1) : 0, z: 0 }, 0.045));
        titleAt = span / 2 + 0.2;
        break;
      }
      case 'scores': {
        const span = Math.min(1.3, 0.3 * n);
        layer.nodes.forEach((id, i) => disc(id, { x, y: n > 1 ? span / 2 - (span * i) / (n - 1) : 0, z: 0 }, 0.13, scene.byId.get(id)?.label));
        titleAt = span / 2 + 0.3;
        break;
      }
      default:
        break;
    }
    layers.push({ layer, x, titleAt, below, depth });
    maxDepth = Math.max(maxDepth, depth);
    cursor = x + half + (depth > 0 ? depth * 0.22 : 0);
  });

  const zMax = maxDepth / 2 + 0.6;
  return { sheets, discs, byId, layers, points, floor: { x0: -0.3, x1: cursor + 0.3, z0: -zMax, z1: zMax, y: -0.62 } };
}

/** The four world corners of a pixel rectangle on a sheet, top-left first. */
export function sheetCells(sheet: Sheet3D, cols: number, rows: number, x: number, y: number, w: number, h: number): [Vec3, Vec3, Vec3, Vec3] {
  const left = sheet.centre.x - sheet.w / 2;
  const top = sheet.centre.y + sheet.h / 2;
  const cw = sheet.w / cols;
  const ch = sheet.h / rows;
  const z = sheet.centre.z + 0.002;
  return [
    { x: left + x * cw, y: top - y * ch, z },
    { x: left + (x + w) * cw, y: top - y * ch, z },
    { x: left + (x + w) * cw, y: top - (y + h) * ch, z },
    { x: left + x * cw, y: top - (y + h) * ch, z },
  ];
}

export function sheetCorners(sheet: Sheet3D): [Vec3, Vec3, Vec3, Vec3] {
  return sheetCells(sheet, 1, 1, 0, 0, 1, 1);
}
