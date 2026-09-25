/** The architecture graph: types, layout arithmetic and hit testing. No canvas code here. */

import type { Palette } from '../lib/viz/palette';

export interface Rect {
  x: number;
  y: number;
  w: number;
  h: number;
}

/** What a node is, which decides its shape and colour. */
export type ArchNodeKind = 'data' | 'input' | 'param' | 'op' | 'unit' | 'output' | 'loss';

export interface ArchNode {
  id: string;
  kind: ArchNodeKind;
  /** The symbol: 'x₁', 'Σ', 'ŷ'. Inside a circle, under a box. */
  label: string;
  /** A second line under a box, used for the data's shape: '60 × 2'. */
  value?: string;
  /** Column this node sits in. Columns lay out left to right. */
  column: number;
  /** Order within the column, top to bottom. */
  row: number;
  /** Total rows in this node's column, so a short column centres itself. */
  rows: number;
  /** Sit on the centre line; the rest of the column stacks around this node. */
  anchor?: boolean;
  /** Paint the node's own picture inside its box. */
  thumbnail?: (ctx: CanvasRenderingContext2D, rect: Rect, palette: Palette) => void;
  /** Signed value in [-1, 1] tinting the box. */
  tint?: number;
  /** 0..1, scales the box. */
  emphasis?: number;
  /** Drawn faded, for parts that are switched off. */
  muted?: boolean;
  /** The number it holds now, shown under the symbol on hover: '= 1.42', '60 × 1'. */
  readout?: string;
}

export interface ArchEdge {
  id: string;
  from: string;
  to: string;
  /** Signed weight: sets the colour (negative/positive) and the thickness. */
  weight?: number;
  /** The symbol shown on hover: 'w', 'ŷ', '∂L/∂w'. */
  symbol?: string;
  /** The number it carries now, shown under the symbol on hover. */
  readout?: string;
  /** A value travelling along this edge animates while the model runs. */
  active?: boolean;
}

export interface ArchColumn {
  /** Small caps heading above the column: 'INPUTS', 'HIDDEN 1', 'OUTPUT'. */
  title: string;
  /** Symbols or a count under the heading: 'wx + b', '60 points'. */
  subtitle?: string;
}

export interface ArchGraph {
  nodes: ArchNode[];
  edges: ArchEdge[];
  columns: ArchColumn[];
  /** One line under the diagram stating the architecture in symbols. */
  summary?: string;
  /** Largest |weight| in the graph, for scaling edge thickness consistently. */
  maxWeight?: number;
}

/** A node with its resolved pixel rectangle. */
export interface PlacedNode extends ArchNode {
  rect: Rect;
  cx: number;
  cy: number;
}

export interface ArchLayout {
  nodes: PlacedNode[];
  byId: Map<string, PlacedNode>;
  columnX: number[];
  columnWidth: number;
  /** Baseline of the column titles, just above the tallest column. */
  headerY: number;
  /** Top and bottom of the block of nodes, for anything drawn around it. */
  top: number;
  bottom: number;
  /** True when the boxes had to shrink below the point of showing text. */
  compact: boolean;
}

/** Preferred box size, shared across explainers. */
export const MAX_BOX = 96;

/** Height of the symbol strip drawn under a box. */
export const LABEL_STRIP = 26;

/** An op with no picture is a small circle. */
export function isCompact(node: Pick<ArchNode, 'kind' | 'thumbnail'>): boolean {
  return (node.kind === 'op' || node.kind === 'unit') && !node.thumbnail;
}

/** True when the symbol is drawn inside the shape rather than under it. */
export function labelInside(node: Pick<ArchNode, 'thumbnail'> & { rect: Rect }): boolean {
  return !node.thumbnail && node.rect.h >= 30;
}

export interface LayoutOptions {
  /** Outer padding in CSS pixels. */
  padX?: number;
  padTop?: number;
  padBottom?: number;
  /** Preferred box size before shrinking to fit. */
  maxBox?: number;
  minBox?: number;
  /** Maximum column pitch. */
  maxPitch?: number;
  /** Wire run kept free per column, so labels have room between boxes. */
  labelGap?: number;
  /** Space between boxes in a column. */
  rowGap?: number;
}

/** Place every node: columns spread across the width, nodes centred down each column. */
export function layoutArchitecture(
  graph: ArchGraph,
  width: number,
  height: number,
  options: LayoutOptions = {},
): ArchLayout {
  const {
    padX = 24,
    padTop = 44,
    padBottom = 30,
    maxBox = MAX_BOX,
    minBox = 18,
    maxPitch = 260,
    labelGap = 36,
    rowGap = 10,
  } = options;

  const columnCount = Math.max(1, graph.columns.length);
  const pad = width < 480 ? Math.min(padX, 12) : padX;
  const usableWidth = Math.max(1, width - pad * 2);
  const columnWidth = Math.min(maxPitch, usableWidth / columnCount);
  const startX = pad + (usableWidth - columnWidth * columnCount) / 2;

  const columnX: number[] = [];
  for (let i = 0; i < columnCount; i++) {
    columnX.push(startX + columnWidth * (i + 0.5));
  }

  // The tallest column decides the box size for everyone.
  let tallest = 1;
  for (const node of graph.nodes) tallest = Math.max(tallest, node.rows);

  const usableHeight = Math.max(1, height - padTop - padBottom);
  const perRow = usableHeight / tallest;
  const boxCap = Math.min(128, Math.max(maxBox, usableHeight * 0.22));
  const box = Math.max(
    minBox,
    Math.min(boxCap, perRow - rowGap, columnWidth - (columnWidth < 90 ? 24 : labelGap)),
  );
  const compact = box < 28;
  const centreY = padTop + usableHeight / 2;

  // Row pitch, capped so a short column reads as a group.
  const spacingFor = (rows: number, anchorRow: number | null): number => {
    if (rows <= 1) return 0;
    let spacing = Math.min(perRow, (usableHeight - box) / (rows - 1), box * 1.6);
    if (anchorRow !== null) {
      // The longer arm of an anchored column must still fit its half of the frame.
      const arm = Math.max(anchorRow, rows - 1 - anchorRow);
      if (arm > 0) spacing = Math.min(spacing, (usableHeight / 2 - box / 2) / arm);
    }
    return spacing;
  };

  const anchors = new Map<number, number>();
  for (const node of graph.nodes) if (node.anchor) anchors.set(node.column, node.row);

  const nodes: PlacedNode[] = [];
  const byId = new Map<string, PlacedNode>();
  let top = Infinity;
  let bottom = -Infinity;

  for (const node of graph.nodes) {
    const rows = Math.max(1, node.rows);
    const anchorRow = anchors.get(node.column) ?? null;
    const spacing = spacingFor(rows, anchorRow);
    const middle = anchorRow ?? (rows - 1) / 2;
    const cy = centreY + (node.row - middle) * spacing;
    const scale = node.emphasis === undefined ? 1 : 0.6 + 0.4 * clamp01(node.emphasis);
    const side = isCompact(node) ? Math.max(Math.min(box, 30), box * 0.5) : box * scale;
    const cx = columnX[Math.min(columnCount - 1, Math.max(0, node.column))];

    const placed: PlacedNode = {
      ...node,
      cx,
      cy,
      rect: { x: cx - side / 2, y: cy - side / 2, w: side, h: side },
    };
    nodes.push(placed);
    byId.set(node.id, placed);
    top = Math.min(top, placed.rect.y);
    bottom = Math.max(bottom, placed.rect.y + placed.rect.h);
  }
  if (nodes.length === 0) {
    top = centreY - box / 2;
    bottom = centreY + box / 2;
  }

  return {
    nodes,
    byId,
    columnX,
    columnWidth,
    headerY: Math.max(12, top - 26),
    top,
    bottom,
    compact,
  };
}

function clamp01(v: number): number {
  return v < 0 ? 0 : v > 1 ? 1 : v;
}

/** The node under a point, or null. Used for hover and click. */
export function hitTestNode(layout: ArchLayout, x: number, y: number): PlacedNode | null {
  for (let i = layout.nodes.length - 1; i >= 0; i--) {
    const n = layout.nodes[i];
    const pad = 4;
    if (
      x >= n.rect.x - pad &&
      x <= n.rect.x + n.rect.w + pad &&
      y >= n.rect.y - pad &&
      y <= n.rect.y + n.rect.h + pad
    ) {
      return n;
    }
  }
  return null;
}

/** Perpendicular distance from a point to a segment, the edge hit test. */
export function distanceToSegment(
  px: number,
  py: number,
  ax: number,
  ay: number,
  bx: number,
  by: number,
): number {
  const dx = bx - ax;
  const dy = by - ay;
  const lengthSquared = dx * dx + dy * dy;
  if (lengthSquared === 0) return Math.hypot(px - ax, py - ay);
  let t = ((px - ax) * dx + (py - ay) * dy) / lengthSquared;
  t = t < 0 ? 0 : t > 1 ? 1 : t;
  return Math.hypot(px - (ax + t * dx), py - (ay + t * dy));
}

export interface Point {
  x: number;
  y: number;
}

/** A cubic bezier from one node to another, plus where its hover plate sits. */
export interface EdgePath {
  start: Point;
  c1: Point;
  c2: Point;
  end: Point;
  /** Direction the arrowhead points at `end`, in radians. */
  angle: number;
  /** True for a return path, routed in an arc under the diagram. */
  backwards: boolean;
  label: Point;
}

/** Where the symbol strip under a node ends, or its box edge when the symbol is inside. */
function underside(node: PlacedNode): number {
  const strip = labelInside(node) ? 0 : node.value ? LABEL_STRIP + 4 : LABEL_STRIP - 8;
  return node.rect.y + node.rect.h + strip;
}

/** The curve an edge follows. Forward wires run left to right; a return path arcs under the diagram. */
export function edgePath(edge: ArchEdge, layout: ArchLayout): EdgePath | null {
  const from = layout.byId.get(edge.from);
  const to = layout.byId.get(edge.to);
  if (!from || !to) return null;

  if (to.cx < from.cx - 1) {
    let arrival = to;
    let lowest = -Infinity;
    let fromIsLowest = true;
    for (const node of layout.nodes) {
      if (node.column === to.column && node.cy > arrival.cy) arrival = node;
      if (node.column === from.column && node.cy > from.cy + 1) fromIsLowest = false;
      if (node.column >= to.column && node.column <= from.column) {
        lowest = Math.max(lowest, underside(node));
      }
    }
    // Out of the bottom when nothing sits under the source; off its right side otherwise.
    const start = fromIsLowest
      ? { x: from.cx, y: underside(from) + 4 }
      : { x: from.rect.x + from.rect.w + 3, y: from.cy };
    const end = { x: arrival.cx, y: underside(arrival) + 4 };
    // Pull the control points down until the arc's lowest point clears the block.
    const floorY = lowest + 22;
    const floor = (8 * floorY - start.y - end.y) / 6;
    return {
      start,
      c1: { x: fromIsLowest ? start.x : start.x + 6, y: floor },
      c2: { x: end.x, y: floor },
      end,
      angle: -Math.PI / 2,
      backwards: true,
      label: { x: (start.x + end.x) / 2, y: floorY },
    };
  }

  if (Math.abs(to.cx - from.cx) < 1) {
    const down = to.cy > from.cy;
    const start = down
      ? { x: from.cx, y: Math.min(underside(from), to.rect.y - 12) }
      : { x: from.cx, y: from.rect.y - 3 };
    const end = down
      ? { x: to.cx, y: to.rect.y - 3 }
      : { x: to.cx, y: to.rect.y + to.rect.h + 3 };
    const midY = (start.y + end.y) / 2;
    return {
      start,
      c1: { x: start.x, y: midY },
      c2: { x: end.x, y: midY },
      end,
      angle: down ? Math.PI / 2 : -Math.PI / 2,
      backwards: false,
      label: { x: start.x, y: midY },
    };
  }

  const start = { x: from.rect.x + from.rect.w + 3, y: from.cy };
  const end = { x: to.rect.x - 3, y: to.cy };
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

/** Points along the curve, for hit testing. */
export function sampleEdge(path: EdgePath, steps = 14): Point[] {
  const points: Point[] = [];
  for (let i = 0; i <= steps; i++) {
    const t = i / steps;
    const u = 1 - t;
    const a = u * u * u;
    const b = 3 * u * u * t;
    const c = 3 * u * t * t;
    const d = t * t * t;
    points.push({
      x: a * path.start.x + b * path.c1.x + c * path.c2.x + d * path.end.x,
      y: a * path.start.y + b * path.c1.y + c * path.c2.y + d * path.end.y,
    });
  }
  return points;
}

/** The edge under a point, within `tolerance` pixels, or null. */
export function hitTestEdge(
  graph: ArchGraph,
  layout: ArchLayout,
  x: number,
  y: number,
  tolerance = 6,
): ArchEdge | null {
  let best: ArchEdge | null = null;
  let bestDistance = tolerance;
  for (const edge of graph.edges) {
    const path = edgePath(edge, layout);
    if (!path) continue;
    const points = sampleEdge(path);
    for (let i = 1; i < points.length; i++) {
      const d = distanceToSegment(
        x,
        y,
        points[i - 1].x,
        points[i - 1].y,
        points[i].x,
        points[i].y,
      );
      if (d < bestDistance) {
        bestDistance = d;
        best = edge;
      }
    }
  }
  return best;
}

/* ---------------- builder ---------------- */

/** Builds an ArchGraph without bookkeeping row counts by hand. */
export class GraphBuilder {
  private readonly nodes: ArchNode[] = [];
  private readonly edges: ArchEdge[] = [];
  private readonly columns: ArchColumn[] = [];
  private readonly counts = new Map<number, number>();

  column(title: string, subtitle?: string): number {
    this.columns.push({ title, subtitle });
    return this.columns.length - 1;
  }

  node(column: number, node: Omit<ArchNode, 'column' | 'row' | 'rows'>): string {
    const row = this.counts.get(column) ?? 0;
    this.counts.set(column, row + 1);
    this.nodes.push({ ...node, column, row, rows: 1 });
    return node.id;
  }

  edge(from: string, to: string, extra: Omit<ArchEdge, 'id' | 'from' | 'to'> = {}): void {
    this.edges.push({ id: from + '->' + to, from, to, ...extra });
  }

  build(summary?: string): ArchGraph {
    // Backfill each node's row count now that every column is known.
    for (const node of this.nodes) {
      node.rows = this.counts.get(node.column) ?? 1;
    }
    let maxWeight = 0;
    for (const edge of this.edges) {
      if (edge.weight !== undefined) maxWeight = Math.max(maxWeight, Math.abs(edge.weight));
    }
    return {
      nodes: this.nodes,
      edges: this.edges,
      columns: this.columns,
      summary,
      maxWeight: maxWeight || 1,
    };
  }
}
