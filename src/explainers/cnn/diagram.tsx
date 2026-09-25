/** The network for one probe image: columns of pictures, or a turntable of sheets. Hover reads, click docks the card. */

import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import type { ReactNode } from 'react';

import { distanceToSegment, sampleEdge } from '../../explainer/architecture';
import type { EdgePath, Rect } from '../../explainer/architecture';
import { Chart } from '../../explainer/components/Chart';
import type { DrawArgs } from '../../explainer/components/Chart';
import { DetailOverlay } from '../../explainer/components/Detail';
import type { ArchSelection } from '../../explainer/components/ArchitectureView';
import { DIM, drawHoverPlate, measurePlate, placePlate, strokeOutline, strokeWire, underWire } from '../../explainer/diagramStyle';
import type { WireEmphasis } from '../../explainer/diagramStyle';
import { useDiagramKeys } from '../../explainer/useDiagramKeys';
import { useElementSize } from '../../explainer/useElementSize';
import type { CnnCache, CnnParams } from '../../lib/ml/conv';
import { FONT_STACK, MONO_STACK, roundRect } from '../../lib/viz/canvas';
import { rgba } from '../../lib/viz/palette';
import type { Palette } from '../../lib/viz/palette';
import { drawPixelGrid } from '../../lib/viz/plots';
import {
  DEFAULT_VOLUME_VIEW,
  dragVolumeView,
  drawVolume,
  drawVolumeFloor,
  fitVolume,
  hitVolume,
  makeVolumeProjector,
  polygonBox,
  volumeViewKey,
} from '../../lib/viz/volume';
import type { PlacedItem, Vec3, VolumeItem, VolumeView } from '../../lib/viz/volume';
import { layoutScene2D, layoutScene3D, sheetCells, sheetCorners, wirePath } from './layout';
import type { Layout2D, Layout3D, Placed, Sheet3D } from './layout';
import { NODE, patchOf } from './scene';
import type { CnnEdge, CnnNode, CnnScene, Picture, Window } from './scene';

export type ArchView = '2d' | '3d';

export interface CnnDiagramProps {
  scene: CnnScene;
  params: CnnParams;
  cache: CnnCache;
  view: ArchView;
  selection: ArchSelection | null;
  onSelect: (selection: ArchSelection | null) => void;
  /** A lesson's spotlight, drawn like a hover until the pointer takes over. */
  spot?: ArchSelection | null;
  /** The cell whose receptive field is drawn. */
  window: Window | null;
  renderDetail?: (selection: ArchSelection, open: (next: ArchSelection) => void) => ReactNode;
  description: string;
  redrawKey?: unknown;
  phase?: number;
}

function same(a: ArchSelection | null, b: ArchSelection | null): boolean {
  if (!a || !b) return a === b;
  return a.kind === b.kind && a.id === b.id;
}

const CLICK_SLOP = 3;

/** The two hover lines of a component: its symbol, then the number it holds. */
function plateLines(scene: CnnScene, target: ArchSelection): string[] {
  if (target.kind === 'node') {
    const node = scene.byId.get(target.id);
    return node ? [node.label, node.readout] : [];
  }
  const edge = scene.edges.find((e) => e.id === target.id);
  return edge ? [edge.symbol ?? '', edge.readout ?? ''].filter(Boolean) : [];
}

/** Which wires recede and which light up around the focused component. */
function emphasisOf(edge: CnnEdge, focus: ArchSelection | null, hot: ArchSelection | null): WireEmphasis {
  if (hot?.kind === 'edge' && hot.id === edge.id) return 'hot';
  if (focus?.kind === 'edge') return focus.id === edge.id ? 'hot' : 'dim';
  if (focus?.kind === 'node') return edge.from === focus.id || edge.to === focus.id ? 'lit' : 'dim';
  return 'plain';
}

function wireStyle(edge: CnnEdge, maxWeight: number, palette: Palette): { colour: string; alpha: number; width: number } {
  if (edge.weight !== undefined) {
    const m = Math.min(1, Math.abs(edge.weight) / maxWeight);
    return { colour: edge.weight < 0 ? palette.negative : palette.positive, alpha: 0.4 + 0.55 * m, width: 1 + 2.6 * m };
  }
  if (!edge.symbol) return { colour: palette.textMuted, alpha: 0.28, width: 1 };
  return { colour: palette.textMuted, alpha: 0.6, width: 1.4 };
}

/** The outline of a class box: the call in green or orange once trained, a plain guess before, quiet otherwise. */
function callColour(scene: CnnScene, node: CnnNode, palette: Palette): string {
  if (!node.emphasis) return rgba(palette.accent, 0.45);
  if (!scene.trained) return palette.textMuted;
  return node.index === scene.label ? palette.green : palette.orange;
}

/** The nodes a stage reads: the image, or the previous stage's outputs. */
function inputsOf(scene: CnnScene, stage: number): CnnNode[] {
  if (stage === 0) return [scene.byId.get(NODE.image)!].filter(Boolean);
  const previous = scene.layers.filter((l) => l.stage === stage - 1 && (l.kind === 'pooled' || l.kind === 'maps'));
  const layer = previous.find((l) => l.kind === 'pooled') ?? previous[0];
  return layer ? layer.nodes.map((id) => scene.byId.get(id)!).filter(Boolean) : [];
}

/** The rectangles a window links: the block read on each input and the cell written. */
function windowRects(scene: CnnScene, params: CnnParams, cache: CnnCache, window: Window): { inputs: Array<{ node: CnnNode; x: number; y: number; w: number; h: number }>; output: { node: CnnNode; x: number; y: number; w: number; h: number } } | null {
  if (window.kind === 'pool') {
    const map = scene.byId.get(NODE.map(window.stage, window.index));
    const pool = scene.byId.get(NODE.pool(window.stage, window.index));
    if (!map || !pool) return null;
    return { inputs: [{ node: map, x: 2 * window.x, y: 2 * window.y, w: 2, h: 2 }], output: { node: pool, x: window.x, y: window.y, w: 1, h: 1 } };
  }
  const map = scene.byId.get(NODE.map(window.stage, window.index));
  const patch = patchOf(params, cache, window);
  if (!map || !patch) return null;
  const inputs = inputsOf(scene, window.stage).map((node) => ({ node, x: patch.x, y: patch.y, w: patch.k, h: patch.k }));
  return { inputs, output: { node: map, x: window.x, y: window.y, w: 1, h: 1 } };
}

/* ---------------- textures ---------------- */

/** A picture as a bitmap the size of its grid, for the sheets. */
function paintTexture(store: Map<string, HTMLCanvasElement>, id: string, picture: Picture, palette: Palette): HTMLCanvasElement | null {
  let canvas = store.get(id);
  if (!canvas) {
    canvas = document.createElement('canvas');
    store.set(id, canvas);
  }
  if (canvas.width !== picture.cols || canvas.height !== picture.rows) {
    canvas.width = picture.cols;
    canvas.height = picture.rows;
  }
  const ctx = canvas.getContext('2d');
  if (!ctx) return null;
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.clearRect(0, 0, picture.cols, picture.rows);
  drawPixelGrid(ctx, { x: 0, y: 0, width: picture.cols, height: picture.rows }, picture.values, picture.cols, picture.rows, palette, { ramp: picture.ramp, gap: 0 });
  return canvas;
}

/* ---------------- 2D ---------------- */

/** The picture's cell grid inside a box, so windows land on the right pixels. */
function pictureRect(placed: Placed, picture: Picture): { x: number; y: number; cell: number } {
  const inner = { x: placed.rect.x + 2, y: placed.rect.y + 2, w: placed.rect.w - 4, h: placed.rect.h - 4 };
  const cell = Math.min(inner.w / picture.cols, inner.h / picture.rows);
  return { x: inner.x + (inner.w - cell * picture.cols) / 2, y: inner.y + (inner.h - cell * picture.rows) / 2, cell };
}

function cellRect(placed: Placed, picture: Picture, x: number, y: number, w: number, h: number): Rect {
  const grid = pictureRect(placed, picture);
  const x0 = Math.max(0, x);
  const y0 = Math.max(0, y);
  const x1 = Math.min(picture.cols, x + w);
  const y1 = Math.min(picture.rows, y + h);
  return { x: grid.x + x0 * grid.cell, y: grid.y + y0 * grid.cell, w: Math.max(0, x1 - x0) * grid.cell, h: Math.max(0, y1 - y0) * grid.cell };
}

function drawHeaders2D(ctx: CanvasRenderingContext2D, layout: Layout2D, palette: Palette): void {
  ctx.save();
  ctx.textAlign = 'center';
  ctx.textBaseline = 'alphabetic';
  for (const column of layout.columns) {
    ctx.font = '700 10px ' + FONT_STACK;
    ctx.fillStyle = palette.textMuted;
    ctx.fillText(column.layer.title.toUpperCase(), column.x, layout.headerY, column.width - 6);
    ctx.font = '10px ' + MONO_STACK;
    ctx.fillStyle = palette.textFaint;
    if (ctx.measureText(column.layer.subtitle).width <= column.width - 4) ctx.fillText(column.layer.subtitle, column.x, layout.headerY + 13);
  }
  ctx.restore();
}

function drawNode2D(ctx: CanvasRenderingContext2D, scene: CnnScene, node: CnnNode, placed: Placed, layout: Layout2D, palette: Palette, state: 'hover' | 'open' | 'spot' | null): void {
  const { rect } = placed;
  const outline = () => {
    ctx.beginPath();
    if (placed.shape === 'circle') ctx.arc(placed.cx, placed.cy, rect.w / 2, 0, Math.PI * 2);
    else roundRect(ctx, rect.x, rect.y, rect.w, rect.h, Math.max(2, rect.w * 0.06));
  };
  ctx.save();
  outline();
  if (node.picture) {
    ctx.fillStyle = palette.surfaceAlt;
    ctx.fill();
    ctx.save();
    ctx.clip();
    const grid = pictureRect(placed, node.picture);
    drawPixelGrid(ctx, { x: grid.x, y: grid.y, width: grid.cell * node.picture.cols, height: grid.cell * node.picture.rows }, node.picture.values, node.picture.cols, node.picture.rows, palette, { ramp: node.picture.ramp, gap: 0 });
    ctx.restore();
    outline();
    ctx.strokeStyle = rgba(palette.accent, 0.35);
    ctx.lineWidth = 1;
    ctx.stroke();
  } else if (node.kind === 'unit') {
    ctx.fillStyle = palette.surface;
    ctx.fill();
    ctx.fillStyle = rgba(palette.positive, 0.1 + 0.8 * (node.value ?? 0));
    ctx.fill();
    ctx.strokeStyle = rgba(palette.accent, 0.5);
    ctx.lineWidth = 1;
    ctx.stroke();
  } else {
    const p = node.value ?? 0;
    ctx.fillStyle = palette.surface;
    ctx.fill();
    ctx.fillStyle = rgba(palette.accent, 0.08 + 0.72 * p);
    ctx.fill();
    ctx.strokeStyle = callColour(scene, node, palette);
    ctx.lineWidth = node.emphasis ? 2.4 : 1;
    ctx.stroke();
    ctx.fillStyle = palette.text;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.font = '600 ' + Math.max(11, Math.round(rect.h * 0.45)) + 'px ' + FONT_STACK;
    ctx.fillText(node.label, placed.cx, placed.cy + 1);
    if (node.index === scene.label && !layout.compact) {
      ctx.font = '600 9px ' + FONT_STACK;
      ctx.fillStyle = palette.textMuted;
      ctx.textBaseline = 'top';
      ctx.fillText('true', placed.cx, rect.y + rect.h + 4);
    }
  }
  if (state) strokeOutline(ctx, outline, state, palette);
  ctx.restore();

  if (node.picture && !layout.compact && placed.shape !== 'strip') {
    ctx.save();
    ctx.textAlign = 'center';
    ctx.textBaseline = 'top';
    ctx.font = '600 10px ' + FONT_STACK;
    ctx.fillStyle = palette.textMuted;
    ctx.fillText(node.label, placed.cx, rect.y + rect.h + 4);
    ctx.restore();
  }
}

/** The block a cell reads, drawn on each input, joined to the cell it writes. */
function drawWindow2D(ctx: CanvasRenderingContext2D, scene: CnnScene, params: CnnParams, cache: CnnCache, layout: Layout2D, window: Window, palette: Palette): void {
  const rects = windowRects(scene, params, cache, window);
  if (!rects) return;
  const outPlaced = layout.byId.get(rects.output.node.id);
  if (!outPlaced || !rects.output.node.picture) return;
  const out = cellRect(outPlaced, rects.output.node.picture, rects.output.x, rects.output.y, rects.output.w, rects.output.h);
  ctx.save();
  ctx.lineWidth = 1.5;
  ctx.strokeStyle = palette.accent;
  for (const input of rects.inputs) {
    const placed = layout.byId.get(input.node.id);
    if (!placed || !input.node.picture) continue;
    const r = cellRect(placed, input.node.picture, input.x, input.y, input.w, input.h);
    ctx.strokeRect(r.x, r.y, r.w, r.h);
    ctx.save();
    ctx.strokeStyle = rgba(palette.accent, 0.55);
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(r.x + r.w, r.y);
    ctx.lineTo(out.x, out.y);
    ctx.moveTo(r.x + r.w, r.y + r.h);
    ctx.lineTo(out.x, out.y + out.h);
    ctx.stroke();
    ctx.restore();
  }
  ctx.strokeRect(out.x - 0.5, out.y - 0.5, out.w + 1, out.h + 1);
  ctx.restore();
}

function drawFooter(ctx: CanvasRenderingContext2D, text: string, width: number, height: number, palette: Palette): void {
  ctx.save();
  ctx.font = '11px ' + MONO_STACK;
  ctx.textBaseline = 'middle';
  ctx.textAlign = 'center';
  ctx.fillStyle = palette.textFaint;
  ctx.fillText(text, width / 2, height - 9, width - 28);
  ctx.restore();
}

/* ---------------- 3D ---------------- */

interface Scene3D {
  items: VolumeItem[];
  titles: Array<{ title: string; subtitle: string; at: Vec3; below: boolean; ids: string[] }>;
}

function nodeAt3D(layout: Layout3D, id: string): Vec3 | null {
  const placed = layout.byId.get(id);
  if (!placed) return null;
  return 'centre' in placed ? placed.centre : placed.at;
}

function buildScene3D(
  scene: CnnScene,
  params: CnnParams,
  cache: CnnCache,
  layout: Layout3D,
  textures: Map<string, HTMLCanvasElement>,
  palette: Palette,
  focus: ArchSelection | null,
  hot: ArchSelection | null,
  window: Window | null,
  scale: number,
): Scene3D {
  const items: VolumeItem[] = [];
  for (const edge of scene.edges) {
    const from = nodeAt3D(layout, edge.from);
    const to = nodeAt3D(layout, edge.to);
    if (!from || !to) continue;
    const emphasis = emphasisOf(edge, focus, hot);
    const style = wireStyle(edge, scene.maxWeight, palette);
    const alpha = emphasis === 'dim' ? style.alpha * DIM : emphasis === 'plain' ? style.alpha : Math.min(1, style.alpha + 0.25);
    if (emphasis === 'hot') items.push({ kind: 'line', from, to, colour: style.colour, width: style.width + 6, alpha: 0.22, bias: 0.01 });
    items.push({ kind: 'line', from, to, colour: style.colour, width: style.width, alpha, bias: 0.01 });
  }
  for (const sheet of layout.sheets) {
    const node = scene.byId.get(sheet.id);
    if (!node) continue;
    const texture = node.picture ? paintTexture(textures, node.id, node.picture, palette) : null;
    items.push({ kind: 'sheet', id: sheet.id, corners: sheetCorners(sheet), texture, fill: palette.surfaceAlt, stroke: rgba(palette.accent, 0.4) });
  }
  const unitR = Math.max(4, Math.min(9, 0.045 * scale));
  const classR = Math.max(9, Math.min(16, 0.13 * scale));
  for (const disc of layout.discs) {
    const node = scene.byId.get(disc.id);
    if (!node) continue;
    if (node.kind === 'unit') {
      items.push({ kind: 'disc', id: disc.id, at: disc.at, r: unitR, fill: rgba(palette.positive, 0.1 + 0.8 * (node.value ?? 0)), stroke: rgba(palette.accent, 0.5) });
    } else {
      const p = node.value ?? 0;
      items.push({
        kind: 'disc',
        id: disc.id,
        at: disc.at,
        r: classR,
        fill: rgba(palette.accent, 0.08 + 0.72 * p),
        stroke: callColour(scene, node, palette),
        label: node.label,
        labelColour: palette.text,
      });
      if (node.index === scene.label) {
        // A dashed halo marks the true class.
        const ring: Vec3[] = [];
        const r = 0.13 * 1.35;
        for (let a = 0; a < 24; a++) ring.push({ x: disc.at.x + r * Math.cos((a / 24) * Math.PI * 2), y: disc.at.y + r * Math.sin((a / 24) * Math.PI * 2), z: disc.at.z });
        items.push({ kind: 'poly', points: ring, stroke: rgba(palette.textMuted, 0.8), width: 1, alpha: 1, bias: -0.02 });
      }
    }
  }
  if (window) {
    const rects = windowRects(scene, params, cache, window);
    const outSheet = rects ? (layout.byId.get(rects.output.node.id) as Sheet3D | undefined) : undefined;
    if (rects && outSheet && rects.output.node.picture) {
      const outPic = rects.output.node.picture;
      const out = sheetCells(outSheet, outPic.cols, outPic.rows, rects.output.x, rects.output.y, rects.output.w, rects.output.h);
      items.push({ kind: 'poly', points: out, stroke: palette.accent, width: 1.5, alpha: 1, bias: -0.05 });
      for (const input of rects.inputs) {
        const sheet = layout.byId.get(input.node.id) as Sheet3D | undefined;
        const pic = input.node.picture;
        if (!sheet || !pic || !('centre' in sheet)) continue;
        const x0 = Math.max(0, input.x);
        const y0 = Math.max(0, input.y);
        const x1 = Math.min(pic.cols, input.x + input.w);
        const y1 = Math.min(pic.rows, input.y + input.h);
        const block = sheetCells(sheet, pic.cols, pic.rows, x0, y0, x1 - x0, y1 - y0);
        items.push({ kind: 'poly', points: block, stroke: palette.accent, width: 1.5, alpha: 1, bias: -0.05 });
        items.push({ kind: 'line', from: block[1], to: out[0], colour: palette.accent, width: 1, alpha: 0.55, bias: -0.05 });
        items.push({ kind: 'line', from: block[2], to: out[3], colour: palette.accent, width: 1, alpha: 0.55, bias: -0.05 });
      }
    }
  }
  const titles = layout.layers.map((bounds) => ({ title: bounds.layer.title, subtitle: bounds.layer.subtitle, at: { x: bounds.x, y: bounds.titleAt, z: 0 }, below: bounds.below, ids: bounds.layer.nodes }));
  return { items, titles };
}

/* ---------------- the component ---------------- */

export function CnnDiagram({ scene, params, cache, view, selection, onSelect, spot = null, window, renderDetail, description, redrawKey, phase = 0 }: CnnDiagramProps) {
  const [setFrame, size] = useElementSize<HTMLDivElement>();
  const [hover, setHover] = useState<ArchSelection | null>(null);
  const [orbit, setOrbit] = useState<VolumeView>(DEFAULT_VOLUME_VIEW);
  const [dragging, setDragging] = useState(false);
  const drag = useRef<{ x: number; y: number; from: VolumeView; moved: boolean } | null>(null);
  const shown = hover ?? spot;
  const spotlit = hover === null && spot !== null;

  const layout2d = useMemo(() => (size.width > 0 ? layoutScene2D(scene, size.width, size.height) : null), [scene, size.width, size.height]);
  const layout3d = useMemo(() => layoutScene3D(scene, cache.input.w), [scene, cache.input.w]);
  const layout2dRef = useRef<Layout2D | null>(null);
  layout2dRef.current = layout2d;
  const placed3dRef = useRef<PlacedItem[]>([]);
  const textures = useRef(new Map<string, HTMLCanvasElement>());

  const draw = useCallback(
    ({ ctx, width, height, palette }: DrawArgs) => {
      const focus = selection ?? shown;
      if (view === '2d') {
        const layout = layout2d && size.width === width && size.height === height ? layout2d : layoutScene2D(scene, width, height);
        drawHeaders2D(ctx, layout, palette);
        for (const edge of scene.edges) {
          const from = layout.byId.get(edge.from);
          const to = layout.byId.get(edge.to);
          if (!from || !to) continue;
          const style = wireStyle(edge, scene.maxWeight, palette);
          strokeWire(ctx, wirePath(from, to), { ...style, emphasis: emphasisOf(edge, focus, hover) });
        }
        for (const placed of layout.nodes) {
          const node = scene.byId.get(placed.id);
          if (!node) continue;
          const isOpen = selection?.kind === 'node' && selection.id === node.id;
          const isShown = shown?.kind === 'node' && shown.id === node.id;
          drawNode2D(ctx, scene, node, placed, layout, palette, isOpen ? 'open' : isShown ? (spotlit ? 'spot' : 'hover') : null);
        }
        if (window) drawWindow2D(ctx, scene, params, cache, layout, window, palette);
        drawFooter(ctx, scene.summary, width, height, palette);
        if (shown && !(selection && same(shown, selection))) {
          const lines = plateLines(scene, shown);
          if (lines.length > 0) {
            const plate = measurePlate(ctx, lines);
            let around: Rect | null = null;
            if (shown.kind === 'node') around = layout.byId.get(shown.id)?.rect ?? null;
            else {
              const edge = scene.edges.find((e) => e.id === shown.id);
              const from = edge && layout.byId.get(edge.from);
              const to = edge && layout.byId.get(edge.to);
              if (from && to) around = underWire(wirePath(from, to).label);
            }
            if (around) {
              const at = placePlate(plate, around, width, height);
              drawHoverPlate(ctx, lines, at.left, at.top, palette);
            }
          }
        }
        return;
      }

      const stage = { w: width, h: height - 22 };
      const fit = fitVolume(layout3d.points, stage.w, stage.h, 0.07);
      const project = makeVolumeProjector(orbit, fit, width / 2, stage.h / 2 + 6);
      drawVolumeFloor(ctx, project, palette, layout3d.floor.x0, layout3d.floor.x1, layout3d.floor.z0, layout3d.floor.z1, layout3d.floor.y, 0.5);
      const built = buildScene3D(scene, params, cache, layout3d, textures.current, palette, focus, hover, window, fit.scale);
      const placed = drawVolume(ctx, built.items, project);
      placed3dRef.current = placed;

      ctx.save();
      ctx.textAlign = 'center';
      // Titles that would overlap a neighbour step up (or down) a line.
      const lastTitle = { above: -Infinity, below: -Infinity };
      let stagger = { above: 0, below: 0 };
      for (const t of built.titles) {
        const p = project(t.at);
        // Clear of the layer's own sheets, wherever the turn has put them.
        let edge = p.y;
        for (const item of placed) {
          if (!t.ids.includes(item.id)) continue;
          for (const q of item.polygon) edge = t.below ? Math.max(edge, q.y) : Math.min(edge, q.y);
        }
        const band = t.below ? 'below' : 'above';
        const close = Math.abs(p.x - lastTitle[band]) < 60;
        stagger = { ...stagger, [band]: close ? (stagger[band] + 1) % 2 : 0 };
        lastTitle[band] = p.x;
        const lift = stagger[band] * 26;
        const y = t.below ? edge + 16 + lift : edge - 18 - lift;
        ctx.font = '700 10px ' + FONT_STACK;
        ctx.fillStyle = palette.textMuted;
        ctx.textBaseline = 'alphabetic';
        ctx.fillText(t.title.toUpperCase(), p.x, y);
        ctx.font = '10px ' + MONO_STACK;
        ctx.fillStyle = palette.textFaint;
        ctx.fillText(t.subtitle, p.x, y + 12);
      }
      ctx.restore();

      const outlineOf = (item: PlacedItem) => () => {
        ctx.beginPath();
        if (item.kind === 'disc') ctx.arc(item.centre.x, item.centre.y, item.r, 0, Math.PI * 2);
        else item.polygon.forEach((q, k) => (k === 0 ? ctx.moveTo(q.x, q.y) : ctx.lineTo(q.x, q.y)));
        ctx.closePath();
      };
      if (selection?.kind === 'node') {
        const item = placed.find((p) => p.id === selection.id);
        if (item) strokeOutline(ctx, outlineOf(item), 'open', palette);
      }
      if (shown?.kind === 'node' && !(selection && same(shown, selection))) {
        const item = placed.find((p) => p.id === shown.id);
        if (item) {
          strokeOutline(ctx, outlineOf(item), spotlit ? 'spot' : 'hover', palette);
          const lines = plateLines(scene, shown);
          const plate = measurePlate(ctx, lines);
          const at = placePlate(plate, polygonBox(item.polygon), width, height);
          drawHoverPlate(ctx, lines, at.left, at.top, palette);
        }
      }
      ctx.save();
      ctx.font = '10px ' + MONO_STACK;
      ctx.fillStyle = palette.textFaint;
      ctx.textAlign = 'left';
      ctx.textBaseline = 'middle';
      ctx.fillText('drag to turn', 38, 18);
      ctx.restore();
      drawFooter(ctx, scene.summary, width, height, palette);
    },
    [view, scene, params, cache, layout2d, layout3d, size.width, size.height, shown, selection, hover, spotlit, window, orbit],
  );

  const locate = useCallback(
    (pos: { x: number; y: number }): ArchSelection | null => {
      if (view === '3d') {
        const hit = hitVolume(placed3dRef.current, pos.x, pos.y);
        return hit ? { kind: 'node', id: hit.id } : null;
      }
      const layout = layout2dRef.current;
      if (!layout) return null;
      for (let i = layout.nodes.length - 1; i >= 0; i--) {
        const n = layout.nodes[i];
        // Small units get a wider margin.
        const pad = n.shape === 'circle' ? 6 : 3;
        if (pos.x >= n.rect.x - pad && pos.x <= n.rect.x + n.rect.w + pad && pos.y >= n.rect.y - pad && pos.y <= n.rect.y + n.rect.h + pad) return { kind: 'node', id: n.id };
      }
      let best: CnnEdge | null = null;
      let bestDistance = 6;
      for (const edge of scene.edges) {
        if (!edge.symbol) continue;
        const from = layout.byId.get(edge.from);
        const to = layout.byId.get(edge.to);
        if (!from || !to) continue;
        const points = sampleEdge(wirePath(from, to) as EdgePath);
        for (let i = 1; i < points.length; i++) {
          const d = distanceToSegment(pos.x, pos.y, points[i - 1].x, points[i - 1].y, points[i].x, points[i].y);
          if (d < bestDistance) {
            bestDistance = d;
            best = edge;
          }
        }
      }
      return best ? { kind: 'edge', id: best.id } : null;
    },
    [view, scene.edges],
  );

  const handleMove = useCallback(
    (pos: { x: number; y: number } | null) => {
      const d = drag.current;
      if (d && pos) {
        const dx = pos.x - d.x;
        const dy = pos.y - d.y;
        if (!d.moved && Math.abs(dx) + Math.abs(dy) > CLICK_SLOP) {
          d.moved = true;
          setDragging(true);
          setHover(null);
        }
        if (d.moved) {
          setOrbit(dragVolumeView(d.from, dx, dy));
          return;
        }
      }
      const found = pos ? locate(pos) : null;
      setHover((prev) => (same(prev, found) ? prev : found));
    },
    [locate],
  );

  const handleDown = useCallback(
    (pos: { x: number; y: number }) => {
      if (view === '3d') {
        drag.current = { x: pos.x, y: pos.y, from: orbit, moved: false };
        return;
      }
      const found = locate(pos);
      onSelect(found && same(found, selection) ? null : found);
    },
    [view, orbit, locate, onSelect, selection],
  );

  const handleUp = useCallback(
    (pos: { x: number; y: number }) => {
      const d = drag.current;
      drag.current = null;
      setDragging(false);
      if (!d || d.moved) return;
      const found = locate(pos);
      onSelect(found && same(found, selection) ? null : found);
    },
    [locate, onSelect, selection],
  );

  const handleLeave = useCallback(() => {
    drag.current = null;
    setDragging(false);
    setHover(null);
  }, []);

  const close = useCallback(() => onSelect(null), [onSelect]);

  const targets = useMemo<ArchSelection[]>(
    () => [...scene.nodes.map((n) => ({ kind: 'node' as const, id: n.id })), ...scene.edges.filter((e) => e.symbol).map((e) => ({ kind: 'edge' as const, id: e.id }))],
    [scene],
  );
  const handleKey = useDiagramKeys<ArchSelection>({
    targets,
    cursor: hover ?? selection,
    same,
    onCursor: setHover,
    onOpen: (target) => onSelect(target),
    onClose: () => {
      onSelect(null);
      setHover(null);
    },
  });
  const cursorText = useMemo(() => {
    const cursor = hover ?? selection;
    return cursor ? plateLines(scene, cursor).join(', ') : '';
  }, [scene, hover, selection]);

  // A card whose component has left the scene closes with it.
  useEffect(() => {
    if (!selection) return;
    const exists = selection.kind === 'node' ? scene.byId.has(selection.id) : scene.edges.some((e) => e.id === selection.id);
    if (!exists) onSelect(null);
  }, [scene, selection, onSelect]);

  const detail = selection && renderDetail ? renderDetail(selection, onSelect) : null;
  const hoverKey = shown ? shown.kind + ':' + shown.id : '';
  const selectionKey = selection ? selection.kind + ':' + selection.id : '';
  const windowKey = window ? window.kind + window.stage + ':' + window.index + ':' + window.y + ',' + window.x : '';

  return (
    <div
      className="mlx-arch"
      tabIndex={0}
      role="group"
      aria-label={description + ' Arrow keys move between components, Enter opens one.'}
      onKeyDown={handleKey}
      onBlur={(event) => {
        if (!event.currentTarget.contains(event.relatedTarget as Node | null)) setHover(null);
      }}
    >
      <span className="mlx-visually-hidden" aria-live="polite">
        {cursorText}
      </span>
      <div className="mlx-arch__stage" ref={setFrame}>
        <Chart
          draw={draw}
          height="fill"
          description={description}
          cursor={view === '3d' ? (dragging ? 'grabbing' : hover ? 'pointer' : 'grab') : hover ? 'pointer' : 'default'}
          drag={view === '3d'}
          onPointerDown={handleDown}
          onPointerMove={handleMove}
          onPointerUp={handleUp}
          onPointerLeave={handleLeave}
          redrawKey={String(redrawKey) + '|' + view + '|' + hoverKey + '|' + selectionKey + '|' + volumeViewKey(orbit) + '|' + windowKey + '|' + phase}
        />
        {view === '3d' ? (
          <button type="button" className="mlx-arch__tool" onClick={() => setOrbit(DEFAULT_VOLUME_VIEW)} title="Reset the view" aria-label="Reset the view">
            ↺
          </button>
        ) : null}
      </div>
      {detail ? (
        <DetailOverlay dock onClose={close}>
          {detail}
        </DetailOverlay>
      ) : null}
    </div>
  );
}
