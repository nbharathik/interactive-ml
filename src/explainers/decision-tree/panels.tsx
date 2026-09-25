/** Decision tree charts: feature space, the tree diagram, the split search, Gini vs entropy, growth. */

import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import type { ReactNode } from 'react';

import { Chart } from '../../explainer/components/Chart';
import type { DrawArgs } from '../../explainer/components/Chart';
import { DetailOverlay } from '../../explainer/components/Detail';
import { useDiagramKeys } from '../../explainer/useDiagramKeys';
import { useElementSize } from '../../explainer/useElementSize';
import {
  clamp,
  clipFrame,
  drawLabelPlate,
  drawPath,
  drawPoint,
  makeFrame,
  drawAxes,
  roundRect,
  FONT_STACK,
  MONO_STACK,
} from '../../lib/viz/canvas';
import type { Frame } from '../../lib/viz/canvas';
import {
  classColour,
  drawEmptyState,
  drawGlow,
  drawScatter,
  drawSeries,
  plateRoom,
} from '../../lib/viz/plots';
import { rgba } from '../../lib/viz/palette';
import type { Palette } from '../../lib/viz/palette';
import type { Point2D, PointData } from '../../lib/datasets/types';
import { nearestPointIndex } from '../../lib/datasets/points';
import { impurityOf, leafRegions, splitLines, walk } from '../../lib/ml/decisionTree';
import type { Criterion, SplitCandidate, TreeNode } from '../../lib/ml/decisionTree';
import { fmt, fmtPercent } from '../../lib/math/stats';

/* ---------------- shared bits ---------------- */

/** A 0 to 1 ramp that restarts whenever `key` changes. Reduced motion skips to 1. */
export function useIntro(key: unknown, duration = 420): number {
  const [progress, setProgress] = useState(1);

  useEffect(() => {
    const reduced =
      typeof window !== 'undefined' &&
      window.matchMedia &&
      window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    if (reduced) {
      setProgress(1);
      return undefined;
    }
    let frame = 0;
    const start = performance.now();
    setProgress(0);
    const tick = (now: number) => {
      const t = Math.min(1, (now - start) / duration);
      // Ease out.
      setProgress(1 - Math.pow(1 - t, 3));
      if (t < 1) frame = requestAnimationFrame(tick);
    };
    frame = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(frame);
  }, [key, duration]);

  return progress;
}

/** Truncate to fit, with an ellipsis. */
function fitText(ctx: CanvasRenderingContext2D, text: string, maxWidth: number): string {
  if (ctx.measureText(text).width <= maxWidth) return text;
  let cut = text;
  while (cut.length > 1 && ctx.measureText(cut + '…').width > maxWidth) cut = cut.slice(0, -1);
  return cut + '…';
}

/** "x₁ ≤ 1.34", the question a node asks. */
export function conditionText(
  feature: 0 | 1,
  threshold: number,
  names: readonly [string, string],
): string {
  return names[feature] + ' ≤ ' + fmt(threshold, 2);
}

/* ---------------- 1. Feature space ---------------- */

export interface FeatureSpaceProps {
  data: PointData;
  /** Training points first, then the held-out ones. */
  points: readonly Point2D[];
  testStart: number;
  root: TreeNode;
  showRegions: boolean;
  showSplitLines: boolean;
  showTestPoints: boolean;
  markMistakes: boolean;
  /** Indices into `points` the tree gets wrong. */
  mistakes: ReadonlySet<number>;
  /** Index into `points` currently under the pointer. */
  hoverIndex: number | null;
  /** Node ids on the hovered point's root-to-leaf path. */
  pathIds: ReadonlySet<number>;
  /** The node clicked in the tree diagram, whose rectangle gets outlined. */
  selected: TreeNode | null;
  /** Node that split most recently; its cut animates in. */
  lastSplitId: number | null;
  intro: number;
  onHover?: (index: number | null) => void;
}

export function FeatureSpace({
  data,
  points,
  testStart,
  root,
  showRegions,
  showSplitLines,
  showTestPoints,
  markMistakes,
  mistakes,
  hoverIndex,
  pathIds,
  selected,
  lastSplitId,
  intro,
  onHover,
}: FeatureSpaceProps) {
  const frameRef = useRef<Frame | null>(null);

  const draw = useCallback(
    ({ ctx, width, height, palette }: DrawArgs) => {
      const frame = makeFrame(width, height, data.xRange, data.yRange, {
        left: 44,
        bottom: 34,
        right: 14,
        top: 14,
      });
      frameRef.current = frame;

      // 1. One filled rectangle per leaf.
      if (showRegions) {
        ctx.save();
        clipFrame(ctx, frame);
        const floor = data.classCount > 1 ? 1 / data.classCount : 0;
        for (const region of leafRegions(root)) {
          const { bounds, node } = region;
          const x0 = frame.x(bounds.x0);
          const x1 = frame.x(bounds.x1);
          const y0 = frame.y(bounds.y1);
          const y1 = frame.y(bounds.y0);
          const purity = floor >= 1 ? 1 : (node.confidence - floor) / (1 - floor);
          ctx.fillStyle = rgba(
            classColour(palette, node.prediction),
            0.1 + 0.26 * clamp(purity, 0, 1),
          );
          ctx.fillRect(x0, y0, x1 - x0, y1 - y0);
        }
        ctx.restore();
      }

      drawAxes(ctx, frame, palette, {
        xLabel: data.xLabel,
        yLabel: data.yLabel,
        xTicks: 6,
        yTicks: 5,
        showGrid: !showRegions,
      });

      // 2. Every cut, clipped to the region the node that made it owns.
      if (showSplitLines) {
        ctx.save();
        clipFrame(ctx, frame);
        for (const cut of splitLines(root)) {
          const onPath = pathIds.has(cut.node.id);
          const newest = cut.node.id === lastSplitId;
          const colour = onPath
            ? palette.accent
            : rgba(palette.text, newest ? 0.85 : 0.3);
          const lineWidth = newest ? 2.6 : onPath ? 2.2 : 1.3;

          let a: { x: number; y: number };
          let b: { x: number; y: number };
          if (cut.feature === 0) {
            const px = frame.x(cut.threshold);
            a = { x: px, y: frame.y(cut.bounds.y0) };
            b = { x: px, y: frame.y(cut.bounds.y1) };
          } else {
            const py = frame.y(cut.threshold);
            a = { x: frame.x(cut.bounds.x0), y: py };
            b = { x: frame.x(cut.bounds.x1), y: py };
          }

          if (newest && intro < 1) {
            // Grow the newest cut out from its middle.
            const mx = (a.x + b.x) / 2;
            const my = (a.y + b.y) / 2;
            a = { x: mx + (a.x - mx) * intro, y: my + (a.y - my) * intro };
            b = { x: mx + (b.x - mx) * intro, y: my + (b.y - my) * intro };
            ctx.save();
            ctx.globalAlpha = 1 - intro;
            drawGlow(ctx, mx, my, 46, palette.accent);
            ctx.restore();
          }
          drawPath(ctx, [a, b], colour, lineWidth);
        }
        ctx.restore();
      }

      // 3. The data, held-out points faded.
      const trainPoints = points.slice(0, testStart);
      const testPoints = points.slice(testStart);
      if (showTestPoints && testPoints.length > 0) {
        drawScatter(ctx, frame, palette, testPoints, {
          radius: 3.6,
          shapes: true,
          opacity: 0.45,
          highlight: hoverIndex !== null && hoverIndex >= testStart ? [hoverIndex - testStart] : undefined,
        });
      }
      drawScatter(ctx, frame, palette, trainPoints, {
        radius: 4,
        shapes: true,
        highlight: hoverIndex !== null && hoverIndex < testStart ? [hoverIndex] : undefined,
      });

      // 4. Mistakes, ringed.
      if (markMistakes && mistakes.size > 0) {
        ctx.save();
        clipFrame(ctx, frame);
        ctx.strokeStyle = palette.red;
        ctx.lineWidth = 1.6;
        for (const index of mistakes) {
          if (index >= testStart && !showTestPoints) continue;
          const p = points[index];
          ctx.beginPath();
          ctx.arc(frame.x(p.x), frame.y(p.y), 7, 0, Math.PI * 2);
          ctx.stroke();
        }
        ctx.restore();
      }

      // 5. The rectangle owned by the node selected in the tree diagram.
      if (selected) {
        const x0 = clamp(frame.x(selected.bounds.x0), frame.left, frame.right);
        const x1 = clamp(frame.x(selected.bounds.x1), frame.left, frame.right);
        const y0 = clamp(frame.y(selected.bounds.y1), frame.top, frame.bottom);
        const y1 = clamp(frame.y(selected.bounds.y0), frame.top, frame.bottom);
        ctx.save();
        ctx.strokeStyle = palette.accent;
        ctx.lineWidth = 2;
        ctx.setLineDash([6, 4]);
        ctx.strokeRect(x0 + 1, y0 + 1, x1 - x0 - 2, y1 - y0 - 2);
        ctx.restore();
        drawLabelPlate(
          ctx,
          (selected.isLeaf ? 'leaf · ' : 'node · ') +
            selected.samples.length +
            ' pts · ' +
            data.classNames[selected.prediction],
          (x0 + x1) / 2,
          Math.max(frame.top + 10, y0 + 12),
          palette,
          { align: 'center', bold: true },
        );
      }

      // 6. The hovered point's path.
      if (hoverIndex !== null && points[hoverIndex]) {
        const p = points[hoverIndex];
        const px = frame.x(p.x);
        const py = frame.y(p.y);
        ctx.save();
        ctx.strokeStyle = palette.accent;
        ctx.lineWidth = 1.6;
        ctx.beginPath();
        ctx.arc(px, py, 9, 0, Math.PI * 2);
        ctx.stroke();
        ctx.restore();
      }
    },
    [
      data,
      points,
      testStart,
      root,
      showRegions,
      showSplitLines,
      showTestPoints,
      markMistakes,
      mistakes,
      hoverIndex,
      pathIds,
      selected,
      lastSplitId,
      intro,
    ],
  );

  const handleMove = useCallback(
    (pos: { x: number; y: number } | null, event?: { pointerType: string }) => {
      if (!onHover) return;
      const frame = frameRef.current;
      if (!pos || !frame) {
        // A tap keeps its point until the next tap.
        if (event?.pointerType !== 'touch') onHover(null);
        return;
      }
      const limit = showTestPoints ? points.length : testStart;
      const x = frame.x.invert(pos.x);
      const y = frame.y.invert(pos.y);
      const index = nearestPointIndex(points.slice(0, limit), x, y, 0.55);
      onHover(index >= 0 ? index : null);
    },
    [onHover, points, showTestPoints, testStart],
  );

  return (
    <Chart
      draw={draw}
      height={(w) => Math.round(Math.min(430, Math.max(280, w * 0.72)))}
      description={
        'Feature space with ' +
        leafRegions(root).length +
        ' leaf rectangles, each tinted with the class it predicts, ' +
        splitLines(root).length +
        ' axis-aligned split lines, and ' +
        points.length +
        ' labelled points drawn with one marker shape per class.'
      }
      onPointerDown={onHover ? handleMove : undefined}
      onPointerMove={onHover ? handleMove : undefined}
      onPointerLeave={
        onHover
          ? (event) => {
              if (event.pointerType !== 'touch') onHover(null);
            }
          : undefined
      }
      redrawKey={
        root.id +
        ':' +
        splitLines(root).length +
        ':' +
        hoverIndex +
        ':' +
        (selected?.id ?? -1) +
        ':' +
        intro.toFixed(2)
      }
    />
  );
}

/* ---------------- 2. The tree diagram ---------------- */

interface PlacedNode {
  node: TreeNode;
  x: number;
  y: number;
  w: number;
  h: number;
  parent: PlacedNode | null;
  branch: 'left' | 'right' | null;
}

type Detail = 'full' | 'compact' | 'minimal';

/** Horizontal room one leaf wants. */
function preferredSlot(leafCount: number): number {
  if (leafCount <= 4) return 132;
  if (leafCount <= 8) return 106;
  if (leafCount <= 16) return 88;
  if (leafCount <= 32) return 58;
  if (leafCount <= 64) return 42;
  return 30;
}

/** How wide the canvas has to be before the diagram needs a scrollbar. */
export function treeCanvasWidth(root: TreeNode): number {
  let leaves = 0;
  walk(root, (n) => {
    if (n.isLeaf) leaves += 1;
  });
  return Math.round(leaves * preferredSlot(leaves) + 56);
}

export function treeCanvasHeight(root: TreeNode): number {
  let depth = 0;
  walk(root, (n) => {
    if (n.depth > depth) depth = n.depth;
  });
  return clamp(96 + (depth + 1) * 60, 220, 520);
}

/** Slot layout: every leaf gets a column, an internal node sits between its children. */
function layoutTree(
  root: TreeNode,
  width: number,
  height: number,
  featureNames: readonly [string, string],
  classNames: readonly string[],
): {
  placed: PlacedNode[];
  slot: number;
  detail: Detail;
} {
  const leaves: TreeNode[] = [];
  let maxDepth = 0;
  walk(root, (n) => {
    if (n.isLeaf) leaves.push(n);
    if (n.depth > maxDepth) maxDepth = n.depth;
  });
  const leafCount = Math.max(1, leaves.length);

  const padX = 16;
  const slot = clamp((width - padX * 2) / leafCount, 22, 148);
  const detail: Detail = slot >= 80 ? 'full' : slot >= 44 ? 'compact' : 'minimal';
  const boxHeight = detail === 'full' ? 42 : detail === 'compact' ? 34 : 16;

  const topPad = boxHeight / 2 + 14;
  const bottomPad = boxHeight / 2 + 10;
  const yStep =
    maxDepth === 0 ? 0 : clamp((height - topPad - bottomPad) / maxDepth, boxHeight + 12, 108);
  const offsetX = (width - leafCount * slot) / 2;
  const total = Math.max(1, root.samples.length);

  const placed: PlacedNode[] = [];
  let cursor = 0;

  const assign = (node: TreeNode, parent: PlacedNode | null, branch: 'left' | 'right' | null): PlacedNode => {
    let centre: number;
    let entry: PlacedNode;
    const share = Math.sqrt(node.samples.length / total);
    // A full box is at least as wide as its label.
    const label = node.split
      ? conditionText(node.split.feature, node.split.threshold, featureNames)
      : classNames[node.prediction] ?? '';
    const need = detail === 'full' ? Math.ceil(label.length * 6.1) + 12 : detail === 'compact' ? slot - 8 : 0;
    const w = clamp(
      Math.max(need, (slot - 8) * (0.55 + 0.45 * share)),
      18,
      Math.min(132, slot - 4),
    );

    if (node.isLeaf || !node.left || !node.right) {
      centre = offsetX + (cursor + 0.5) * slot;
      cursor += 1;
      entry = { node, x: centre, y: topPad + node.depth * yStep, w, h: boxHeight, parent, branch };
      placed.push(entry);
      return entry;
    }

    // Children first; the parent sits above their midpoint.
    entry = { node, x: 0, y: topPad + node.depth * yStep, w, h: boxHeight, parent, branch };
    placed.push(entry);
    const left = assign(node.left, entry, 'left');
    const right = assign(node.right, entry, 'right');
    entry.x = (left.x + right.x) / 2;
    return entry;
  };

  assign(root, null, null);

  // Slots position centres, so an outer box can spill; squeeze the spread until it fits.
  let minEdge = Infinity;
  let maxEdge = -Infinity;
  for (const item of placed) {
    minEdge = Math.min(minEdge, item.x - item.w / 2);
    maxEdge = Math.max(maxEdge, item.x + item.w / 2);
  }

  if (Number.isFinite(minEdge) && Number.isFinite(maxEdge)) {
    const spread = maxEdge - minEdge;
    const available = width - 4;
    if (spread > available && spread > 0) {
      const centre = width / 2;
      const squeeze = available / spread;
      for (const item of placed) {
        item.x = centre + (item.x - centre) * squeeze;
        item.w = Math.max(14, item.w * squeeze);
      }
    } else {
      // Off-centre rather than too wide: nudge it back inside.
      const shift = minEdge < 2 ? 2 - minEdge : maxEdge > width - 2 ? width - 2 - maxEdge : 0;
      if (shift !== 0) for (const item of placed) item.x += shift;
    }
  }

  return { placed, slot, detail };
}

/** The stacked bar showing what a node holds, one segment per class. */
function drawClassMix(
  ctx: CanvasRenderingContext2D,
  palette: Palette,
  counts: readonly number[],
  x: number,
  y: number,
  w: number,
  h: number,
): void {
  const total = counts.reduce((a, b) => a + b, 0);
  if (total === 0) return;
  let cursor = x;
  for (let k = 0; k < counts.length; k++) {
    if (counts[k] === 0) continue;
    const segment = (counts[k] / total) * w;
    ctx.fillStyle = classColour(palette, k);
    ctx.fillRect(cursor, y, Math.max(1, segment), h);
    cursor += segment;
  }
  ctx.strokeStyle = rgba(palette.border, 0.9);
  ctx.lineWidth = 1;
  ctx.strokeRect(x + 0.5, y + 0.5, w - 1, h - 1);
}

export interface TreeDiagramProps {
  root: TreeNode;
  classNames: readonly string[];
  featureNames: readonly [string, string];
  selectedId: number | null;
  pathIds: ReadonlySet<number>;
  lastSplitId: number | null;
  intro: number;
  onSelect?: (id: number | null) => void;
  /** The card for the selected node. */
  renderDetail?: (node: TreeNode, close: () => void) => ReactNode;
}

export function TreeDiagram({
  root,
  classNames,
  featureNames,
  selectedId,
  pathIds,
  lastSplitId,
  intro,
  onSelect,
  renderDetail,
}: TreeDiagramProps) {
  const layoutRef = useRef<PlacedNode[]>([]);
  const [setScroller, scrollerSize, scrollRef] = useElementSize<HTMLDivElement>();
  const [setInner, innerSize] = useElementSize<HTMLDivElement>();
  const [hoverId, setHoverId] = useState<number | null>(null);
  const [scroll, setScroll] = useState({ left: 0, top: 0 });
  const canvasHeight = treeCanvasHeight(root);

  let leafCount = 0;
  let nodeCount = 0;
  walk(root, (n) => {
    nodeCount += 1;
    if (n.isLeaf) leafCount += 1;
  });

  // Drop to compact detail before overflowing.
  const neededWidth = Math.min(treeCanvasWidth(root), Math.max(scrollerSize.width, leafCount * 52 + 32));
  const overflowing = scrollerSize.width > 0 && neededWidth > scrollerSize.width + 1;

  // Where the selected node sits, for the card.
  const anchor = useMemo(() => {
    if (selectedId === null || innerSize.width === 0) return null;
    const { placed } = layoutTree(root, innerSize.width, innerSize.height, featureNames, classNames);
    const item = placed.find((entry) => entry.node.id === selectedId);
    return item ? { x: item.x - item.w / 2, y: item.y - item.h / 2, w: item.w, h: item.h } : null;
  }, [root, selectedId, innerSize.width, innerSize.height, featureNames, classNames]);
  const selectedNode = useMemo(() => {
    if (selectedId === null) return null;
    let found: TreeNode | null = null;
    walk(root, (n) => {
      if (n.id === selectedId) found = n;
    });
    return found;
  }, [root, selectedId]);

  // Keep the root in view when the tree overflows.
  useEffect(() => {
    const el = scrollRef.current;
    if (!el) return;
    const overflow = el.scrollWidth - el.clientWidth;
    if (overflow > 0) el.scrollLeft = overflow / 2;
  }, [neededWidth, scrollRef]);

  const draw = useCallback(
    ({ ctx, width, height, palette }: DrawArgs) => {
      const { placed, detail } = layoutTree(root, width, height, featureNames, classNames);
      layoutRef.current = placed;
      const total = Math.max(1, root.samples.length);

      // Edges first, thickness by data flow.
      for (const item of placed) {
        if (!item.parent) continue;
        const from = item.parent;
        const onPath = pathIds.has(item.node.id) && pathIds.has(from.node.id);
        const share = Math.sqrt(item.node.samples.length / total);
        ctx.save();
        ctx.beginPath();
        ctx.moveTo(from.x, from.y + from.h / 2);
        const midY = (from.y + from.h / 2 + item.y - item.h / 2) / 2;
        ctx.bezierCurveTo(from.x, midY, item.x, midY, item.x, item.y - item.h / 2);
        ctx.strokeStyle = onPath ? palette.accent : rgba(palette.textFaint, 0.65);
        ctx.lineWidth = (onPath ? 1.4 : 0.8) + 2.2 * share;
        ctx.stroke();
        ctx.restore();

        if (detail === 'full') {
          ctx.save();
          ctx.font = '9px ' + FONT_STACK;
          ctx.fillStyle = palette.textFaint;
          ctx.textAlign = item.branch === 'left' ? 'right' : 'left';
          ctx.textBaseline = 'middle';
          const text = item.branch === 'left' ? 'yes' : 'no';
          const tx = item.x + (item.branch === 'left' ? -5 : 5);
          const tw = ctx.measureText(text).width;
          const x0 = item.branch === 'left' ? tx - tw : tx;
          // Lifted off the wire when a neighbouring box would cover it.
          const touches = placed.some(
            (other) =>
              other !== item &&
              Math.abs(other.y - midY) < other.h / 2 + 5 &&
              x0 < other.x + other.w / 2 &&
              x0 + tw > other.x - other.w / 2,
          );
          ctx.fillText(text, tx, touches ? midY - 6 : midY);
          ctx.restore();
        }
      }

      // Nodes.
      for (const item of placed) {
        const { node } = item;
        const left = item.x - item.w / 2;
        const top = item.y - item.h / 2;
        const colour = classColour(palette, node.prediction);
        const newest = node.id === lastSplitId;

        if (newest) {
          ctx.save();
          ctx.globalAlpha = 0.35 + 0.65 * (1 - intro);
          drawGlow(ctx, item.x, item.y, 26 + 26 * (1 - intro), palette.accent);
          ctx.restore();
        }

        roundRect(ctx, left, top, item.w, item.h, 3);
        ctx.fillStyle = node.isLeaf
          ? rgba(colour, 0.16 + 0.24 * node.confidence)
          : palette.surface;
        ctx.fill();
        ctx.strokeStyle =
          selectedId === node.id
            ? palette.accent
            : newest
              ? palette.accent
              : pathIds.has(node.id)
                ? rgba(palette.accent, 0.75)
                : node.isLeaf
                  ? rgba(colour, 0.75)
                  : rgba(palette.border, 1);
        ctx.lineWidth = selectedId === node.id || newest ? 2 : 1;
        ctx.stroke();

        if (detail === 'minimal') {
          drawClassMix(ctx, palette, node.counts, left + 2, top + 3, item.w - 4, item.h - 6);
          continue;
        }

        const innerWidth = item.w - 10;
        ctx.save();
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';

        if (detail === 'full') {
          const asks = !node.isLeaf && node.split;
          ctx.font = asks ? '10px ' + MONO_STACK : '600 10px ' + FONT_STACK;
          ctx.fillStyle = palette.text;
          const label = node.split
            ? conditionText(node.split.feature, node.split.threshold, featureNames)
            : classNames[node.prediction] ?? 'class ' + node.prediction;
          ctx.fillText(fitText(ctx, label, innerWidth), item.x, top + 11);

          drawClassMix(ctx, palette, node.counts, left + 5, top + 19, innerWidth, 6);

          ctx.font = '9px ' + FONT_STACK;
          ctx.fillStyle = palette.textFaint;
          const detailText = node.isLeaf
            ? node.samples.length + ' pts'
            : node.samples.length + ' pts · ' + fmt(node.impurity, 2);
          ctx.fillText(fitText(ctx, detailText, innerWidth), item.x, top + 34);
        } else {
          const asks = !node.isLeaf && node.split;
          ctx.font = asks ? '9px ' + MONO_STACK : '600 9px ' + FONT_STACK;
          ctx.fillStyle = palette.text;
          let label = node.split
            ? conditionText(node.split.feature, node.split.threshold, featureNames)
            : classNames[node.prediction] ?? 'class ' + node.prediction;
          if (node.split && ctx.measureText(label).width > item.w - 6) {
            label = featureNames[node.split.feature] + '≤' + fmt(node.split.threshold, 1);
          }
          ctx.fillText(fitText(ctx, label, item.w - 6), item.x, top + 9);
          drawClassMix(ctx, palette, node.counts, left + 4, top + 17, item.w - 8, 5);
          ctx.font = '9px ' + FONT_STACK;
          ctx.fillStyle = palette.textMuted;
          ctx.fillText(fitText(ctx, String(node.samples.length), item.w - 8), item.x, top + 27);
        }
        ctx.restore();
      }

      // The hovered node's readout.
      const hovered = hoverId !== null && hoverId !== selectedId ? placed.find((item) => item.node.id === hoverId) : undefined;
      if (hovered) {
        const node = hovered.node;
        const majority = classNames[node.prediction] ?? '?';
        const text = node.isLeaf
          ? 'Leaf · ' + node.samples.length + ' pts · ' + majority + ' ' + fmtPercent(node.confidence, 0) + ' · impurity ' + fmt(node.impurity, 2)
          : (node.split ? conditionText(node.split.feature, node.split.threshold, featureNames) : 'Node') +
            ' · ' + node.samples.length + ' pts · impurity ' + fmt(node.impurity, 2) +
            (node.split ? ' · gain ' + fmt(node.split.gain, 3) : '');
        // Centred under the node, slid just enough to keep the whole plate on the canvas.
        const label = text + ' · click for more';
        ctx.save();
        ctx.font = '600 11px ' + FONT_STACK;
        const half = ctx.measureText(label).width / 2 + 6;
        ctx.restore();
        drawLabelPlate(
          ctx,
          label,
          Math.max(half, Math.min(width - half, hovered.x)),
          Math.min(height - 10, hovered.y + hovered.h / 2 + 12),
          palette,
          { align: 'center', bold: true },
        );
      }
    },
    [root, classNames, featureNames, selectedId, pathIds, lastSplitId, intro, hoverId],
  );

  const nodeAt = useCallback((pos: { x: number; y: number }): PlacedNode | null => {
    for (const item of layoutRef.current) {
      if (
        pos.x >= item.x - item.w / 2 - 3 &&
        pos.x <= item.x + item.w / 2 + 3 &&
        pos.y >= item.y - item.h / 2 - 3 &&
        pos.y <= item.y + item.h / 2 + 3
      ) {
        return item;
      }
    }
    return null;
  }, []);

  const handleDown = useCallback(
    (pos: { x: number; y: number }) => {
      if (!onSelect) return;
      const hit = nodeAt(pos);
      onSelect(hit ? (hit.node.id === selectedId ? null : hit.node.id) : null);
    },
    [nodeAt, onSelect, selectedId],
  );

  const handleMove = useCallback(
    (pos: { x: number; y: number } | null) => {
      const hit = pos ? nodeAt(pos) : null;
      setHoverId(hit ? hit.node.id : null);
    },
    [nodeAt],
  );

  const close = useCallback(() => onSelect?.(null), [onSelect]);
  const detail = selectedNode && renderDetail ? renderDetail(selectedNode, close) : null;

  // Arrows walk the nodes top to bottom, left to right.
  const targets = useMemo(() => {
    const ids: number[] = [];
    walk(root, (n) => {
      ids.push(n.id);
    });
    return ids;
  }, [root]);
  const handleKey = useDiagramKeys<number>({
    targets,
    cursor: hoverId ?? selectedId,
    same: (a, b) => a === b,
    onCursor: (id) => {
      setHoverId(id);
      // Keep the cursor in view when the tree is wider than its panel.
      const el = scrollRef.current;
      const placed = layoutRef.current.find((item) => item.node.id === id);
      if (el && placed) el.scrollLeft = placed.x - el.clientWidth / 2;
    },
    onOpen: (id) => onSelect?.(id),
    onClose: () => {
      onSelect?.(null);
      setHoverId(null);
    },
  });
  const cursorText = useMemo(() => {
    const id = hoverId ?? selectedId;
    if (id === null) return '';
    let found: TreeNode | null = null;
    walk(root, (n) => {
      if (n.id === id) found = n;
    });
    const node = found as TreeNode | null;
    if (!node) return '';
    return node.isLeaf
      ? 'Leaf predicting ' + (classNames[node.prediction] ?? node.prediction) + ', ' + node.samples.length + ' points'
      : (node.split ? conditionText(node.split.feature, node.split.threshold, featureNames) : 'Node') + ', ' + node.samples.length + ' points';
  }, [root, hoverId, selectedId, classNames, featureNames]);

  return (
    <div
      className="mlx-tree"
      ref={setScroller}
      data-overflow={overflowing || undefined}
      tabIndex={0}
      role="group"
      aria-label="The tree. Arrow keys move between nodes, Enter opens one."
      onKeyDown={handleKey}
      onScroll={(event) => setScroll({ left: event.currentTarget.scrollLeft, top: event.currentTarget.scrollTop })}
      onBlur={(event) => {
        if (!event.currentTarget.contains(event.relatedTarget as Node | null)) setHoverId(null);
      }}
    >
      <span className="mlx-visually-hidden" aria-live="polite">
        {cursorText}
      </span>
      <div className="mlx-tree__inner" ref={setInner} style={{ minWidth: neededWidth, minHeight: canvasHeight }}>
        <Chart
          draw={draw}
          height="fill"
          cursor={hoverId !== null ? 'pointer' : 'default'}
          onPointerDown={onSelect ? handleDown : undefined}
          onPointerMove={handleMove}
          onPointerLeave={() => setHoverId(null)}
          description={
            'Node-link diagram of a decision tree with ' +
            nodeCount +
            ' nodes and ' +
            leafCount +
            ' leaves. Each internal node shows the feature and threshold it tests, its sample count, and a stacked bar of its class mix; leaves show the class they predict.'
          }
          redrawKey={nodeCount + ':' + selectedId + ':' + lastSplitId + ':' + intro.toFixed(2) + ':' + hoverId}
        />
        {anchor && detail ? (
          <DetailOverlay
            anchor={anchor}
            bounds={innerSize}
            viewport={{
              x: scroll.left,
              y: scroll.top,
              width: scrollerSize.width || innerSize.width,
              height: scrollerSize.height || innerSize.height,
            }}
            onClose={close}
          >
            {detail}
          </DetailOverlay>
        ) : null}
      </div>
      {overflowing ? (
        <p className="mlx-note mlx-tree__cue">Scroll sideways to see the whole tree.</p>
      ) : null}
    </div>
  );
}

/* ---------------- 3. The split search ---------------- */

export interface SplitSearchProps {
  candidates: readonly SplitCandidate[];
  chosen: SplitCandidate | null;
  parentImpurity: number;
  featureNames: readonly [string, string];
  xDomain: [number, number];
  minGain: number;
  mode: 'gain' | 'impurity';
  criterion: Criterion;
}

export function SplitSearch({
  candidates,
  chosen,
  parentImpurity,
  featureNames,
  xDomain,
  minGain,
  mode,
  criterion,
}: SplitSearchProps) {
  const draw = useCallback(
    ({ ctx, width, height, palette }: DrawArgs) => {
      if (candidates.length === 0) {
        drawEmptyState(
          ctx,
          width,
          height,
          palette.textFaint,
          'No candidate splits: this node is either pure, too small to split, or every threshold is blocked by the minimum-samples rules.',
        );
        return;
      }

      const values = candidates.map((c) => (mode === 'gain' ? c.gain : c.impurity));
      const top = Math.max(mode === 'gain' ? 0.02 : parentImpurity, ...values) * 1.12;
      // Show only the span this node's points cover.
      const thresholds = candidates.map((c) => c.threshold);
      const lo = Math.min(...thresholds);
      const hi = Math.max(...thresholds);
      const pad = Math.max(0.05, (hi - lo) * 0.06);
      const domain: [number, number] = [
        Math.max(xDomain[0], lo - pad),
        Math.min(xDomain[1], hi + pad),
      ];
      const frame = makeFrame(width, height, domain, [0, top], {
        left: 48,
        bottom: 34,
        right: 14,
        top: 14,
      });

      drawAxes(ctx, frame, palette, {
        xLabel: 'threshold',
        yLabel: mode === 'gain' ? 'gain' : 'weighted child impurity',
        xTicks: 6,
        yTicks: 4,
      });

      ctx.save();
      clipFrame(ctx, frame);

      // The parent's impurity, the baseline for every candidate.
      if (mode === 'impurity') {
        const y = frame.y(parentImpurity);
        drawPath(
          ctx,
          [
            { x: frame.left, y },
            { x: frame.right, y },
          ],
          rgba(palette.text, 0.45),
          1.4,
          [5, 4],
        );
      } else if (minGain > 0) {
        const y = frame.y(minGain);
        drawPath(
          ctx,
          [
            { x: frame.left, y },
            { x: frame.right, y },
          ],
          palette.red,
          1.4,
          [5, 4],
        );
      }

      for (const feature of [0, 1] as const) {
        const series = candidates
          .filter((c) => c.feature === feature)
          .sort((a, b) => a.threshold - b.threshold)
          .map((c) => ({
            x: frame.x(c.threshold),
            y: frame.y(mode === 'gain' ? c.gain : c.impurity),
          }));
        if (series.length < 2) continue;
        drawPath(ctx, series, feature === 0 ? palette.violet : palette.cyan, 1.9);
      }
      ctx.restore();

      if (chosen) {
        const x = frame.x(chosen.threshold);
        const y = frame.y(mode === 'gain' ? chosen.gain : chosen.impurity);
        const colour = chosen.feature === 0 ? palette.violet : palette.cyan;
        drawPath(
          ctx,
          [
            { x, y: frame.bottom },
            { x, y: frame.top },
          ],
          rgba(colour, 0.55),
          1.4,
          [4, 4],
        );
        drawPoint(ctx, x, y, 5, colour, rgba(palette.surface, 0.95), 2);
        if (plateRoom(frame)) {
          drawLabelPlate(
            ctx,
            conditionText(chosen.feature, chosen.threshold, featureNames) +
              '  ·  gain ' +
              fmt(chosen.gain, 3),
            clamp(x, frame.left + 4, frame.right - 4),
            clamp(y - 16, frame.top + 10, frame.bottom - 10),
            palette,
            { align: x > (frame.left + frame.right) / 2 ? 'right' : 'left', bold: true },
          );
        }
      }
    },
    [candidates, chosen, parentImpurity, featureNames, xDomain, minGain, mode],
  );

  return (
    <Chart
      draw={draw}
      height={230}
      description={
        candidates.length === 0
          ? 'Split search chart, empty because this node has no usable candidate splits.'
          : 'Two curves showing ' +
            (mode === 'gain' ? 'the gain' : 'weighted child ' + criterion) +
            ' at every candidate threshold, one curve per feature, with the greedy winner marked' +
            (chosen
              ? ' at ' + conditionText(chosen.feature, chosen.threshold, featureNames) + '.'
              : '.')
      }
      redrawKey={candidates.length + ':' + mode + ':' + (chosen?.threshold ?? 0)}
    />
  );
}

/* ---------------- 4. Gini against entropy ---------------- */

/** Gini when one of k classes holds share p and the others split the rest evenly. */
function giniOf(p: number, k: number): number {
  return 1 - p * p - ((1 - p) * (1 - p)) / (k - 1);
}

/** Entropy in bits for the same split. */
function entropyOf(p: number, k: number): number {
  const rest = (1 - p) / (k - 1);
  return -(p > 0 ? p * Math.log2(p) : 0) - (rest > 0 ? (1 - p) * Math.log2(rest) : 0);
}

export interface ImpurityCurvesProps {
  /** Class counts of the node under discussion. */
  counts: readonly number[];
  criterion: Criterion;
  /** Scale Gini by two so its peak lines up with entropy's. */
  scaleGini: boolean;
  nodeLabel: string;
}

export function ImpurityCurves({ counts, criterion, scaleGini, nodeLabel }: ImpurityCurvesProps) {
  const total = counts.reduce((a, b) => a + b, 0);
  const largest = total === 0 ? 0.5 : Math.max(...counts) / total;
  const actualGini = impurityOf(counts, 'gini');
  const actualEntropy = impurityOf(counts, 'entropy');
  const k = Math.max(2, counts.length);

  const draw = useCallback(
    ({ ctx, width, height, palette }: DrawArgs) => {
      // Entropy peaks at log₂ k bits; scaled Gini is stretched to the same peak.
      const peak = Math.log2(k);
      const stretch = peak / (1 - 1 / k);
      const frame = makeFrame(width, height, [0, 1], [0, Math.max(1, peak) * 1.08], {
        left: 44,
        bottom: 34,
        right: 14,
        top: 14,
      });
      drawAxes(ctx, frame, palette, {
        xLabel: 'p, share of the largest class',
        yLabel: 'impurity',
        xTicks: 5,
        yTicks: 4,
      });

      ctx.save();
      clipFrame(ctx, frame);

      const curve = (fn: (p: number) => number, colour: string, lineWidth: number, dash?: number[]) => {
        const pts: Array<{ x: number; y: number }> = [];
        for (let i = 0; i <= 200; i++) {
          const p = i / 200;
          pts.push({ x: frame.x(p), y: frame.y(fn(p)) });
        }
        drawPath(ctx, pts, colour, lineWidth, dash);
      };

      const giniActive = criterion === 'gini';
      curve((p) => giniOf(p, k), rgba(palette.violet, giniActive ? 1 : 0.55), giniActive ? 2.6 : 1.6);
      if (scaleGini) {
        curve(
          (p) => giniOf(p, k) * stretch,
          rgba(palette.violet, 0.6),
          1.6,
          [5, 4],
        );
      }
      curve((p) => entropyOf(p, k), rgba(palette.cyan, giniActive ? 0.55 : 1), giniActive ? 1.6 : 2.6);

      // Where this node actually sits.
      const px = frame.x(largest);
      drawPath(
        ctx,
        [
          { x: px, y: frame.bottom },
          { x: px, y: frame.top },
        ],
        rgba(palette.text, 0.4),
        1.3,
        [4, 4],
      );
      drawPoint(ctx, px, frame.y(actualGini), 4.5, palette.violet, rgba(palette.surface, 0.95), 1.8);
      drawPoint(ctx, px, frame.y(actualEntropy), 4.5, palette.cyan, rgba(palette.surface, 0.95), 1.8);
      ctx.restore();

      if (plateRoom(frame)) {
        drawLabelPlate(
          ctx,
          nodeLabel + ' · p = ' + fmt(largest, 2),
          clamp(px, frame.left + 6, frame.right - 6),
          frame.top + 10,
          palette,
          { align: largest > 0.6 ? 'right' : 'left', bold: true },
        );
      }
    },
    [criterion, scaleGini, largest, actualGini, actualEntropy, nodeLabel, k],
  );

  return (
    <Chart
      draw={draw}
      height={230}
      description={
        'Gini and entropy plotted against the class balance p, both zero at a pure node and largest at an even mix. The current node sits at p = ' +
        fmt(largest, 2) +
        ', Gini ' +
        fmt(actualGini, 3) +
        ', entropy ' +
        fmt(actualEntropy, 3) +
        '.'
      }
      redrawKey={largest + ':' + criterion + ':' + String(scaleGini) + ':' + k}
    />
  );
}

/* ---------------- 5. Accuracy as the tree grows ---------------- */

export interface GrowthPoint {
  train: number;
  test: number;
  leaves: number;
}

export interface GrowthCurveProps {
  history: readonly GrowthPoint[];
  /** How many splits the transport has taken. */
  current: number;
  /** Index of the best held-out score, or -1 when there is no held-out data. */
  best: number;
  hasTest: boolean;
}

export function GrowthCurve({ history, current, best, hasTest }: GrowthCurveProps) {
  const draw = useCallback(
    ({ ctx, width, height, palette }: DrawArgs) => {
      if (history.length < 2) {
        drawEmptyState(
          ctx,
          width,
          height,
          palette.textFaint,
          'This tree cannot be split at all; there is no growth curve to draw.',
        );
        return;
      }

      const lows = history.flatMap((h) => (hasTest ? [h.train, h.test] : [h.train]));
      const lo = Math.max(0, Math.min(...lows) - 0.06);
      const frame = makeFrame(width, height, [0, history.length - 1], [lo, 1.02], {
        left: 46,
        bottom: 34,
        right: 14,
        top: 14,
      });
      drawAxes(ctx, frame, palette, {
        xLabel: 'splits added',
        yLabel: 'accuracy',
        xTicks: 6,
        yTicks: 4,
        formatY: (v) => Math.round(v * 100) + '%',
      });

      const series = [
        {
          values: history.map((h) => h.train),
          colour: palette.violet,
          width: 2,
          dash: [5, 4],
          label: 'train',
        },
      ];
      if (hasTest) {
        series.push({
          values: history.map((h) => h.test),
          colour: palette.orange,
          width: 2.4,
          dash: [],
          label: 'held out',
        });
      }
      drawSeries(ctx, frame, palette, series, { markLast: false });

      ctx.save();
      clipFrame(ctx, frame);
      if (hasTest && best >= 0 && best < history.length) {
        const x = frame.x(best);
        const y = frame.y(history[best].test);
        drawPoint(ctx, x, y, 4.5, palette.orange, rgba(palette.surface, 0.95), 2);
        if (plateRoom(frame)) {
          drawLabelPlate(
            ctx,
            'best held-out ' + fmtPercent(history[best].test, 0) + ' at ' + best + ' splits',
            clamp(x, frame.left + 4, frame.right - 4),
            clamp(y - 15, frame.top + 10, frame.bottom - 10),
            palette,
            { align: best > history.length / 2 ? 'right' : 'left' },
          );
        }
      }

      const cx = frame.x(clamp(current, 0, history.length - 1));
      drawPath(
        ctx,
        [
          { x: cx, y: frame.bottom },
          { x: cx, y: frame.top },
        ],
        rgba(palette.text, 0.5),
        1.4,
      );
      ctx.restore();
    },
    [history, current, best, hasTest],
  );

  return (
    <Chart
      draw={draw}
      height={230}
      description={
        history.length < 2
          ? 'Growth curve, empty because this tree cannot be split.'
          : 'Training and held-out accuracy plotted against the number of splits added, over ' +
            (history.length - 1) +
            ' splits, with a marker at the current step.'
      }
      redrawKey={history.length + ':' + current + ':' + String(hasTest)}
    />
  );
}
