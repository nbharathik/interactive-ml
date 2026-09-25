/** The model as a map: data, features, weights, prediction and error down one spine, the penalty branching off the weights and joining at the plus. Live numbers, one station per hover. */

import { useCallback, useRef } from 'react';

import { Chart } from '../../explainer/components/Chart';
import type { DrawArgs } from '../../explainer/components/Chart';
import { drawHoverPlate, measurePlate, strokeOutline } from '../../explainer/diagramStyle';
import type { Rect } from '../../explainer/architecture';
import { FONT_STACK, MONO_STACK, roundRect } from '../../lib/viz/canvas';
import { rgba } from '../../lib/viz/palette';
import { fmtCompact, fmtKnob } from '../../lib/math/stats';
import { featureColour } from './panels';
import { featureOf, nodeOf, sameTarget, targetKey } from './targets';
import type { ModelNode, RegTarget, Spot } from './targets';

export interface ModelMapProps {
  featureNames: readonly string[];
  weights: readonly number[];
  /** The curve's powers of x, or the table's columns. */
  curve: boolean;
  trainCount: number;
  testCount: number;
  lambda: number;
  alpha: number;
  /** The smooth half of the objective, the penalty, and their sum, as the solver sees them. */
  error: number;
  penalty: number;
  objective: number;
  /** The two weights the plane draws, marked under their cells. */
  pair: [number, number];
  hover: RegTarget | null;
  open: RegTarget | null;
  spot: Spot | null;
  onHover: (target: RegTarget | null) => void;
  onOpen: (target: RegTarget | null) => void;
}

interface Station {
  id: ModelNode;
  rect: Rect;
  round: boolean;
  dashed?: boolean;
  symbol: string;
  /** The name over the value, to the right of the station. */
  name: string;
  value: string[];
  /** Where the name and value go. */
  labelAt: 'right' | 'below';
}

interface Cell {
  index: number;
  rect: Rect;
}

const CELL = 18;
const CELL_COLUMNS = 5;
const BOX_W = 48;
const BOX_H = 26;
const PEN_W = 84;
const PAD = 10;
const GAP = 16;

export function penaltyName(alpha: number): string {
  return alpha === 1 ? 'Σ|w|' : alpha === 0 ? '½Σw²' : 'R(w)';
}

/** The map's height follows the number of feature cells. */
export function mapHeight(p: number): number {
  const rows = Math.max(1, Math.ceil(p / CELL_COLUMNS));
  return 234 + Math.max(rows * CELL + 8, 36);
}

export function ModelMap(props: ModelMapProps) {
  const { featureNames, weights, curve, trainCount, testCount, lambda, alpha, error, penalty, objective, pair, hover, open, spot, onHover, onOpen } = props;
  const stationsRef = useRef<Station[]>([]);
  const cellsRef = useRef<Cell[]>([]);
  const p = weights.length;
  const nonZero = weights.filter((v) => Number.isFinite(v) && v !== 0).length;
  const maxAbs = weights.reduce((m, v) => (Number.isFinite(v) ? Math.max(m, Math.abs(v)) : m), 0);

  const draw = useCallback(
    ({ ctx, width, height, palette }: DrawArgs) => {
      const rows = Math.max(1, Math.ceil(p / CELL_COLUMNS));
      const featW = Math.min(CELL_COLUMNS, Math.max(1, p)) * CELL + 8;
      const featH = rows * CELL + 8;
      const spine = PAD + BOX_W / 2;
      const rectAt = (cx: number, cy: number, w: number, h: number): Rect => ({ x: cx - w / 2, y: cy - h / 2, w, h });
      // Down the spine: data, features, sum, prediction, error, objective.
      let y = PAD;
      const dataY = y + BOX_H / 2;
      y += BOX_H + GAP;
      // The features row is at least as tall as its three label lines.
      const featRow = Math.max(featH, 36);
      const featY = y + featRow / 2;
      y += featRow + GAP + 10;
      const sumY = y + BOX_H / 2;
      y += BOX_H + GAP;
      const predY = y + BOX_H / 2;
      y += BOX_H + GAP;
      const errY = y + BOX_H / 2;
      y += BOX_H + GAP;
      const objY = y + BOX_H / 2;
      // On a wide column the map keeps its shape and leaves the rest empty.
      const penX = Math.min(width, 320) - PAD - PEN_W / 2;
      const penY = predY;
      const weightAt = (j: number) => (Number.isFinite(weights[j]) ? weights[j] : 0);
      const rawPenalty = penalty / Math.max(lambda, 1e-12);

      const stations: Station[] = [
        {
          id: 'data',
          rect: rectAt(spine, dataY, BOX_W, BOX_H),
          round: false,
          dashed: true,
          symbol: curve ? 'x, y' : 'X, y',
          name: 'DATA',
          value: [trainCount + ' fitted + ' + testCount + ' held out'],
          labelAt: 'right',
        },
        {
          id: 'features',
          rect: { x: PAD, y: featY - featH / 2, w: featW, h: featH },
          round: false,
          symbol: '',
          name: 'FEATURES',
          value: [p + (curve ? ' powers of x' : ' columns'), nonZero + ' of ' + p + ' non-zero'],
          labelAt: 'right',
        },
        { id: 'sum', rect: rectAt(spine, sumY, BOX_H, BOX_H), round: true, symbol: 'Σ', name: 'SUM', value: ['Σ wⱼxⱼ + b'], labelAt: 'right' },
        { id: 'prediction', rect: rectAt(spine, predY, BOX_W, BOX_H), round: false, symbol: 'ŷ', name: 'PREDICTION', value: [curve ? 'the curve' : 'one per row'], labelAt: 'right' },
        { id: 'error', rect: rectAt(spine, errY, BOX_W + 14, BOX_H), round: false, symbol: '(y − ŷ)²', name: 'ERROR', value: [fmtCompact(error)], labelAt: 'right' },
        {
          id: 'penalty',
          rect: rectAt(penX, penY, PEN_W, BOX_H),
          round: false,
          symbol: 'λ · ' + penaltyName(alpha),
          name: 'PENALTY',
          value: [fmtKnob(lambda) + ' × ' + fmtCompact(rawPenalty), '= ' + fmtCompact(penalty)],
          labelAt: 'below',
        },
        { id: 'objective', rect: rectAt(spine, objY, BOX_H, BOX_H), round: true, symbol: '+', name: 'OBJECTIVE', value: [fmtCompact(objective)], labelAt: 'right' },
      ];
      stationsRef.current = stations;
      const at = (id: ModelNode) => stations.find((s) => s.id === id)!;

      const litNode = nodeOf(open) ?? nodeOf(hover);
      const litFeature = featureOf(open) ?? featureOf(hover);
      const wireColour = rgba(palette.textMuted, 0.65);
      const arrowDown = (x: number, yEnd: number) => {
        ctx.beginPath();
        ctx.moveTo(x, yEnd + 1);
        ctx.lineTo(x - 3.3, yEnd - 5);
        ctx.lineTo(x, yEnd - 3.4);
        ctx.lineTo(x + 3.3, yEnd - 5);
        ctx.closePath();
        ctx.fillStyle = wireColour;
        ctx.fill();
      };
      const arrowLeft = (xEnd: number, yy: number) => {
        ctx.beginPath();
        ctx.moveTo(xEnd - 1, yy);
        ctx.lineTo(xEnd + 5, yy - 3.3);
        ctx.lineTo(xEnd + 3.4, yy);
        ctx.lineTo(xEnd + 5, yy + 3.3);
        ctx.closePath();
        ctx.fillStyle = wireColour;
        ctx.fill();
      };
      ctx.save();
      ctx.strokeStyle = wireColour;
      ctx.lineWidth = 1.5;
      ctx.lineCap = 'round';
      ctx.lineJoin = 'round';
      const spineOrder: ModelNode[] = ['data', 'features', 'sum', 'prediction', 'error', 'objective'];
      for (let k = 1; k < spineOrder.length; k++) {
        const from = at(spineOrder[k - 1]).rect;
        const to = at(spineOrder[k]).rect;
        ctx.beginPath();
        ctx.moveTo(spine, from.y + from.h);
        ctx.lineTo(spine, to.y);
        ctx.stroke();
        arrowDown(spine, to.y);
      }
      // The penalty reads the weights off the wire into the sum, and joins the plus from the right.
      const wY = (at('features').rect.y + at('features').rect.h + at('sum').rect.y) / 2;
      const pen = at('penalty').rect;
      const obj = at('objective').rect;
      ctx.beginPath();
      ctx.moveTo(spine + 4, wY);
      ctx.lineTo(penX - 6, wY);
      ctx.quadraticCurveTo(penX, wY, penX, wY + 6);
      ctx.lineTo(penX, pen.y);
      ctx.stroke();
      arrowDown(penX, pen.y);
      ctx.beginPath();
      ctx.moveTo(penX, pen.y + pen.h);
      ctx.lineTo(penX, objY - 6);
      ctx.quadraticCurveTo(penX, objY, penX - 6, objY);
      ctx.lineTo(obj.x + obj.w, objY);
      ctx.stroke();
      arrowLeft(obj.x + obj.w, objY);
      ctx.restore();

      // The weight symbol on the wire into the sum.
      ctx.save();
      ctx.font = '600 11px ' + FONT_STACK;
      ctx.fillStyle = litFeature !== null ? palette.accent : palette.textMuted;
      ctx.textAlign = 'left';
      ctx.textBaseline = 'alphabetic';
      ctx.fillText('× w', spine + 10, wY - 4);
      ctx.restore();

      for (const s of stations) {
        const { rect } = s;
        const outline = () => {
          if (s.round) {
            ctx.beginPath();
            ctx.arc(rect.x + rect.w / 2, rect.y + rect.h / 2, rect.w / 2, 0, Math.PI * 2);
          } else roundRect(ctx, rect.x, rect.y, rect.w, rect.h, 4);
        };
        const lit = litNode === s.id;
        ctx.save();
        outline();
        ctx.fillStyle = palette.surface;
        ctx.fill();
        ctx.fillStyle = rgba(palette.accent, lit ? 0.16 : s.id === 'data' ? 0.04 : s.round ? 0.1 : 0.08);
        ctx.fill();
        outline();
        ctx.strokeStyle = rgba(palette.accent, lit ? 0.85 : 0.5);
        ctx.lineWidth = 1.1;
        if (s.dashed) ctx.setLineDash([4, 3]);
        ctx.stroke();
        if (lit) strokeOutline(ctx, outline, open && nodeOf(open) === s.id ? 'open' : 'hover', palette);
        if (spot && spot.target.kind === 'node' && spot.target.id === s.id) strokeOutline(ctx, outline, 'spot', palette);
        ctx.restore();

        if (s.id === 'features') {
          const cells: Cell[] = [];
          for (let j = 0; j < p; j++) {
            const col = j % CELL_COLUMNS;
            const row = Math.floor(j / CELL_COLUMNS);
            const cr: Rect = { x: rect.x + 4 + col * CELL, y: rect.y + 4 + row * CELL, w: CELL, h: CELL };
            cells.push({ index: j, rect: cr });
            const w = weightAt(j);
            const share = maxAbs > 0 ? Math.abs(w) / maxAbs : 0;
            const emphasised = litFeature === null || litFeature === j;
            ctx.save();
            ctx.globalAlpha = emphasised ? 1 : 0.35;
            roundRect(ctx, cr.x + 1, cr.y + 1, cr.w - 2, cr.h - 2, 3);
            ctx.fillStyle = w === 0 ? rgba(palette.textFaint, 0.12) : rgba(featureColour(palette, j), 0.18 + 0.7 * share);
            ctx.fill();
            if (litFeature === j) {
              ctx.strokeStyle = palette.accent;
              ctx.lineWidth = 1.5;
              ctx.stroke();
            }
            ctx.font = '600 8px ' + MONO_STACK;
            ctx.textAlign = 'center';
            ctx.textBaseline = 'middle';
            ctx.fillStyle = w === 0 ? palette.textFaint : palette.text;
            ctx.fillText(featureNames[j] ?? String(j + 1), cr.x + cr.w / 2, cr.y + cr.h / 2 + 0.5, cr.w - 2);
            // The two weights the plane draws carry a tick under their cell.
            if (j === pair[0] || j === pair[1]) {
              ctx.fillStyle = rgba(palette.accent, 0.9);
              ctx.fillRect(cr.x + 4, cr.y + cr.h - 2.5, cr.w - 8, 1.5);
            }
            ctx.restore();
            if (spot && spot.target.kind === 'feature' && spot.target.index === j) {
              strokeOutline(ctx, () => roundRect(ctx, cr.x + 1, cr.y + 1, cr.w - 2, cr.h - 2, 3), 'spot', palette);
            }
          }
          cellsRef.current = cells;
        } else if (s.symbol) {
          ctx.save();
          ctx.textAlign = 'center';
          ctx.textBaseline = 'middle';
          ctx.font = (s.round ? '500 16px ' : '650 11px ') + FONT_STACK;
          ctx.fillStyle = palette.text;
          ctx.fillText(s.symbol, rect.x + rect.w / 2, rect.y + rect.h / 2 + 1, rect.w - 6);
          ctx.restore();
        }

        // The station's name, then its numbers.
        ctx.save();
        const nameColour = lit ? palette.text : palette.textMuted;
        if (s.labelAt === 'right') {
          const lx = rect.x + rect.w + 9;
          const room = (s.id === 'data' || s.id === 'features' ? Math.min(width, 320) - PAD : pen.x - 8) - lx;
          ctx.textAlign = 'left';
          ctx.textBaseline = 'alphabetic';
          ctx.font = '700 9px ' + FONT_STACK;
          ctx.fillStyle = nameColour;
          const lines = 1 + s.value.length;
          const top = rect.y + rect.h / 2 - (lines * 11) / 2 + 9;
          ctx.fillText(s.name, lx, top, room);
          ctx.font = '9px ' + MONO_STACK;
          ctx.fillStyle = palette.textFaint;
          s.value.forEach((line, k) => ctx.fillText(line, lx, top + 11 * (k + 1), room));
        } else {
          ctx.textAlign = 'center';
          ctx.textBaseline = 'alphabetic';
          ctx.font = '700 9px ' + FONT_STACK;
          ctx.fillStyle = nameColour;
          ctx.fillText(s.name, rect.x + rect.w / 2, rect.y - 5, rect.w + 20);
          ctx.font = '9px ' + MONO_STACK;
          ctx.fillStyle = palette.textFaint;
          s.value.forEach((line, k) => ctx.fillText(line, rect.x + rect.w / 2, rect.y + rect.h + 11 + 11 * k, rect.w + 24));
        }
        ctx.restore();
      }

      // The lesson's label under its station.
      if (spot && spot.target.kind !== 'pane') {
        let r: Rect | undefined;
        if (spot.target.kind === 'node') r = at(spot.target.id).rect;
        else if (spot.target.kind === 'feature') {
          const index = spot.target.index;
          r = cellsRef.current.find((c) => c.index === index)?.rect;
        }
        if (r) {
          const size = measurePlate(ctx, [spot.label]);
          const below = r.y + r.h + 4 + size.h <= height - 2;
          const left = Math.max(4, Math.min(width - size.w - 4, r.x + r.w / 2 - size.w / 2));
          drawHoverPlate(ctx, [spot.label], left, below ? r.y + r.h + 4 : Math.max(2, r.y - size.h - 4), palette);
        }
      }
    },
    [p, curve, trainCount, testCount, nonZero, error, alpha, lambda, penalty, objective, pair, spot, weights, maxAbs, hover, open, featureNames],
  );

  const locate = useCallback((pos: { x: number; y: number }): RegTarget | null => {
    const inside = (r: Rect, slack = 0) => pos.x >= r.x - slack && pos.x <= r.x + r.w + slack && pos.y >= r.y - slack && pos.y <= r.y + r.h + slack;
    const cell = cellsRef.current.find((c) => inside(c.rect));
    if (cell) return { kind: 'feature', index: cell.index };
    const station = stationsRef.current.find((s) => inside(s.rect, 3));
    return station ? { kind: 'node', id: station.id } : null;
  }, []);

  return (
    <div className="mlx-inspector__map">
      <Chart
        draw={draw}
        height={mapHeight(p)}
        description={
          'The model as a map: ' + trainCount + ' rows in, ' + p + ' features times their weights added up into a prediction, its error plus λ times the penalty into the objective. ' +
          nonZero + ' of ' + p + ' weights are non-zero. Hover a station for its numbers, click to open it.'
        }
        cursor="pointer"
        onPointerDown={(pos) => {
          const found = locate(pos);
          onOpen(found && sameTarget(found, open) ? null : found);
        }}
        onPointerMove={(pos) => onHover(pos ? locate(pos) : null)}
        onPointerLeave={() => onHover(null)}
        redrawKey={[weights.map((v) => v.toFixed(3)).join(','), lambda, alpha, error, penalty, pair.join(), spot?.label, targetKey(hover), targetKey(open)].join('|')}
      />
    </div>
  );
}
