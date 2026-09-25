/** The detail card and the pieces that fill it: vector tables, scalar chips, the weighted sum. */

import { Fragment, createContext, useContext, useEffect, useId, useMemo, useRef, useState } from 'react';
import type { CSSProperties, ReactNode } from 'react';

import { usePalette } from '../../components/ThemeProvider';
import { fmt, fmtCell } from '../../lib/math/stats';
import { readPalette, readableOn, rgba } from '../../lib/viz/palette';
import { useElementSize } from '../useElementSize';
import { columnScale, vectorColour } from '../vectors';
import type { VectorColumn } from '../vectors';
import { IconClose } from './Icons';
import { Math as Tex } from './Panels';

/* ---------------- placement ---------------- */

export interface Anchor {
  x: number;
  y: number;
  w: number;
  h: number;
}

export interface DetailOverlayProps {
  /** Sit in the diagram's side column instead of floating next to `anchor`. */
  dock?: boolean;
  /** The component's rectangle, in the diagram's own pixels. */
  anchor?: Anchor;
  /** The diagram's size, so a floating card stays inside it. */
  bounds?: { width: number; height: number };
  /** Visible window inside bounds, in anchor pixels; defaults to all of bounds. */
  viewport?: { x: number; y: number; width: number; height: number };
  onClose: () => void;
  children: ReactNode;
}

const GAP = 14;
const EDGE = 8;

const DetailTitleId = createContext<string | undefined>(undefined);

export function DetailOverlay({ dock, anchor, bounds, viewport, onClose, children }: DetailOverlayProps) {
  const [setCard, cardSize, cardRef] = useElementSize<HTMLDivElement>();
  const titleId = useId();

  // Capture phase so the card takes Escape first, unless a full-screen panel elsewhere is the top layer.
  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if (event.key !== 'Escape') return;
      // An open menu or tooltip closes first.
      if (document.querySelector('.mlx-popover__pop, .mlx-tip__bubble')) return;
      const layer = document.querySelector('.mlx-panel--full');
      if (layer && !layer.contains(cardRef.current)) return;
      event.stopImmediatePropagation();
      event.preventDefault();
      onClose();
    };
    window.addEventListener('keydown', onKey, true);
    return () => window.removeEventListener('keydown', onKey, true);
  }, [onClose, cardRef]);

  // Focus moves into the card once placed, and back out on close.
  const focused = useRef(false);
  useEffect(() => {
    const previous = document.activeElement as HTMLElement | null;
    return () => {
      if (previous?.isConnected) previous.focus({ preventScroll: true });
    };
  }, []);
  useEffect(() => {
    if (cardSize.height && !focused.current) {
      focused.current = true;
      cardRef.current?.focus({ preventScroll: true });
    }
  }, [cardSize.height, cardRef]);

  const style: CSSProperties = dock ? {} : pinned(anchor, bounds, viewport, cardSize.height);

  return (
    <div
      ref={setCard}
      className={'mlx-detail' + (dock ? ' mlx-detail--dock' : '')}
      style={{ ...style, visibility: dock || cardSize.height ? 'visible' : 'hidden' }}
      role="dialog"
      aria-label="Component detail"
      aria-labelledby={titleId}
      tabIndex={-1}
      onPointerDown={(event) => event.stopPropagation()}
    >
      <button type="button" className="mlx-detail__close" onClick={onClose} aria-label="Close">
        <IconClose size={13} />
      </button>
      <div className="mlx-detail__body">
        <DetailTitleId.Provider value={titleId}>{children}</DetailTitleId.Provider>
      </div>
    </div>
  );
}

/** Where a floating card sits: beside the component where there is room, always inside the frame. */
function pinned(
  anchor: Anchor | undefined,
  bounds: { width: number; height: number } | undefined,
  viewport: DetailOverlayProps['viewport'],
  measured: number,
): CSSProperties {
  if (!anchor || !bounds) return {};
  const view = viewport ?? { x: 0, y: 0, width: bounds.width, height: bounds.height };
  // A wider card when the diagram has the room.
  const width = Math.min(view.width > 1000 ? 360 : 300, Math.max(200, view.width - EDGE * 2));
  const height = measured || 160;
  const cx = anchor.x + anchor.w / 2;
  const cy = anchor.y + anchor.h / 2;

  let side: 'right' | 'left' | 'below' | 'above' = 'right';
  if (anchor.x + anchor.w + GAP + width > view.x + view.width - EDGE) {
    side = anchor.x - GAP - width >= view.x + EDGE ? 'left' : 'below';
  }
  if (side === 'below' && anchor.y + anchor.h + GAP + height > view.y + view.height - EDGE) {
    side = 'above';
  }

  let left: number;
  let top: number;
  if (side === 'right') {
    left = anchor.x + anchor.w + GAP;
    top = cy - height / 2;
  } else if (side === 'left') {
    left = anchor.x - GAP - width;
    top = cy - height / 2;
  } else if (side === 'below') {
    left = cx - width / 2;
    top = anchor.y + anchor.h + GAP;
  } else {
    left = cx - width / 2;
    top = anchor.y - GAP - height;
  }
  left = Math.max(view.x + EDGE, Math.min(view.x + view.width - width - EDGE, left));
  top = Math.max(view.y + EDGE, Math.min(Math.max(view.y + EDGE, view.y + view.height - height - EDGE), top));
  return { left, top, width };
}

/* ---------------- content pieces ---------------- */

export function DetailHead({
  eyebrow,
  title,
  shape,
}: {
  eyebrow: string;
  title: ReactNode;
  /** The sizes involved: 'unit 2 of 4 · 2 weights + bias'. */
  shape?: string;
}) {
  const titleId = useContext(DetailTitleId);
  return (
    <header className="mlx-detail__head">
      <p className="mlx-detail__eyebrow">{eyebrow}</p>
      <h3 id={titleId} className="mlx-detail__title">
        {title}
      </h3>
      {shape ? (
        <p className="mlx-detail__shape">
          {shape.split(' · ').map((part, i) => (
            <Fragment key={i}>
              {i > 0 ? ' ' : null}
              <span className="mlx-detail__shape-part" style={{ whiteSpace: 'nowrap' }}>
                {i > 0 ? '· ' : ''}
                {part}
              </span>
            </Fragment>
          ))}
        </p>
      ) : null}
    </header>
  );
}

export interface WeightedRow {
  /** 'x₁', 'x²', '1'. */
  name: string;
  value: number;
  weight: number;
  /** Symbol of the weight, for the header and the jump. */
  weightName?: string;
}

/** The weighted sum as one table: input, weight, product, then the sum and what follows it. */
export function WeightedSumTable({
  rows,
  bias,
  columns = ['input', 'value', 'weight', 'product'],
  sumLabel = 'z',
  noTotal,
  result,
  decimals = 2,
  onSelectInput,
  onSelectWeight,
}: {
  rows: readonly WeightedRow[];
  bias?: number;
  /** Column headings, when the table is not literally inputs and weights. */
  columns?: readonly [string, string, string, string];
  sumLabel?: string;
  /** Leave the total out, for tables whose rows do not add up to anything. */
  noTotal?: boolean;
  /** The line after the sum: ['tanh(z)', 0.917]. */
  result?: { label: string; value: number | string; tone?: 'accent' | 'good' | 'bad' };
  decimals?: number;
  onSelectInput?: (index: number) => void;
  onSelectWeight?: (index: number) => void;
}) {
  const palette = usePalette();
  let scale = 0;
  for (const row of rows) scale = Math.max(scale, Math.abs(row.weight));
  let total = bias ?? 0;
  for (const row of rows) total += row.value * row.weight;

  const cell = (text: string, index: number, onSelect?: (index: number) => void, style?: CSSProperties) =>
    onSelect ? (
      <button type="button" className="mlx-wsum__jump" style={style} onClick={() => onSelect(index)} title="Open it">
        {text}
      </button>
    ) : (
      <span style={style}>{text}</span>
    );

  return (
    <table className="mlx-wsum mlx-num">
      <thead>
        <tr>
          {columns.map((heading) => (
            <th key={heading} scope="col">
              {heading}
            </th>
          ))}
        </tr>
      </thead>
      <tbody>
        {rows.map((row, index) => (
          <tr key={row.name + index}>
            <th scope="row">{cell(row.name, index, onSelectInput)}</th>
            <td>{fmt(row.value, decimals)}</td>
            <td>{cell(fmt(row.weight, decimals), index, onSelectWeight, cellStyle(row.weight, scale, palette))}</td>
            <td>{fmt(row.value * row.weight, decimals)}</td>
          </tr>
        ))}
        {bias !== undefined ? (
          <tr>
            <th scope="row">bias</th>
            <td />
            <td>{fmt(bias, decimals)}</td>
            <td>{fmt(bias, decimals)}</td>
          </tr>
        ) : null}
      </tbody>
      <tfoot>
        {noTotal ? null : (
          <tr>
            <th scope="row" colSpan={3}>
              {sumLabel} = sum
            </th>
            <td>{fmt(total, 3)}</td>
          </tr>
        )}
        {result ? (
          <tr data-tone={result.tone ?? 'accent'}>
            <th scope="row" colSpan={3}>
              {result.label}
            </th>
            <td>{typeof result.value === 'number' ? fmt(result.value, 3) : result.value}</td>
          </tr>
        ) : null}
      </tfoot>
    </table>
  );
}

/** One or two sentences saying what the component does. */
export function Annotation({ children }: { children: ReactNode }) {
  return <p className="mlx-detail__note">{children}</p>;
}

/** Colour a cell by its value: sign picks the hue, size picks the depth. */
function cellStyle(value: number, scale: number, palette = readPalette()): CSSProperties {
  const t = scale > 0 ? Math.max(-1, Math.min(1, value / scale)) : 0;
  const hue = t < 0 ? palette.negative : palette.positive;
  return { background: rgba(hue, 0.12 + 0.55 * Math.abs(t)) };
}

/** The equation a component computes, with its live numbers. */
export function Formula({ tex }: { tex: string }) {
  return (
    <div className="mlx-formula">
      <Tex tex={tex} display />
    </div>
  );
}

type Sort = { name: string; descending: boolean } | null;

function meanOf(values: readonly number[], rows: readonly number[]): number {
  let sum = 0;
  let count = 0;
  for (const i of rows) {
    const v = values[i];
    if (Number.isFinite(v)) {
      sum += v;
      count += 1;
    }
  }
  return count > 0 ? sum / count : NaN;
}

function sdOf(values: readonly number[], rows: readonly number[], mean: number): number {
  let sum = 0;
  let count = 0;
  for (const i of rows) {
    const v = values[i];
    if (Number.isFinite(v)) {
      sum += (v - mean) ** 2;
      count += 1;
    }
  }
  return count > 0 ? Math.sqrt(sum / count) : NaN;
}

/** A vector over the dataset: one row per point, with stats in the head and sortable columns. */
export function VectorTable({
  columns,
  order,
  active,
  decimals = 3,
  classes,
  extra,
  limit,
}: {
  columns: readonly VectorColumn[];
  /** Row order, indices into each column. */
  order: readonly number[];
  /** Rows the last step trained on; the rest are dimmed. */
  active?: ReadonlySet<number> | null;
  decimals?: number;
  /** 0 or 1 per point: the stats split by class instead of mean and sd. */
  classes?: readonly number[];
  /** The inputs, folded behind one chip; open to start with when there are few. */
  extra?: readonly VectorColumn[];
  /** Show this many rows after the stats, the rest behind a toggle. */
  limit?: number;
}) {
  const palette = usePalette();
  const [sort, setSort] = useState<Sort>(null);
  const [showExtra, setShowExtra] = useState((extra?.length ?? 0) <= 2);
  const [showAll, setShowAll] = useState(false);
  const shown = useMemo(
    () => (extra && extra.length > 0 && showExtra ? [...extra, ...columns] : columns),
    [columns, extra, showExtra],
  );
  const scales = shown.map(columnScale);

  const rows = useMemo(() => {
    const column = sort ? shown.find((c) => c.name === sort.name) : undefined;
    if (!sort || !column) return order;
    const values = column.values;
    const sorted = order.slice();
    sorted.sort((a, b) => ((values[a] ?? 0) - (values[b] ?? 0)) * (sort.descending ? -1 : 1));
    return sorted;
  }, [order, sort, shown]);
  const capped = limit !== undefined && !showAll && rows.length > limit;
  const visible = capped ? rows.slice(0, limit) : rows;

  const stats = useMemo<Array<{ label: string; tint?: string; values: number[] }>>(() => {
    if (classes) {
      const groups = [0, 1].map((c) => order.filter((i) => Math.round(classes[i] ?? 0) === c));
      return groups.map((rowsOf, c) => ({
        label: 'μ' + (c === 0 ? '₀' : '₁'),
        tint: c === 0 ? palette.classA : palette.classB,
        values: shown.map((column) => meanOf(column.values, rowsOf)),
      }));
    }
    const means = shown.map((column) => meanOf(column.values, order));
    return [
      { label: 'μ', values: means },
      { label: 'σ', values: shown.map((column, c) => sdOf(column.values, order, means[c])) },
    ];
  }, [classes, order, shown, palette.classA, palette.classB]);

  const toggleSort = (name: string) =>
    setSort((prev) =>
      prev && prev.name === name
        ? prev.descending
          ? { name, descending: false }
          : null
        : { name, descending: true },
    );

  const show = (column: VectorColumn, value: number) =>
    column.ramp === 'class' ? fmt(value, 0) : fmtCell(value, column.decimals ?? decimals);

  return (
    <div className="mlx-vtable">
      {extra && extra.length > 0 ? (
        <div className="mlx-vtable__bar">
          <button
            type="button"
            className="mlx-vtable__toggle"
            data-on={showExtra || undefined}
            aria-pressed={showExtra}
            onClick={() => setShowExtra((v) => !v)}
            title={showExtra ? 'Hide the inputs' : 'Show the inputs'}
          >
            {extra.length <= 2 ? extra.map((c) => c.name).join(', ') : extra[0].name + ' … ' + extra[extra.length - 1].name}
          </button>
        </div>
      ) : null}
      <table className="mlx-wsum mlx-num">
        <thead>
          <tr>
            <th scope="col">
              <button
                type="button"
                className="mlx-vtable__sort"
                data-on={sort === null || undefined}
                onClick={() => setSort(null)}
                title="Diagram order"
              >
                pt
              </button>
            </th>
            {shown.map((column, c) => (
              <th key={c} scope="col">
                <button
                  type="button"
                  className="mlx-vtable__sort"
                  data-on={sort?.name === column.name || undefined}
                  onClick={() => toggleSort(column.name)}
                  title="Sort by this column"
                >
                  {column.name}
                  {sort?.name === column.name ? (sort.descending ? ' ▾' : ' ▴') : ''}
                </button>
              </th>
            ))}
          </tr>
          {stats.map((row) => (
            <tr key={row.label} className="mlx-vtable__stat">
              <th scope="row" style={row.tint ? { color: row.tint } : undefined}>
                {row.label}
              </th>
              {row.values.map((value, c) => (
                <td key={c}>
                  {Number.isFinite(value) ? fmtCell(value, shown[c].ramp === 'class' ? 2 : Math.min(3, shown[c].decimals ?? decimals)) : '·'}
                </td>
              ))}
            </tr>
          ))}
        </thead>
        <tbody>
          {visible.map((index) => (
            <tr key={index} data-dim={active && !active.has(index) ? '' : undefined}>
              <th scope="row">{index + 1}</th>
              {shown.map((column, c) => {
                const value = column.values[index] ?? NaN;
                const background = vectorColour(palette, column.ramp, value, scales[c]);
                return (
                  <td key={c}>
                    <span className="mlx-vtable__cell" style={{ background, color: readableOn(background) }}>
                      {show(column, value)}
                    </span>
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
      {limit !== undefined && rows.length > limit ? (
        <button type="button" className="mlx-vtable__toggle mlx-vtable__more" onClick={() => setShowAll((v) => !v)}>
          {showAll ? 'First ' + limit + ' rows' : 'All ' + rows.length + ' rows'}
        </button>
      ) : null}
    </div>
  );
}

/** A single number with its symbol, for biases, scores and outputs. */
export function Scalar({
  label,
  value,
  decimals = 3,
  tone,
}: {
  label: ReactNode;
  value: number | string;
  decimals?: number;
  tone?: 'accent' | 'good' | 'bad';
}) {
  return (
    <span className="mlx-scalar" data-tone={tone}>
      <span className="mlx-scalar__label">{label}</span>
      <span className="mlx-scalar__value">
        {typeof value === 'number' ? fmtCell(value, decimals, true) : value}
      </span>
    </span>
  );
}

/** Scalars side by side. */
export function Scalars({ children }: { children: ReactNode }) {
  return <div className="mlx-scalars">{children}</div>;
}
