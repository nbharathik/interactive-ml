/** A fixed column beside a diagram: the open card or a quiet reference, with the hover numbers and the setup folded at the foot. */

import { useState } from 'react';
import type { ReactNode } from 'react';

import { readFlag, writeFlag } from '../storage';
import { ChartFillContext } from './Chart';
import { DetailOverlay } from './Detail';

const HOVER_KEY = 'mlx-hover-numbers';

export interface ReadoutRow {
  /** The symbol. */
  label: string;
  value: string;
}

export interface Readout {
  title: string;
  rows: ReadoutRow[];
}

/** The settings behind the picture: one line folded, the rows when opened. */
export interface Setup {
  summary: string;
  rows: ReadoutRow[];
}

export interface InspectorProps {
  /** What is under the pointer, or null. */
  readout: Readout | null;
  /** Fills the readout while nothing is under the pointer. */
  hint: string;
  /** The settings in use, folded at the foot. */
  setup?: Setup;
  /** The open card, or null. */
  card: ReactNode | null;
  onClose: () => void;
  /** Sits under the header while no card is open. */
  idle?: ReactNode;
  /** Always at the top of the column, such as a map of the whole model. */
  header?: ReactNode;
  /** Always at the foot of the column. */
  footer?: ReactNode;
}

export function Inspector({ readout, hint, setup, card, onClose, idle, header, footer }: InspectorProps) {
  // Folded by default so moving the pointer changes nothing; the choice carries across pages.
  const [numbers, setNumbers] = useState(() => readFlag(HOVER_KEY));
  const toggleNumbers = (open: boolean) => {
    if (open === numbers) return;
    writeFlag(HOVER_KEY, open);
    setNumbers(open);
  };

  // Charts in the column keep their own height, even inside a panel that fills the screen.
  return (
    <ChartFillContext.Provider value={false}>
      <aside className="mlx-inspector" aria-label="Inspector">
        {header ? <div className="mlx-inspector__head">{header}</div> : null}
        {card ? (
          <DetailOverlay dock onClose={onClose}>
            {card}
          </DetailOverlay>
        ) : idle ? (
          <div className="mlx-inspector__idle">{idle}</div>
        ) : null}
        <div className="mlx-inspector__foot">
          {setup ? (
            <details className="mlx-fold mlx-fold--setup">
              <summary className="mlx-fold__summary">
                <span className="mlx-fold__label">Setup</span>
                <span className="mlx-fold__line">{setup.summary}</span>
              </summary>
              <dl className="mlx-fold__rows">
                {setup.rows.map((row) => (
                  <div key={row.label} className="mlx-fold__row">
                    <dt>{row.label}</dt>
                    <dd>{row.value}</dd>
                  </div>
                ))}
              </dl>
            </details>
          ) : null}
          <details className="mlx-fold" open={numbers} onToggle={(event) => toggleNumbers(event.currentTarget.open)}>
            <summary className="mlx-fold__summary">
              <span className="mlx-fold__label">Hover</span>
              <span className="mlx-fold__line">numbers under the pointer</span>
            </summary>
            {numbers ? (
              <div className="mlx-readout" data-live={readout ? '' : undefined}>
                <p className="mlx-readout__title">{readout ? readout.title : hint}</p>
                <p className="mlx-readout__rows">
                  {readout
                    ? readout.rows.map((row) => (
                        <span key={row.label} className="mlx-readout__row">
                          <span className="mlx-readout__label">{row.label}</span>
                          <span className="mlx-readout__value">{row.value}</span>
                        </span>
                      ))
                    : ' '}
                </p>
              </div>
            ) : null}
          </details>
          {footer}
        </div>
      </aside>
    </ChartFillContext.Provider>
  );
}
