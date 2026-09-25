/** A fixed column beside a diagram: the readout of what is under the pointer, then the open card or a quiet reference. */

import type { ReactNode } from 'react';

import { ChartFillContext } from './Chart';
import { DetailOverlay } from './Detail';

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
  /** Fills the strip while nothing is under the pointer. */
  hint: string;
  /** The settings in use, folded under the strip. */
  setup?: Setup;
  /** The open card, or null. */
  card: ReactNode | null;
  onClose: () => void;
  /** Sits under the strip while no card is open. */
  idle?: ReactNode;
  /** Always at the top of the column, such as a map of the whole model. */
  header?: ReactNode;
  /** Always at the foot of the column. */
  footer?: ReactNode;
}

export function Inspector({ readout, hint, setup, card, onClose, idle, header, footer }: InspectorProps) {
  // Charts in the column keep their own height, even inside a panel that fills the screen.
  return (
    <ChartFillContext.Provider value={false}>
      <aside className="mlx-inspector" aria-label="Inspector">
        {header ? <div className="mlx-inspector__head">{header}</div> : null}
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
              : ' '}
          </p>
        </div>
        {setup ? (
          <details className="mlx-setup">
            <summary className="mlx-setup__summary">
              <span className="mlx-setup__label">Setup</span>
              <span className="mlx-setup__line">{setup.summary}</span>
            </summary>
            <dl className="mlx-setup__rows">
              {setup.rows.map((row) => (
                <div key={row.label} className="mlx-setup__row">
                  <dt>{row.label}</dt>
                  <dd>{row.value}</dd>
                </div>
              ))}
            </dl>
          </details>
        ) : null}
        {card ? (
          <DetailOverlay dock onClose={onClose}>
            {card}
          </DetailOverlay>
        ) : idle ? (
          <div className="mlx-inspector__idle">{idle}</div>
        ) : null}
        {footer ? <div className="mlx-inspector__foot">{footer}</div> : null}
      </aside>
    </ChartFillContext.Provider>
  );
}
