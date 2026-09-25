/** The sequence the diagram follows: step through the held-out ones, or type your own. */

import { useEffect, useId, useState } from 'react';

import type { SequenceDataset } from '../../lib/datasets/sequences';
import { spellTokens } from '../../lib/datasets/sequences';
import type { Probe } from './model';

export function ProbeStepper({ index, count, onChange }: { index: number | null; count: number; onChange: (next: number) => void }) {
  const last = Math.max(0, count - 1);
  const at = index ?? -1;
  return (
    <span className="mlx-stepper mlx-probe">
      <button type="button" className="mlx-stepper__button" disabled={at <= 0} onClick={() => onChange(Math.max(0, at - 1))} aria-label="Previous held-out sequence" title="Previous held-out sequence">
        ‹
      </button>
      <span className="mlx-stepper__label">{index === null ? 'yours' : 'seq ' + (index + 1) + ' / ' + count}</span>
      <button type="button" className="mlx-stepper__button" disabled={at >= last} onClick={() => onChange(Math.min(last, at + 1))} aria-label="Next held-out sequence" title="Next held-out sequence">
        ›
      </button>
    </span>
  );
}

/** A text field for a sequence of your own; Enter applies it, the button goes back to the held-out ones. A letter model can write on from it. */
export function TokenInput({
  data,
  probe,
  value,
  onChange,
  onWrite,
  writing = false,
  onStop,
}: {
  data: SequenceDataset;
  probe: Probe;
  value: string;
  onChange: (next: string) => void;
  onWrite?: (from: string) => void;
  writing?: boolean;
  onStop?: () => void;
}) {
  const id = useId();
  const shown = spellTokens(data, probe.tokens);
  const [draft, setDraft] = useState(value || shown);
  // A new probe or task rewrites the field unless the reader is mid-edit.
  const [editing, setEditing] = useState(false);
  useEffect(() => {
    if (!editing) setDraft(value || shown);
  }, [value, shown, editing]);
  const words = data.task === 'animals';
  const letters = data.task === 'letters';
  const apply = () => {
    setEditing(false);
    // A letter prompt keeps its trailing space: after "says " comes the sound.
    const next = letters ? draft.replace(/^\s+/, '') : draft.trim();
    onChange(next === shown && value === '' ? '' : next);
  };
  return (
    <div className="mlx-tf-input">
      <label htmlFor={id} className="mlx-tf-input__label">
        {letters ? 'Your text' : words ? 'Your words' : 'Your digits'}
      </label>
      <input
        id={id}
        className="mlx-tf-input__field"
        value={draft}
        spellCheck={false}
        autoComplete="off"
        placeholder={letters ? 'the old dog s' : words ? 'the old dog says' : '0'.repeat(data.size)}
        onFocus={() => setEditing(true)}
        onChange={(event) => setDraft(event.target.value)}
        onBlur={apply}
        onKeyDown={(event) => {
          if (event.key === 'Enter') {
            event.preventDefault();
            (event.target as HTMLInputElement).blur();
          } else if (event.key === 'Escape') {
            setEditing(false);
            setDraft(value || shown);
          }
        }}
        aria-describedby={id + '-hint'}
      />
      <span id={id + '-hint'} className="mlx-tf-input__hint" data-error={probe.error ? '' : undefined}>
        {probe.error ??
          (letters
            ? 'The start of a sentence, then Enter; the model writes on letter by letter.'
            : words
              ? 'Any words the model knows; it predicts each next one.'
              : data.size + ' digits from 0 to ' + (data.vocab.length - 2) + ', then Enter.')}
      </span>
      {onWrite ? (
        <button
          type="button"
          className="mlx-scalar mlx-scalar--jump"
          data-tone={writing ? 'accent' : undefined}
          onClick={() => (writing ? onStop?.() : onWrite(letters ? draft.replace(/^\s+/, '') : draft))}
          title={writing ? 'Stop writing' : 'Let the model write on, one letter at a time'}
        >
          <span className="mlx-scalar__label">{writing ? 'stop' : 'write ▸'}</span>
        </button>
      ) : null}
      {value !== '' ? (
        <button type="button" className="mlx-scalar mlx-scalar--jump" onClick={() => onChange('')} title="Follow the held-out sequences again">
          <span className="mlx-scalar__label">held-out</span>
        </button>
      ) : null}
    </div>
  );
}
