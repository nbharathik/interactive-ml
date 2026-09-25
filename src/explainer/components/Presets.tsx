/** The free-play rail head: the run status, a preset dropdown and the blurb of the one on show. */

import type { GuideTone, Preset } from '../types';
import type { ParamsRecord } from '../useExplainerParams';
import { SelectField } from './Controls';

export interface PresetPickerProps<P extends ParamsRecord> {
  presets: readonly Preset<P>[];
  /** The preset matched exactly, or null once the reader has edited. */
  activeId: string | null;
  /** The preset most recently on show, kept after the params drift. */
  lastPreset?: Preset<P> | null;
  onPick: (id: string) => void;
  status: string;
  tone?: GuideTone;
  /** Just the dropdown, for a Setup tab that sits beside a Lesson tab. */
  compact?: boolean;
}

const CUSTOM = '';

export function PresetPicker<P extends ParamsRecord>({
  presets,
  activeId,
  lastPreset = null,
  onPick,
  status,
  tone = 'neutral',
  compact = false,
}: PresetPickerProps<P>) {
  const active = presets.find((p) => p.id === activeId) ?? null;
  const flagged = tone !== 'neutral' && status !== '';
  const options = presets.map((p) => ({ value: p.id, label: p.name }));
  if (!active) options.unshift({ value: CUSTOM, label: 'Custom' });
  const note = active
    ? active.blurb
    : lastPreset
      ? 'Edited from ' + lastPreset.name + '. Every control is live.'
      : 'Every control is live. Pick a preset for a setting worth watching.';

  return (
    <header className="mlx-lesson__head mlx-presets" data-compact={compact || undefined}>
      {compact ? null : <h3 className="mlx-lesson__title">Free play</h3>}
      {flagged && !compact ? (
        <p className="mlx-lesson__status" data-tone={tone}>
          {status}
        </p>
      ) : null}
      <SelectField
        label="Preset"
        value={active ? active.id : CUSTOM}
        options={options}
        onChange={(id) => {
          if (id !== CUSTOM) onPick(id);
        }}
        note={note}
      />
    </header>
  );
}
