/** Renders a declared `ControlGroup[]` against a params object, as the top row or the rail. */

import { useContext, useEffect, useMemo, useState } from 'react';
import type { ReactNode } from 'react';

import type { Control, ControlGroup, Preset } from '../types';
import type { ParamsRecord } from '../useExplainerParams';
import { LessonFocusContext, focusesControl, useLessonTarget } from '../lessonFocus';
import { readFlag, writeFlag } from '../storage';
import { COMPACT_QUERY } from '../useMediaQuery';
import { SegmentedField, SelectField, SeedField, SliderField, StepperField, ToggleField } from './Controls';
import { IconChevron } from './Icons';

/** The groups a page wants in the single top row, with their titles kept. */
export function topGroupsOf<K extends string>(
  groups: ReadonlyArray<ControlGroup<K>>,
): ControlGroup<K>[] {
  return groups.filter((g) => g.zone === 'top');
}

/** Everything else, which stays grouped in the left rail. */
export function railGroupsOf<K extends string>(
  groups: ReadonlyArray<ControlGroup<K>>,
): ControlGroup<K>[] {
  return groups.filter((g) => g.zone !== 'top');
}

export interface ControlPanelProps<P extends ParamsRecord> {
  groups: ReadonlyArray<ControlGroup<Extract<keyof P, string>>>;
  params: P;
  onChange: <K extends keyof P>(key: K, value: P[K]) => void;
  /** Rendered at the top of the rail, above the first group. */
  header?: ReactNode;
  /** Rendered at the very bottom. */
  footer?: ReactNode;
}

export function ControlPanel<P extends ParamsRecord>({
  groups,
  params,
  onChange,
  header,
  footer,
}: ControlPanelProps<P>) {
  return (
    <div className="mlx-controls">
      {header}
      {groups.map((group) => (
        <ControlGroupBlock key={group.title} group={group} params={params} onChange={onChange} />
      ))}
      {footer}
    </div>
  );
}

/** Remembers whether the reader wants the advanced knobs, across pages. */
const ADVANCED_KEY = 'mlx-advanced-controls';

/** The top row of knobs. Advanced ones sit behind More; inapplicable ones dim rather than vanish. */
export function ControlRow<P extends ParamsRecord>({
  groups,
  params,
  onChange,
  lesson,
  named,
  children,
}: {
  groups: ReadonlyArray<ControlGroup<Extract<keyof P, string>>>;
  params: P;
  onChange: <K extends keyof P>(key: K, value: P[K]) => void;
  /** The preset on show; loading one that sets an advanced knob reveals the set. */
  lesson?: Preset<P> | null;
  /** The knobs the open lesson names; the others dim in a hero layout. */
  named?: readonly string[] | null;
  /** Extra cells appended after the declared controls. */
  children?: ReactNode;
}) {
  const [open, setOpen] = useState(() => readFlag(ADVANCED_KEY));
  const advancedKeys = useMemo(
    () => groups.flatMap((g) => g.controls.filter((c) => c.advanced).map((c) => c.key as string)),
    [groups],
  );
  useEffect(() => {
    if (lesson && advancedKeys.some((key) => key in lesson.params)) setOpen(true);
  }, [lesson, advancedKeys]);
  // A lesson step pointing at an advanced knob reveals the set too.
  const focus = useContext(LessonFocusContext);
  useEffect(() => {
    if (focusesControl(focus, advancedKeys)) setOpen(true);
  }, [focus, advancedKeys]);
  const toggle = () =>
    setOpen((v) => {
      writeFlag(ADVANCED_KEY, !v);
      return !v;
    });
  // Dim only when the lesson names at least one knob in this row; a lesson about the rail leaves the row alone.
  const rowKeys = useMemo(() => groups.flatMap((g) => g.controls.map((c) => c.key as string)), [groups]);
  const dims = named ? named.some((key) => rowKeys.includes(key)) : false;

  return (
    <div className="mlx-toprow">
      {groups.map((group) => {
        const basics = group.controls.filter((c) => !c.advanced);
        const shown = open ? basics.concat(group.controls.filter((c) => c.advanced)) : basics;
        if (shown.length === 0) return null;
        return (
          <div className="mlx-toprow__group" role="group" aria-label={group.title} key={group.title}>
            <span className="mlx-toprow__caption" title={group.blurb}>
              {group.title}
            </span>
            <div className="mlx-toprow__cells">
              {shown.map((control) => (
                <TopCell
                  key={control.key}
                  control={control}
                  params={params}
                  onChange={onChange}
                  named={dims && named ? named.includes(control.key) : undefined}
                />
              ))}
            </div>
          </div>
        );
      })}
      {advancedKeys.length > 0 ? (
        <button
          type="button"
          className="mlx-toprow__more"
          aria-expanded={open}
          onClick={toggle}
          title={open ? 'Hide the advanced knobs' : 'Show ' + advancedKeys.length + ' more knobs'}
        >
          {open ? 'Less' : 'More'}
          <IconChevron size={11} />
        </button>
      ) : null}
      {children}
    </div>
  );
}

function TopCell<P extends ParamsRecord>({
  control,
  params,
  onChange,
  named,
}: {
  control: Control<Extract<keyof P, string>>;
  params: P;
  onChange: <K extends keyof P>(key: K, value: P[K]) => void;
  named?: boolean;
}) {
  const target = useLessonTarget('control', control.key);
  return (
    <div
      className="mlx-toprow__cell"
      data-kind={control.kind}
      data-advanced={control.advanced || undefined}
      data-named={named === undefined ? undefined : named ? '' : 'no'}
      {...target.attrs}
    >
      <ControlRenderer
        control={control}
        params={params}
        onChange={onChange}
        zone="top"
        inactive={Boolean(control.visibleWhen && !control.visibleWhen(params))}
      />
    </div>
  );
}

/** A rail knob, wrapped so a lesson step can point at it. */
function RailControl<P extends ParamsRecord>({
  control,
  params,
  onChange,
}: {
  control: Control<Extract<keyof P, string>>;
  params: P;
  onChange: <K extends keyof P>(key: K, value: P[K]) => void;
}) {
  const target = useLessonTarget('control', control.key);
  return (
    <div className="mlx-control" {...target.attrs}>
      <ControlRenderer control={control} params={params} onChange={onChange} />
    </div>
  );
}

function ControlGroupBlock<P extends ParamsRecord>({
  group,
  params,
  onChange,
}: {
  group: ControlGroup<Extract<keyof P, string>>;
  params: P;
  onChange: <K extends keyof P>(key: K, value: P[K]) => void;
}) {
  // Collapsed by default only where height is scarce.
  const [open, setOpen] = useState(
    () => !(group.collapsedByDefault && typeof window !== 'undefined' && window.matchMedia(COMPACT_QUERY).matches),
  );
  const visible = group.controls.filter((control) => !control.visibleWhen || control.visibleWhen(params));
  // A lesson step pointing inside a collapsed group opens it.
  const focus = useContext(LessonFocusContext);
  const keys = useMemo(() => group.controls.map((c) => c.key as string), [group]);
  useEffect(() => {
    if (focusesControl(focus, keys)) setOpen(true);
  }, [focus, keys]);
  if (visible.length === 0) return null;

  return (
    <section className="mlx-control-group" data-open={open || undefined}>
      <button
        type="button"
        className="mlx-control-group__head"
        aria-expanded={open}
        title={group.blurb}
        onClick={() => setOpen((v) => !v)}
      >
        <span className="mlx-control-group__title">{group.title}</span>
        <span className="mlx-control-group__chevron"><IconChevron size={12} /></span>
      </button>
      {open ? (
        <div className="mlx-control-group__body">
          {visible.map((control) => (
            <RailControl key={control.key} control={control} params={params} onChange={onChange} />
          ))}
        </div>
      ) : null}
    </section>
  );
}

/** The active option's own explanation, folded into the tooltip. */
function withOptionHint(
  help: string | undefined,
  options: ReadonlyArray<{ value: string; label: string; hint?: string }>,
  value: string,
): string | undefined {
  const active = options.find((o) => o.value === value);
  if (!active?.hint) return help;
  return [help, active.label + ': ' + active.hint].filter(Boolean).join(' ');
}

function ControlRenderer<P extends ParamsRecord>({
  control,
  params,
  onChange,
  zone = 'rail',
  inactive = false,
}: {
  control: Control<Extract<keyof P, string>>;
  params: P;
  onChange: <K extends keyof P>(key: K, value: P[K]) => void;
  zone?: 'top' | 'rail';
  /** Present but not in use under the current setting; drawn dimmed. */
  inactive?: boolean;
}) {
  const disabled = inactive || (control.disabledWhen ? control.disabledWhen(params) : false);
  const key = control.key as keyof P;
  const raw = params[key];

  switch (control.kind) {
    case 'stepper':
      return (
        <StepperField
          label={control.label}
          help={control.help}
          value={typeof raw === 'number' ? raw : control.min}
          min={control.min}
          max={control.max}
          step={control.step}
          scale={control.scale}
          format={control.format}
          disabled={disabled}
          onChange={(value) => onChange(key, value as P[keyof P])}
        />
      );
    case 'slider':
      return (
        <SliderField
          label={control.label}
          help={control.help}
          value={typeof raw === 'number' ? raw : control.min}
          min={control.min}
          max={control.max}
          step={control.step}
          format={control.format}
          disabled={disabled}
          onChange={(value) => onChange(key, value as P[keyof P])}
        />
      );
    case 'select':
      return (
        <SelectField
          label={control.label}
          help={withOptionHint(control.help, control.options, String(raw))}
          value={String(raw)}
          options={control.options}
          disabled={disabled}
          onChange={(value) => onChange(key, (typeof raw === 'number' ? Number(value) : value) as P[keyof P])}
        />
      );
    case 'segmented': {
      // A numeric param gets its number back.
      const pick = (value: string) => onChange(key, (typeof raw === 'number' ? Number(value) : value) as P[keyof P]);
      // The top row draws a segmented knob as a dropdown.
      if (zone === 'top') {
        return (
          <SelectField
            label={control.label}
            help={withOptionHint(control.help, control.options, String(raw))}
            value={String(raw)}
            options={control.options}
            disabled={disabled}
            onChange={pick}
          />
        );
      }
      return (
        <SegmentedField
          label={control.label}
          help={withOptionHint(control.help, control.options, String(raw))}
          value={String(raw)}
          options={control.options}
          disabled={disabled}
          onChange={pick}
        />
      );
    }
    case 'toggle':
      if (zone === 'top') {
        return (
          <SelectField
            label={control.label}
            help={control.help}
            value={raw ? 'on' : 'off'}
            options={[
              { value: 'on', label: control.onLabel ?? 'On' },
              { value: 'off', label: control.offLabel ?? 'Off' },
            ]}
            disabled={disabled}
            onChange={(value) => onChange(key, (value === 'on') as P[keyof P])}
          />
        );
      }
      return (
        <ToggleField
          label={control.label}
          help={control.help}
          value={Boolean(raw)}
          disabled={disabled}
          onChange={(value) => onChange(key, value as P[keyof P])}
        />
      );
    case 'seed':
      return (
        <SeedField
          label={control.label}
          help={control.help}
          value={typeof raw === 'number' ? raw : 0}
          disabled={disabled}
          onChange={(value) => onChange(key, value as P[keyof P])}
        />
      );
    default:
      return null;
  }
}
