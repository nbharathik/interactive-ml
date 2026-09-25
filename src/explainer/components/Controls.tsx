/** Control primitives: a label, a live readout, an optional tip, and a native input. */

import { useEffect, useId, useLayoutEffect, useRef, useState } from 'react';
import type { CSSProperties, ReactNode } from 'react';
import { createPortal } from 'react-dom';

import { IconChevron, IconInfo, IconShuffle } from './Icons';

/* ---------------- info tip ---------------- */

const TIP_GAP = 8;

interface TipPlace {
  left: number;
  top: number;
  arrow: number;
  side: 'above' | 'below';
}

/** The bubble lives in a portal with fixed coordinates so nothing clips it. */
export function InfoTip({ text, label }: { text: string; label?: string }) {
  const [open, setOpen] = useState(false);
  const [place, setPlace] = useState<TipPlace | null>(null);
  const buttonRef = useRef<HTMLButtonElement>(null);
  const bubbleRef = useRef<HTMLSpanElement>(null);
  // A tap fires a synthetic mouseenter first.
  const touch = useRef(false);
  const id = useId();

  useLayoutEffect(() => {
    if (!open) {
      setPlace(null);
      return undefined;
    }
    const target = buttonRef.current?.getBoundingClientRect();
    const bubble = bubbleRef.current?.getBoundingClientRect();
    if (!target || !bubble) return undefined;
    const headerBottom = document.querySelector('.mlx-header')?.getBoundingClientRect().bottom ?? 0;
    const side = target.top - TIP_GAP - bubble.height >= headerBottom + 4 ? 'above' : 'below';
    // Below, the bubble clears the whole field.
    const field = buttonRef.current?.closest('.mlx-field')?.getBoundingClientRect();
    const top =
      side === 'above' ? target.top - TIP_GAP - bubble.height : (field?.bottom ?? target.bottom) + TIP_GAP;
    const centre = target.left + target.width / 2;
    const left = Math.min(Math.max(8, centre - bubble.width / 2), window.innerWidth - bubble.width - 8);
    setPlace({ left, top, side, arrow: centre - left });
    // A fixed bubble goes stale as soon as anything scrolls.
    const close = () => setOpen(false);
    // Touch buttons never blur; a tap elsewhere closes it.
    const onDown = (event: PointerEvent) => {
      if (!buttonRef.current?.contains(event.target as Node)) setOpen(false);
    };
    window.addEventListener('scroll', close, true);
    document.addEventListener('pointerdown', onDown);
    return () => {
      window.removeEventListener('scroll', close, true);
      document.removeEventListener('pointerdown', onDown);
    };
  }, [open, text]);

  const style: CSSProperties = {
    left: place?.left ?? 0,
    top: place?.top ?? 0,
    visibility: place ? 'visible' : 'hidden',
    ['--mlx-tip-arrow' as string]: (place?.arrow ?? 0) + 'px',
  };

  return (
    <span className="mlx-tip">
      <button
        ref={buttonRef}
        type="button"
        className="mlx-tip__button"
        aria-label={'About ' + (label ?? 'this')}
        aria-expanded={open}
        aria-describedby={open ? id : undefined}
        onPointerDown={(event) => {
          touch.current = event.pointerType === 'touch';
        }}
        onClick={() => setOpen(true)}
        onFocus={() => setOpen(true)}
        onBlur={() => setOpen(false)}
        onMouseEnter={() => {
          if (!touch.current) setOpen(true);
        }}
        onMouseLeave={() => setOpen(false)}
        onKeyDown={(event) => {
          if (event.key === 'Escape' && open) {
            event.stopPropagation();
            setOpen(false);
          }
        }}
      >
        <IconInfo size={13} />
      </button>
      {open
        ? createPortal(
            <span
              ref={bubbleRef}
              role="tooltip"
              id={id}
              className="mlx-tip__bubble"
              data-side={place?.side ?? 'above'}
              style={style}
            >
              {text}
            </span>,
            document.body,
          )
        : null}
    </span>
  );
}

/* ---------------- field wrapper ---------------- */

interface FieldProps {
  label: string;
  htmlFor?: string;
  help?: string;
  value?: ReactNode;
  children: ReactNode;
  disabled?: boolean;
  /** Rendered under the control. */
  note?: ReactNode;
}

export function Field({ label, htmlFor, help, value, children, disabled, note }: FieldProps) {
  return (
    <div className="mlx-field" data-disabled={disabled || undefined}>
      <div className="mlx-field__head">
        <label className="mlx-field__label" htmlFor={htmlFor}>
          {label}
        </label>
        {help ? <InfoTip text={help} label={label} /> : null}
        {value !== undefined ? <span className="mlx-field__value mlx-num">{value}</span> : null}
      </div>
      {children}
      {note ? <p className="mlx-field__note">{note}</p> : null}
    </div>
  );
}

/* ---------------- stepper ---------------- */

export interface StepperFieldProps {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  onChange: (value: number) => void;
  help?: string;
  format?: (value: number) => string;
  /** Log: each press moves to the next of 1, 2, 3, 5 in the current decade. */
  scale?: 'linear' | 'log';
  disabled?: boolean;
  note?: ReactNode;
}

const LOG_MANTISSAS = [1, 2, 3, 5];

/** The grid a log stepper walks: 1, 2, 3, 5 per decade, from `step` up to `max`. */
function logGrid(step: number, max: number): number[] {
  const grid: number[] = [];
  const lo = Math.floor(Math.log10(Math.max(step, 1e-9)));
  const hi = Math.ceil(Math.log10(Math.max(max, 1e-9)));
  for (let e = lo; e <= hi; e++) {
    for (const m of LOG_MANTISSAS) {
      const v = Number((m * 10 ** e).toPrecision(4));
      if (v >= step - 1e-12 && v <= max + 1e-12) grid.push(v);
    }
  }
  return grid;
}

function decimalsOf(step: number): number {
  const text = String(step);
  if (text.includes('e-')) return Number(text.split('e-')[1]);
  return text.includes('.') ? text.split('.')[1].length : 0;
}

/** The value one press away, clamped to the range. */
export function stepValue(
  value: number,
  direction: 1 | -1,
  { min, max, step, scale }: { min: number; max: number; step: number; scale?: 'linear' | 'log' },
): number {
  if (scale === 'log') {
    const grid = logGrid(step, max).filter((v) => v >= min - 1e-12);
    if (min <= 0) grid.unshift(0);
    if (direction > 0) return grid.find((v) => v > value * (1 + 1e-9)) ?? grid[grid.length - 1];
    for (let i = grid.length - 1; i >= 0; i--) if (grid[i] < value * (1 - 1e-9)) return grid[i];
    return grid[0];
  }
  const next = Math.min(max, Math.max(min, value + direction * step));
  return Number(next.toFixed(decimalsOf(step)));
}

export function StepperField({
  label,
  value,
  min,
  max,
  step,
  onChange,
  help,
  format,
  scale = 'linear',
  disabled,
  note,
}: StepperFieldProps) {
  const id = useId();
  const display = format ? format(value) : String(value).replace('-', '−');
  const range = { min, max, step, scale };
  const latest = useRef(value);
  latest.current = value;
  const timer = useRef<number | null>(null);

  const stop = () => {
    if (timer.current !== null) window.clearTimeout(timer.current);
    timer.current = null;
  };
  // Holding a button keeps stepping, faster after the first few.
  const press = (direction: 1 | -1) => {
    stop();
    const go = () => {
      const next = stepValue(latest.current, direction, range);
      if (next === latest.current) {
        stop();
        return;
      }
      latest.current = next;
      onChange(next);
    };
    go();
    let delay = 320;
    const repeat = () => {
      go();
      delay = Math.max(50, delay * 0.8);
      timer.current = window.setTimeout(repeat, delay);
    };
    timer.current = window.setTimeout(repeat, delay);
  };
  useEffect(() => stop, []);
  // A mouse press already stepped; keys, screen readers and voice control arrive as a bare click.
  const pressed = useRef(false);
  const release = () => {
    stop();
    pressed.current = false;
  };
  const click = (direction: 1 | -1, blocked: boolean) => {
    if (pressed.current) {
      pressed.current = false;
      return;
    }
    if (!blocked) onChange(stepValue(value, direction, range));
  };

  const atMin = value <= min;
  const atMax = value >= max;

  return (
    <Field label={label} htmlFor={id} help={help} disabled={disabled} note={note}>
      <div className="mlx-stepper mlx-stepper--field" role="group" aria-label={label}>
        <button
          type="button"
          className="mlx-stepper__button"
          disabled={disabled}
          aria-disabled={atMin || undefined}
          aria-label={'Decrease ' + label}
          onPointerDown={(event) => {
            if (atMin || event.button !== 0) return;
            pressed.current = true;
            press(-1);
          }}
          onPointerUp={stop}
          onPointerLeave={release}
          onPointerCancel={release}
          onClick={() => click(-1, atMin)}
        >
          −
        </button>
        <output id={id} className="mlx-stepper__label mlx-num" aria-live="polite">
          {display}
        </output>
        <button
          type="button"
          className="mlx-stepper__button"
          disabled={disabled}
          aria-disabled={atMax || undefined}
          aria-label={'Increase ' + label}
          onPointerDown={(event) => {
            if (atMax || event.button !== 0) return;
            pressed.current = true;
            press(1);
          }}
          onPointerUp={stop}
          onPointerLeave={release}
          onPointerCancel={release}
          onClick={() => click(1, atMax)}
        >
          +
        </button>
      </div>
    </Field>
  );
}

/* ---------------- slider ---------------- */

export interface SliderFieldProps {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  onChange: (value: number) => void;
  help?: string;
  format?: (value: number) => string;
  disabled?: boolean;
  note?: ReactNode;
}

export function SliderField({
  label,
  value,
  min,
  max,
  step,
  onChange,
  help,
  format,
  disabled,
  note,
}: SliderFieldProps) {
  const id = useId();
  const display = format ? format(value) : String(value).replace('-', '−');
  const fill = max > min ? ((value - min) / (max - min)) * 100 : 0;
  const style = { ['--mlx-fill' as string]: Math.min(100, Math.max(0, fill)) + '%' } as CSSProperties;
  return (
    <Field label={label} htmlFor={id} help={help} value={display} disabled={disabled} note={note}>
      <input
        id={id}
        className="mlx-slider"
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        disabled={disabled}
        style={style}
        aria-valuetext={display}
        onChange={(event) => onChange(Number(Number(event.target.value).toFixed(decimalsOf(step))))}
      />
    </Field>
  );
}

/* ---------------- select ---------------- */

export interface SelectFieldProps {
  label: string;
  value: string;
  options: Array<{ value: string; label: string; hint?: string }>;
  onChange: (value: string) => void;
  help?: string;
  disabled?: boolean;
  note?: ReactNode;
}

export function SelectField({
  label,
  value,
  options,
  onChange,
  help,
  disabled,
  note,
}: SelectFieldProps) {
  const id = useId();
  return (
    <Field label={label} htmlFor={id} help={help} disabled={disabled} note={note}>
      <div className="mlx-select">
        <select
          id={id}
          value={value}
          disabled={disabled}
          onChange={(event) => onChange(event.target.value)}
        >
          {options.map((option) => (
            <option key={option.value} value={option.value}>
              {option.label}
            </option>
          ))}
        </select>
        <IconChevron size={12} />
      </div>
    </Field>
  );
}

/* ---------------- segmented ---------------- */

export interface SegmentedFieldProps {
  label: string;
  value: string;
  options: Array<{ value: string; label: string; hint?: string }>;
  onChange: (value: string) => void;
  help?: string;
  disabled?: boolean;
  note?: ReactNode;
}

export function SegmentedField({
  label,
  value,
  options,
  onChange,
  help,
  disabled,
  note,
}: SegmentedFieldProps) {
  return (
    <Field label={label} help={help} disabled={disabled} note={note}>
      <div className="mlx-segmented" role="group" aria-label={label}>
        {options.map((option) => (
          <button
            key={option.value}
            type="button"
            className="mlx-segmented__item"
            data-active={option.value === value || undefined}
            aria-pressed={option.value === value}
            disabled={disabled}
            onClick={() => onChange(option.value)}
          >
            {option.label}
          </button>
        ))}
      </div>
    </Field>
  );
}

/* ---------------- toggle ---------------- */

export interface ToggleFieldProps {
  label: string;
  value: boolean;
  onChange: (value: boolean) => void;
  help?: string;
  disabled?: boolean;
  note?: ReactNode;
}

export function ToggleField({ label, value, onChange, help, disabled, note }: ToggleFieldProps) {
  const id = useId();
  return (
    <div className="mlx-field mlx-field--inline" data-disabled={disabled || undefined}>
      <button
        id={id}
        type="button"
        role="switch"
        aria-checked={value}
        className="mlx-switch"
        disabled={disabled}
        onClick={() => onChange(!value)}
      >
        <span className="mlx-switch__track">
          <span className="mlx-switch__thumb" />
        </span>
      </button>
      <label className="mlx-field__label mlx-field__label--inline" htmlFor={id}>
        {label}
      </label>
      {help ? <InfoTip text={help} label={label} /> : null}
      {note ? <p className="mlx-field__note">{note}</p> : null}
    </div>
  );
}

/* ---------------- seed ---------------- */

export function SeedField({
  label,
  value,
  onChange,
  help,
  disabled,
}: {
  label: string;
  value: number;
  onChange: (value: number) => void;
  help?: string;
  disabled?: boolean;
}) {
  const id = useId();
  // The value only moves on digits, so the box can be cleared while typing.
  const [draft, setDraft] = useState(String(value));
  useEffect(() => {
    setDraft(String(value));
  }, [value]);
  return (
    <Field label={label} htmlFor={id} help={help} disabled={disabled}>
      <div className="mlx-seed">
        <input
          id={id}
          className="mlx-seed__input mlx-num"
          type="text"
          inputMode="numeric"
          value={draft}
          disabled={disabled}
          onChange={(event) => {
            const text = event.target.value;
            setDraft(text);
            if (/^\d+$/.test(text)) onChange(Math.min(99999, Number(text)));
          }}
          onBlur={() => setDraft(String(value))}
        />
        <button
          type="button"
          className="mlx-button mlx-button--secondary mlx-button--sm mlx-seed__shuffle"
          disabled={disabled}
          aria-label="New seed"
          title="New seed"
          onClick={() => onChange(Math.floor(Math.random() * 99999))}
        >
          <IconShuffle size={14} />
        </button>
      </div>
    </Field>
  );
}

/* ---------------- button ---------------- */

export interface ButtonProps {
  children: ReactNode;
  onClick?: () => void;
  variant?: 'primary' | 'secondary' | 'ghost' | 'danger';
  size?: 'sm' | 'md';
  disabled?: boolean;
  /** Looks disabled and does nothing, but stays focusable for keyboard users. */
  softDisabled?: boolean;
  title?: string;
  ariaLabel?: string;
  type?: 'button' | 'submit';
  full?: boolean;
}

export function Button({
  children,
  onClick,
  variant = 'secondary',
  size = 'md',
  disabled,
  softDisabled,
  title,
  ariaLabel,
  type = 'button',
  full,
}: ButtonProps) {
  return (
    <button
      type={type}
      className={
        'mlx-button mlx-button--' + variant + ' mlx-button--' + size + (full ? ' mlx-button--full' : '')
      }
      onClick={softDisabled ? undefined : onClick}
      disabled={disabled}
      aria-disabled={softDisabled || undefined}
      title={title}
      aria-label={ariaLabel}
    >
      {children}
    </button>
  );
}
