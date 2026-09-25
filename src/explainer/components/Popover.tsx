/** A button and the bubble it opens under itself. Esc or clicking elsewhere closes it. */

import { useEffect, useId, useLayoutEffect, useRef, useState } from 'react';
import type { ReactNode } from 'react';

export function Popover({
  button,
  label,
  className,
  align = 'left',
  children,
}: {
  /** The button's content. */
  button: ReactNode;
  /** Accessible name for the button and the bubble. */
  label: string;
  className?: string;
  align?: 'left' | 'right';
  children: ReactNode;
}) {
  const [open, setOpen] = useState(false);
  const id = useId();
  const rootRef = useRef<HTMLDivElement>(null);
  const buttonRef = useRef<HTMLButtonElement>(null);
  const popRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!open) return undefined;
    const onDown = (event: PointerEvent) => {
      if (!rootRef.current?.contains(event.target as Node)) setOpen(false);
    };
    // Capture phase, so an open bubble takes Escape before the layer under it.
    const onKey = (event: KeyboardEvent) => {
      if (event.key !== 'Escape') return;
      event.stopImmediatePropagation();
      setOpen(false);
      buttonRef.current?.focus();
    };
    document.addEventListener('pointerdown', onDown);
    window.addEventListener('keydown', onKey, true);
    return () => {
      document.removeEventListener('pointerdown', onDown);
      window.removeEventListener('keydown', onKey, true);
    };
  }, [open]);

  // Slide the bubble back inside the viewport before paint.
  useLayoutEffect(() => {
    const pop = popRef.current;
    if (!open || !pop) return;
    pop.style.removeProperty('translate');
    const rect = pop.getBoundingClientRect();
    const pad = 16;
    let shift = 0;
    if (rect.right > window.innerWidth - pad) shift = window.innerWidth - pad - rect.right;
    if (rect.left + shift < pad) shift = pad - rect.left;
    if (shift) pop.style.translate = shift + 'px 0';
  }, [open]);

  return (
    <div
      className={'mlx-popover' + (className ? ' ' + className : '')}
      ref={rootRef}
      onBlur={(event) => {
        if (!event.currentTarget.contains(event.relatedTarget as Node | null)) setOpen(false);
      }}
    >
      <button
        ref={buttonRef}
        type="button"
        className="mlx-popover__button"
        aria-label={label}
        title={label}
        aria-expanded={open}
        aria-controls={id}
        onClick={() => setOpen((v) => !v)}
      >
        {button}
      </button>
      {open ? (
        <div
          id={id}
          ref={popRef}
          className="mlx-popover__pop"
          data-align={align}
          role="dialog"
          aria-label={label}
          tabIndex={-1}
        >
          {children}
        </div>
      ) : null}
    </div>
  );
}
