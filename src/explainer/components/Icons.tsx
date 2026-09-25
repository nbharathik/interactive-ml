/** The icon set: one stroke weight, one grid, currentColor. */

import type { ReactNode } from 'react';

function Icon({ size = 16, children }: { size?: number; children: ReactNode }) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 16 16"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.6"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
    >
      {children}
    </svg>
  );
}

export function IconReset({ size }: { size?: number }) {
  return (
    <Icon size={size}>
      <path d="M2.8 8a5.2 5.2 0 1 0 1.5-3.7" />
      <path d="M2.5 2.6v3.2h3.2" />
    </Icon>
  );
}

export function IconReplay({ size }: { size?: number }) {
  return (
    <Icon size={size}>
      <path d="M2.8 8a5.2 5.2 0 1 0 1.5-3.7" />
      <path d="M2.5 2.6v3.2h3.2" />
      <path d="M6.8 6v4l3.4-2Z" fill="currentColor" stroke="none" />
    </Icon>
  );
}

export function IconKeyboard({ size }: { size?: number }) {
  return (
    <Icon size={size}>
      <rect x="1.5" y="4" width="13" height="8.5" rx="1.5" />
      <path d="M4 6.8h.01M6.7 6.8h.01M9.3 6.8h.01M12 6.8h.01M4 9.6h.01M12 9.6h.01M6.2 9.6h3.6" />
    </Icon>
  );
}

export function IconSliders({ size }: { size?: number }) {
  return (
    <Icon size={size}>
      <path d="M2.5 4.5h11M2.5 8h11M2.5 11.5h11" />
      <circle cx="6" cy="4.5" r="1.5" fill="var(--bg-secondary)" />
      <circle cx="10.5" cy="8" r="1.5" fill="var(--bg-secondary)" />
      <circle cx="5" cy="11.5" r="1.5" fill="var(--bg-secondary)" />
    </Icon>
  );
}

export function IconPlay({ size }: { size?: number }) {
  return (
    <Icon size={size}>
      <path d="M5 3.2v9.6l7.4-4.8Z" fill="currentColor" stroke="none" />
    </Icon>
  );
}

export function IconPause({ size }: { size?: number }) {
  return (
    <Icon size={size}>
      <rect x="3.6" y="3" width="3.2" height="10" rx="0.8" fill="currentColor" stroke="none" />
      <rect x="9.2" y="3" width="3.2" height="10" rx="0.8" fill="currentColor" stroke="none" />
    </Icon>
  );
}

export function IconStop({ size }: { size?: number }) {
  return (
    <Icon size={size}>
      <rect x="3.5" y="3.5" width="9" height="9" rx="1.2" fill="currentColor" stroke="none" />
    </Icon>
  );
}

export function IconStep({ size }: { size?: number }) {
  return (
    <Icon size={size}>
      <path d="M3.5 3.4v9.2l6.4-4.6Z" fill="currentColor" stroke="none" />
      <path d="M12.5 3.2v9.6" />
    </Icon>
  );
}

export function IconEnd({ size }: { size?: number }) {
  return (
    <Icon size={size}>
      <path d="M2.5 3.6v8.8L8 8Z" fill="currentColor" stroke="none" />
      <path d="M8 3.6v8.8L13.5 8Z" fill="currentColor" stroke="none" opacity="0.5" />
    </Icon>
  );
}

export function IconExpand({ size }: { size?: number }) {
  return (
    <Icon size={size}>
      <path d="M9.5 2.5h4v4M13.5 2.5 9 7M6.5 13.5h-4v-4M2.5 13.5 7 9" />
    </Icon>
  );
}

export function IconCollapse({ size }: { size?: number }) {
  return (
    <Icon size={size}>
      <path d="M13.5 6.5h-4v-4M9.5 6.5 14 2M2.5 9.5h4v4M6.5 9.5 2 14" />
    </Icon>
  );
}

export function IconFullscreen({ size }: { size?: number }) {
  return (
    <Icon size={size}>
      <path d="M2.5 6V2.5H6M10 2.5h3.5V6M13.5 10v3.5H10M6 13.5H2.5V10" />
    </Icon>
  );
}

export function IconExitFullscreen({ size }: { size?: number }) {
  return (
    <Icon size={size}>
      <path d="M6 2.5V6H2.5M10 2.5V6h3.5M13.5 10H10v3.5M6 13.5V10H2.5" />
    </Icon>
  );
}

export function IconClose({ size }: { size?: number }) {
  return (
    <Icon size={size}>
      <path d="M4 4l8 8M12 4l-8 8" />
    </Icon>
  );
}

export function IconShuffle({ size }: { size?: number }) {
  return (
    <Icon size={size}>
      <path d="M2.5 4.5h2.2c1.4 0 2.2.7 3 1.8l1.6 2.4c.8 1.1 1.6 1.8 3 1.8h1.2" />
      <path d="M2.5 11.5h2.2c1.1 0 1.8-.4 2.4-1.1M9.4 5.6c.7-.7 1.4-1.1 2.5-1.1h1.6" />
      <path d="M12 2.6l1.9 1.9-1.9 1.9M12 8.6l1.9 1.9-1.9 1.9" />
    </Icon>
  );
}

export function IconInfo({ size }: { size?: number }) {
  return (
    <Icon size={size}>
      <circle cx="8" cy="8" r="6.2" />
      <path d="M8 7.2v3.8" />
      <circle cx="8" cy="5" r="0.6" fill="currentColor" stroke="none" />
    </Icon>
  );
}

export function IconChevron({ size }: { size?: number }) {
  return (
    <Icon size={size}>
      <path d="M4 6.2 8 10.2l4-4" />
    </Icon>
  );
}

export function IconExternal({ size }: { size?: number }) {
  return (
    <Icon size={size}>
      <path d="M6.5 3.5h6v6M12.5 3.5 4 12" />
    </Icon>
  );
}

export function IconSun({ size }: { size?: number }) {
  return (
    <Icon size={size}>
      <circle cx="8" cy="8" r="3" />
      <path d="M8 1.5v1.8M8 12.7v1.8M1.5 8h1.8M12.7 8h1.8M3.4 3.4l1.3 1.3M11.3 11.3l1.3 1.3M3.4 12.6l1.3-1.3M11.3 4.7l1.3-1.3" />
    </Icon>
  );
}

export function IconMoon({ size }: { size?: number }) {
  return (
    <Icon size={size}>
      <path d="M13.2 9.6A5.6 5.6 0 0 1 6.4 2.8a5.6 5.6 0 1 0 6.8 6.8Z" />
    </Icon>
  );
}

export function IconArrow({ size }: { size?: number }) {
  return (
    <Icon size={size}>
      <path d="M3 8h10M9.5 4.5 13 8l-3.5 3.5" />
    </Icon>
  );
}

export function IconLink({ size }: { size?: number }) {
  return (
    <Icon size={size}>
      <path d="M6.6 9.4a2.9 2.9 0 0 0 4.1 0l2.1-2.1a2.9 2.9 0 0 0-4.1-4.1l-.9.9" />
      <path d="M9.4 6.6a2.9 2.9 0 0 0-4.1 0L3.2 8.7a2.9 2.9 0 0 0 4.1 4.1l.9-.9" />
    </Icon>
  );
}

export function IconArticle({ size }: { size?: number }) {
  return (
    <Icon size={size}>
      <path d="M3.5 2h6l3 3v9h-9Z" />
      <path d="M9.5 2v3h3" />
      <path d="M6 8.5h4M6 11h4" />
    </Icon>
  );
}

export function IconCheck({ size }: { size?: number }) {
  return (
    <Icon size={size}>
      <path d="M3 8.4l3.2 3.2L13 5" />
    </Icon>
  );
}

export function IconTarget({ size }: { size?: number }) {
  return (
    <Icon size={size}>
      <circle cx="8" cy="8" r="5.2" />
      <circle cx="8" cy="8" r="1.6" fill="currentColor" stroke="none" />
      <path d="M8 1.6v2.2M8 12.2v2.2M1.6 8h2.2M12.2 8h2.2" />
    </Icon>
  );
}
