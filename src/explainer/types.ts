/** The explainer framework's shared types: catalog, controls, presets, metrics. */

/* ---------------- catalog metadata ---------------- */

export type ExplainerStatus = 'ready' | 'preview' | 'planned';

export interface ExplainerMeta {
  slug: string;
  title: string;
  /** One sentence, sentence case, no trailing period, used on cards. */
  tagline: string;
  /** Two or three sentences for the page hero. */
  summary: string;
  status: ExplainerStatus;
  difficulty: 'gentle' | 'core' | 'deep';
  /** Concepts this page teaches, rendered as pills on the card. */
  concepts: string[];
  /** Path of the matching blog post, relative to the blog root. */
  blogPath?: string;
  /** The post link is only rendered when this is 'live'. */
  blogStatus?: 'live' | 'draft';
  /** Ordering inside its category. */
  order: number;
  /** Distinctive mark drawn on the catalog card. */
  symbol: string;
}

/* ---------------- controls ---------------- */

interface ControlBase<K extends string> {
  /** Key into the explainer's params object. */
  key: K;
  label: string;
  /** Tooltip text: what the knob does. */
  help?: string;
  /** Hide the control when the current params make it irrelevant. */
  visibleWhen?: (params: Record<string, unknown>) => boolean;
  /** Grey the control out but keep it visible, with a reason. */
  disabledWhen?: (params: Record<string, unknown>) => boolean;
  /** Changing this control invalidates training and forces a reset. */
  resets?: boolean;
  /** Hidden in the top row until the reader asks for More. Ignored in the rail. */
  advanced?: boolean;
}

export interface StepperControl<K extends string = string> extends ControlBase<K> {
  kind: 'stepper';
  min: number;
  max: number;
  step: number;
  /** Format the live value readout. */
  format?: (value: number) => string;
  /** Step through orders of magnitude (1, 2, 3, 5 per decade) for learning rates and penalties. */
  scale?: 'linear' | 'log';
}

/** A number the reader drags. The rail's numeric knob; the top row steps instead. */
export interface SliderControl<K extends string = string> extends ControlBase<K> {
  kind: 'slider';
  min: number;
  max: number;
  step: number;
  /** Format the live value readout. */
  format?: (value: number) => string;
}

export interface SelectControl<K extends string = string> extends ControlBase<K> {
  kind: 'select';
  options: Array<{ value: string; label: string; hint?: string }>;
}

/** A few choices side by side; the top row draws it as a dropdown. */
export interface SegmentedControl<K extends string = string> extends ControlBase<K> {
  kind: 'segmented';
  options: Array<{ value: string; label: string; hint?: string }>;
}

export interface ToggleControl<K extends string = string> extends ControlBase<K> {
  kind: 'toggle';
  /** Optional names for the two states; the top row offers them in a dropdown. */
  onLabel?: string;
  offLabel?: string;
}

export interface SeedControl<K extends string = string> extends ControlBase<K> {
  kind: 'seed';
}

export type Control<K extends string = string> =
  | StepperControl<K>
  | SliderControl<K>
  | SelectControl<K>
  | SegmentedControl<K>
  | ToggleControl<K>
  | SeedControl<K>;

/** A titled group of controls. */
export interface ControlGroup<K extends string = string> {
  title: string;
  /** Optional one-liner, shown as the group title's tooltip. */
  blurb?: string;
  controls: Control<K>[];
  /** Start collapsed on small screens. */
  collapsedByDefault?: boolean;
  /** 'top' hoists the group's controls into the studio's single top row. */
  zone?: 'top' | 'rail';
}

/* ---------------- presets ---------------- */

export interface Preset<P> {
  id: string;
  name: string;
  /** What the reader should notice after loading it. */
  blurb: string;
  params: Partial<P>;
  /** Autoplay after loading. */
  autoRun?: boolean;
}

export type GuideTone = 'neutral' | 'good' | 'warn' | 'bad';

/** An event a page flags: converged, diverged, stopped, a warning. */
export interface GuideStatus {
  text: string;
  tone?: GuideTone;
}

/* ---------------- metrics ---------------- */

export type MetricTone = 'neutral' | 'good' | 'warn' | 'bad' | 'accent';

export interface Metric {
  key: string;
  label: string;
  value: string;
  /** Small caption under the value, e.g. "train" / "held-out". */
  caption?: string;
  tone?: MetricTone;
  help?: string;
  /** Direction of the last change, for a subtle up/down mark. */
  trend?: 'up' | 'down' | 'flat';
  /** Swatch of the curve this number is read off, shown when it is a headline. */
  colour?: string;
}
