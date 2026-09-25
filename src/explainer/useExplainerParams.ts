/** Parameter state for an explainer: defaults, presets, reset and a shareable URL encoding. */

import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';

import type { ControlGroup, Preset } from './types';

export type ParamValue = number | string | boolean;
export type ParamsRecord = Record<string, ParamValue>;

export interface ExplainerParamsApi<P extends ParamsRecord> {
  params: P;
  /** Update a single parameter. */
  set: <K extends keyof P>(key: K, value: P[K]) => void;
  /** Update several at once (used by presets and linked controls). */
  merge: (patch: Partial<P>) => void;
  /** Back to the explainer's defaults. */
  reset: () => void;
  /** Load a named preset. Returns the preset so the caller can autoplay. */
  applyPreset: (id: string) => Preset<P> | undefined;
  /** Id of the preset currently matched exactly, or null once the user deviates. */
  activePresetId: string | null;
  /** The preset most recently matched or applied, kept after the params drift. */
  lastPreset: Preset<P> | null;
  /** True when params differ from the defaults. */
  isDirty: boolean;
  /** A shareable absolute URL carrying the current configuration. */
  shareUrl: string;
  /** Query keys owned by someone else (the lesson) that travel with the params. */
  extras: Record<string, string>;
  /** Set or delete (undefined) extra keys; they share the single URL writer. */
  setExtras: (patch: Record<string, string | undefined>) => void;
  /** True when the URL carried at least one parameter on mount. */
  fromUrl: boolean;
}

function encode(params: ParamsRecord, defaults: ParamsRecord): string {
  const search = new URLSearchParams();
  for (const [key, value] of Object.entries(params)) {
    // Only non-default values travel.
    if (Object.is(value, defaults[key])) continue;
    if (typeof value === 'boolean') search.set(key, value ? '1' : '0');
    else if (typeof value === 'number') search.set(key, String(Number(value.toFixed(6))));
    else search.set(key, value);
  }
  return search.toString();
}

type Limit = { min: number; max: number } | { options: string[] };

/** What each knob allows, so a link can only ask for a setting the controls could reach. */
function limitsOf(groups: readonly ControlGroup[]): Record<string, Limit> {
  const out: Record<string, Limit> = {};
  for (const group of groups) {
    for (const c of group.controls) {
      if (c.kind === 'stepper' || c.kind === 'slider') out[c.key] = { min: c.min, max: c.max };
      else if (c.kind === 'select' || c.kind === 'segmented') out[c.key] = { options: c.options.map((o) => o.value) };
    }
  }
  return out;
}

function decode<P extends ParamsRecord>(query: string, defaults: P, limits: Record<string, Limit>): Partial<P> {
  const search = new URLSearchParams(query);
  const out: Record<string, ParamValue> = {};
  for (const [key, raw] of search.entries()) {
    if (!(key in defaults)) continue;
    const fallback = defaults[key];
    const limit = limits[key];
    if (limit && 'options' in limit && !limit.options.includes(typeof fallback === 'number' ? String(Number(raw)) : raw)) continue;
    if (typeof fallback === 'boolean') out[key] = raw === '1' || raw === 'true';
    else if (typeof fallback === 'number') {
      const n = Number(raw);
      if (!Number.isFinite(n)) continue;
      out[key] = limit && 'min' in limit ? Math.min(limit.max, Math.max(limit.min, n)) : n;
    } else out[key] = raw;
  }
  return out as Partial<P>;
}

function sameParams(a: ParamsRecord, b: ParamsRecord): boolean {
  for (const key of Object.keys(a)) {
    if (!Object.is(a[key], b[key])) return false;
  }
  return true;
}

export function useExplainerParams<P extends ParamsRecord>(
  defaults: P,
  presets: readonly Preset<P>[] = [],
  extraKeys: readonly string[] = [],
  controls: readonly ControlGroup[] = [],
): ExplainerParamsApi<P> {
  const location = useLocation();
  const navigate = useNavigate();

  // Read the URL once on mount; afterwards state pushes to the URL.
  const initial = useMemo(() => {
    const query = location.search.replace(/^\?/, '');
    const fromUrl = decode(query, defaults, limitsOf(controls));
    const search = new URLSearchParams(query);
    const extras: Record<string, string> = {};
    for (const key of extraKeys) {
      const value = search.get(key);
      if (value !== null) extras[key] = value;
    }
    return { params: { ...defaults, ...fromUrl }, extras, fromUrl: Object.keys(fromUrl).length > 0 };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const [params, setParams] = useState<P>(initial.params);
  const [extras, setExtrasState] = useState<Record<string, string>>(initial.extras);
  const defaultsRef = useRef(defaults);

  const setExtras = useCallback((patch: Record<string, string | undefined>) => {
    setExtrasState((prev) => {
      const next = { ...prev };
      for (const [key, value] of Object.entries(patch)) {
        if (value === undefined) delete next[key];
        else next[key] = value;
      }
      const same =
        Object.keys(next).length === Object.keys(prev).length &&
        Object.keys(next).every((key) => prev[key] === next[key]);
      return same ? prev : next;
    });
  }, []);

  const set = useCallback(<K extends keyof P>(key: K, value: P[K]) => {
    setParams((prev) => (Object.is(prev[key], value) ? prev : { ...prev, [key]: value }));
  }, []);

  const merge = useCallback((patch: Partial<P>) => {
    setParams((prev) => ({ ...prev, ...patch }));
  }, []);

  const reset = useCallback(() => {
    setParams({ ...defaultsRef.current });
  }, []);

  const applyPreset = useCallback(
    (id: string) => {
      const preset = presets.find((p) => p.id === id);
      if (!preset) return undefined;
      setParams({ ...defaultsRef.current, ...preset.params });
      return preset;
    },
    [presets],
  );

  // Params first, then the extra keys in their declared order.
  const query = useMemo(() => {
    const search = new URLSearchParams(encode(params, defaultsRef.current));
    for (const key of extraKeys) {
      if (extras[key] !== undefined) search.set(key, extras[key]);
    }
    return search.toString();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [params, extras]);

  // Mirror state into the address bar without adding history entries, once a drag settles:
  // browsers throttle a stream of history rewrites.
  useEffect(() => {
    const target = query ? '?' + query : '';
    if (location.search === target) return undefined;
    const id = window.setTimeout(() => navigate({ pathname: location.pathname, search: target }, { replace: true }), 150);
    return () => window.clearTimeout(id);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [query]);

  const activePresetId = useMemo(() => {
    for (const preset of presets) {
      const candidate = { ...defaultsRef.current, ...preset.params };
      if (sameParams(candidate, params)) return preset.id;
    }
    return null;
  }, [params, presets]);

  // Follows the exact match while there is one and keeps it once the reader edits.
  const [lastPresetId, setLastPresetId] = useState<string | null>(activePresetId);
  useEffect(() => {
    if (activePresetId) setLastPresetId(activePresetId);
  }, [activePresetId]);
  const lastPreset = useMemo(
    () => presets.find((p) => p.id === lastPresetId) ?? null,
    [presets, lastPresetId],
  );

  const isDirty = useMemo(() => !sameParams(defaultsRef.current, params), [params]);

  const shareUrl = useMemo(() => {
    if (typeof window === 'undefined') return '';
    // The address bar already holds the full path, base included.
    const { origin, pathname } = window.location;
    return origin + pathname + (query ? '?' + query : '');
  }, [query]);

  return {
    params,
    set,
    merge,
    reset,
    applyPreset,
    activePresetId,
    lastPreset,
    isDirty,
    shareUrl,
    extras,
    setExtras,
    fromUrl: initial.fromUrl,
  };
}
