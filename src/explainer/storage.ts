/** Tiny localStorage flags; private mode and quota errors read as unset. */

export function readFlag(key: string): boolean {
  try {
    return localStorage.getItem(key) === '1';
  } catch {
    return false;
  }
}

export function writeFlag(key: string, value: boolean): void {
  try {
    localStorage.setItem(key, value ? '1' : '0');
  } catch {
    // Best effort.
  }
}

/** A small list of ids, for progress marks. */
export function readList(key: string): string[] {
  try {
    const raw = localStorage.getItem(key);
    const parsed: unknown = raw ? JSON.parse(raw) : [];
    return Array.isArray(parsed) ? parsed.filter((v): v is string => typeof v === 'string') : [];
  } catch {
    return [];
  }
}

export function writeList(key: string, list: readonly string[]): void {
  try {
    localStorage.setItem(key, JSON.stringify(list));
  } catch {
    // Best effort.
  }
}
