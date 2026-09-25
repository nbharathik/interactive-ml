import { useEffect } from 'react';
import type { RefObject } from 'react';

const ATTR = 'data-nav';

/** Tints the site header while a band is under it: the whole page, or only while `ref` is in view. */
export function useBandNav(band?: RefObject<HTMLElement>) {
  useEffect(() => {
    const root = document.documentElement;
    const el = band?.current;
    if (!el) {
      root.setAttribute(ATTR, 'band');
      return () => root.removeAttribute(ATTR);
    }
    const header = parseInt(getComputedStyle(root).getPropertyValue('--header-height'), 10) || 0;
    const observer = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting) root.setAttribute(ATTR, 'band');
        else root.removeAttribute(ATTR);
      },
      { rootMargin: '-' + header + 'px 0px 0px 0px' },
    );
    observer.observe(el);
    return () => {
      observer.disconnect();
      root.removeAttribute(ATTR);
    };
  }, [band]);
}
