import { useEffect, useState } from 'react';
import { Link, NavLink, Outlet, useLocation, useNavigation } from 'react-router-dom';

import { BLOG_ORIGIN } from '../explainers/registry';
import { IconExternal } from '../explainer/components/Icons';
import { Logo, LogoMark } from './Logo';
import { ThemeToggle } from './ThemeToggle';

/** Scroll to the top on navigation, but leave in-page anchor jumps alone. */
function useScrollReset() {
  const { pathname, hash } = useLocation();
  useEffect(() => {
    if (hash) return;
    window.scrollTo({ top: 0, behavior: 'instant' });
  }, [pathname, hash]);
}

export function AppLayout() {
  useScrollReset();

  return (
    <>
      <a className="mlx-skip-link" href="#main">
        Skip to content
      </a>
      <LoadBar />
      <SiteHeader />
      <main id="main" className="mlx-main">
        <Outlet />
      </main>
      <SiteFooter />
    </>
  );
}

function SiteHeader() {
  // Explainer pages light the catalogue tab.
  const { pathname } = useLocation();
  const inExplainers = pathname === '/' || pathname.startsWith('/explainer');
  return (
    <header className="mlx-header">
      <div className="mlx-header__inner">
        <Link className="mlx-brand" to="/" aria-label="Interactive ML home">
          <Logo size={26} />
        </Link>

        <nav className="mlx-nav" aria-label="Main">
          <Link to="/" className={'mlx-nav__link' + (inExplainers ? ' active' : '')} aria-current={inExplainers ? 'page' : undefined}>
            Explainers
          </Link>
          <NavLink to="/about" className="mlx-nav__link">
            About
          </NavLink>
          <a className="mlx-nav__link mlx-nav__link--external" href={BLOG_ORIGIN} target="_blank" rel="noreferrer">
            Blog
            <IconExternal size={11} />
          </a>
          <ThemeToggle />
        </nav>
      </div>
    </header>
  );
}

/** A thin bar at the top while the next page's chunk loads. Loads under 150 ms never show it. */
function LoadBar() {
  const pending = useNavigation().state === 'loading';
  const [phase, setPhase] = useState<'idle' | 'run' | 'done'>('idle');

  useEffect(() => {
    if (pending) {
      const id = window.setTimeout(() => setPhase('run'), 150);
      return () => window.clearTimeout(id);
    }
    setPhase((current) => (current === 'run' ? 'done' : 'idle'));
    return undefined;
  }, [pending]);

  useEffect(() => {
    if (phase !== 'done') return undefined;
    const id = window.setTimeout(() => setPhase('idle'), 400);
    return () => window.clearTimeout(id);
  }, [phase]);

  if (phase === 'idle') return null;
  return <div className="mlx-loadbar" data-phase={phase} role="progressbar" aria-label="Loading" />;
}

function SiteFooter() {
  return (
    <footer className="mlx-footer">
      <div className="mlx-footer__inner">
        <div className="mlx-footer__brand">
          <LogoMark size={28} />
          <p className="mlx-footer__note">
            © {new Date().getFullYear()} Bharathi Kannan
          </p>
        </div>
        <nav className="mlx-caps mlx-footer__meta" aria-label="Footer">
          <Link to="/about">About</Link>
          <a href={BLOG_ORIGIN} target="_blank" rel="noreferrer">
            Blog
          </a>
          <a href="https://github.com/nbharathik" target="_blank" rel="noreferrer">
            GitHub
          </a>
        </nav>
      </div>
      <p className="mlx-footer__wordmark" aria-hidden="true">
        Interactive ML
      </p>
    </footer>
  );
}
