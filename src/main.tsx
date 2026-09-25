import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import { RouterProvider } from 'react-router-dom';

import { createAppRouter } from './App';
import { PageSkeleton } from './components/PageSkeleton';
import { ThemeProvider } from './components/ThemeProvider';

import 'katex/dist/katex.min.css';
import './styles/tokens.css';
import './styles/base.css';
import './styles/app.css';
import './styles/explainer.css';
import './styles/studio.css';

const rootElement = document.getElementById('root');
if (!rootElement) throw new Error('#root not found');

// public/404.html bounces GitHub Pages deep links here with the path stashed away.
function restoreDeepLink() {
  let stored: string | null = null;
  try {
    stored = sessionStorage.getItem('mlx-redirect');
    if (stored) sessionStorage.removeItem('mlx-redirect');
  } catch {
    return;
  }
  if (stored && stored.startsWith('/')) window.history.replaceState(null, '', stored);
}

restoreDeepLink();
const router = createAppRouter();

createRoot(rootElement).render(
  <StrictMode>
    <ThemeProvider>
      <RouterProvider
        router={router}
        future={{ v7_startTransition: true }}
        fallbackElement={<PageSkeleton />}
      />
    </ThemeProvider>
  </StrictMode>,
);
