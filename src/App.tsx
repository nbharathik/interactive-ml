import { Navigate, Route, createBrowserRouter, createRoutesFromElements } from 'react-router-dom';

import { AppLayout } from './components/AppLayout';
import { RouteError } from './components/RouteError';
import { HomePage } from './pages/HomePage';
import { NotFoundPage } from './pages/NotFoundPage';
import { AboutPage } from './pages/AboutPage';
import { PAGE_MODULES } from './explainers/pages';

// Each explainer is its own chunk, loaded before the route renders so the previous page stays up.
const explainerRoutes = Object.entries(PAGE_MODULES).map(([slug, load]) => (
  <Route
    key={slug}
    path={'explainer/' + slug}
    lazy={() => load().then((module) => ({ Component: module.default }))}
  />
));

/** Built after any deep-link restore, so the first match is the page the user asked for. */
export function createAppRouter() {
  return createBrowserRouter(
    createRoutesFromElements(
      <Route element={<AppLayout />}>
        <Route errorElement={<RouteError />}>
          <Route index element={<HomePage />} />
          <Route path="about" element={<AboutPage />} />
          {explainerRoutes}
          {/* Old-style links land on the catalog rather than a dead end. */}
          <Route path="explainer/*" element={<Navigate to="/" replace />} />
          <Route path="*" element={<NotFoundPage />} />
        </Route>
      </Route>,
    ),
    { basename: import.meta.env.BASE_URL, future: { v7_relativeSplatPath: true } },
  );
}
