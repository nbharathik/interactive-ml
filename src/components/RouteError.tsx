import { Link, useRouteError } from 'react-router-dom';

import { useDocumentTitle } from '../lib/useDocumentTitle';

/** A page that failed to load, usually a chunk that changed under a stale tab after a deploy. */
export function RouteError() {
  const error = useRouteError();
  useDocumentTitle('Something went wrong');
  const detail = error instanceof Error ? error.message : String(error);
  return (
    <div className="mlx-prose-page mlx-prose-page--centred">
      <h1>This page could not load</h1>
      <p className="mlx-prose-page__lede">
        Reloading usually fixes it. If it keeps happening, the site may have been updated
        while this tab was open.
      </p>
      <p className="mlx-route-error__detail">{detail}</p>
      <p>
        <button
          type="button"
          className="mlx-button mlx-button--primary mlx-button--md"
          onClick={() => window.location.reload()}
        >
          Reload
        </button>{' '}
        <Link className="mlx-button mlx-button--md" to="/">
          Back to the catalog
        </Link>
      </p>
    </div>
  );
}
