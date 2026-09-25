import { Link } from 'react-router-dom';

import { useDocumentTitle } from '../lib/useDocumentTitle';

export function NotFoundPage() {
  useDocumentTitle('Not found');
  return (
    <div className="mlx-prose-page mlx-prose-page--centred">
      <p className="mlx-notfound__code">404</p>
      <h1>No explainer at this address</h1>
      <p className="mlx-prose-page__lede">
        The page you asked for is not here. It may have moved, or it may be one of
        the explainers still on the roadmap.
      </p>
      <p>
        <Link className="mlx-button mlx-button--primary mlx-button--md" to="/">
          Back to the catalog
        </Link>
      </p>
    </div>
  );
}
