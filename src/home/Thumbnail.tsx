import { useMemo } from 'react';

import { Chart } from '../explainer/components/Chart';

import { getPreview } from './previews';

const RATIO = 9 / 16;

/** A card's picture: the explainer's real output, or its glyph when there is none yet. */
export function Thumbnail({ slug, symbol, alt }: { slug: string; symbol: string; alt: string }) {
  const preview = useMemo(() => getPreview(slug), [slug]);
  if (!preview) {
    return (
      <span className="mlx-card__plate" aria-hidden="true">
        {symbol}
      </span>
    );
  }
  return (
    <div className="mlx-card__thumb">
      <Chart draw={preview} height={(w) => w * RATIO} description={alt} />
    </div>
  );
}
