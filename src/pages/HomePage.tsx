/** The front page: hero with a live model, the gallery, the roadmap. */

import { useEffect, useRef } from 'react';
import { Link } from 'react-router-dom';

import { IconArrow } from '../explainer/components/Icons';
import type { ExplainerMeta } from '../explainer/types';
import { preloadOnIdle, warmMathFonts, warmOn } from '../explainers/pages';
import { EXPLAINERS, READY_EXPLAINERS } from '../explainers/registry';
import { Playground } from '../home/Playground';
import { Thumbnail } from '../home/Thumbnail';
import { useBandNav } from '../lib/useBandNav';

const PLANNED = EXPLAINERS.filter((e) => e.status === 'planned').sort((a, b) => a.order - b.order);

export function HomePage() {
  const first = READY_EXPLAINERS[0];
  const hero = useRef<HTMLElement>(null);
  useBandNav(hero);

  // Warm every explainer while the catalog is idle, so a click only has to render.
  useEffect(() => {
    warmMathFonts();
    return preloadOnIdle(READY_EXPLAINERS.map((meta) => meta.slug));
  }, []);

  return (
    <div className="mlx-home">
      <section className="mlx-band mlx-home__hero" ref={hero}>
        <div className="mlx-home__hero-inner">
          <div className="mlx-home__copy">
            <h1 className="mlx-home__title">Interactive Machine Learning</h1>
            <p className="mlx-home__tagline">See algorithms in action</p>
            <p className="mlx-home__lede">
              Visualise, train and inspect machine-learning models as they learn, live in the browser.
            </p>
            {first ? (
              <div className="mlx-home__cta">
                <Link
                  className="mlx-button mlx-button--primary mlx-button--md"
                  to={'/explainer/' + first.slug}
                  {...warmOn(first.slug)}
                >
                  Start with {first.title.toLowerCase()}
                  <IconArrow size={14} />
                </Link>
              </div>
            ) : null}
          </div>
          <Playground />
        </div>
      </section>

      <section
        className="mlx-home__gallery"
        id="explainers"
        aria-labelledby="gallery-title"
      >
        <h2 className="mlx-home__heading" id="gallery-title">
          Explainers
        </h2>
        <ul className="mlx-cards">
          {READY_EXPLAINERS.map((meta) => (
            <li key={meta.slug}>
              <ExplainerCard meta={meta} />
            </li>
          ))}
        </ul>
      </section>

      {PLANNED.length > 0 ? (
        <section className="mlx-home__roadmap" aria-labelledby="roadmap-title">
          <h2 className="mlx-caps mlx-home__label" id="roadmap-title">
            On the roadmap
          </h2>
          <ul className="mlx-roadmap">
            {PLANNED.map((meta) => (
              <li key={meta.slug}>
                <span className="mlx-roadmap__symbol" aria-hidden="true">
                  {meta.symbol}
                </span>
                {meta.title}
              </li>
            ))}
          </ul>
        </section>
      ) : null}
    </div>
  );
}

function ExplainerCard({ meta }: { meta: ExplainerMeta }) {
  return (
    <Link className="mlx-card" to={'/explainer/' + meta.slug} {...warmOn(meta.slug)}>
      <Thumbnail slug={meta.slug} symbol={meta.symbol} alt={meta.title + ': ' + meta.tagline} />
      <div className="mlx-card__body">
        <h3 className="mlx-card__title">{meta.title}</h3>
        <p className="mlx-card__tagline">{meta.tagline}</p>
      </div>
    </Link>
  );
}
