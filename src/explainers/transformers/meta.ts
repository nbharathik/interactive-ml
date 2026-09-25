/** The parked chapter's metadata. It stays out of the catalog, so the site lists and links it nowhere; only its own URL opens it. */

import type { ExplainerMeta } from '../../explainer/types';

export const META: ExplainerMeta = {
  slug: 'transformers',
  title: 'Attention and transformers',
  tagline: 'Let every token look at every other and decide what matters',
  summary:
    'A tiny transformer reads a small language letter by letter, guesses each next letter and writes whole sentences, the way a large language model writes words. Open every part of it in place, watch each letter decide where to look, and see what the mask, the positions and the feed-forward block are for.',
  status: 'preview',
  difficulty: 'deep',
  concepts: ['Next-token prediction', 'Self-attention', 'Queries, keys, values', 'Positional encoding', 'Causal mask'],
  blogPath: '/transformers/',
  blogStatus: 'draft',
  order: 11,
  symbol: '⇶',
};
